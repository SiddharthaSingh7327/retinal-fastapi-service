# Retinal Diagnostic API

A diabetic retinopathy (DR) classification service: a ResNet18 model fine-tuned on
fundus images, served behind a FastAPI endpoint with a Streamlit demo, containerized
with Docker.

Built as a starter project for the VSI Lab.

---

## How it works

```
Streamlit UI  ──upload image──▶  FastAPI /predict  ──▶  preprocess_for_inference()
(app_frontend.py)                (src/api/app.py)        (src/imgutils.py:
                                                            crop + resize with cv2,
                                                            mirrors preprocess.py)
                                                                    │
                                                                    ▼
                                                          ResNet18 (fc → 5 classes)
                                                          models/model.pth
                                                                    │
                                                                    ▼
                                              JSON: {predicted_class, diagnosis_label, confidence}
```

**Offline training pipeline** (run once before the API can serve real predictions):
```
download_data.py → eda.py (optional) → preprocess.py → train.py → models/model.pth
```

**Key design choices:**
- **ResNet18**, not a custom CNN — a pretrained backbone converges fast on a small
  (~250 image) sample, which training from scratch wouldn't.
- **Shared preprocessing (`src/imgutils.py`)** — `preprocess.py` and `app.py` both
  call the same `preprocess_for_inference()` function, so training and inference see
  images processed identically. This matters more than it sounds like (see "Bugs
  found and fixed" below).
- **Class-weighted loss + augmentation** in `train.py` — the dataset is ~50% "No DR,"
  so an unweighted model just learns to guess the majority class.
- **`test_api.sh`** — automated test suite covering schema validation, error
  handling, idempotency, latency, and live batch accuracy.

---

## Project structure

```
retina-dr-api/
├── data/                    # raw/processed images (gitignored, excluded from Docker builds)
├── models/model.pth         # trained weights
├── src/
│   ├── api/app.py           # FastAPI service
│   ├── imgutils.py          # shared crop + resize logic
│   ├── download_data.py, eda.py, preprocess.py, train.py
├── app_frontend.py          # Streamlit demo (project root, not src/)
├── check_crop_consistency.py  # verifies train/inference preprocessing match (project root)
├── test_api.sh
├── Dockerfile, docker-compose.yml, .dockerignore
├── requirements.txt
└── README.md
```
`app_frontend.py` and `check_crop_consistency.py` must stay at the project root —
the Dockerfile, `streamlit run`, and `test_api.sh` all expect them there.

---

## Setup

```bash
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```
Needs `torch`, `torchvision`, `fastapi`, `uvicorn[standard]`, `pydantic`, `pillow`,
`opencv-python`, `pandas`, `numpy`, `scikit-learn`, `streamlit`, `requests`,
`python-multipart`. Also needs a [Kaggle API token](https://www.kaggle.com/docs/api)
(`~/.kaggle/kaggle.json`) for `download_data.py`.

## Run the pipeline
```bash
python src/download_data.py
python src/eda.py            # optional
python src/preprocess.py
python src/train.py          # produces models/model.pth
```

## Run the API

**Locally:**
```bash
uvicorn src.api.app:app --reload --host 0.0.0.0 --port 8000
```
```bash
curl -X POST "http://127.0.0.1:8000/predict" -F "file=@data/raw/train_images/000c1434d8d7.png"
```
```json
{"filename": "000c1434d8d7.png", "predicted_class": 2, "diagnosis_label": "Moderate Diabetic Retinopathy", "confidence": 0.8123}
```

**With Docker** (confirmed working end to end):
```bash
docker compose up --build
```
Starts two containers — `dr_fastapi_backend` (port 8000) and `dr_streamlit_frontend`
(port 8501). `.dockerignore` excludes `data/` to keep builds fast.

## Run the demo
```bash
streamlit run app_frontend.py
```
Override the API target with `API_URL=http://your-host:8000/predict streamlit run app_frontend.py`.

## Testing
```bash
chmod +x test_api.sh
./test_api.sh                 # local: http://127.0.0.1:8000
API_URL="http://127.0.0.1:<port>" ./test_api.sh   # or against Docker

python check_crop_consistency.py data/raw/train_images/<some_id>.png
```

---

## Results

200-train / 50-val split of a 250-image sample. Overall accuracy is misleading here
since the dataset is ~50% "No DR" — per-class recall tells the real story.

| Metric | Baseline | + class weighting & augmentation |
|---|---|---|
| Overall accuracy | 0.76 | 0.66 |
| Class 0 (No DR) recall | 1.00 | 0.81 |
| Class 1 (Mild) recall | 0.25 | 0.75 |
| Class 2 (Moderate) recall | 0.77 | 0.69 |
| Class 3 (Severe) recall | 0.00 | 0.00 |
| Class 4 (Proliferative) recall | 0.25 | 0.00 |

Class weighting substantially improved Class 1 recall but traded off overall
accuracy and Class 0/4 recall — expected on a dataset this small. Class 3 stayed at
zero recall in both configurations (only 3 validation examples). A live batch check
against the running API (10 random images vs. `train.csv`) independently landed
around 0.70–0.80 accuracy across runs, consistent with the `train.py` numbers.

---

## Bugs found and fixed

1. **Train/inference crop mismatch.** `preprocess.py` crops dark borders before
   training; an early API version skipped that crop at inference time, so the model
   saw a different image distribution live than it did during training. Fixed by
   moving the crop logic into `src/imgutils.py`, shared by both.
2. **Validation accuracy bug.** Per-epoch val accuracy didn't match the final
   report — traced to a typo (`p == 1` instead of `p == l`) that silently computed
   the wrong thing. Fixed, and added a manual cross-check as a standing guard.
3. **Resize interpolation mismatch.** Even after fixing #1, `check_crop_consistency.py`
   still showed a real gap (max pixel diff of 65) — `preprocess.py` resizes with
   `cv2.resize`, but the API was resizing with `torchvision`'s PIL-based resize, a
   different interpolation algorithm. Fixed by adding `preprocess_for_inference()`
   to `imgutils.py`, which does crop *and* resize with `cv2`, exactly matching
   `preprocess.py`. `check_crop_consistency.py` now reports a max pixel difference
   of 0.

---

## Known limitations
- Trained on ~250 images, not the full APTOS dataset — a proof of concept, not a
  clinical benchmark.
- Class 3 (Severe) has almost no validation support and near-zero recall so far.
- Not validated against clinical ground truth beyond the Kaggle labels; not intended
  for real diagnostic use.

## Next steps
- Try oversampling minority classes alongside loss weighting.
- Use the full dataset, particularly to get more Class 3 examples.
- Compare full fine-tuning against freezing most of ResNet18 and only training the
  final layer.