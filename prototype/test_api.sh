#!/usr/bin/env bash
#
# test_api.sh — full test suite for the Retinal Diagnostic API.
#
# Usage:
#   chmod +x test_api.sh
#   ./test_api.sh
#
# Assumes:
#   - The FastAPI server is already running (uvicorn src.api.app:app --port 8000)
#   - Run this from the project root (same level as data/, src/, models/)
#
# Optional but recommended: install jq for proper JSON field checks
#   macOS: brew install jq

set -uo pipefail

API_URL="http://127.0.0.1:8000"
IMAGE_DIR="data/raw/train_images"
CSV_PATH="data/raw/train.csv"
SAMPLE_SIZE=10          # how many images to use for the batch accuracy check
LATENCY_WARN_SECONDS=2  # flag any single prediction slower than this
PASS_COUNT=0
FAIL_COUNT=0
HAS_JQ=0

command -v jq >/dev/null 2>&1 && HAS_JQ=1

green() { printf "\033[32m%s\033[0m\n" "$1"; }
red()   { printf "\033[31m%s\033[0m\n" "$1"; }
blue()  { printf "\033[34m%s\033[0m\n" "$1"; }
yellow(){ printf "\033[33m%s\033[0m\n" "$1"; }

check() {
    local desc="$1" expected="$2" actual="$3"
    if [ "$expected" = "$actual" ]; then
        green "  PASS: $desc (got $actual)"
        PASS_COUNT=$((PASS_COUNT + 1))
    else
        red "  FAIL: $desc (expected $expected, got $actual)"
        FAIL_COUNT=$((FAIL_COUNT + 1))
    fi
}

check_bool() {
    local desc="$1" ok="$2"
    if [ "$ok" = "1" ]; then
        green "  PASS: $desc"
        PASS_COUNT=$((PASS_COUNT + 1))
    else
        red "  FAIL: $desc"
        FAIL_COUNT=$((FAIL_COUNT + 1))
    fi
}

if [ "$HAS_JQ" = "0" ]; then
    yellow "Note: jq not found -- schema/field checks will be skipped. Install with 'brew install jq' for full coverage."
fi

blue "=== 1. Root endpoint ==="
ROOT_RESPONSE=$(curl -s -o /tmp/root_body.json -w "%{http_code}" "$API_URL/")
check "GET / returns 200" "200" "$ROOT_RESPONSE"
cat /tmp/root_body.json
echo ""

blue "=== 2. /predict on a few real images (spot check with true labels) ==="
if [ ! -d "$IMAGE_DIR" ]; then
    red "  Image directory not found at $IMAGE_DIR -- skipping predict tests"
else
    SPOT_IMAGES=$(ls "$IMAGE_DIR"/*.png 2>/dev/null | head -3)
    for IMG in $SPOT_IMAGES; do
        RESPONSE=$(curl -s -o /tmp/predict_body.json -w "%{http_code}" -X POST "$API_URL/predict" -F "file=@$IMG")
        check "POST /predict on $(basename "$IMG")" "200" "$RESPONSE"
        if [ "$RESPONSE" = "200" ]; then
            echo "    $(cat /tmp/predict_body.json)"
            if [ -f "$CSV_PATH" ]; then
                ID_CODE=$(basename "$IMG" .png)
                TRUE_LABEL=$(grep "^$ID_CODE," "$CSV_PATH" | cut -d',' -f2)
                [ -n "$TRUE_LABEL" ] && echo "    (true diagnosis label from train.csv: $TRUE_LABEL)"
            fi
        fi
    done
fi

blue "=== 3. Response schema validation ==="
if [ "$HAS_JQ" = "1" ] && [ -s /tmp/predict_body.json ]; then
    HAS_FILENAME=$(jq 'has("filename")' /tmp/predict_body.json)
    HAS_PRED_CLASS=$(jq 'has("predicted_class")' /tmp/predict_body.json)
    HAS_LABEL=$(jq 'has("diagnosis_label")' /tmp/predict_body.json)
    HAS_CONF=$(jq 'has("confidence")' /tmp/predict_body.json)
    [ "$HAS_FILENAME" = "true" ] && check_bool "response has 'filename' field" 1 || check_bool "response has 'filename' field" 0
    [ "$HAS_PRED_CLASS" = "true" ] && check_bool "response has 'predicted_class' field" 1 || check_bool "response has 'predicted_class' field" 0
    [ "$HAS_LABEL" = "true" ] && check_bool "response has 'diagnosis_label' field" 1 || check_bool "response has 'diagnosis_label' field" 0
    [ "$HAS_CONF" = "true" ] && check_bool "response has 'confidence' field" 1 || check_bool "response has 'confidence' field" 0

    PRED_CLASS=$(jq '.predicted_class' /tmp/predict_body.json)
    CONF=$(jq '.confidence' /tmp/predict_body.json)

    if [[ "$PRED_CLASS" =~ ^[0-4]$ ]]; then
        check_bool "predicted_class ($PRED_CLASS) is in valid range 0-4" 1
    else
        check_bool "predicted_class ($PRED_CLASS) is in valid range 0-4" 0
    fi

    CONF_OK=$(python3 -c "print(1 if 0.0 <= float('$CONF') <= 1.0 else 0)" 2>/dev/null || echo 0)
    check_bool "confidence ($CONF) is between 0.0 and 1.0" "$CONF_OK"
else
    yellow "  Skipped (jq not installed or no prior response to check)"
fi

blue "=== 4. Error path: non-image file ==="
echo "this is not an image" > /tmp/test_not_image.txt
RESPONSE=$(curl -s -o /tmp/error_body.json -w "%{http_code}" -X POST "$API_URL/predict" -F "file=@/tmp/test_not_image.txt")
check "POST /predict with .txt file returns 400" "400" "$RESPONSE"
cat /tmp/error_body.json
echo ""
rm -f /tmp/test_not_image.txt

blue "=== 5. Error path: missing file field ==="
RESPONSE=$(curl -s -o /tmp/error_body2.json -w "%{http_code}" -X POST "$API_URL/predict")
check "POST /predict with no file returns 422" "422" "$RESPONSE"
cat /tmp/error_body2.json
echo ""

blue "=== 6. Error path: empty file ==="
touch /tmp/test_empty.png
RESPONSE=$(curl -s -o /tmp/error_body3.json -w "%{http_code}" -X POST "$API_URL/predict" -F "file=@/tmp/test_empty.png;type=image/png")
if [ "$RESPONSE" = "400" ] || [ "$RESPONSE" = "500" ]; then
    check_bool "empty .png file is rejected cleanly (got $RESPONSE, not a hang or 200)" 1
else
    check_bool "empty .png file is rejected cleanly (got $RESPONSE)" 0
fi
cat /tmp/error_body3.json
echo ""
rm -f /tmp/test_empty.png

blue "=== 7. Idempotency: same image predicted twice gives same result ==="
if [ -n "${SPOT_IMAGES:-}" ]; then
    FIRST_IMAGE=$(echo "$SPOT_IMAGES" | head -1)
    R1=$(curl -s -X POST "$API_URL/predict" -F "file=@$FIRST_IMAGE")
    R2=$(curl -s -X POST "$API_URL/predict" -F "file=@$FIRST_IMAGE")
    if [ "$R1" = "$R2" ]; then
        check_bool "same image produces identical prediction on repeat calls" 1
    else
        check_bool "same image produces identical prediction on repeat calls" 0
        echo "    First:  $R1"
        echo "    Second: $R2"
    fi
else
    yellow "  Skipped (no test image available)"
fi

blue "=== 8. Latency check on a single prediction ==="
if [ -n "${SPOT_IMAGES:-}" ]; then
    FIRST_IMAGE=$(echo "$SPOT_IMAGES" | head -1)
    ELAPSED=$(curl -s -o /dev/null -w "%{time_total}" -X POST "$API_URL/predict" -F "file=@$FIRST_IMAGE")
    SLOW=$(python3 -c "print(1 if float('$ELAPSED') > $LATENCY_WARN_SECONDS else 0)" 2>/dev/null || echo 0)
    if [ "$SLOW" = "0" ]; then
        check_bool "prediction latency (${ELAPSED}s) is under ${LATENCY_WARN_SECONDS}s" 1
    else
        check_bool "prediction latency (${ELAPSED}s) is under ${LATENCY_WARN_SECONDS}s" 0
    fi
else
    yellow "  Skipped (no test image available)"
fi

blue "=== 9. Batch accuracy check against train.csv (sample of $SAMPLE_SIZE images) ==="
if [ -d "$IMAGE_DIR" ] && [ -f "$CSV_PATH" ]; then
    BATCH_IMAGES=$(ls "$IMAGE_DIR"/*.png 2>/dev/null | shuf -n "$SAMPLE_SIZE" 2>/dev/null || ls "$IMAGE_DIR"/*.png | head -"$SAMPLE_SIZE")
    CORRECT=0
    TOTAL=0
    for IMG in $BATCH_IMAGES; do
        ID_CODE=$(basename "$IMG" .png)
        TRUE_LABEL=$(grep "^$ID_CODE," "$CSV_PATH" | cut -d',' -f2)
        [ -z "$TRUE_LABEL" ] && continue
        RESP=$(curl -s -X POST "$API_URL/predict" -F "file=@$IMG")
        if [ "$HAS_JQ" = "1" ]; then
            PRED=$(echo "$RESP" | jq '.predicted_class')
        else
            PRED=$(echo "$RESP" | sed -n 's/.*"predicted_class":\([0-9]*\).*/\1/p')
        fi
        TOTAL=$((TOTAL + 1))
        if [ "$PRED" = "$TRUE_LABEL" ]; then
            CORRECT=$((CORRECT + 1))
        fi
    done
    if [ "$TOTAL" -gt 0 ]; then
        ACC=$(python3 -c "print(f'{$CORRECT/$TOTAL:.2f}')" 2>/dev/null || echo "n/a")
        blue "  Batch accuracy: $CORRECT / $TOTAL correct ($ACC)"
        echo "  (This is a quick live-API sanity check, not a substitute for the full validation report from train.py)"
    else
        yellow "  No matching images/labels found for batch check"
    fi
else
    yellow "  Skipped (image dir or train.csv not found)"
fi

blue "=== 10. Crop consistency check (train vs. inference preprocessing) ==="
if [ -f "check_crop_consistency.py" ]; then
    if [ -n "${SPOT_IMAGES:-}" ]; then
        FIRST_IMAGE=$(echo "$SPOT_IMAGES" | head -1)
        python check_crop_consistency.py "$FIRST_IMAGE"
        if [ $? -eq 0 ]; then
            green "  Crop consistency script ran (check PASS/FAIL output above)"
        else
            red "  Crop consistency script errored out"
        fi
    else
        yellow "  Skipped (no test image available)"
    fi
else
    red "  check_crop_consistency.py not found in project root -- skipping"
fi

echo ""
blue "=== Summary ==="
green "Passed: $PASS_COUNT"
if [ "$FAIL_COUNT" -gt 0 ]; then
    red "Failed: $FAIL_COUNT"
else
    echo "Failed: $FAIL_COUNT"
fi

echo ""
blue "Not automated -- do these manually:"
echo "  - streamlit run app_frontend.py, upload an image, click Analyze"
echo "  - stop the API, click Analyze again, confirm a clean error message (not a crash)"
echo "  - docker compose up --build, then re-run this script against the containerized API"

if [ "$FAIL_COUNT" -gt 0 ]; then
    exit 1
fi
exit 0