#!/usr/bin/env bash
#
# test_api.sh — runs through the manual test checklist automatically.
#
# Usage:
#   chmod +x test_api.sh
#   ./test_api.sh
#
# Assumes:
#   - The FastAPI server is already running (uvicorn src.api.app:app --port 8000)
#   - Run this from the project root (same level as data/, src/, models/)

set -uo pipefail

API_URL="http://127.0.0.1:8000"
IMAGE_DIR="data/raw/train_images"
CSV_PATH="data/raw/train.csv"
PASS_COUNT=0
FAIL_COUNT=0

green() { printf "\033[32m%s\033[0m\n" "$1"; }
red()   { printf "\033[31m%s\033[0m\n" "$1"; }
blue()  { printf "\033[34m%s\033[0m\n" "$1"; }

check() {
    # check "description" <expected_status> <actual_status>
    local desc="$1" expected="$2" actual="$3"
    if [ "$expected" = "$actual" ]; then
        green "  PASS: $desc (got $actual)"
        PASS_COUNT=$((PASS_COUNT + 1))
    else
        red "  FAIL: $desc (expected $expected, got $actual)"
        FAIL_COUNT=$((FAIL_COUNT + 1))
    fi
}

blue "=== 1. Root endpoint ==="
ROOT_RESPONSE=$(curl -s -o /tmp/root_body.json -w "%{http_code}" "$API_URL/")
check "GET / returns 200" "200" "$ROOT_RESPONSE"
cat /tmp/root_body.json
echo ""

blue "=== 2. /predict on real images (spanning a few IDs) ==="
if [ ! -d "$IMAGE_DIR" ]; then
    red "  Image directory not found at $IMAGE_DIR -- skipping predict tests"
else
    # Grab up to 3 real image files to test against
    IMAGES=$(ls "$IMAGE_DIR"/*.png 2>/dev/null | head -3)
    if [ -z "$IMAGES" ]; then
        red "  No .png files found in $IMAGE_DIR -- skipping predict tests"
    else
        for IMG in $IMAGES; do
            RESPONSE=$(curl -s -o /tmp/predict_body.json -w "%{http_code}" -X POST "$API_URL/predict" -F "file=@$IMG")
            check "POST /predict on $(basename "$IMG")" "200" "$RESPONSE"
            if [ "$RESPONSE" = "200" ]; then
                echo "    $(cat /tmp/predict_body.json)"
                # Cross-reference true label from CSV if available
                if [ -f "$CSV_PATH" ]; then
                    ID_CODE=$(basename "$IMG" .png)
                    TRUE_LABEL=$(grep "^$ID_CODE," "$CSV_PATH" | cut -d',' -f2)
                    if [ -n "$TRUE_LABEL" ]; then
                        echo "    (true diagnosis label from train.csv: $TRUE_LABEL)"
                    fi
                fi
            fi
        done
    fi
fi

blue "=== 3. Error path: non-image file ==="
echo "this is not an image" > /tmp/test_not_image.txt
RESPONSE=$(curl -s -o /tmp/error_body.json -w "%{http_code}" -X POST "$API_URL/predict" -F "file=@/tmp/test_not_image.txt")
check "POST /predict with .txt file returns 400" "400" "$RESPONSE"
cat /tmp/error_body.json
echo ""
rm -f /tmp/test_not_image.txt

blue "=== 4. Error path: missing file field ==="
RESPONSE=$(curl -s -o /tmp/error_body2.json -w "%{http_code}" -X POST "$API_URL/predict")
check "POST /predict with no file returns 422" "422" "$RESPONSE"
cat /tmp/error_body2.json
echo ""

blue "=== 5. Crop consistency check (train vs. inference preprocessing) ==="
if [ -f "check_crop_consistency.py" ]; then
    if [ -n "${IMAGES:-}" ]; then
        FIRST_IMAGE=$(echo "$IMAGES" | head -1)
        python check_crop_consistency.py "$FIRST_IMAGE"
        if [ $? -eq 0 ]; then
            green "  Crop consistency script ran (check PASS/FAIL output above)"
        else
            red "  Crop consistency script errored out"
        fi
    else
        red "  No test image available to run crop consistency check"
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