#!/bin/bash
set -euo pipefail

echo "### Starting Baybayin LSTM Training ###"

MODEL_NAME="${MODEL_NAME:-baybayin}"
SOURCE_DATASET="${SOURCE_DATASET:-kaggle_dataset}"
BASE_DIR="${BASE_DIR:-tesseract_training}"
LANGDATA_DIR_REL="$BASE_DIR/langdata"
EPOCHS="${EPOCHS:-10}"

# --- 1. Repositories ---
echo "--- Checking repositories ---"
if [ ! -d "$BASE_DIR" ]; then
    git clone https://github.com/tesseract-ocr/tesstrain.git "$BASE_DIR"
fi
if [ ! -d "$LANGDATA_DIR_REL" ]; then
    git clone https://github.com/tesseract-ocr/langdata_lstm.git "$LANGDATA_DIR_REL"
fi
echo "✅ Repositories ready."

BASE_DIR="$(realpath "$BASE_DIR")"
LANGDATA_DIR="$BASE_DIR/langdata"
SOURCE_DATASET="$(realpath "$SOURCE_DATASET")"
export TESSDATA_PREFIX="$BASE_DIR/data"

# --- 2. Data preparation ---
echo -e "\n--- Preparing dataset ---"
python3 prepare_data.py --source "$SOURCE_DATASET" \
    --base-dir "$BASE_DIR" \
    --model-name "$MODEL_NAME" \
    --langdata-dir "$LANGDATA_DIR"
echo "✅ Data prepared."

# --- 3. Check if unicharset exists ---
UNI_FILE="$BASE_DIR/data/$MODEL_NAME/unicharset"
if [ ! -f "$UNI_FILE" ]; then
    echo "❌ Unicharset not found: $UNI_FILE"
    echo "Run 'make unicharset_only' before training."
    exit 1
fi
echo "✅ Found unicharset: $UNI_FILE"

# --- 4. Start Training ---
echo -e "\n--- Starting training for $EPOCHS epochs ---"
cd "$BASE_DIR"
make training MODEL_NAME="$MODEL_NAME" START_MODEL=eng EPOCHS="$EPOCHS"
echo "✅ Training finished."

# --- 5. Export final model ---
echo -e "\n--- Exporting trained model ---"
make traineddata MODEL_NAME="$MODEL_NAME"
FINAL_MODEL="$BASE_DIR/data/$MODEL_NAME.traineddata"

if [ -f "$FINAL_MODEL" ]; then
    echo "🎉 Model ready: $FINAL_MODEL"
    ls -lh "$FINAL_MODEL"
else
    echo "🛑 Model export failed. Check logs."
fi
