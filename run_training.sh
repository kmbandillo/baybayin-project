#!/bin/bash
set -euo pipefail

echo "### Starting Baybayin Tesseract 5 Training Pipeline ###"

MODEL_NAME="${MODEL_NAME:-baybayin}"
SOURCE_DATASET="${SOURCE_DATASET:-kaggle_dataset}"
BASE_DIR="${BASE_DIR:-tesseract_training}"
LANGDATA_DIR_REL="$BASE_DIR/langdata"
MAX_ITERATIONS="${MAX_ITERATIONS:-30000}"

# --- 1. Clone Repositories ---
echo "--- 1. Cloning Tesseract training repositories ---"
if [ ! -d "$BASE_DIR" ]; then
    git clone https://github.com/tesseract-ocr/tesstrain.git "$BASE_DIR"
fi
if [ ! -d "$LANGDATA_DIR_REL" ]; then
    git clone https://github.com/tesseract-ocr/langdata_lstm.git "$LANGDATA_DIR_REL"
fi
echo "✅ Repositories are ready."

BASE_DIR="$(realpath "$BASE_DIR")"
LANGDATA_DIR="$BASE_DIR/langdata"
SOURCE_DATASET="$(realpath "$SOURCE_DATASET")"

export TESSDATA_PREFIX="$BASE_DIR/data"
CONFIG_PATH="$TESSDATA_PREFIX/configs/lstm.train"

# --- 2. Download a base 'eng.traineddata' file for the Makefile to use. ---
echo -e "\n--- 2. Downloading base language data ---"
# The tesstrain script needs a base model to start from. We'll put it where it expects it.
mkdir -p "$TESSDATA_PREFIX"
wget -q -O "$TESSDATA_PREFIX/eng.traineddata" https://github.com/tesseract-ocr/tessdata_best/raw/main/eng.traineddata
echo "✅ Base 'eng.traineddata' downloaded."

# Ensure lstm.train config is available for feature extraction
if [ ! -f "$CONFIG_PATH" ]; then
    echo -e "\n--- 2b. Downloading lstm.train config ---"
    mkdir -p "$(dirname "$CONFIG_PATH")"
    curl -sSfL https://raw.githubusercontent.com/tesseract-ocr/tesseract/main/tessdata/configs/lstm.train -o "$CONFIG_PATH"
    echo "✅ lstm.train config saved to $CONFIG_PATH"
fi

# --- 3. Prepare Data ---
echo -e "\n--- 3. Running data preparation script ---"
python3 prepare_data.py --source "$SOURCE_DATASET" --base-dir "$BASE_DIR" --model-name "$MODEL_NAME" --langdata-dir "$LANGDATA_DIR"
echo "✅ Data preparation script finished."

# --- 4. Execute Training ---
echo -e "\n--- 4. Starting the training process (This will take many hours!) ---"
cd "$BASE_DIR"
make training MODEL_NAME="$MODEL_NAME" START_MODEL=eng TESSDATA="$TESSDATA_PREFIX" MAX_ITERATIONS="$MAX_ITERATIONS"
echo "✅ Training command finished."

# --- 5. Locate Final Model ---
echo -e "\n--- 5. Locating final model ---"
FINAL_MODEL_PATH="data/$MODEL_NAME.traineddata"
if [ -f "$FINAL_MODEL_PATH" ]; then
    echo "🎉 EPIC SUCCESS! Your production model is ready at: $BASE_DIR/$FINAL_MODEL_PATH"
    ls -lh "$FINAL_MODEL_PATH"
else
    echo "🛑 Model training may have encountered an error. Check the logs."
fi
