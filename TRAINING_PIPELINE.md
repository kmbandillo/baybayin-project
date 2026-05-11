# Tesseract Training Pipeline - From Scratch
## Baybayin OCR Model Training Guide

---

## Overview

This pipeline will train a Tesseract OCR model from scratch using your comprehensive Baybayin datasets. The training uses approximately **64,893 images** across multiple styles and types.

### Dataset Summary

| Dataset | Location | Count | Type |
|---------|----------|-------|------|
| Handwritten Characters | `baybayin_dataset/handwritten/char_unbundled/` | 59,942 | Individual character images |
| Handwritten Words | `baybayin_dataset/handwritten/hw_words/` | ~700 | Word images |
| Baybayin Namin | `baybayin_dataset/baybayin_namin/` | 2,481 | Synthetic printed (words/phrases/chars) |
| Tagalog Stylized | `baybayin_dataset/tagalog_stylized/` | 2,470 | Stylized printed (words/phrases/chars) |
| **Total Images** | | **~64,893** | |
| **Wordlist Entries** | `baybayin_dataset/training_wordlist.txt` | **39,329** | Words, characters, phrases |
| **Unique Characters** | From wordlist | **20** | Baybayin unicharset |

### Character Set (20 characters)
- **Vowels (3)**: ᜀ (a), ᜁ (i), ᜂ (u)
- **Consonants (14)**: ᜊ (ba), ᜃ (ka), ᜇ (da), ᜄ (ga), ᜑ (ha), ᜎ (la), ᜋ (ma), ᜈ (na), ᜅ (nga), ᜉ (pa), ᜍ (ra), ᜐ (sa), ᜆ (ta), ᜏ (wa), ᜌ (ya)
- **Diacritics (2)**: ᜒ (kudlit i), ᜓ (kudlit u)
- **Special (1)**: ᜔ (vowel canceller/virama)

---

## Pipeline Stages

### Stage 1: Data Preparation & Organization

#### 1.1 Create Training Directory Structure
```bash
# Create base training directory
mkdir -p tesseract_training_v2
cd tesseract_training_v2

# Clone official repositories
git clone https://github.com/tesseract-ocr/tesstrain.git .
git clone https://github.com/tesseract-ocr/langdata_lstm.git langdata
```

#### 1.2 Set Up Model-Specific Directories
```bash
# Define your model name
MODEL_NAME="baybayin_full"

# Create langdata structure
mkdir -p langdata/${MODEL_NAME}

# Create data directories
mkdir -p data/${MODEL_NAME}-ground-truth
mkdir -p data/${MODEL_NAME}_eval
```

#### 1.3 Consolidate All Ground Truth Data
```bash
# Copy all datasets to central ground truth folder
# Handwritten characters (59,942 images)
cp -r ../baybayin_dataset/handwritten/char_unbundled/* data/${MODEL_NAME}-ground-truth/

# Handwritten words (~700 images)
cp -r ../baybayin_dataset/handwritten/hw_words/* data/${MODEL_NAME}-ground-truth/

# Baybayin Namin synthetic data (2,481 images)
find ../baybayin_dataset/baybayin_namin -name "*.tif" -o -name "*.box" -o -name "*.gt.txt" | \
    xargs -I {} cp {} data/${MODEL_NAME}-ground-truth/

# Tagalog Stylized data (2,470 images)
find ../baybayin_dataset/tagalog_stylized -name "*.tif" -o -name "*.box" -o -name "*.gt.txt" | \
    xargs -I {} cp {} data/${MODEL_NAME}-ground-truth/

# Verify total file count
echo "Total .tif files: $(find data/${MODEL_NAME}-ground-truth -name "*.tif" | wc -l)"
echo "Total .box files: $(find data/${MODEL_NAME}-ground-truth -name "*.box" | wc -l)"
echo "Total .gt.txt files: $(find data/${MODEL_NAME}-ground-truth -name "*.gt.txt" | wc -l)"
```

#### 1.4 Copy Wordlist
```bash
# Copy comprehensive wordlist to langdata
cp ../baybayin_dataset/training_wordlist.txt langdata/${MODEL_NAME}/${MODEL_NAME}.wordlist

# Verify wordlist
echo "Wordlist entries: $(wc -l < langdata/${MODEL_NAME}/${MODEL_NAME}.wordlist)"
```

#### 1.5 Sanitize Ground Truth Files
```bash
# Remove BOM and ensure UTF-8 encoding
cd data/${MODEL_NAME}-ground-truth
for file in *.gt.txt; do
    if [ -f "$file" ]; then
        # Remove BOM and save as UTF-8
        sed '1s/^\xEF\xBB\xBF//' "$file" > "${file}.tmp"
        mv "${file}.tmp" "$file"
    fi
done
cd ../..
```

#### 1.6 Verify Image DPI Settings
```bash
# Set DPI to 300 for all TIFF images (required by Tesseract)
find data/${MODEL_NAME}-ground-truth -name "*.tif" | \
    xargs mogrify -set density 300

# Verify DPI setting
identify -format "%f: %x x %y\n" data/${MODEL_NAME}-ground-truth/*.tif | head -10
```

---

### Stage 2: Generate Training Files (LSTMF)

#### 2.1 Create Training File List
```bash
# Generate list of all training files (without extensions)
cd data/${MODEL_NAME}-ground-truth
ls -1 *.tif | sed 's/\.tif$//' > ../all_gt.list
cd ../..

echo "Total training samples: $(wc -l < data/all_gt.list)"
```

#### 2.2 Split Training and Validation Sets
```bash
# Create training/validation split (90/10)
cd data
total_lines=$(wc -l < all_gt.list)
train_lines=$((total_lines * 9 / 10))

# Shuffle and split
shuf all_gt.list > all_gt_shuffled.list
head -n $train_lines all_gt_shuffled.list > ${MODEL_NAME}_train.list
tail -n +$((train_lines + 1)) all_gt_shuffled.list > ${MODEL_NAME}_eval.list

echo "Training samples: $(wc -l < ${MODEL_NAME}_train.list)"
echo "Validation samples: $(wc -l < ${MODEL_NAME}_eval.list)"
cd ..
```

#### 2.3 Generate LSTMF Files
```bash
# Use tesstrain's built-in mechanism or generate manually
# This converts images + ground truth to LSTM training format

# Option A: Let tesstrain handle it during training (recommended)
# (Skip to Stage 3)

# Option B: Pre-generate LSTMF files manually
make training MODEL_NAME=${MODEL_NAME} \
    START_MODEL=eng \
    TESSDATA=/usr/share/tesseract-ocr/5/tessdata \
    GROUND_TRUTH_DIR=data/${MODEL_NAME}-ground-truth \
    OUTPUT_DIR=data \
    EPOCHS=1 \
    --dry-run  # Check commands first

# Then run without --dry-run
```

---

### Stage 3: Configure Training Parameters

#### 3.1 Create Training Configuration File
```bash
cat > training_config.txt <<'EOF'
# Baybayin Full Model Training Configuration

# Model Settings
MODEL_NAME=baybayin_full
START_MODEL=eng  # Use English LSTM as base
TESSDATA=/usr/share/tesseract-ocr/5/tessdata

# Training Parameters
MAX_ITERATIONS=100000
NET_SPEC="[1,36,0,1 Ct3,3,16 Mp3,3 Lfys48 Lfx96 Lrx96 Lfx256 O1c\${UNICHARSET_SIZE}]"

# Page/segmentation modes
# Dataset composition ≈92% characters, 1% words, 7% phrases.
# Leave PSM unset during training to let `tools/run_lstmf_with_auto_psm.sh`
# pick the right PSM (10 for characters, 8 for words, 7 for phrases/lines).
# Inference tips:
#   * PSM 10 for single characters
#   * PSM 8 for single words
#   * PSM 7 for lines/phrases

# Training rates and stopping criteria
LEARNING_RATE=0.0001
TARGET_ERROR_RATE=0.01

# Dataset settings
RATIO_TRAIN=0.90
WORDLIST_FILE=langdata/${MODEL_NAME}/${MODEL_NAME}.wordlist

# Debugging (keep 0 to suppress ScrollView pop-ups)
DEBUG_INTERVAL=0
EOF

echo "Training configuration created."
```

# Set Environment Variables
```bash
# Export training parameters
export MODEL_NAME="baybayin_full"
export START_MODEL="eng"
export TESSDATA="/usr/share/tesseract-ocr/5/tessdata"
export MAX_ITERATIONS=100000
export RATIO_TRAIN=0.90
export PSM=""           # Blank => auto-select (10/8/7) via helper script
export DEBUG_INTERVAL=0 # Avoid ScrollView UI pop-ups during training
# (Inference) Use PSM 10 for characters, 8 for words, 7 for lines.
```

---

### Stage 4: Run Training

#### 4.1 Initial Training from Scratch
```bash
# Start training using make
# Note: Leave PSM unset to let tools/run_lstmf_with_auto_psm.sh use 10/8/7 heuristics
make training \
    MODEL_NAME=${MODEL_NAME} \
    START_MODEL=${START_MODEL} \
    TESSDATA=${TESSDATA} \
    MAX_ITERATIONS=${MAX_ITERATIONS} \
    RATIO_TRAIN=${RATIO_TRAIN} 2>&1 | tee training_${MODEL_NAME}_$(date +%Y%m%d_%H%M%S).log
```

**Expected Training Time**: 24-72 hours depending on hardware (with 64K+ images)

#### 4.2 Monitor Training Progress
```bash
# Watch training log in real-time
tail -f training_${MODEL_NAME}_*.log

# Check current checkpoint performance
ls -lht data/${MODEL_NAME}/checkpoints/ | head -10

# Extract error rates from log
grep "char train" training_${MODEL_NAME}_*.log | tail -20
```

#### 4.3 Understanding Training Metrics

**Example training output:**
```
At iteration 3476/95900/95904, 
Mean rms=0.231%, 
delta=0.807%, 
char train=3.206%, 
word train=4.2%, 
skip ratio=0%,  
New worst char error = 3.206 
wrote checkpoint.
```

**Metric Explanations:**
- `3476/95900/95904`: Current iteration / Sub-trainer iteration / Total iterations
- `Mean rms=0.231%`: Root mean square error of the learning rate (how stable)
- `delta=0.807%`: Change in error rate (want this decreasing)
- `char train=3.206%`: **Character error rate on training set** (primary metric)
- `word train=4.2%`: Word error rate on training set
- `skip ratio=0%`: Percentage of samples skipped (should stay at 0%)
- `New worst char error`: Checkpoint saved when error is higher than best

**Target Goals:**
- Character error rate: < 1% (excellent), < 2% (good), < 5% (acceptable)
- Word error rate: < 5% (excellent), < 10% (good)
- Training should converge around 50,000-100,000 iterations

**Note on "Epochs":**
Tesseract **does NOT use traditional epochs**. Instead:
- It uses **iterations** where each iteration processes a single sample
- With 64,893 images, one "pseudo-epoch" = 64,893 iterations
- Training for 100,000 iterations ≈ 1.5 "pseudo-epochs"
- The training randomly samples from the dataset each iteration
- This is why you see iteration counts like `3476/95900/95904` (complex internal sub-trainer logic)

**PSM Strategy Recap:**
- ~92% of the dataset are single-character glyphs, ~1% are single words, ~7% are lines/phrases.
- During training, omit the `PSM` flag entirely so tesstrain automatically chooses the best segmentation for each sample.
- During inference, pick `--psm 10` for isolated characters, `--psm 8` for single words, and `--psm 7` for multi-character lines/phrases.

---

### Stage 5: Checkpointing & Model Selection

#### 5.1 Monitor Checkpoints
```bash
# List all checkpoints with error rates
cd data/${MODEL_NAME}/checkpoints
ls -lt *.checkpoint

# Find best checkpoint (lowest error rate)
# Checkpoint names contain error rate: baybayin_full_1.234.checkpoint
ls -1 *.checkpoint | sort -t_ -k3 -n | head -5
```

#### 5.2 Extract Best Model
```bash
# Once training completes or plateaus
# Convert best checkpoint to traineddata
BEST_CHECKPOINT=$(ls -1 data/${MODEL_NAME}/checkpoints/*.checkpoint | sort -t_ -k3 -n | head -1)

lstmtraining \
    --stop_training \
    --continue_from ${BEST_CHECKPOINT} \
    --traineddata data/${MODEL_NAME}/${MODEL_NAME}.traineddata \
    --model_output data/${MODEL_NAME}.traineddata

echo "Final model saved to: data/${MODEL_NAME}.traineddata"
```

---

### Stage 6: Model Evaluation

#### 6.1 Test on Validation Set
```bash
# Create evaluation script
cat > evaluate_model.sh <<'EOF'
#!/bin/bash
MODEL_PATH="$1"
EVAL_LIST="data/${MODEL_NAME}_eval.list"
GT_DIR="data/${MODEL_NAME}-ground-truth"

total=0
errors=0

while IFS= read -r basename; do
    gt_file="${GT_DIR}/${basename}.gt.txt"
    img_file="${GT_DIR}/${basename}.tif"
    
    if [[ ! -f "$gt_file" || ! -f "$img_file" ]]; then
        continue
    fi
    
    # Run OCR (default PSM 10 for characters; set PSM_MODE=8 or 7 for words/lines)
    PSM_MODE="${PSM_MODE:-10}"
    result=$(tesseract "$img_file" stdout --tessdata-dir $(dirname "$MODEL_PATH") \
             -l $(basename "$MODEL_PATH" .traineddata) --psm ${PSM_MODE} 2>/dev/null)
    
    # Compare with ground truth
    expected=$(cat "$gt_file")
    
    if [[ "$result" != "$expected" ]]; then
        ((errors++))
    fi
    ((total++))
    
    if ((total % 100 == 0)); then
        echo "Processed $total samples, errors: $errors"
    fi
done < "$EVAL_LIST"

accuracy=$(echo "scale=4; 100 * (1 - $errors / $total)" | bc)
echo "========================================"
echo "Evaluation Results:"
echo "Total samples: $total"
echo "Errors: $errors"
echo "Accuracy: ${accuracy}%"
echo "========================================"
EOF

chmod +x evaluate_model.sh

# Run evaluation
./evaluate_model.sh data/${MODEL_NAME}.traineddata
```

#### 6.2 Visual Inspection
```bash
# Test on sample images
mkdir -p evaluation_results

# Pick 10 random validation samples
shuf -n 10 data/${MODEL_NAME}_eval.list > sample_eval.list

# Run OCR and save results
while read basename; do
    PSM_FLAG="--psm 10"  # default for single characters
    # Switch to --psm 8 for words or --psm 7 for lines/phrases as needed
    tesseract "data/${MODEL_NAME}-ground-truth/${basename}.tif" \
        "evaluation_results/${basename}" \
        --tessdata-dir data \
        -l ${MODEL_NAME} \
        ${PSM_FLAG}
done < sample_eval.list

# Compare results
echo "Checking sample predictions..."
for file in evaluation_results/*.txt; do
    basename=$(basename "$file" .txt)
    echo "=== $basename ==="
    echo "Ground Truth:"
    cat "data/${MODEL_NAME}-ground-truth/${basename}.gt.txt"
    echo "Prediction:"
    cat "$file"
    echo ""
done
```

---

### Stage 7: Fine-tuning (Optional)

If initial results are not satisfactory:

#### 7.1 Fine-tune from Best Checkpoint
```bash
# Continue training from best checkpoint
make training \
    MODEL_NAME=${MODEL_NAME}_v2 \
    CONTINUE_FROM=data/${MODEL_NAME}/checkpoints/baybayin_full_XXXX.checkpoint \
    MAX_ITERATIONS=150000
```

#### 7.2 Adjust Learning Rate
```bash
# For fine-tuning, use lower learning rate
# Edit training parameters in Makefile or pass directly:
make training \
    MODEL_NAME=${MODEL_NAME}_finetune \
    START_MODEL=data/${MODEL_NAME}.traineddata \
    LEARNING_RATE=0.00001 \
    MAX_ITERATIONS=120000
```

---

### Stage 8: Model Deployment

#### 8.1 Install Model System-Wide
```bash
# Copy to system tessdata directory
sudo cp data/${MODEL_NAME}.traineddata /usr/share/tesseract-ocr/5/tessdata/

# Verify installation
tesseract --list-langs | grep ${MODEL_NAME}
```

#### 8.2 Test Deployment
```bash
# Test on new image
# Choose the PSM that matches your sample (10=char, 8=word, 7=line)
tesseract sample_baybayin.tif output -l ${MODEL_NAME} --psm 10

# View result
cat output.txt
```

#### 8.3 Create Release Package
```bash
# Package model with documentation
mkdir -p releases/baybayin_full_v1.0
cp data/${MODEL_NAME}.traineddata releases/baybayin_full_v1.0/
cp training_${MODEL_NAME}_*.log releases/baybayin_full_v1.0/training.log

# Create README
cat > releases/baybayin_full_v1.0/README.md <<EOF
# Baybayin Full Model v1.0

## Training Details
- **Date**: $(date +%Y-%m-%d)
- **Training Images**: 64,893
- **Wordlist Entries**: 39,329
- **Characters**: 20 Baybayin characters
- **Iterations**: ${MAX_ITERATIONS}
- **Final Character Error Rate**: [Add from log]
- **Final Word Error Rate**: [Add from log]

## Datasets Used
1. Handwritten Characters: 59,942 images
2. Handwritten Words: ~700 images
3. Baybayin Namin Synthetic: 2,481 images
4. Tagalog Stylized: 2,470 images

## Usage
\`\`\`bash
# For single characters:
tesseract input.tif output -l baybayin_full --psm 10

# For single words:
tesseract input.tif output -l baybayin_full --psm 8

# For text lines/phrases:
tesseract input.tif output -l baybayin_full --psm 7

# For automatic detection:
tesseract input.tif output -l baybayin_full
\`\`\`

## Performance
- Validation Accuracy: [Add after evaluation]
- Best suited for: Character recognition, word recognition, short phrases
EOF

# Create archive
tar -czf baybayin_full_v1.0.tar.gz -C releases baybayin_full_v1.0/
echo "Release package created: baybayin_full_v1.0.tar.gz"
```

---

## Training Schedule & Resource Requirements

### Computational Requirements
- **CPU**: 4+ cores recommended (8+ cores ideal)
- **RAM**: 8GB minimum (16GB+ recommended)
- **Storage**: 10GB+ free space
- **Time**: 24-72 hours for 100,000 iterations

### Monitoring Schedule
| Time | Action |
|------|--------|
| Hour 1 | Verify training started, check first 1000 iterations |
| Hour 6 | Check error rates are decreasing |
| Hour 12 | Review first checkpoint, ensure convergence |
| Hour 24 | Evaluate progress, consider adjusting parameters |
| Daily | Monitor error rates, check for plateaus |

### Troubleshooting

**Training not starting:**
- Verify all .tif files have corresponding .box or .gt.txt files
- Check DPI is set to 300
- Ensure START_MODEL exists in TESSDATA directory

**High error rates (>10%):**
- Increase MAX_ITERATIONS (try 150,000 or 200,000)
- Check ground truth quality
- Verify wordlist contains all characters used in training

**Training crashes/OOM:**
- Reduce batch size (edit lstmtraining parameters)
- Use fewer images (create subset)
- Increase system RAM or use swap

**Error rates plateau:**
- Model has converged - extract current checkpoint
- Try fine-tuning with lower learning rate
- Add more diverse training data

---

## Quality Assurance Checklist

### Before Training
- [ ] All images are 300 DPI TIFF format
- [ ] Every .tif has corresponding .box or .gt.txt file
- [ ] Ground truth files are UTF-8 encoded (no BOM)
- [ ] Wordlist contains all characters present in training data
- [ ] Training/validation split is appropriate (90/10)
- [ ] START_MODEL (eng) is available in TESSDATA

### During Training
- [ ] Training log shows decreasing error rates
- [ ] No excessive "skip ratio" warnings
- [ ] Checkpoints are being saved regularly
- [ ] Character error rate < 5% by 50,000 iterations

### After Training
- [ ] Final model file (.traineddata) is created
- [ ] Validation accuracy is acceptable (>90%)
- [ ] Visual inspection shows good recognition
- [ ] Model is documented with training parameters

### Deployment
- [ ] Model installed in system tessdata directory
- [ ] Model appears in `tesseract --list-langs`
- [ ] Test images recognized correctly
- [ ] README documentation is complete

---

## Next Steps

1. **Start with this pipeline** to train your base model
2. **Evaluate performance** on validation set and real-world samples
3. **Iterate** based on results:
   - If good: Deploy and document
   - If poor: Analyze errors and fine-tune
4. **Create specialized models** if needed:
   - Character-only model
   - Word-only model
   - Handwritten-specific model
5. **Build a testing suite** with diverse samples
6. **Version control** your models and track performance

---

## References

- [Tesseract Documentation](https://tesseract-ocr.github.io/)
- [Tesstrain Repository](https://github.com/tesseract-ocr/tesstrain)
- [Training Guide](https://tesseract-ocr.github.io/tessdoc/Training-Tesseract.html)
- Your project README: `/home/kmbandillo/baybayin_project/README.md`
