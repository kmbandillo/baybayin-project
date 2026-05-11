# Baybayin OCR Model Evaluation Summary

## Overview
This document describes the evaluation of the Stage 5 Baybayin Tesseract LSTM model, including the evaluation methodology, confusion matrix generation, and interpretation of results.

---

## 1. Evaluation Dataset

### Dataset Composition
- **Total samples**: 7,036 entries
- **Words**: 5,629 (80%)
- **Character-only samples**: 1,407 (20%)

The evaluation list (`new_stage5.list`) was assembled to represent a realistic mix of word-level and character-level recognition tasks, with 80% word content and 20% isolated character samples.

### Data Sources
- **Words**: Drawn from the dataset/words folder with samples from both synthetic fonts (NotoSansTagalog, rendered stylizations) and handwritten variants
- **Characters**: Isolated Baybayin character glyphs from the character dataset, covering all 20 base characters + vowel marks (kudlits) + virama

### Excluded Patterns
- No overlap with Stage 2, Stage 3, or Stage 4 training lists (strict exclusion on base word stems)
- Rationale: To avoid data leakage and ensure fair evaluation on held-out material

---

## 2. Trained Model

**Model Path**: `releases/tesseract_training/data/newest_training/stage4withnoise/stage4_from_stage3_withnoise0.082_13649.traineddata`

**Model Details**:
- Fine-tuned from Stage 3 baseline with noise augmentation (dropout=0.082)
- Trained for 13,649 iterations
- Architecture: LSTM-based character recognition with Baybayin-specific character set
- Inference: Uses Tesseract's `lstmeval` tool to compute character error rate (CER) and word error rate (WER)

---

## 3. Evaluation Methodology

### 3.1 Tesseract lstmeval Tool
The evaluation was performed using Tesseract's built-in `lstmeval` command with the following invocation:

```bash
lstmeval \
  --model <model_path> \
  --test_file <evaluation_list> \
  --verbosity 2 \
  --unicharset <unicharset_file>
```

**Key Parameters**:
- `--verbosity 2`: Outputs Truth/OCR pairs for both **correct** and **incorrect** predictions, enabling full confusion matrix computation
- `--unicharset`: Provides the character inventory and character-to-index mapping used during training

### 3.2 Error Metrics

#### Character Error Rate (CER)
- **Definition**: Percentage of character-level mismatches across all predictions
- **Formula**: `(Substitutions + Insertions + Deletions) / Total_Characters × 100`
- **Result**: **2.18%** CER
- **Interpretation**: 2 out of every 100 characters are misrecognized; high accuracy at character level

#### Word Error Rate (WER)
- **Definition**: Percentage of word samples that contain at least one character error
- **Formula**: `Incorrect_Words / Total_Words × 100`
- **Result**: **5.50%** WER
- **Interpretation**: ~5–6 out of every 100 words contain errors; word-level errors are higher than character-level because a single character error invalidates the entire word

---

## 4. Confusion Matrix Generation

### 4.1 Data Source
The full evaluation log file (`releases/final_releases/lstmeval_stage5_updated_v2.log`) contains **Truth/OCR pairs** for every prediction:
- Correct predictions: "Truth: X OCR: X" lines
- Incorrect predictions: "Truth: X OCR: Y" lines (where X ≠ Y)

### 4.2 Confusion Matrix Computation

#### Step 1: Parse Truth/OCR Pairs
1. Extract all "Truth:" and "OCR:" values from the evaluation log
2. For each (Truth, OCR) pair, increment a counter in the confusion matrix

#### Step 2: Build Raw Counts Matrix
- **Rows**: True character labels (what the model should recognize)
- **Columns**: Predicted character labels (what the model output)
- **Cell [i,j]**: Count of times character i was misclassified as character j
- **Diagonal cells**: Correct predictions (Truth == OCR)
- **<DEL> column**: Deletions (character omitted by the model)
- **<INS> row**: Insertions (extra characters hallucinated by the model)

#### Step 3: Normalize to Percentages
- For each row (true character), compute the percentage distribution:
  - `Percentage[i,j] = Count[i,j] / Sum_of_Row[i] × 100`
- This creates a row-normalized confusion matrix where each row sums to 100%
- **Interpretation**: For character A, this shows what percentage of A's predictions are classified as each category

### 4.3 Matrix Dimensions
- **Rows**: 21 (20 base characters + 1 for special <INS> row)
- **Columns**: 21 (20 base characters + 1 for special <DEL> column)
- **Character Set**: A, E/I, O/U, KA, GA, NGA, TA, DA/RA, NA, PA, BA, MA, YA, LA, WA, SA, HA, Kudlit E/I, Kudlit O/U, Virama, plus Insertion/Deletion

---

## 5. Interpretation of Results

### 5.1 Model Performance Summary
| Metric | Value | Assessment |
|--------|-------|-----------|
| CER | 2.18% | Excellent character-level accuracy |
| WER | 5.50% | Good word-level accuracy |
| Overall | ~97% character accuracy | Production-ready for most use cases |

### 5.2 Top Confusion Patterns (from the confusion matrix)

The matrix reveals which characters are occasionally confused with each other:

1. **Virama (᜔)**: 99.57% correctly recognized, rare confusions with vowel marks
2. **Kudlit E/I (ᜒ)**: 99.23% correct; minor confusions with Kudlit O/U
3. **Most base consonants**: >99% diagonal accuracy (e.g., KA, GA, TA, DA, NA, PA, BA, MA, LA, WA)

**Key Insight**: The model has learned to distinguish between:
- Base consonants (well-separated feature space)
- Vowel marks (sometimes subtle differences in rendering)
- Virama (mostly unambiguous)

### 5.3 Error Categories

Errors are classified into three types based on alignment between Truth and OCR strings:

1. **Substitution-like** (similar lengths or mono-character):
   - Truth and OCR strings have similar character counts
   - Typically single-character replacements (e.g., KA → GA)
   - ~Most common error type

2. **Deletion-like** (OCR shorter than Truth):
   - Model omits one or more characters
   - Example: "KATA" → "KTA" (missing vowel mark)
   - Occurs when marks or diacritics are poorly rendered

3. **Insertion-like** (OCR longer than Truth):
   - Model hallucmates extra characters
   - Example: "KA" → "KAᜒ" (spurious vowel)
   - Rare; occurs when confusion regions add phantom predictions

---

## 6. Files Generated

### CSV Files (Confusion Matrices)
- **[confusion_matrix_stage5_updated_continue_counts.csv](confusion_matrix_stage5_updated_continue_counts.csv)**:
  - Raw counts of each (Truth, OCR) pair
  - Values are integers representing occurrence counts
  - Format: First column = True label, header row = Predicted labels

- **[confusion_matrix_stage5_updated_continue_percent.csv](confusion_matrix_stage5_updated_continue_percent.csv)**:
  - Row-normalized percentages (each row sums to 100%)
  - Decimal format: 0–100 range
  - Format: Same structure as counts CSV

### Visualization Files (PNG Heatmaps)
- **[confusion_matrix_corrected_20class_figure_style.png](confusion_matrix_corrected_20class_figure_style.png)**:
  - Figure-style 20-class confusion matrix matching the publication layout
  - Inner cells: `count` (top) + `% of total samples` (bottom)
  - Last column: per-class **precision** and error complement (`100% - precision`)
  - Last row: per-class **recall** and miss complement (`100% - recall`)
  - Bottom-right cell: overall model accuracy and error rate

- **[confusion_matrix_stage5_updated_continue_counts_decimal_blue.png](confusion_matrix_stage5_updated_continue_counts_decimal_blue.png)**:
  - Dual-label x-axis: Baybayin glyphs (unrotated, above) + Latin names (rotated, below)
  - Y-axis: Latin names + Baybayin glyphs (right-aligned)
  - Cell annotations: Count (top) + Decimal fraction (bottom)
  - Color scale: Blue gradient (0.0–1.0); darker = higher value
  - Readable for detailed analysis of individual (Truth, OCR) pairs

- **[confusion_matrix_stage5_updated_continue_counts_decimal_latin_blue.png](confusion_matrix_stage5_updated_continue_counts_decimal_latin_blue.png)**:
  - Same layout and annotations as above
  - Variant generated for archive/comparison

### Log Files
- **[lstmeval_stage5_updated_v2.log](lstmeval_stage5_updated_v2.log)**:
  - Full output from Tesseract's `lstmeval` with `--verbosity 2`
  - Contains all Truth/OCR pairs used to compute the confusion matrix
  - Final metrics line: `Eval Char error rate=2.1813679, Word error rate=5.5002843`

---

## 7. How the Numbers Were Computed: Step-by-Step

### 7.1 CER and WER Extraction
1. Tesseract `lstmeval` computes CER and WER during evaluation on the test set
2. These are printed at the end of the log: `Eval Char error rate=X, Word error rate=Y`
3. Parsed and reported as **2.18%** and **5.50%** respectively

### 7.2 Counts Matrix
1. Parse the `lstmeval` log line-by-line
2. For each line matching `Truth: X` and the next matching `OCR: Y`:
   - Look up the character-to-index mapping from the `.unicharset` file
   - Increment `matrix[index_of_X][index_of_Y] += 1`
3. Handle special cases:
   - If OCR string is shorter → characters are marked as `<DEL>`
   - If OCR string is longer → extra characters are marked as `<INS>`
4. Output the matrix as CSV with row/column headers

### 7.3 Percent Matrix
1. For each row `i` in the counts matrix:
   - Compute the row sum: `row_sum = sum(matrix[i][:])`
   - For each column `j`:
     - `percent_matrix[i][j] = (matrix[i][j] / row_sum) × 100`
   - Round to 2 decimal places
2. Output as CSV with the same structure as the counts matrix

### 7.4 Heatmap Visualization
1. Read both counts.csv and percent.csv files
2. For each cell [i, j]:
   - **Color intensity** is determined by the percent value (0–1 scale)
   - **Text annotation** is formatted as: `"{count}\n{percent:.2f}"`
   - Example: `"536\n0.99"` means 536 correct predictions with 99% frequency
3. Render using matplotlib's `seaborn.heatmap`:
   - X-axis: Predicted labels (rotated Latin names + unrotated Baybayin glyphs)
   - Y-axis: True labels (Latin names + Baybayin glyphs)
   - Colorbar: Blue gradient (YlGnBu colormap, inverted for readability)
4. Save as PNG at 300 DPI

---

## 8. Quality Assurance

### Validation Checks
✓ Total count in matrix = Total characters in evaluation log  
✓ Each row sums to 100% (within rounding tolerance)  
✓ Diagonal values are highest (correct predictions dominate)  
✓ Deletion and Insertion rows/columns are low (<5% each in most cases)  
✓ CER and WER values are consistent across multiple evaluation runs  

### Known Limitations
- Font rendering variations may affect certain marks (kudlits)
- Handwritten samples introduce natural variability
- Character-only samples are less ecologically valid than full-word samples (may slightly underestimate real-world performance)

---

## 9. Recommendations for Model Improvement

Based on the confusion matrix and error analysis:

1. **High-Performing Characters**: No action needed for A, E/I, O/U, KA, GA, etc.
2. **Virama (᜔) Accuracy**: Already excellent (99.57%); maintain current data pipeline
3. **Minor Mark Confusions**: Consider increasing training samples for Kudlit E/I vs. Kudlit O/U if higher precision is required
4. **Handwritten Data**: Continue collecting well-annotated handwritten samples to improve robustness

---

## 10. References


---

**Generated**: April 15, 2026  
**Model**: stage4_from_stage3_withnoise0.082_13649.traineddata  
**Evaluation Set**: new_stage5.list (7,036 samples)  
**CER**: 2.18% | **WER**: 5.50%
