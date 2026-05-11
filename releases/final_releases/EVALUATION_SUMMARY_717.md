# Baybayin OCR Model Evaluation Summary

## Overview
This document describes the evaluation of the Stage 3 Baybayin Tesseract LSTM checkpoint on the Stage 3 eval list, including the evaluation methodology and results.

---

## 1. Evaluation Dataset

### Dataset Composition
- **Total samples**: 14,033 entries
- **Words**: Not categorized for this run
- **Character-only samples**: Not categorized for this run

The evaluation list (`stage3/eval.list`) was used as provided for Stage 3 validation.

### Data Sources
- **Evaluation list**: `releases/tesseract_training/data/newest_training/stage3/eval.list`

### Excluded Patterns
- Not assessed for this run

---

## 2. Trained Model

**Model Path**: `releases/tesseract_training/data/newest_training/kaggle_stage3/new_stage3_from_new_stage2_balanced0.377_525121.71_7.traineddata`

**Model Details**:
- Stage 3 checkpoint from the new_stage3_from_new_stage2_balanced run
- LSTM-based Baybayin OCR model
- Inference: Uses Tesseract's `lstmeval` tool to compute character error rate (CER) and word error rate (WER)

---

## 3. Evaluation Methodology

### 3.1 Tesseract lstmeval Tool
The evaluation was performed using Tesseract's built-in `lstmeval` command with the following invocation:

```bash
TESSDATA_PREFIX=/home/kmbandillo/baybayin_project/releases/tesseract_training/data \
  lstmeval \
    --model releases/tesseract_training/data/newest_training/kaggle_stage3/checkpoints/new_stage3_from_new_stage2_balanced0.377_525121.71_7.checkpoint \
    --traineddata releases/tesseract_training/data/newest_training/kaggle_stage3/new_stage3_from_new_stage2_balanced0.377_525121.71_7.traineddata \
    --eval_listfile releases/tesseract_training/data/newest_training/stage3/eval.list \
    --verbosity 2
```

**Key Parameters**:
- `--verbosity 2`: Outputs Truth/OCR pairs for both correct and incorrect predictions
- `--eval_listfile`: Points to the Stage 3 eval list file

### 3.2 Error Metrics

#### Character Error Rate (CER)
- **Definition**: Percentage of character-level mismatches across all predictions
- **Formula**: `(Substitutions + Insertions + Deletions) / Total_Characters * 100`
- **Result**: **1.150%** CER

#### Word Error Rate (WER)
- **Definition**: Percentage of word samples that contain at least one character error
- **Formula**: `Incorrect_Words / Total_Words * 100`
- **Result**: **3.135%** WER

---

## 4. Confusion Matrix Generation

### 4.1 Data Source
The evaluation log file (`releases/tesseract_training/data/newest_training/kaggle_stage3/evals/new_stage5/new_stage3_from_new_stage2_balanced0.377_525121.71_7.checkpoint.stage3_eval.log`) contains Truth/OCR pairs for every prediction.

### 4.2 Confusion Matrix Computation
- Not generated for this run

### 4.3 Matrix Dimensions
- Not applicable for this run

---

## 5. Interpretation of Results

### 5.1 Model Performance Summary
| Metric | Value | Assessment |
|--------|-------|------------|
| CER | 1.150% | Excellent character-level accuracy |
| WER | 3.135% | Good word-level accuracy |
| Overall | ~98.9% character accuracy | Strong performance on this eval list |

### 5.2 Top Confusion Patterns (from the confusion matrix)
- Not available (confusion matrix not generated)

### 5.3 Error Categories

Errors are classified into three types based on alignment between Truth and OCR strings:

1. **Substitution-like** (similar lengths or mono-character):
   - Truth and OCR strings have similar character counts
   - Typically single-character replacements (for example, KA to GA)

2. **Deletion-like** (OCR shorter than Truth):
   - Model omits one or more characters
   - Example: "KATA" to "KTA" (missing vowel mark)

3. **Insertion-like** (OCR longer than Truth):
   - Model hallucinated extra characters
   - Example: "KA" to "KAI" (spurious vowel)

---

## 6. Files Generated

### Log Files
- **Log**: `releases/tesseract_training/data/newest_training/kaggle_stage3/evals/new_stage5/new_stage3_from_new_stage2_balanced0.377_525121.71_7.checkpoint.stage3_eval.log`
- Final metrics line: `At iteration 0, stage 0, Eval Char error rate=1.1498238, Word error rate=3.1354664`

### CSV Files (Confusion Matrices)
- Not generated for this run

### Visualization Files (PNG Heatmaps)
- Not generated for this run

---

## 7. How the Numbers Were Computed: Step-by-Step

### 7.1 CER and WER Extraction
1. Tesseract `lstmeval` computes CER and WER during evaluation on the test set
2. These are printed at the end of the log: `Eval Char error rate=X, Word error rate=Y`
3. Parsed and reported as **1.150%** and **3.135%** respectively

### 7.2 Counts Matrix
- Not generated for this run

### 7.3 Percent Matrix
- Not generated for this run

### 7.4 Heatmap Visualization
- Not generated for this run

---

## 8. Quality Assurance

### Validation Checks
- Total count in matrix = Total characters in evaluation log: Not checked
- Each row sums to 100%: Not checked
- Diagonal values are highest: Not checked
- Deletion and Insertion rows/columns are low: Not checked
- CER and WER values are consistent across multiple evaluation runs: Not checked

### Known Limitations
- Font rendering variations may affect certain marks (kudlits)
- Handwritten samples introduce natural variability
- Character-only samples are less ecologically valid than full-word samples (may slightly underestimate real-world performance)

---

## 9. Recommendations for Model Improvement

No confusion matrix was generated for this run, so targeted recommendations are not available. Consider generating a confusion matrix for deeper error analysis.

---

## 10. References

- **Tesseract LSTM Evaluation**: [Tesseract OCR Documentation](https://github.com/UB-Mannheim/tesseract/wiki)
- **Confusion Matrix Standard**: ISO/IEC metrics for OCR quality (row-normalized, per-character breakdown)
- **Baybayin Character Set**: Unicode Block U+1700-U+171F (Tagalog)

---

**Generated**: April 17, 2026  \
**Model**: new_stage3_from_new_stage2_balanced0.377_525121.71_7.traineddata  \
**Evaluation Set**: stage3/eval.list (14,033 samples)  \
**CER**: 1.150% | **WER**: 3.135%
