# Legacy Offline Training Pipeline

This folder contains a Jupyter notebook that reproduces the Colab-based Baybayin OCR training flow without relying on any cloud-only conveniences or network access.

## Prerequisites
- Tesseract 4.1+ built with the training tools (`unicharset_extractor`, `lstmtraining`, `combine_lang_model`, etc.)
- `eng.traineddata` (and any other start models) placed in `tesseract_training/data/`
- Python packages available locally: `python-bidi`, `arabic-reshaper`, `Pillow`
- The cleaned dataset at `kaggle_dataset_corrected_full/`

If you need to install Python packages with pip in an offline environment, copy the wheels in advance or use your organisation’s package mirror.

## Notebook
- `legacy_offline_training.ipynb` – step-by-step workflow that:
  1. Uses `prepare_data.py` to sanitise the ground-truth dataset
  2. Generates the aggregated GT corpus and `unicharset`
  3. Runs the tesstrain Makefile for fine-tuning
  4. Summarises training logs

By default the notebook stages a new model named `baybayin_legacy`, so existing experiments remain untouched. Adjust the constants in the “Configure Paths and Hyperparameters” cell to reuse a previous run.
