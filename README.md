# baybayin-project

## Mask Latin text before Baybayin OCR

Use `tools/mask_latin_regions.py` to run the recommended 2-stage pipeline:

1. **Detection (script-agnostic):** EAST finds every text region on the page, regardless of script.
2. **Classification (cheap Latin check):** Each detected crop is fed through a lightweight Tesseract-ENG pass. High-confidence Latin detections are then masked so the Baybayin recognizer never sees them.

### Prerequisites

- Python 3 with OpenCV (`pip install opencv-python`) and NumPy.
- Tesseract CLI (already present for training) accessible in `PATH`.
- EAST model weights (`frozen_east_text_detection.pb`). Download once from the OpenCV model zoo or `https://github.com/opencv/opencv_extra/tree/master/testdata/dnn/text`.

### Example

```bash
python3 tools/mask_latin_regions.py dataset/words/withnoise/kapit.png \
    --east /path/to/frozen_east_text_detection.pb \
    --output dataset/words/withnoise/kapit_masked.png \
    --overlay dataset/words/withnoise/kapit_overlay.png
```

The script prints how many detections were blanked out, writes the masked image, and (optionally) saves an overlay where red boxes were masked and green boxes were kept for Baybayin OCR. Adjust `--latin-threshold`, `--min-ascii-letters`, or `--mask-color` if you need stricter or looser filtering. For dense documents increase `--width/--height` to 960 or 1280 (must be multiples of 32).
