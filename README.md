# Computer Vision Project

This is the main branch for our computer vision project.

## Current State

The current direction is a camera-based OMR reader for sheets with 10, 20, 30, 50, or 100 questions.

`omr_bubble_detector.py` contains the production pipeline:

- normalize a phone photo of the sheet
- recover the printed frame or paper contour
- detect the header and answer regions from printed lines
- detect the OMR bubble layout inside those regions
- infer the sheet length from the detected geometry
- read student ID and test ID bubble regions
- grade answers from an editable answer key
- flag blank, multiple, and unclear answers for review

`streamlit_omr_detector.py` is the interactive demo app.

## Presentation Notebooks

Use these notebooks for the project explanation and professor demo:

- `notebooks/01_omr_pipeline_walkthrough.ipynb` explains the full computer-vision pipeline with intermediate plots: raw photo, warped sheet, printed-line evidence, recovered regions, and bubble detections.
- `notebooks/02_grading_and_review_dashboard.ipynb` explains grading, Student ID/Test ID decoding, `fill_score`, answer-key comparison, review flags, and the graded overlay.

Both notebooks use the real camera samples in `samples/` and import the same functions used by the app, so the presentation stays connected to the working implementation.

The sections below document earlier project checkpoints and experiments.

`preprocessing.py` still stores the initial helper functions to preprocess images and run basic computer vision algorithms such as grayscale conversion, Gaussian blur, and Canny edge detection.

The `samples/` folder will store sample images now and later more samples that we collect from datasets.

## Preprocessing Notebook

Use `preprocessing_tests.ipynb` to test the helper functions in `preprocessing.py`.

The notebook uses images from `samples/` and shows the output of:

- resizing
- grayscale conversion
- Gaussian blur
- Canny edge detection
- blank debug image creation

If you use the included environment, select `cv_env` as the notebook kernel.

## Contour Detection Notebook

Use `detecting_counters.ipynb` to test the functions in `counter_detection.py`.

This notebook starts from a sample sheet, runs the preprocessing pipeline to produce a saved Canny image, and then displays:

- all detected contours
- rectangle-like contours
- corner points from the largest rectangle
- reordered corner points for later perspective correction

## Bubble Detection Notebook

Use `detecting_bubbles.ipynb` for the next stage of the OMR pipeline.

This notebook detects the two largest rectangles, warps both sections, and then uses `bubble_detection.py` to find bubble candidates and highlight strongly marked candidates.

Current checkpoint:

- largest rectangle warped
- second largest rectangle warped
- bubble candidates detected inside both rectangles
- simple dark-pixel mark scoring available for the first algorithmic method

## Classical Grading Notebook

Use `classical_grading.ipynb` to test the first full classical grading method.

This stage follows the same general path as the reference paper after bubble detection:

- group detected bubbles into answer rows
- assign answers as `A`, `B`, `C`, or `D`
- flag blank answers as `X`
- flag multiple marks as `M`
- flag unclear answers as `?` for human review
- compare detected answers with an editable answer key

The demo answer key is stored in `answer_keys/demo_100_questions.json`. Replace its values with the real correct answers before using the generated score as an actual grade.

## Phone Image Alignment Notebook

Use `phone_alignment.ipynb` to test the new robustness stage for phone-captured sheets.

This stage uses lecture-style classical computer vision methods:

- illumination normalization
- Canny edge detection
- threshold-based paper segmentation
- morphology for connecting document regions
- contour extraction
- perspective correction

The goal is to detect the paper inside a phone photo, warp it to a frontal view, and then pass the corrected sheet to the existing OMR stages.

## Streamlit Demo

Use `streamlit_omr_detector.py` for the interactive class demo.

The app supports:

- file upload
- phone perspective correction
- printed-region detection
- Student ID and Test ID decoding
- bubble detection, grading, and review flags
- graded overlay export

Run it with:

```bash
source cv_env/bin/activate
streamlit run streamlit_omr_detector.py
```
