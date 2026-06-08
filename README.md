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
