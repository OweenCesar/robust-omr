# Computer Vision Project

This is the main branch for our computer vision project.

## Current State

`preprocessing.py` stores the initial main functions to preprocess the data and run basic computer vision algorithms such as grayscale conversion, Gaussian blur, and Canny edge detection.

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
