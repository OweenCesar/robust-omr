# robust-omr

Computer vision utilities for reading and grading bubble answer sheets from an image.

The pipeline now does four things:

1. Detects the four black registration squares on the sheet.
2. Warps the photo into the template coordinate system.
3. Reads the inner area of each answer bubble.
4. Grades the detected answers against `answer_key.json`.

## Setup

Create and activate a virtual environment, then install the dependencies:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

This repo also includes a `cv_env/` environment. If you use that one, run commands with `cv_env/bin/python`.

## Run

Grade the bundled sample sheet:

```bash
python src/main.py
```

Or pass your own photo:

```bash
python src/main.py \
  --image data/raw/student_sheet.jpg \
  --template templates/template_20q_abcd.json \
  --answer-key answer_key.json \
  --output-dir outputs
```

With the included environment:

```bash
cv_env/bin/python src/main.py --image data/raw/student_sheet.jpg
```

## Outputs

After each run, check:

- `outputs/results.json` for the machine-readable report.
- `outputs/warped/warped_sheet.png` for the aligned sheet.
- `outputs/debug/thresholded.png` for the thresholded image used by the bubble reader.
- `outputs/annotated/annotated_result.png` for visual verification.

## Human Review

The grader separates confident answers from answers that need a human check.
If a student marks more than one bubble, or the mark is too weak/ambiguous, the console and `results.json` include a review item such as:

```text
Q7: multiple marks detected (A, C) - please review manually.
Q12: weak mark detected near B - please review manually.
```

In `outputs/results.json`, look at the `review` section for all manual-review items.

## Template Format

Templates live in `templates/`. A template defines the warped page size, answer options, bubble radius, answer blocks, thresholds, and registration marker positions.

For a new sheet layout, create a new JSON file and update:

- `warped_width` and `warped_height`: normalized page size used after alignment.
- `corner_markers`: where the four black square marker centers should land after warping.
- `bubble_blocks`: one block per column/section of questions.
- `bubble_radius`: radius in warped-template pixels.
- `thresholds`: tune only if real student marks are being missed or false positives appear.

The current `template_20q_abcd.json` describes a 20-question, A-D sheet with two columns.

## Photo Tips

- Keep all four black corner squares visible.
- Use a dark pen and fill one bubble per question.
- Avoid heavy shadows, motion blur, and cropped page edges.
- If the image is already perfectly in template coordinates, use `--no-align`.
