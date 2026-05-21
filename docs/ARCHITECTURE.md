# Robust OMR Architecture

This document explains how the webapp files and functions work together. The
Graphviz source diagram is in [app_architecture.dot](app_architecture.dot).

To render the diagram after installing Graphviz:

```powershell
dot -Tsvg docs\app_architecture.dot -o docs\app_architecture.svg
```

## High-Level Flow

1. The teacher opens the Flask webapp in a browser.
2. The teacher creates a test and answer-key variation.
3. The app stores the test, question text, and answer key in SQLite.
4. The teacher prints the SVG sheet generated from `sheet_layout.py`.
5. The student marks the printed sheet.
6. The teacher scans the completed sheet from the phone.
7. JavaScript optionally compresses the image before upload.
8. Flask sends the image bytes to the OpenCV pipeline.
9. OpenCV detects the four ArUco markers and warps the photo into the fixed sheet coordinate system.
10. The scanner reads answer boxes, interprets answers, grades the sheet, and creates an annotated image.
11. The teacher reviews detected answers and saves the final result.
12. Results are stored in SQLite and can be exported as CSV.

## Files And Responsibilities

| File | Responsibility |
| --- | --- |
| `app.py` | Flask application, routes, form parsing, scan preview/save flow, CSV export. |
| `src/db.py` | SQLite schema, migrations, test storage, variation storage, result storage. |
| `src/sheet_layout.py` | Single source of truth for sheet size, marker IDs, marker images, answer-box coordinates. |
| `src/omr_engine.py` | High-level OMR pipeline that accepts image bytes and returns score, details, warnings, and annotated image. |
| `src/preprocessing.py` | Image decode, grayscale conversion, blur/brightness checks, ArUco detection, perspective warp, thresholding. |
| `src/bubble_reader.py` | Measures square answer boxes and interprets them as valid, blank, multiple, or unclear. |
| `src/grader.py` | Compares interpreted answers to the saved answer key and returns score summaries. |
| `src/visualization.py` | Draws the colored debug overlay on the warped sheet. |
| `templates/*.html` | Jinja pages for create, print, scan, review, and results. |
| `static/app.js` | Browser behavior: question-row visibility, camera capture, optional image compression. |
| `static/styles.css` | Simple mobile-first UI styles and print styles. |
| `requirements.txt` | Python packages required by the app. |

## `app.py` Route Functions

| Function | Route | What it does |
| --- | --- | --- |
| `create_app()` | app factory | Configures Flask, initializes SQLite, creates runtime folders, registers routes. |
| `index()` | `/` | Shows the home page with recent tests and recent results. |
| `tests_index()` | `/tests` | Lists saved tests. |
| `new_test()` | `/tests/new` | Validates the create-test form and saves a test plus its first variation. |
| `test_detail()` | `/tests/<test_id>` | Shows variations and saved question text for one test. |
| `new_variation()` | `/tests/<test_id>/variations/new` | Adds another answer-key version to an existing test. |
| `print_sheet()` | `/tests/<test_id>/variations/<variation_id>/sheet` | Builds sheet layout data and renders the printable SVG answer sheet. |
| `scan()` | `/scan` | Shows the phone-friendly scan form. |
| `scan_preview()` | `/scan/preview` | Receives image upload, runs OMR, saves temporary image files, and shows review page. |
| `scan_save()` | `/scan/save` | Saves teacher-confirmed answers and final score. |
| `results_index()` | `/results` | Shows saved scan results. |
| `results_csv()` | `/results.csv` | Exports saved results as CSV. |
| `media_file()` | `/media/<kind>/<filename>` | Serves uploaded and annotated scan images. |

## `src/sheet_layout.py` Functions

| Function | What it does |
| --- | --- |
| `validate_question_count()` | Ensures tests stay within the supported 10-40 question range. |
| `get_columns()` | Splits questions into one or two printed columns. |
| `question_y()` | Calculates a question row's Y coordinate in the fixed 1000 x 1400 sheet. |
| `generate_answer_box_centers()` | Produces the exact answer-box coordinates used by OpenCV. |
| `marker_rects()` | Produces marker rectangles and embedded marker images for the SVG template. |
| `marker_corner_points()` | Produces destination marker-corner coordinates for perspective correction. |
| `marker_id_to_name()` | Maps detected ArUco IDs back to sheet corners. |
| `aruco_marker_data_url()` | Generates embedded PNG marker data URLs for the printable SVG sheet. |
| `sheet_context()` | Packages all printable sheet geometry for `templates/sheet.html`. |

## `src/omr_engine.py` Functions

| Function | What it does |
| --- | --- |
| `_quality_warnings()` | Converts blur and brightness numbers into teacher-friendly warnings. |
| `_normalize_answer_key()` | Converts answer-key keys to strings and keeps only the selected question range. |
| `process_scan()` | Runs the full image-processing and grading pipeline for one uploaded sheet. |
| `build_manual_interpretation()` | Converts teacher-edited final answers into the same shape used by the grader. |

## `src/preprocessing.py` Functions

| Function | What it does |
| --- | --- |
| `load_image_from_bytes()` | Decodes uploaded JPEG/PNG bytes into an OpenCV image. |
| `convert_to_gray()` | Converts OpenCV BGR images to grayscale. |
| `threshold_dark_marks()` | Converts dark ink to white pixels on a black background for mark measurement. |
| `compute_blur_score()` | Uses Laplacian variance to estimate blur. |
| `compute_brightness()` | Computes average grayscale brightness. |
| `detect_corner_markers()` | Detects the four expected ArUco marker IDs in the uploaded image. |
| `warp_to_sheet()` | Uses marker corners to warp an angled phone photo into the fixed sheet coordinate system. |

## `src/bubble_reader.py` Functions

| Function | What it does |
| --- | --- |
| `create_rectangular_mask()` | Builds the inner measurement region for one square answer box. |
| `read_single_box()` | Measures the ratio of dark pixels inside one answer box. |
| `read_all_boxes()` | Measures every answer option for every question. |
| `interpret_answers()` | Converts raw mark ratios into valid, blank, multiple, or unclear answer statuses. |

## `src/db.py` Functions

| Function | What it does |
| --- | --- |
| `ensure_storage_dirs()` | Creates `data/` and `data/uploads/` if missing. |
| `get_connection()` | Opens a SQLite connection with dictionary-like rows. |
| `init_db()` | Creates or migrates the SQLite schema. |
| `_ensure_column()` | Adds missing columns to older local databases. |
| `create_variation()` | Inserts another answer-key variation for a test. |
| `create_test_with_variation()` | Inserts a test and first answer-key variation in one transaction. |
| `list_tests()` | Lists tests with variation counts. |
| `get_test()` | Fetches one test. |
| `list_variations()` | Lists variations for one test. |
| `get_variation()` | Fetches one variation joined with its test. |
| `save_scan_result()` | Saves a confirmed scan result. |
| `list_results()` | Lists saved results and decodes JSON fields. |

## Templates

| Template | What it displays |
| --- | --- |
| `base.html` | Shared page shell, navigation, messages, CSS/JS includes. |
| `index.html` | Home page actions and recent activity. |
| `tests.html` | All saved tests. |
| `test_form.html` | Test creation and variation answer-key form. |
| `test_detail.html` | Variations and saved question text for a test. |
| `sheet.html` | Printable SVG OMR sheet. |
| `scan.html` | Phone camera/file upload form. |
| `scan_preview.html` | Annotated image, score summary, and editable final answers. |
| `results.html` | Saved scores and annotated image links. |

## Removed Prototype Files

The old command-line prototype files were removed because they were no longer
used by the Flask app:

- `src/main.py`
- `src/template_loader.py`
- `answer_key.json`
- `templates/template_20q_abcd.json`
- `templates/template_40q_abcd_square.json`
- `templates/sheet_20q_abcd_square.tex`

The runtime folders `data/` and `outputs/` were not deleted because they may
contain saved local results, uploads, or debug images.
