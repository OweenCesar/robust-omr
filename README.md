# Robust OMR Webapp

Robust OMR is a local webapp for phone-based optical mark recognition. A teacher
runs the app on a computer, opens it from a phone, creates tests, prints coded
OMR sheets, scans completed student sheets, reviews detected answers, and saves
scores locally.

## Features

- Create tests with 10 to 40 questions.
- Store optional question text for each question.
- Save answer-key variations, such as Version A and Version B.
- Print a matching SVG answer sheet from the webapp.
- Use four coded ArUco corner markers for perspective correction.
- Scan with a phone camera or image upload.
- Optionally compress photos before upload.
- Review detected answers before saving.
- Store results in local SQLite.
- Export saved results as CSV.

## Requirements

Install Python dependencies from `requirements.txt`:

```text
Flask
numpy
opencv-python
```

The current OpenCV install must include `cv2.aruco`, because the sheet uses
ArUco markers for reliable angled-photo correction.

## Setup On Windows

From the project folder:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

If the virtual environment already exists:

```powershell
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

## Start The App

Run:

```powershell
.\.venv\Scripts\python.exe app.py
```

Then open this on the computer:

```text
http://127.0.0.1:5000
```

The app starts on `0.0.0.0:5000`, so another device on the same Wi-Fi can use
the computer's Wi-Fi IP address:

```text
http://YOUR-COMPUTER-IP:5000
```

Example:

```text
http://192.168.178.36:5000
```

If Windows Firewall asks about Python, allow it on private networks.

## Phone Access With Ngrok

For camera access on many phone browsers, use HTTPS through ngrok:

```powershell
ngrok http 5000
```

Open the HTTPS ngrok URL on the phone.

If ngrok says the request is too large, keep `Compress image before upload`
checked on the scan page.

## Normal Workflow

1. Create a test.
2. Optionally type the question text.
3. Choose the correct answer for each question.
4. Print the generated OMR sheet.
5. Give the sheet to a student.
6. Open the scan page on a phone.
7. Select the test variation.
8. Capture or upload the completed sheet.
9. Review detected answers and warnings.
10. Save the result.
11. View or export results.

## Important Scanning Notes

- Print sheets from this app. Older plain-square sheets are no longer valid.
- Keep all four coded corner markers visible in the photo.
- The app rejects scans if it cannot find all four coded markers.
- Normal side-angle photos are corrected by OpenCV perspective warping.
- Very blurry, cropped, folded, or heavily shadowed photos should be retaken.

## Local Data

Runtime data is intentionally stored locally:

- SQLite database: `data/omr_demo.sqlite3`
- Uploaded scans: `data/uploads/`
- Annotated scan images: `outputs/annotated/`
- Flask logs: `outputs/flask.log` and `outputs/flask.err.log`

The `data/` and `outputs/` folders are ignored by git because they contain local
runtime data, not source code.

## Architecture Documentation

Detailed file/function documentation is in:

[docs/ARCHITECTURE.md](docs/ARCHITECTURE.md)

The Graphviz DOT source for the architecture diagram is in:

[docs/app_architecture.dot](docs/app_architecture.dot)

If Graphviz is installed, render it with:

```powershell
dot -Tsvg docs\app_architecture.dot -o docs\app_architecture.svg
```

