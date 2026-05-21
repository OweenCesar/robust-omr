# Robust OMR Webapp Requirements

## Purpose

Build a simple web-based OMR system for teachers using a phone in a low-resource environment.

The teacher should run the app on a computer, expose it to a phone with ngrok, create a test, print a matching OMR sheet, scan completed student sheets with the phone camera, and save the scores locally.

This local version does not need login, sign-up, cloud hosting, or a complex user interface.

## Scope

The first working version should:

- Run locally on the teacher's computer.
- Be accessible from a phone browser through an ngrok HTTPS link.
- Use a simple mobile-friendly web interface.
- Let the teacher create tests with 10 to 40 questions.
- Let the teacher choose correct answers for each question.
- Support test variations, such as Version A, Version B, and Version C.
- Generate or use a matching printable OMR sheet.
- Let the teacher scan each student's completed sheet.
- Automatically detect marked answers.
- Calculate and show the student's score.
- Save student results locally.

## Out Of Scope

The current app does not need:

- Teacher login or sign-up.
- Student accounts.
- Cloud database.
- Payment features.
- Multi-school administration.
- Fancy dashboard UI.
- Full offline sync.
- Advanced reporting.

These can be added later if the local app works well.

## Recommended Frameworks And Tools

Use a small Python webapp stack:

- Flask for the web application.
- Jinja templates for simple HTML pages.
- Plain HTML, CSS, and JavaScript for the frontend.
- Browser camera access with `navigator.mediaDevices.getUserMedia()`.
- OpenCV and NumPy for image processing and OMR detection.
- SQLite for local storage.
- ngrok for phone access when HTTPS camera access is needed.

Flask is used because the app needs controlled routes, printable pages, file uploads, camera capture, and result storage.

## Main Pages

### Home Page

The home page should be simple and show clear actions:

- Create Test
- View Tests
- Print Sheet
- Scan Sheet
- View Results

### Create Test Page

The teacher should be able to enter:

- Test name.
- Optional question text for each question.
- Number of questions from 10 to 40.
- Answer options, initially A, B, C, and D.
- Test variation name, such as A, B, or C.
- Correct answer for every question.

The app should validate that:

- The number of questions is between 10 and 40.
- Every question has one correct answer.
- Each variation has its own saved answer key.

### Print Sheet Page

The app should show or generate a printable OMR sheet for the selected test.

The sheet should include:

- Test name or test ID.
- Variation field.
- Student name field.
- Student ID field.
- Question boxes from 1 to the selected question count.
- Answer options A to D.
- Four coded ArUco corner markers for alignment and orientation.
- Clear instructions for students.

The sheet should be black and white so it is cheap to print.

### Scan Sheet Page

The teacher should be able to:

- Select the test and variation.
- Open the phone camera.
- Capture one completed sheet.
- Send the photo to the Flask backend.
- See detected answers and score.
- Retake the photo if the image is blurry, too dark, cropped, or missing markers.
- Confirm and save the result.

### Results Page

The results page should show saved student results in a table:

- Student name or ID.
- Test name.
- Variation.
- Score.
- Correct count.
- Wrong count.
- Blank count.
- Multiple-mark count.
- Scan date and time.

It should also support CSV export if time allows.

## OMR Sheet Requirements

The printed sheet is a core part of the system. OpenCV will work best if the app controls the sheet layout.

The sheet should:

- Have a fixed layout.
- Use large answer boxes or bubbles.
- Keep enough spacing between options.
- Use four coded ArUco corner markers.
- Avoid decorative graphics.
- Avoid gray backgrounds.
- Avoid small text near answer boxes.
- Print consistently at the same scale.

For the current project, the LaTeX sheet and the JSON template must match exactly. If the LaTeX sheet has 40 questions in two columns, the JSON template and answer key must also support 40 questions in two columns.

## OMR Detection Requirements

The OpenCV pipeline should:

- Load the uploaded camera image.
- Convert it to grayscale.
- Check blur and brightness.
- Detect all four coded ArUco corner markers.
- Correct rotation and perspective.
- Warp the sheet to a known size.
- Locate the answer boxes using the template coordinates.
- Measure how much each answer box is filled.
- Detect valid answers.
- Detect blank answers.
- Detect multiple marked answers.
- Detect unclear answers.
- Return confidence values for review.

For the current square-box sheet, the detector should use square or rectangular regions instead of only circular masks.

## Scoring Requirements

The scoring system should:

- Compare detected answers to the saved answer key.
- Count correct answers.
- Count wrong answers.
- Count blank answers.
- Count multiple marks.
- Count unclear answers.
- Calculate a total score.
- Show detailed question-by-question results.

Unclear answers should be shown to the teacher for review instead of being silently marked wrong.

## Storage Requirements

Use SQLite for local storage.

The database should store:

- Tests.
- Optional question text.
- Test variations.
- Answer keys.
- Scan results.
- Student name or ID.
- Detected answers.
- Final corrected answers, if the teacher edits them.
- Score summary.
- Scan timestamp.
- Optional path to the uploaded scan image.

## Low-Resource Requirements

The app should work well in a low-resource setting:

- Simple UI with large buttons.
- Mobile-first layout.
- Small number of screens.
- Minimal typing during scanning.
- Low data usage.
- Images should be resized or compressed before processing when possible.
- Backend processing should happen on the computer, not the phone.
- The printed sheet should be cheap and black-and-white.

## Current Project Notes

The current project already has useful pieces:

- Python OMR pipeline.
- OpenCV preprocessing.
- Bubble reading logic.
- Grading logic.
- Debug image output.
- A sample OMR image.
- A LaTeX sheet template.

Before building the webapp, the existing OMR code should be aligned with the new sheet:

- Update the template JSON to support 40 questions.
- Support the two-column layout.
- Match the LaTeX coordinates.
- Update answer key handling for 10 to 40 questions.
- Add perspective correction using coded corner markers.
- Adjust detection for square answer boxes.

## Suggested Build Phases

### Phase 1: Align Sheet And OMR Template

- Make the LaTeX sheet, JSON template, and answer key describe the same layout.
- Support 10 to 40 questions.
- Support two-column reading for 40 questions.
- Improve detection for square boxes.
- Reject scans when the four coded markers are not visible.

### Phase 2: Build Flask Webapp

- Add Flask app structure.
- Add home page.
- Add create test page.
- Save tests and answer keys to SQLite.
- Add results page.

### Phase 3: Add Printable Sheet Flow

- Show printable OMR sheet from the webapp.
- Include test ID and variation.
- Keep the print layout stable.

### Phase 4: Add Phone Scanning

- Add camera capture page.
- Upload captured image to backend.
- Process image with OpenCV.
- Show detected answers and score.

### Phase 5: Save And Export Results

- Save confirmed scan results.
- Show results table.
- Add CSV export.

### Phase 6: Demo Polish

- Improve error messages.
- Add retake warnings for bad photos.
- Make the UI cleaner on mobile.
- Test through ngrok on a phone.
