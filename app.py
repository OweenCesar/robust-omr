"""
Flask webapp for the local Robust OMR system.

Run this file on the computer, then expose it to a phone with ngrok for the
teacher phone. The app has no login by design for the current single-teacher
local workflow: create tests, print sheets, scan completed sheets, review the
detected answers, and save scores.
"""

from __future__ import annotations

import base64
import csv
import io
import json
import sys
import uuid
from datetime import datetime
from pathlib import Path

import cv2
from flask import (
    Flask,
    Response,
    flash,
    redirect,
    render_template,
    request,
    send_from_directory,
    session,
    url_for,
)
from werkzeug.utils import secure_filename


ROOT_DIR = Path(__file__).resolve().parent
SRC_DIR = ROOT_DIR / "src"

# The original project keeps helper modules in src/ without packaging them.
# Adding src/ to sys.path lets app.py import those modules without forcing a
# larger project restructure.
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from db import (  # noqa: E402
    UPLOAD_DIR,
    create_test_with_variation,
    create_variation,
    get_test,
    get_variation,
    init_db,
    list_results,
    list_tests,
    list_variations,
    save_scan_result,
)
from grader import grade_answers  # noqa: E402
from omr_engine import build_manual_interpretation, process_scan  # noqa: E402
from sheet_layout import OPTIONS, sheet_context, validate_question_count  # noqa: E402


OUTPUT_DIR = ROOT_DIR / "outputs"
ANNOTATED_DIR = OUTPUT_DIR / "annotated"


def create_app() -> Flask:
    """Create and configure the Flask application."""

    app = Flask(__name__)

    # A fixed development secret is fine for this no-login local app. If the
    # app later stores private data or moves online, this should come from an
    # environment variable.
    app.secret_key = "robust-omr-demo-local-secret"
    # Phones can cache old JavaScript aggressively. During local use, disable
    # static-file caching so scan-page changes, such as compression settings,
    # appear immediately after refresh.
    app.config["SEND_FILE_MAX_AGE_DEFAULT"] = 0

    init_db()
    UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
    ANNOTATED_DIR.mkdir(parents=True, exist_ok=True)

    @app.context_processor
    def inject_globals():
        """Expose shared constants to templates."""

        return {"default_options": OPTIONS}

    @app.route("/")
    def index():
        """Simple landing page with the main teacher actions."""

        tests = list_tests()
        results = list_results()

        return render_template(
            "index.html",
            tests=tests,
            results=results,
        )

    @app.route("/tests")
    def tests_index():
        """List all saved tests."""

        return render_template("tests.html", tests=list_tests())

    @app.route("/tests/new", methods=["GET", "POST"])
    def new_test():
        """Create a test and its first answer-key variation."""

        if request.method == "POST":
            form_result = _parse_test_form(request.form)

            if form_result["errors"]:
                for error in form_result["errors"]:
                    flash(error, "error")

                return render_template(
                    "test_form.html",
                    mode="create",
                    test=None,
                    question_count=form_result["question_count"],
                    variation_name=form_result["variation_name"],
                    question_texts=form_result["question_texts"],
                    answer_key=form_result["answer_key"],
                    options=OPTIONS,
                )

            test_id, variation_id = create_test_with_variation(
                name=form_result["name"],
                question_count=form_result["question_count"],
                options=list(OPTIONS),
                question_texts=form_result["question_texts"],
                variation_name=form_result["variation_name"],
                answer_key=form_result["answer_key"],
            )

            flash("Test created. You can print the OMR sheet now.", "success")
            return redirect(url_for("print_sheet", test_id=test_id, variation_id=variation_id))

        return render_template(
            "test_form.html",
            mode="create",
            test=None,
            question_count=20,
            variation_name="A",
            question_texts={},
            answer_key={},
            options=OPTIONS,
        )

    @app.route("/tests/<int:test_id>")
    def test_detail(test_id: int):
        """Show a test, its variations, and quick actions."""

        test = get_test(test_id)

        if test is None:
            flash("Test not found.", "error")
            return redirect(url_for("tests_index"))

        variations = list_variations(test_id)

        return render_template(
            "test_detail.html",
            test=test,
            variations=variations,
            question_texts=json.loads(test["questions_json"]),
        )

    @app.route("/tests/<int:test_id>/variations/new", methods=["GET", "POST"])
    def new_variation(test_id: int):
        """Add another answer-key variation to an existing test."""

        test = get_test(test_id)

        if test is None:
            flash("Test not found.", "error")
            return redirect(url_for("tests_index"))

        if request.method == "POST":
            form_result = _parse_variation_form(request.form, test["question_count"])

            if form_result["errors"]:
                for error in form_result["errors"]:
                    flash(error, "error")

                return render_template(
                    "test_form.html",
                    mode="variation",
                    test=test,
                    question_count=test["question_count"],
                    variation_name=form_result["variation_name"],
                    question_texts=json.loads(test["questions_json"]),
                    answer_key=form_result["answer_key"],
                    options=OPTIONS,
                )

            variation_id = create_variation(
                test_id=test_id,
                name=form_result["variation_name"],
                answer_key=form_result["answer_key"],
            )

            flash("Variation saved.", "success")
            return redirect(url_for("print_sheet", test_id=test_id, variation_id=variation_id))

        return render_template(
            "test_form.html",
            mode="variation",
            test=test,
            question_count=test["question_count"],
            variation_name="B",
            question_texts=json.loads(test["questions_json"]),
            answer_key={},
            options=OPTIONS,
        )

    @app.route("/tests/<int:test_id>/variations/<int:variation_id>/sheet")
    def print_sheet(test_id: int, variation_id: int):
        """
        Render the printable SVG answer sheet.

        This page is intentionally plain. The printed SVG uses the exact same
        coordinates as the OpenCV detector, so changing the sheet layout should
        happen in src/sheet_layout.py rather than only in HTML/CSS.
        """

        test = get_test(test_id)
        variation = get_variation(variation_id)

        if test is None or variation is None or variation["test_id"] != test_id:
            flash("Sheet not found.", "error")
            return redirect(url_for("tests_index"))

        layout = sheet_context(test["question_count"], OPTIONS)

        return render_template(
            "sheet.html",
            test=test,
            variation=variation,
            layout=layout,
        )

    @app.route("/scan", methods=["GET"])
    def scan():
        """Show the phone-friendly scan page."""

        variation_id = request.args.get("variation_id", type=int)
        selected_variation = get_variation(variation_id) if variation_id else None

        choices = _scan_choices()

        return render_template(
            "scan.html",
            choices=choices,
            selected_variation=selected_variation,
        )

    @app.route("/scan/preview", methods=["POST"])
    def scan_preview():
        """
        Process an uploaded/captured sheet and show a confirmation screen.

        The scan is not saved immediately. The teacher first sees the detected
        answers and may correct any unclear choices before pressing Save.
        """

        variation_id = request.form.get("variation_id", type=int)
        variation = get_variation(variation_id) if variation_id else None

        if variation is None:
            flash("Choose a test variation before scanning.", "error")
            return redirect(url_for("scan"))

        image_bytes, original_name = _read_scan_image(request)

        if not image_bytes:
            flash("Take a photo or choose an image file first.", "error")
            return redirect(url_for("scan", variation_id=variation_id))

        answer_key = json.loads(variation["answer_key_json"])
        question_count = int(variation["question_count"])

        try:
            scan_result = process_scan(
                image_bytes=image_bytes,
                answer_key=answer_key,
                question_count=question_count,
                options=OPTIONS,
            )
        except Exception as exc:
            flash(f"Could not process the image: {exc}", "error")
            return redirect(url_for("scan", variation_id=variation_id))

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        safe_original = secure_filename(original_name or "scan.jpg") or "scan.jpg"
        unique_stem = f"{timestamp}_{uuid.uuid4().hex[:8]}"

        upload_filename = f"{unique_stem}_{safe_original}"
        upload_path = UPLOAD_DIR / upload_filename
        upload_path.write_bytes(image_bytes)

        annotated_filename = f"{unique_stem}_annotated.png"
        annotated_path = ANNOTATED_DIR / annotated_filename
        cv2.imwrite(str(annotated_path), scan_result["annotated_image"])

        pending_scan = {
            "variation_id": variation_id,
            "test_id": int(variation["test_id"]),
            "test_name": variation["test_name"],
            "variation_name": variation["name"],
            "question_count": question_count,
            "student_name": request.form.get("student_name", "").strip(),
            "student_id": request.form.get("student_id", "").strip(),
            "upload_filename": upload_filename,
            "annotated_filename": annotated_filename,
            "summary": scan_result["summary"],
            "details": scan_result["details"],
            "detected_answers": _compact_detection(scan_result["detected_answers"]),
            "warnings": scan_result["warnings"],
            "quality": scan_result["quality"],
        }

        session["pending_scan"] = pending_scan

        return render_template(
            "scan_preview.html",
            pending=pending_scan,
            options=OPTIONS,
        )

    @app.route("/scan/save", methods=["POST"])
    def scan_save():
        """Save a scan after the teacher has reviewed the detected answers."""

        pending = session.get("pending_scan")

        if not pending:
            flash("There is no scan waiting to save.", "error")
            return redirect(url_for("scan"))

        variation = get_variation(int(pending["variation_id"]))

        if variation is None:
            flash("The selected variation no longer exists.", "error")
            session.pop("pending_scan", None)
            return redirect(url_for("scan"))

        question_count = int(pending["question_count"])
        answer_key = json.loads(variation["answer_key_json"])

        final_answers = {}
        for question_number in range(1, question_count + 1):
            key = str(question_number)
            selected = request.form.get(f"final_{key}", "")
            final_answers[key] = selected if selected in OPTIONS else None

        final_interpretation = build_manual_interpretation(final_answers, question_count)
        final_summary, final_details = grade_answers(final_interpretation, answer_key)

        # Save both the original machine detection and the teacher-confirmed
        # answer set. This is useful because you can explain when a
        # manual correction changed the final score.
        detected_payload = {
            "original_detection": pending["detected_answers"],
            "final_answers": final_interpretation,
        }

        save_scan_result(
            test_id=int(pending["test_id"]),
            variation_id=int(pending["variation_id"]),
            student_name=request.form.get("student_name", pending["student_name"]).strip(),
            student_id=request.form.get("student_id", pending["student_id"]).strip(),
            image_path=pending["upload_filename"],
            annotated_image_path=pending["annotated_filename"],
            detected_answers=detected_payload,
            detailed_results=final_details,
            summary=final_summary,
        )

        session.pop("pending_scan", None)
        flash("Result saved.", "success")
        return redirect(url_for("results_index"))

    @app.route("/results")
    def results_index():
        """Show saved scan results."""

        return render_template("results.html", results=list_results())

    @app.route("/results.csv")
    def results_csv():
        """Export saved results as CSV for spreadsheet use."""

        output = io.StringIO()
        writer = csv.writer(output)
        writer.writerow(
            [
                "date",
                "student_name",
                "student_id",
                "test",
                "variation",
                "score_percent",
                "correct",
                "wrong",
                "blank",
                "multiple",
                "unclear",
                "total",
            ]
        )

        for result in list_results():
            summary = result["summary"]
            writer.writerow(
                [
                    result["created_at"],
                    result["student_name"],
                    result["student_id"],
                    result["test_name"],
                    result["variation_name"],
                    summary["score_percent"],
                    summary["correct"],
                    summary["wrong"],
                    summary["blank"],
                    summary["multiple"],
                    summary["unclear"],
                    summary["total"],
                ]
            )

        return Response(
            output.getvalue(),
            mimetype="text/csv",
            headers={"Content-Disposition": "attachment; filename=omr_results.csv"},
        )

    @app.route("/media/<kind>/<filename>")
    def media_file(kind: str, filename: str):
        """Serve uploaded and annotated scan images inside the local app."""

        if kind == "uploads":
            return send_from_directory(UPLOAD_DIR, filename)

        if kind == "annotated":
            return send_from_directory(ANNOTATED_DIR, filename)

        flash("File not found.", "error")
        return redirect(url_for("index"))

    return app


def _parse_test_form(form) -> dict:
    """Validate the create-test form and return normalized values."""

    errors = []
    name = form.get("name", "").strip()
    variation_name = form.get("variation_name", "A").strip() or "A"

    try:
        question_count = int(form.get("question_count", "20"))
        validate_question_count(question_count)
    except ValueError:
        question_count = 20
        errors.append("Question count must be between 10 and 40.")

    if not name:
        errors.append("Enter a test name.")

    variation_result = _parse_variation_form(form, question_count)
    question_texts = _parse_question_texts(form, question_count)
    errors.extend(variation_result["errors"])

    return {
        "name": name,
        "question_count": question_count,
        "variation_name": variation_name,
        "question_texts": question_texts,
        "answer_key": variation_result["answer_key"],
        "errors": errors,
    }


def _parse_variation_form(form, question_count: int) -> dict:
    """Validate a variation form and return its answer key."""

    errors = []
    variation_name = form.get("variation_name", "A").strip() or "A"
    answer_key = {}

    if not variation_name:
        errors.append("Enter a variation name.")

    for question_number in range(1, question_count + 1):
        answer = form.get(f"answer_{question_number}", "")

        if answer not in OPTIONS:
            errors.append(f"Choose a correct answer for question {question_number}.")
        else:
            answer_key[str(question_number)] = answer

    return {
        "variation_name": variation_name,
        "answer_key": answer_key,
        "errors": errors,
    }


def _parse_question_texts(form, question_count: int) -> dict[str, str]:
    """
    Collect optional teacher-written question text.

    The OMR answer sheet only needs the answer boxes, but the teacher may still
    want the app to remember what each question was. Empty question text is
    allowed because some demonstrations only need answer keys.
    """

    question_texts = {}

    for question_number in range(1, question_count + 1):
        text = form.get(f"question_text_{question_number}", "").strip()

        if text:
            question_texts[str(question_number)] = text

    return question_texts


def _scan_choices() -> list[dict]:
    """Build flat test/variation choices for the scan page select menu."""

    choices = []

    for test in list_tests():
        variations = list_variations(test["id"])

        for variation in variations:
            choices.append(
                {
                    "variation_id": variation["id"],
                    "label": f"{test['name']} - Version {variation['name']}",
                }
            )

    return choices


def _compact_detection(detected_answers: dict) -> dict:
    """
    Remove raw per-option scores before storing a pending scan in the session.

    Flask's default session is cookie-based, so keeping it small matters. The
    teacher confirmation page needs selected answer, status, and confidence;
    the large raw score dictionary is useful for debugging but not for saving a
    reviewed result.
    """

    compact = {}

    for question, result in detected_answers.items():
        compact[question] = {
            "selected": result.get("selected"),
            "status": result.get("status"),
            "confidence": result.get("confidence"),
        }

    return compact


def _read_scan_image(current_request) -> tuple[bytes | None, str | None]:
    """
    Read a scan image from either camera data or a file upload.

    The camera flow and compressed file-upload flow send a base64 data URL in a
    hidden field. If the teacher turns compression off, the browser sends the
    original image as a normal multipart file. Supporting both paths lets the
    teacher choose between ngrok-friendly uploads and maximum original quality.
    """

    captured_data = current_request.form.get("captured_image_data", "")

    if captured_data.startswith("data:image/") and "," in captured_data:
        header, encoded = captured_data.split(",", 1)
        extension = "jpg"

        if "image/png" in header:
            extension = "png"

        try:
            return base64.b64decode(encoded), f"camera_capture.{extension}"
        except ValueError:
            return None, None

    uploaded_file = current_request.files.get("sheet_image")

    if uploaded_file and uploaded_file.filename:
        return uploaded_file.read(), uploaded_file.filename

    return None, None


app = create_app()


if __name__ == "__main__":
    # host=0.0.0.0 makes the app reachable from ngrok and from other devices on
    # the same network. The ngrok public URL will provide HTTPS for phone camera
    # access.
    app.run(host="0.0.0.0", port=5000, debug=False)
