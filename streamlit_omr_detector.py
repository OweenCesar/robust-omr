"""
Streamlit UI for the OMR bubble detector.

Run with:
    streamlit run streamlit_omr_detector.py

Keep this file in the same folder as omr_bubble_detector.py.
"""

from __future__ import annotations

import json
import re
from io import BytesIO
from typing import Dict, Optional

import cv2
import numpy as np
import pandas as pd
import streamlit as st

from omr_bubble_detector import (
    detect_omr_bubbles,
    draw_detection_overlay,
    draw_grading_overlay,
    grade_omr_result,
    make_json_safe,
    warp_sheet,
)


st.set_page_config(page_title="OMR Bubble Detection", layout="wide")

ANSWER_OPTIONS = ["A", "B", "C", "D", "E"]


def _decode_uploaded_image(uploaded_file) -> np.ndarray:
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    image_bgr = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
    if image_bgr is None:
        raise ValueError("Could not decode the uploaded image. Please upload a JPG or PNG.")
    return image_bgr


def _bgr_to_rgb(image_bgr: np.ndarray) -> np.ndarray:
    return cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)


def _image_to_png_bytes(image_bgr: np.ndarray) -> bytes:
    ok, encoded = cv2.imencode(".png", image_bgr)
    if not ok:
        raise ValueError("Could not encode image as PNG.")
    return encoded.tobytes()


def _section_to_dataframe(result: dict, section_key: str) -> pd.DataFrame:
    bubbles = result.get(section_key, {}).get("bubbles", [])
    if not bubbles:
        return pd.DataFrame()
    return pd.DataFrame(bubbles)


def _selected_question_count(label: str) -> Optional[int]:
    if label.startswith("Auto"):
        return None
    return int(label)


def _layout_label(method: str) -> str:
    labels = {
        "printed_line_layout": "Printed lines",
        "hybrid_layout": "Hybrid",
        "fallback_normalized_roi": "Fallback",
    }
    return labels.get(method, method.replace("_", " ").title())


def _demo_answer_key(question_count: int) -> Dict[int, str]:
    return {question: ANSWER_OPTIONS[(question - 1) % len(ANSWER_OPTIONS)] for question in range(1, question_count + 1)}


def _answer_key_to_text(answer_key: Dict[int, str]) -> str:
    return "\n".join(f"{question}: {answer_key[question]}" for question in sorted(answer_key))


def _parse_answer_key_text(text: str, question_count: int) -> Dict[int, str]:
    text = text.strip()
    if not text:
        return {}

    try:
        loaded = json.loads(text)
        if isinstance(loaded, dict):
            return {
                int(question): str(option).strip().upper()
                for question, option in loaded.items()
                if str(option).strip().upper() in ANSWER_OPTIONS
            }
        if isinstance(loaded, list):
            return {
                idx: str(option).strip().upper()
                for idx, option in enumerate(loaded, start=1)
                if str(option).strip().upper() in ANSWER_OPTIONS
            }
    except json.JSONDecodeError:
        pass

    pairs = re.findall(r"(?m)(\d+)\s*[:=,.)-]\s*([A-Ea-e])\b", text)
    if pairs:
        return {int(question): option.upper() for question, option in pairs}

    tokens = re.findall(r"[A-Ea-e]", text)
    return {
        idx: option.upper()
        for idx, option in enumerate(tokens[:question_count], start=1)
    }


def _decode_file_text(uploaded_file) -> str:
    if uploaded_file is None:
        return ""
    return uploaded_file.read().decode("utf-8", errors="replace")


def _review_dataframe(grading: dict) -> pd.DataFrame:
    rows = grading.get("questions", [])
    if not rows:
        return pd.DataFrame()
    columns = [
        "question",
        "detected_answer",
        "correct_answer",
        "outcome",
        "status",
        "confidence",
        "top_fill_score",
        "second_fill_score",
    ]
    return pd.DataFrame(rows)[columns]


st.title("OMR Bubble Detection")
st.caption(
    "Phone-photo pipeline: frame detection → perspective correction → printed-region discovery → bubble-grid inference → grading. "
    "Short sheets are handled by detecting the real rows inside each answer column group."
)

with st.sidebar:
    st.header("Settings")
    question_label = st.selectbox(
        "Question count mode",
        ["Auto detect from printed bubbles", "10", "20", "30", "50", "100"],
        index=0,
        help=(
            "Auto mode now detects rows per answer column independently. "
            "This avoids generating fake bubbles for shorter sheets such as 30, 20, or 10 questions. "
            "Use a number only when you want to cap detections at a known sheet length."
        ),
    )
    expected_questions = _selected_question_count(question_label)
    draw_regions = st.checkbox("Draw detected regions", value=True)
    draw_labels = st.checkbox("Draw question numbers on overlay", value=False)
    show_tables = st.checkbox("Show detection tables", value=True)

uploaded = st.file_uploader("Upload a phone photo of the OMR sheet", type=["jpg", "jpeg", "png"])

if uploaded is None:
    st.info("Upload an answer sheet image to start.")
    st.stop()

try:
    image_bgr = _decode_uploaded_image(uploaded)
    result = detect_omr_bubbles(image_bgr, expected_questions=expected_questions)
    warped_bgr, _ = warp_sheet(image_bgr)
except Exception as exc:
    st.error(f"Detection failed: {exc}")
    st.stop()

metadata = result.get("metadata", {})
layout = result.get("layout", {})
answers = result.get("answers", {})
student_id = result.get("student_id", {})
test_id = result.get("test_id", {})
question_count = int(answers.get("question_count", 0))

with st.sidebar:
    st.header("Grading")
    grading_enabled = st.checkbox("Enable grading dashboard", value=True)
    key_source = st.selectbox(
        "Answer key source",
        ["Demo A/B/C/D/E pattern", "Paste or edit key", "Upload JSON/TXT", "Read only"],
        index=0,
        help=(
            "Use the demo key only for testing the workflow. "
            "Paste or upload the real key for real scoring."
        ),
        disabled=not grading_enabled,
    )

    answer_key: Dict[int, str] = {}
    if grading_enabled and key_source == "Demo A/B/C/D/E pattern":
        answer_key = _demo_answer_key(question_count)
        with st.expander("Demo key preview"):
            st.code(_answer_key_to_text(answer_key), language="text")
    elif grading_enabled and key_source == "Paste or edit key":
        key_text = st.text_area(
            "Answer key",
            value="",
            placeholder="1:A\n2:C\n3:B\n...\nor ABCDEABCDE...",
            height=180,
        )
        answer_key = _parse_answer_key_text(key_text, question_count)
    elif grading_enabled and key_source == "Upload JSON/TXT":
        key_file = st.file_uploader("Answer key file", type=["json", "txt", "csv"], key="answer_key_file")
        answer_key = _parse_answer_key_text(_decode_file_text(key_file), question_count)

    if grading_enabled:
        st.caption(f"Answer key coverage: {len(answer_key)} / {question_count} questions")

if grading_enabled:
    result["grading"] = grade_omr_result(result, answer_key=answer_key)
    overlay_bgr = draw_grading_overlay(
        warped_bgr,
        result,
        draw_rois=draw_regions,
        draw_labels=draw_labels,
        show_correct_answers=bool(answer_key),
    )
else:
    overlay_bgr = draw_detection_overlay(warped_bgr, result, draw_rois=draw_regions, draw_labels=draw_labels)

m1, m2, m3, m4, m5 = st.columns(5)
m1.metric("Warp method", metadata.get("warp_method", "unknown"))
m2.metric("Layout", _layout_label(metadata.get("layout_method", "unknown")))
m3.metric("Answer questions", answers.get("question_count", 0))
m4.metric("Answer bubbles", answers.get("bubble_count", 0))
m5.metric("ID/Test bubbles", student_id.get("bubble_count", 0) + test_id.get("bubble_count", 0))

rows_per_group = answers.get("rows_per_group", [])
if rows_per_group:
    st.caption(f"Detected answer rows per printed column group: {rows_per_group}")

layout_source = layout.get("source", {})
if layout_source:
    st.caption(
        "Region sources: "
        f"header={layout_source.get('header', 'unknown')}, "
        f"answers={layout_source.get('answers', 'unknown')}, "
        f"student_id={layout_source.get('student_id', 'unknown')}, "
        f"test_id={layout_source.get('test_id', 'unknown')}"
    )

grading = result.get("grading", {})
if grading:
    summary = grading.get("summary", {})
    identity = grading.get("identity", {})
    student_reading = identity.get("student_id", {})
    test_reading = identity.get("test_id", {})

    st.subheader("Grading dashboard")
    g1, g2, g3, g4, g5 = st.columns(5)
    g1.metric("Student ID", student_reading.get("value", "") or "No grid")
    g2.metric("Test ID", test_reading.get("value", "") or "No grid")
    score_percent = summary.get("score_percent")
    g3.metric("Score", "No key" if score_percent is None else f"{score_percent:.2f}%")
    g4.metric("Correct", summary.get("correct", 0))
    g5.metric("Review", summary.get("needs_review", 0))

    if score_percent is not None:
        st.progress(min(max(score_percent / 100.0, 0.0), 1.0))
        st.caption(
            f"{summary.get('correct', 0)} correct, {summary.get('incorrect', 0)} incorrect, "
            f"{summary.get('blank', 0)} blank, {summary.get('multiple', 0)} multiple, "
            f"{summary.get('unclear', 0)} unclear."
        )
    else:
        st.info("Add an answer key to score the sheet and show the correct answers on the overlay.")

col1, col2 = st.columns(2)
with col1:
    st.subheader("Original image")
    st.image(_bgr_to_rgb(image_bgr), use_container_width=True)
with col2:
    st.subheader("Graded sheet overlay" if grading and answer_key else "Warped + detected bubbles")
    st.image(_bgr_to_rgb(overlay_bgr), use_container_width=True)

json_result = json.dumps(make_json_safe(result), indent=2)

download_col1, download_col2 = st.columns(2)
with download_col1:
    st.download_button(
        "Download detections JSON",
        data=json_result.encode("utf-8"),
        file_name="omr_result.json",
        mime="application/json",
    )
with download_col2:
    st.download_button(
        "Download overlay PNG",
        data=_image_to_png_bytes(overlay_bgr),
        file_name="omr_graded_overlay.png" if grading else "omr_detection_overlay.png",
        mime="image/png",
    )

if show_tables:
    if grading:
        st.subheader("Question review")
        review_df = _review_dataframe(grading)
        if review_df.empty:
            st.warning("No question rows available for review.")
        else:
            st.dataframe(review_df, use_container_width=True, height=360)

    st.subheader("Answer bubble detections")
    answer_df = _section_to_dataframe(result, "answers")
    if answer_df.empty:
        st.warning(answers.get("warning", "No answer bubbles detected."))
    else:
        st.dataframe(answer_df, use_container_width=True, height=360)

    tab1, tab2 = st.tabs(["Student ID grid", "Test ID grid"])
    with tab1:
        df = _section_to_dataframe(result, "student_id")
        if df.empty:
            st.warning(student_id.get("warning", "No Student ID bubbles detected."))
        else:
            st.dataframe(df, use_container_width=True, height=300)
    with tab2:
        df = _section_to_dataframe(result, "test_id")
        if df.empty:
            st.warning(test_id.get("warning", "No Test ID bubbles detected."))
        else:
            st.dataframe(df, use_container_width=True, height=300)

with st.expander("Raw metadata"):
    st.json(make_json_safe({"metadata": metadata, "layout": layout}))
