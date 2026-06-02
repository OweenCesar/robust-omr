import io
import json
from pathlib import Path

import cv2
import numpy as np
import streamlit as st
from PIL import Image

from bubble_detection import (
    detect_bubble_candidates_hough,
    draw_mark_scores,
    filter_bubbles_by_color_presence,
    score_bubble_marks,
)
from classical_grading import (
    build_answer_rows,
    build_review_messages,
    grade_answers,
    interpret_answer_rows,
    load_answer_key,
    normalize_answer_key,
)
from counter_detection import (
    build_warped_rectangle_sections,
    draw_all_contours,
    find_external_contours,
    prepare_rectangle_detection_image,
    select_largest_rectangular_contours,
)
from preprocessing import (
    add_gaussian_blur,
    apply_canny_edge_detection,
    image_to_grayscale,
    resize_image,
)


APP_ROOT = Path(__file__).resolve().parent
SAMPLES_DIR = APP_ROOT / "samples"
DEMO_KEY_PATH = APP_ROOT / "answer_keys" / "demo_100_questions.json"


def setup_page():
    st.set_page_config(
        page_title="Accessible OMR Demo",
        page_icon=None,
        layout="wide",
    )
    st.markdown(
        """
        <style>
        :root {
            --project-blue: #183052;
            --project-green: #2a755d;
            --project-gold: #bf8430;
            --project-red: #a63d40;
            --soft-gray: #f5f7fa;
            --ink: #2a3038;
        }
        .main .block-container {
            padding-top: 1.4rem;
            padding-bottom: 2.5rem;
        }
        .hero {
            border-left: 5px solid var(--project-blue);
            padding: 0.7rem 0 0.6rem 1rem;
            margin-bottom: 1rem;
        }
        .hero h1 {
            color: var(--project-blue);
            font-size: 2.05rem;
            margin: 0 0 0.15rem 0;
        }
        .hero p {
            color: var(--ink);
            font-size: 1rem;
            margin: 0;
        }
        .stage-strip {
            display: grid;
            grid-template-columns: repeat(6, minmax(0, 1fr));
            gap: 0.45rem;
            margin: 0.6rem 0 1rem 0;
        }
        .stage-pill {
            background: var(--soft-gray);
            border: 1px solid #d8dee8;
            border-top: 3px solid var(--project-green);
            padding: 0.55rem 0.55rem;
            min-height: 3.6rem;
        }
        .stage-pill strong {
            color: var(--project-blue);
            display: block;
            font-size: 0.83rem;
        }
        .stage-pill span {
            color: #4b5563;
            display: block;
            font-size: 0.76rem;
            line-height: 1.15rem;
        }
        .note-box {
            background: #f5f7fa;
            border-left: 4px solid var(--project-gold);
            padding: 0.7rem 0.9rem;
            margin: 0.7rem 0;
        }
        .review-box {
            background: #fff7ed;
            border-left: 4px solid var(--project-red);
            padding: 0.7rem 0.9rem;
            margin: 0.6rem 0;
        }
        .ok-box {
            background: #f0f9f5;
            border-left: 4px solid var(--project-green);
            padding: 0.7rem 0.9rem;
            margin: 0.6rem 0;
        }
        div[data-testid="stMetric"] {
            background: #f8fafc;
            border: 1px solid #e2e8f0;
            padding: 0.55rem 0.7rem;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def show_header():
    st.markdown(
        """
        <div class="hero">
          <h1>Towards Accessible Optical Mark Recognition</h1>
          <p>Interactive computer vision demo: from scanned answer sheet to review-aware classical grading.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.markdown(
        """
        <div class="stage-strip">
          <div class="stage-pill"><strong>1. Input</strong><span>sample sheet or uploaded scan</span></div>
          <div class="stage-pill"><strong>2. Preprocess</strong><span>resize, grayscale, blur, edges</span></div>
          <div class="stage-pill"><strong>3. Rectangles</strong><span>detect answer and ID regions</span></div>
          <div class="stage-pill"><strong>4. Warp</strong><span>normalize perspective</span></div>
          <div class="stage-pill"><strong>5. Bubbles</strong><span>detect and score marks</span></div>
          <div class="stage-pill"><strong>6. Grade</strong><span>Method 1 with review flags</span></div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def sample_candidates():
    if not SAMPLES_DIR.exists():
        return []

    allowed_suffixes = {".jpg", ".jpeg", ".png"}
    ignored_prefixes = ("canny_", "rectangle_")
    ignored_names = {"shapes_debug.png"}
    paths = []

    for path in sorted(SAMPLES_DIR.iterdir()):
        if path.suffix.lower() not in allowed_suffixes:
            continue
        if path.name in ignored_names:
            continue
        if path.name.startswith(ignored_prefixes):
            continue
        paths.append(path)

    return paths


def read_image_from_path(path):
    image = Image.open(path).convert("RGB")
    return np.array(image)


def read_image_from_upload(uploaded_file):
    uploaded_file.seek(0)
    image = Image.open(uploaded_file).convert("RGB")
    return np.array(image)


def image_to_png_bytes(image):
    if image.ndim == 2:
        pil_image = Image.fromarray(image)
    else:
        pil_image = Image.fromarray(image.astype(np.uint8))

    buffer = io.BytesIO()
    pil_image.save(buffer, format="PNG")
    return buffer.getvalue()


def safe_json_bytes(data):
    return json.dumps(data, indent=2).encode("utf-8")


def load_selected_answer_key(mode, uploaded_key):
    if mode == "No grading":
        return None, "No answer key selected"

    if mode == "Demo 100-question key":
        if DEMO_KEY_PATH.exists():
            return load_answer_key(DEMO_KEY_PATH), DEMO_KEY_PATH.name
        st.sidebar.warning("Demo answer key was not found.")
        return None, "Missing demo key"

    if uploaded_key is None:
        st.sidebar.warning("Upload a JSON answer key or switch to the demo key.")
        return None, "No uploaded key"

    uploaded_key.seek(0)
    answer_key_data = json.load(uploaded_key)
    return normalize_answer_key(answer_key_data), uploaded_key.name


def make_report_payload(result):
    grading = result.get("grading")
    rows = grading["answers"] if grading else result["answers"]

    compact_rows = []
    for row in rows:
        compact_rows.append({
            "question": row["question"],
            "answer": row["answer"],
            "status": row["status"],
            "scores": row["scores"],
            "confidence": row["confidence"],
            "needs_review": row["needs_review"],
            "review_reason": row["review_reason"],
            "quality_flags": row["quality_flags"],
            "correct_answer": row.get("correct_answer"),
            "is_correct": row.get("is_correct"),
            "grading_status": row.get("grading_status"),
        })

    return {
        "sheet_name": result["name"],
        "layout": result["layout"],
        "summary": None if grading is None else grading["summary"],
        "review_messages": result["review_messages"],
        "answers": compact_rows,
    }


@st.cache_data(show_spinner=False)
def process_sheet_cached(
    name,
    image_bytes,
    answer_key_json,
    working_width,
    working_height,
    min_rectangle_area,
    use_color_filter,
    mark_threshold,
    weak_threshold,
    ambiguity_margin,
    x_tolerance,
    y_tolerance,
):
    image = np.array(Image.open(io.BytesIO(image_bytes)).convert("RGB"))
    answer_key = None if answer_key_json is None else normalize_answer_key(
        json.loads(answer_key_json)
    )

    resized_image = resize_image(image, size=(working_width, working_height))
    grayscale_image = image_to_grayscale(resized_image)
    blurred_image = add_gaussian_blur(grayscale_image)
    canny_image = apply_canny_edge_detection(blurred_image)
    rectangle_binary = prepare_rectangle_detection_image(grayscale_image)
    contours = find_external_contours(rectangle_binary)
    rectangular_contours = select_largest_rectangular_contours(
        contours=contours,
        count=2,
        min_area=min_rectangle_area,
    )
    rectangle_overlay = draw_all_contours(
        resized_image,
        rectangular_contours,
        color=(42, 117, 93),
        thickness=8,
    )
    sections = build_warped_rectangle_sections(
        image=resized_image,
        rectangular_contours=rectangular_contours,
        count=2,
    )

    if not sections:
        raise ValueError("No reference rectangles were detected.")

    answer_section = max(sections, key=lambda section: section["area"])
    other_sections = [
        section for section in sections if section["name"] != answer_section["name"]
    ]
    id_section = min(other_sections, key=lambda section: section["area"]) if other_sections else None
    answer_section_image = answer_section["warped_image"]

    raw_bubbles = detect_bubble_candidates_hough(
        answer_section_image,
        dp=1.2,
        min_distance=10,
        param1=50,
        param2=12,
        min_radius=3,
        max_radius=12,
    )

    if use_color_filter:
        filtered_bubbles = filter_bubbles_by_color_presence(
            answer_section_image,
            raw_bubbles,
        )
    else:
        filtered_bubbles = raw_bubbles

    scored_bubbles = score_bubble_marks(
        answer_section_image,
        filtered_bubbles,
    )
    bubble_overlay = draw_mark_scores(
        answer_section_image,
        scored_bubbles,
        mark_threshold=mark_threshold,
    )
    answer_rows, layout = build_answer_rows(
        scored_bubbles,
        x_tolerance=x_tolerance,
        y_tolerance=y_tolerance,
    )
    interpreted_rows = interpret_answer_rows(
        answer_rows,
        mark_threshold=mark_threshold,
        weak_threshold=weak_threshold,
        ambiguity_margin=ambiguity_margin,
    )
    grading = None

    if answer_key is not None:
        grading = grade_answers(interpreted_rows, answer_key)

    return {
        "name": name,
        "original": image,
        "resized": resized_image,
        "grayscale": grayscale_image,
        "blurred": blurred_image,
        "canny": canny_image,
        "rectangle_binary": rectangle_binary,
        "rectangle_overlay": rectangle_overlay,
        "sections": sections,
        "answer_section": answer_section,
        "id_section": id_section,
        "raw_bubbles": raw_bubbles,
        "filtered_bubbles": filtered_bubbles,
        "scored_bubbles": scored_bubbles,
        "bubble_overlay": bubble_overlay,
        "layout": layout,
        "answers": interpreted_rows,
        "review_messages": build_review_messages(interpreted_rows),
        "grading": grading,
    }


def prepare_image_sources(selected_sample_names, uploaded_files):
    sources = []
    paths_by_name = {path.name: path for path in sample_candidates()}

    for sample_name in selected_sample_names:
        path = paths_by_name[sample_name]
        image = read_image_from_path(path)
        sources.append({
            "name": path.name,
            "image": image,
            "bytes": image_to_png_bytes(image),
            "source": "sample",
        })

    for uploaded_file in uploaded_files:
        image = read_image_from_upload(uploaded_file)
        sources.append({
            "name": uploaded_file.name,
            "image": image,
            "bytes": image_to_png_bytes(image),
            "source": "upload",
        })

    if len(sources) > 2:
        st.sidebar.warning("The demo compares up to two sheets at a time. Showing the first two selected files.")
        sources = sources[:2]

    return sources


def render_sidebar():
    st.sidebar.header("Demo controls")

    sample_paths = sample_candidates()
    sample_names = [path.name for path in sample_paths]
    default_samples = sample_names[:1]

    selected_samples = st.sidebar.multiselect(
        "Preloaded sample sheets",
        options=sample_names,
        default=default_samples,
        help="Select one or two sheets for the live demo.",
    )

    uploaded_files = st.sidebar.file_uploader(
        "Upload answer sheets",
        type=["jpg", "jpeg", "png"],
        accept_multiple_files=True,
    )
    uploaded_files = uploaded_files or []

    st.sidebar.divider()
    answer_key_mode = st.sidebar.radio(
        "Answer key",
        ["Demo 100-question key", "Upload JSON key", "No grading"],
        index=0,
    )
    uploaded_key = None

    if answer_key_mode == "Upload JSON key":
        uploaded_key = st.sidebar.file_uploader(
            "Upload answer-key JSON",
            type=["json"],
            accept_multiple_files=False,
        )

    answer_key, answer_key_name = load_selected_answer_key(answer_key_mode, uploaded_key)

    st.sidebar.divider()
    with st.sidebar.expander("Algorithm settings", expanded=False):
        working_width = st.number_input("Working width", 500, 1600, 900, 50)
        working_height = st.number_input("Working height", 700, 2200, 1200, 50)
        min_rectangle_area = st.number_input("Minimum rectangle area", 1000, 50000, 5000, 500)
        use_color_filter = st.checkbox("Use color filter for orange bubbles", value=True)
        mark_threshold = st.slider("Marked bubble threshold", 0.05, 0.60, 0.18, 0.01)
        weak_threshold = st.slider("Weak mark threshold", 0.01, 0.30, 0.08, 0.01)
        ambiguity_margin = st.slider("Ambiguity margin", 0.01, 0.30, 0.08, 0.01)
        x_tolerance = st.slider("Column grouping tolerance", 4, 24, 10, 1)
        y_tolerance = st.slider("Row grouping tolerance", 4, 24, 10, 1)

    sources = prepare_image_sources(selected_samples, uploaded_files)
    settings = {
        "answer_key": answer_key,
        "answer_key_name": answer_key_name,
        "working_width": int(working_width),
        "working_height": int(working_height),
        "min_rectangle_area": int(min_rectangle_area),
        "use_color_filter": bool(use_color_filter),
        "mark_threshold": float(mark_threshold),
        "weak_threshold": float(weak_threshold),
        "ambiguity_margin": float(ambiguity_margin),
        "x_tolerance": int(x_tolerance),
        "y_tolerance": int(y_tolerance),
    }

    return sources, settings


def render_sheet_metrics(result):
    grading = result.get("grading")
    summary = grading["summary"] if grading else {}
    layout = result["layout"]

    metric_columns = st.columns(5)
    metric_columns[0].metric("Questions", layout["questions_detected"])
    metric_columns[1].metric("Option columns", layout["option_column_count"])
    metric_columns[2].metric("Raw bubbles", len(result["raw_bubbles"]))
    metric_columns[3].metric("Filtered bubbles", len(result["filtered_bubbles"]))

    if grading:
        score = summary["score_percent"]
        metric_columns[4].metric("Demo score", "n/a" if score is None else f"{score:.2f}%")
    else:
        metric_columns[4].metric("Review flags", len(result["review_messages"]))


def render_summary(results, settings):
    st.subheader("Live processing summary")
    st.caption(
        f"Answer key: {settings['answer_key_name']} | "
        f"Color filter: {'on' if settings['use_color_filter'] else 'off'} | "
        f"Mark threshold: {settings['mark_threshold']:.2f}"
    )

    for result in results:
        with st.container():
            st.markdown(f"### {result['name']}")
            render_sheet_metrics(result)

            warnings = result["layout"].get("warnings", [])
            if warnings:
                st.markdown(
                    "<div class='review-box'><strong>Layout warnings:</strong><br>"
                    + "<br>".join(warnings)
                    + "</div>",
                    unsafe_allow_html=True,
                )
            elif result["review_messages"]:
                st.markdown(
                    "<div class='review-box'><strong>Human review needed:</strong><br>"
                    + "<br>".join(result["review_messages"][:8])
                    + "</div>",
                    unsafe_allow_html=True,
                )
            else:
                st.markdown(
                    "<div class='ok-box'><strong>Checkpoint:</strong> the current settings produced no review flags for this sheet.</div>",
                    unsafe_allow_html=True,
                )


def render_image_grid(results, key, caption, grayscale=False):
    columns = st.columns(len(results))

    for column, result in zip(columns, results):
        with column:
            image_kwargs = {
                "caption": f"{result['name']} - {caption}",
                "width": "stretch",
                "clamp": True,
            }

            if not grayscale:
                image_kwargs["channels"] = "RGB"

            st.image(result[key], **image_kwargs)


def render_preprocessing_tab(results):
    st.markdown(
        "<div class='note-box'>The first stage standardizes the image and makes printed structure easier to detect.</div>",
        unsafe_allow_html=True,
    )
    st.write("Original sheet")
    render_image_grid(results, "original", "original")
    st.write("Grayscale")
    render_image_grid(results, "grayscale", "grayscale", grayscale=True)
    st.write("Gaussian blur")
    render_image_grid(results, "blurred", "blurred", grayscale=True)
    st.write("Canny edges")
    render_image_grid(results, "canny", "canny edges", grayscale=True)


def render_rectangles_tab(results):
    st.markdown(
        "<div class='note-box'>The two largest rectangle-like contours define the answer and ID regions.</div>",
        unsafe_allow_html=True,
    )
    st.write("Binary image for rectangle detection")
    render_image_grid(results, "rectangle_binary", "rectangle binary", grayscale=True)
    st.write("Detected reference rectangles")
    render_image_grid(results, "rectangle_overlay", "rectangle overlay")


def render_warp_tab(results):
    st.markdown(
        "<div class='note-box'>Perspective correction turns each detected region into a clean rectangular section.</div>",
        unsafe_allow_html=True,
    )
    columns = st.columns(len(results))

    for column, result in zip(columns, results):
        with column:
            st.markdown(f"### {result['name']}")
            st.image(
                result["answer_section"]["warped_image"],
                caption="Warped answer section",
                width="stretch",
            )

            if result["id_section"] is not None:
                st.image(
                    result["id_section"]["warped_image"],
                    caption="Warped ID section",
                    width="stretch",
                )
            else:
                st.info("Only one reference rectangle was detected.")


def render_bubbles_tab(results):
    st.markdown(
        "<div class='note-box'>Green circles are detected candidates. Red circles pass the dark-mark threshold.</div>",
        unsafe_allow_html=True,
    )
    render_image_grid(results, "bubble_overlay", "bubble candidates and marked bubbles")

    for result in results:
        with st.expander(f"Bubble counts for {result['name']}"):
            st.write({
                "raw_candidates": len(result["raw_bubbles"]),
                "filtered_candidates": len(result["filtered_bubbles"]),
                "scored_candidates": len(result["scored_bubbles"]),
            })


def answer_table_rows(rows, limit=None):
    if limit is not None:
        rows = rows[:limit]

    table_rows = []

    for row in rows:
        table_rows.append({
            "Q": row["question"],
            "Answer": row["answer"],
            "Status": row["status"],
            "Confidence": row["confidence"],
            "A": row["scores"].get("A"),
            "B": row["scores"].get("B"),
            "C": row["scores"].get("C"),
            "D": row["scores"].get("D"),
            "Correct": row.get("correct_answer"),
            "Needs review": row["needs_review"],
            "Reason": row["review_reason"],
        })

    return table_rows


def render_grading_tab(results):
    st.markdown(
        "<div class='note-box'>Method 1 groups bubbles into rows and produces answer values plus review flags.</div>",
        unsafe_allow_html=True,
    )

    for result in results:
        st.markdown(f"### {result['name']}")
        grading = result.get("grading")
        rows = grading["answers"] if grading else result["answers"]

        if grading:
            st.write("Summary")
            st.json(grading["summary"], expanded=False)

        review_messages = result["review_messages"]
        if review_messages:
            st.markdown(
                "<div class='review-box'><strong>Review messages</strong><br>"
                + "<br>".join(review_messages)
                + "</div>",
                unsafe_allow_html=True,
            )
        else:
            st.markdown(
                "<div class='ok-box'><strong>No review flags</strong> with the current thresholds.</div>",
                unsafe_allow_html=True,
            )

        st.dataframe(
            answer_table_rows(rows),
            width="stretch",
            hide_index=True,
        )

        report_payload = make_report_payload(result)
        st.download_button(
            "Download JSON report",
            data=safe_json_bytes(report_payload),
            file_name=f"{Path(result['name']).stem}_omr_report.json",
            mime="application/json",
            key=f"download-{result['name']}",
        )


def render_comparison_tab(results):
    if len(results) < 2:
        st.info("Select or upload two sheets to compare them side by side.")
        return

    left, right = results[0], results[1]
    st.markdown(
        "<div class='note-box'>This view compares detected answers question by question.</div>",
        unsafe_allow_html=True,
    )

    rows_left = left["grading"]["answers"] if left.get("grading") else left["answers"]
    rows_right = right["grading"]["answers"] if right.get("grading") else right["answers"]
    max_rows = min(len(rows_left), len(rows_right))
    differences = []

    for index in range(max_rows):
        left_row = rows_left[index]
        right_row = rows_right[index]

        if left_row["answer"] != right_row["answer"] or left_row["status"] != right_row["status"]:
            differences.append({
                "Q": left_row["question"],
                left["name"]: left_row["answer"],
                f"{left['name']} status": left_row["status"],
                right["name"]: right_row["answer"],
                f"{right['name']} status": right_row["status"],
            })

    metric_columns = st.columns(3)
    metric_columns[0].metric(left["name"], len(rows_left))
    metric_columns[1].metric(right["name"], len(rows_right))
    metric_columns[2].metric("Different rows", len(differences))

    if differences:
        st.dataframe(differences, width="stretch", hide_index=True)
    else:
        st.success("The detected answers match for the compared question range.")


def render_method_note():
    st.markdown(
        """
        <div class="note-box">
        <strong>Demo story:</strong> this application is not just a grader. It is a visual explanation of the computer vision path:
        preprocessing, rectangle detection, perspective correction, bubble detection, and Method 1 classical grading.
        </div>
        """,
        unsafe_allow_html=True,
    )


def main():
    setup_page()
    show_header()
    sources, settings = render_sidebar()
    render_method_note()

    if not sources:
        st.info("Select a preloaded sample or upload one or two answer sheets to start the demo.")
        return

    answer_key_json = None
    if settings["answer_key"] is not None:
        answer_key_json = json.dumps(settings["answer_key"], sort_keys=True)

    results = []

    with st.spinner("Running the OMR pipeline..."):
        for source in sources:
            try:
                result = process_sheet_cached(
                    source["name"],
                    source["bytes"],
                    answer_key_json,
                    settings["working_width"],
                    settings["working_height"],
                    settings["min_rectangle_area"],
                    settings["use_color_filter"],
                    settings["mark_threshold"],
                    settings["weak_threshold"],
                    settings["ambiguity_margin"],
                    settings["x_tolerance"],
                    settings["y_tolerance"],
                )
                results.append(result)
            except Exception as exc:
                st.error(f"Could not process {source['name']}: {exc}")

    if not results:
        return

    render_summary(results, settings)

    tabs = st.tabs([
        "Preprocessing",
        "Rectangles",
        "Warped sections",
        "Bubbles",
        "Classical grading",
        "Compare sheets",
    ])

    with tabs[0]:
        render_preprocessing_tab(results)
    with tabs[1]:
        render_rectangles_tab(results)
    with tabs[2]:
        render_warp_tab(results)
    with tabs[3]:
        render_bubbles_tab(results)
    with tabs[4]:
        render_grading_tab(results)
    with tabs[5]:
        render_comparison_tab(results)


if __name__ == "__main__":
    main()
