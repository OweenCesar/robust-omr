import json


REVIEW_STATUSES = {"unclear", "multiple", "missing"}


def load_answer_key(path: str) -> dict:
    """
    Loads the answer key from JSON.
    """
    with open(path, "r", encoding="utf-8") as file:
        answer_key = json.load(file)

    return {
        str(question): str(answer).upper()
        for question, answer in answer_key.items()
    }


def _format_answer(answer):
    if answer is None:
        return "none"

    if isinstance(answer, list):
        return ", ".join(str(option) for option in answer)

    return str(answer)


def _top_scores(scores, limit=2):
    ranked_scores = sorted(
        scores.items(),
        key=lambda item: item[1],
        reverse=True
    )

    return [
        {
            "option": option,
            "score": round(float(score), 3)
        }
        for option, score in ranked_scores[:limit]
    ]


def build_review_message(question, result):
    status = result["status"]

    if status == "multiple":
        selected = _format_answer(result["student_answer"])
        return f"Q{question}: multiple marks detected ({selected}) - please review manually."

    if status == "unclear":
        top_scores = result.get("top_scores", [])
        review_reason = result.get("review_reason")

        if review_reason == "weak_mark":
            selected = _format_answer(result["student_answer"])
            return f"Q{question}: weak mark detected near {selected} - please review manually."

        if len(top_scores) >= 2:
            first = top_scores[0]
            second = top_scores[1]
            return (
                f"Q{question}: unclear mark between {first['option']} "
                f"and {second['option']} - please review manually."
            )

        selected = _format_answer(result["student_answer"])
        return f"Q{question}: unclear mark near {selected} - please review manually."

    if status == "missing":
        return f"Q{question}: answer region was not detected - check the photo/template."

    return ""


def build_review_items(detailed_results):
    """
    Builds a compact list of answers that should be checked by a human.
    """
    review_items = []

    for question, result in detailed_results.items():
        if result["status"] not in REVIEW_STATUSES:
            continue

        review_items.append({
            "question": question,
            "status": result["status"],
            "reason": result.get("review_reason"),
            "student_answer": result["student_answer"],
            "correct_answer": result["correct_answer"],
            "message": build_review_message(question, result),
            "top_scores": result.get("top_scores", [])
        })

    return review_items


def grade_answers(interpreted_answers, answer_key):
    """
    Grades interpreted OMR answers against the answer key.
    """
    correct = 0
    wrong = 0
    blank = 0
    multiple = 0
    unclear = 0

    detailed_results = {}

    for question, correct_answer in answer_key.items():
        result = interpreted_answers.get(question)

        if result is None:
            detailed_results[question] = {
                "student_answer": None,
                "correct_answer": correct_answer,
                "is_correct": False,
                "status": "missing",
                "review_required": True,
                "review_message": f"Q{question}: answer region was not detected - check the photo/template."
            }
            wrong += 1
            continue

        status = result["status"]
        selected = result["selected"]
        top_scores = _top_scores(result.get("scores", {}))

        is_correct = status == "valid" and selected == correct_answer

        if is_correct:
            correct += 1
        elif status == "blank":
            blank += 1
        elif status == "multiple":
            multiple += 1
        elif status == "unclear":
            unclear += 1
        else:
            wrong += 1

        detailed_result = {
            "student_answer": selected,
            "correct_answer": correct_answer,
            "is_correct": is_correct,
            "status": status,
            "confidence": result["confidence"],
            "review_required": status in REVIEW_STATUSES,
            "review_reason": result.get("review_reason"),
            "top_scores": top_scores
        }

        if detailed_result["review_required"]:
            detailed_result["review_message"] = build_review_message(
                question,
                detailed_result
            )

        detailed_results[question] = detailed_result

    total = len(answer_key)
    score_percent = round((correct / total) * 100, 2) if total else 0.0
    review_items = build_review_items(detailed_results)

    summary = {
        "correct": correct,
        "wrong": wrong,
        "blank": blank,
        "multiple": multiple,
        "unclear": unclear,
        "needs_review": len(review_items),
        "total": total,
        "score": correct,
        "score_percent": score_percent
    }

    return summary, detailed_results
