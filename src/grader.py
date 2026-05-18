import json


def load_answer_key(path: str) -> dict:
    """
    Loads the answer key from JSON.
    """
    with open(path, "r", encoding="utf-8") as file:
        return json.load(file)


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
                "status": "missing"
            }
            wrong += 1
            continue

        status = result["status"]
        selected = result["selected"]

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

        detailed_results[question] = {
            "student_answer": selected,
            "correct_answer": correct_answer,
            "is_correct": is_correct,
            "status": status,
            "confidence": result["confidence"]
        }

    total = len(answer_key)
    score_percent = round((correct / total) * 100, 2)

    summary = {
        "correct": correct,
        "wrong": wrong,
        "blank": blank,
        "multiple": multiple,
        "unclear": unclear,
        "total": total,
        "score_percent": score_percent
    }

    return summary, detailed_results