def grade_answers(interpreted_answers, answer_key):
    """
    Grades interpreted OMR answers against the answer key.

    interpreted_answers comes from the scanner and includes both the selected
    answer and a status. Only a status of "valid" can count as correct. Blank,
    multiple, and unclear answers are kept separate so the teacher can see what
    needs review instead of getting a single unexplained wrong count.
    """
    correct = 0
    wrong = 0
    blank = 0
    multiple = 0
    unclear = 0

    detailed_results = {}

    for question, correct_answer in sorted(answer_key.items(), key=lambda item: int(item[0])):
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
            "confidence": result.get("confidence", 0)
        }

    total = len(answer_key)
    score_percent = round((correct / total) * 100, 2) if total else 0

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
