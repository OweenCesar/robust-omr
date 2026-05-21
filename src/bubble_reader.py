import cv2
import numpy as np


def create_circular_mask(shape, center, radius):
    """
    Creates a circular mask for a bubble.
    """
    height, width = shape[:2]

    mask = np.zeros((height, width), dtype=np.uint8)

    cv2.circle(
        mask,
        center,
        radius,
        255,
        thickness=-1
    )

    return mask


def read_single_bubble(thresholded_image, center, radius, inner_radius_ratio=0.55):
    """
    Reads one bubble by calculating the ratio of dark pixels inside it.

    Since the thresholded image uses THRESH_BINARY_INV:
    - black marks become white pixels
    - white paper becomes black pixels
    
    So countNonZero tells us how much ink/mark is inside the bubble.

    The mask uses the inner part of the bubble, which avoids counting the
    printed circle outline as a student's mark.
    """
    x, y = center
    inner_radius = max(2, int(round(radius * inner_radius_ratio)))

    mask = create_circular_mask(
        thresholded_image.shape,
        center=(x, y),
        radius=inner_radius
    )

    bubble_pixels = cv2.bitwise_and(thresholded_image, thresholded_image, mask=mask)

    total_area = np.count_nonzero(mask)
    marked_area = cv2.countNonZero(bubble_pixels)

    mark_ratio = marked_area / total_area if total_area > 0 else 0

    return mark_ratio


def read_all_bubbles(thresholded_image, bubble_centers, radius, inner_radius_ratio=0.55):
    """
    Reads all bubbles and returns mark ratios for every question and option.
    """
    results = {}

    for question, options in bubble_centers.items():
        results[question] = {}

        for option, center in options.items():
            mark_ratio = read_single_bubble(
                thresholded_image=thresholded_image,
                center=center,
                radius=radius,
                inner_radius_ratio=inner_radius_ratio
            )

            results[question][option] = mark_ratio

    return results


def interpret_answers(
    bubble_results,
    blank_threshold=0.08,
    marked_threshold=0.35,
    ambiguity_margin=0.12
):
    """
    Converts bubble mark ratios into answers.

    Possible statuses:
    - valid
    - blank
    - multiple
    - unclear
    """
    interpreted = {}

    for question, options in bubble_results.items():
        sorted_options = sorted(
            options.items(),
            key=lambda item: item[1],
            reverse=True
        )

        best_option, best_score = sorted_options[0]
        second_score = sorted_options[1][1] if len(sorted_options) > 1 else 0

        marked_options = [
            option for option, score in options.items()
            if score >= marked_threshold
        ]

        if best_score < blank_threshold:
            interpreted[question] = {
                "selected": None,
                "status": "blank",
                "confidence": round(best_score, 3),
                "scores": options
            }

        elif len(marked_options) > 1:
            interpreted[question] = {
                "selected": marked_options,
                "status": "multiple",
                "review_reason": "multiple_marks",
                "confidence": round(best_score, 3),
                "scores": options
            }

        elif best_score < marked_threshold or best_score - second_score < ambiguity_margin:
            review_reason = "weak_mark" if best_score < marked_threshold else "ambiguous_mark"
            interpreted[question] = {
                "selected": best_option,
                "status": "unclear",
                "review_reason": review_reason,
                "confidence": round(best_score - second_score, 3),
                "scores": options
            }

        else:
            interpreted[question] = {
                "selected": best_option,
                "status": "valid",
                "confidence": round(best_score, 3),
                "scores": options
            }

    return interpreted
