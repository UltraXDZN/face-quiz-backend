"""Pure scoring helper for live exam state.

This MUST stay numerically identical to the per-task math in
`api.solutions.routes._calculate_user_results`. The two formulas are kept
in separate places only because `_calculate_user_results` also builds the
per-task/per-group breakdown response payload. If you change point
semantics, change both.

Defaults (`positive_points=1.0`, `negative_points=0.3`) and the
`max(0, achieved)` floor mirror the existing finalize path so admin live
scores and final saved scores match.
"""

from __future__ import annotations


def compute_live_score(
    exam_data: dict, answers: dict[str, bool | None]
) -> tuple[float, float]:
    """Returns `(achieved_points, total_points)` for the given answer map.

    `answers` is keyed by task id; missing or `None` values are treated as
    unanswered (no points added, no points subtracted).
    """
    total = 0.0
    achieved = 0.0
    for group in exam_data.get("groups", []):
        for task in group.get("tasks", []):
            positive = task.get("positive_points", 1.0)
            negative = task.get("negative_points", 0.3)
            total += positive

            user_answer = answers.get(task.get("id"))
            if user_answer is None:
                continue
            if user_answer == task.get("state"):
                achieved += positive
            else:
                achieved -= negative
    return max(0.0, achieved), total
