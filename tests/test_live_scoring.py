"""Tests for `compute_live_score` — the pure helper that must stay numerically
identical to `_calculate_user_results`."""

from api.live.scoring import compute_live_score


def _exam(*tasks: tuple[str, bool | None, float, float]) -> dict:
    """Build a minimal exam_data with a single group, given a list of
    `(task_id, correct_state, positive_points, negative_points)`."""
    return {
        "groups": [
            {
                "tasks": [
                    {
                        "id": tid,
                        "state": state,
                        "positive_points": pos,
                        "negative_points": neg,
                    }
                    for tid, state, pos, neg in tasks
                ]
            }
        ]
    }


def test_all_correct_full_score():
    exam = _exam(("t1", True, 2.0, 0.5), ("t2", False, 3.0, 0.5))
    achieved, total = compute_live_score(exam, {"t1": True, "t2": False})
    assert (achieved, total) == (5.0, 5.0)


def test_all_wrong_floors_at_zero():
    exam = _exam(("t1", True, 1.0, 0.3), ("t2", True, 1.0, 0.3))
    achieved, total = compute_live_score(exam, {"t1": False, "t2": False})
    # -0.6 raw, floored to 0
    assert achieved == 0.0
    assert total == 2.0


def test_mixed_correct_and_wrong():
    exam = _exam(("t1", True, 2.0, 0.5), ("t2", False, 2.0, 0.5))
    achieved, total = compute_live_score(exam, {"t1": True, "t2": True})
    # +2 correct, -0.5 wrong = 1.5
    assert achieved == 1.5
    assert total == 4.0


def test_unanswered_neither_adds_nor_subtracts():
    exam = _exam(("t1", True, 1.0, 0.3), ("t2", False, 1.0, 0.3))
    achieved, total = compute_live_score(exam, {"t1": True})  # t2 not in dict
    assert (achieved, total) == (1.0, 2.0)


def test_explicit_none_is_unanswered():
    exam = _exam(("t1", True, 1.0, 0.3))
    achieved, total = compute_live_score(exam, {"t1": None})
    assert (achieved, total) == (0.0, 1.0)


def test_defaults_match_calculate_user_results():
    # When positive_points/negative_points are absent, defaults must match
    # _calculate_user_results: positive=1.0, negative=0.3.
    exam = {"groups": [{"tasks": [{"id": "t1", "state": True}]}]}
    achieved_correct, _ = compute_live_score(exam, {"t1": True})
    achieved_wrong, _ = compute_live_score(exam, {"t1": False})
    assert achieved_correct == 1.0
    assert achieved_wrong == 0.0  # -0.3, floored to 0


def test_empty_exam():
    achieved, total = compute_live_score({"groups": []}, {})
    assert (achieved, total) == (0.0, 0.0)
