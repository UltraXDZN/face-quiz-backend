"""Tests for solution-driven scoring.

Scoring and result ordering are now driven from the submitted solutions list
(stored in Firestore in the exact order the student saw the tasks). There is
no shuffle re-computation on the scoring path, so PYTHONHASHSEED instability
across server restarts cannot affect results.
"""

from api.solutions.routes import _build_task_lookup, _calculate_user_results


def _make_exam(n_groups: int, tasks_per_group: int, **kwargs) -> dict:
    return {
        "id": "exam1",
        "title": "Test Exam",
        "groups": [
            {
                "id": f"g{g}",
                "text": f"Group {g}",
                "tasks": [
                    {
                        "id": f"g{g}t{t}",
                        "text": f"Task g{g}t{t}",
                        "state": True,
                        "positive_points": 1.0,
                        "negative_points": 0.3,
                    }
                    for t in range(tasks_per_group)
                ],
            }
            for g in range(n_groups)
        ],
        **kwargs,
    }


def _make_solutions(task_ids: list, states: list) -> list:
    return [{"id": tid, "state": s} for tid, s in zip(task_ids, states)]


# ── _build_task_lookup ────────────────────────────────────────────────────────

def test_build_task_lookup_all_tasks():
    exam = _make_exam(3, 4)
    task_lookup, task_to_group = _build_task_lookup(exam)
    assert len(task_lookup) == 12
    assert len(task_to_group) == 12


def test_build_task_lookup_correct_group_mapping():
    exam = _make_exam(2, 3)
    _, task_to_group = _build_task_lookup(exam)
    assert task_to_group["g0t0"]["id"] == "g0"
    assert task_to_group["g1t2"]["id"] == "g1"


def test_build_task_lookup_task_data_intact():
    exam = _make_exam(1, 2)
    task_lookup, _ = _build_task_lookup(exam)
    assert task_lookup["g0t0"]["state"] is True
    assert task_lookup["g0t0"]["positive_points"] == 1.0


# ── _calculate_user_results: ordering ────────────────────────────────────────

class _FakeDB:
    """Minimal stub so _calculate_user_results can run without Firestore."""
    class _Col:
        def document(self, *a): return self
        def get(self): return type("Doc", (), {"exists": False})()
    def collection(self, *a): return self._Col()


def test_results_follow_submission_order():
    """Groups in the result must match the order of the submitted solutions."""
    exam = _make_exam(3, 2)
    # Submit in reverse group order: g2, g1, g0
    solutions = (
        _make_solutions(["g2t0", "g2t1"], [True, False])
        + _make_solutions(["g1t0", "g1t1"], [None, True])
        + _make_solutions(["g0t0", "g0t1"], [False, True])
    )
    result = _calculate_user_results("exam1", "pw", "s@x", exam, solutions, _FakeDB())
    assert [g.id for g in result.groups] == ["g2", "g1", "g0"]


def test_results_task_order_within_group():
    """Task order inside a group must match submission order."""
    exam = _make_exam(1, 4)
    # Submit tasks in reverse order: t3, t2, t1, t0
    solutions = _make_solutions(["g0t3", "g0t2", "g0t1", "g0t0"], [True, False, None, True])
    result = _calculate_user_results("exam1", "pw", "s@x", exam, solutions, _FakeDB())
    task_ids = [t.id for t in result.groups[0].tasks]
    assert task_ids == ["g0t3", "g0t2", "g0t1", "g0t0"]


# ── _calculate_user_results: scoring ─────────────────────────────────────────

def test_correct_answer_scores_positive_points():
    exam = _make_exam(1, 1)
    solutions = _make_solutions(["g0t0"], [True])  # correct answer is True
    result = _calculate_user_results("exam1", "pw", "s@x", exam, solutions, _FakeDB())
    assert result.achieved_points == 1.0
    assert result.correct_count == 1
    assert result.incorrect_count == 0


def test_wrong_answer_deducts_negative_points():
    exam = _make_exam(1, 1)
    solutions = _make_solutions(["g0t0"], [False])  # correct is True, student says False
    result = _calculate_user_results("exam1", "pw", "s@x", exam, solutions, _FakeDB())
    assert result.achieved_points == 0.0  # clamped at 0
    assert result.incorrect_count == 1


def test_null_answer_counts_as_unanswered():
    exam = _make_exam(1, 1)
    solutions = _make_solutions(["g0t0"], [None])
    result = _calculate_user_results("exam1", "pw", "s@x", exam, solutions, _FakeDB())
    assert result.achieved_points == 0.0
    assert result.unanswered_count == 1


def test_denominator_matches_submitted_task_count():
    """Only submitted tasks count toward the denominator; full pool is irrelevant."""
    exam = _make_exam(7, 7)  # 49 tasks in pool
    # Student submitted only 25 tasks (e.g. 5 groups × 5 tasks)
    submitted_ids = [f"g{g}t{t}" for g in range(5) for t in range(5)]
    solutions = _make_solutions(submitted_ids, [True] * 25)
    result = _calculate_user_results("exam1", "pw", "s@x", exam, solutions, _FakeDB())
    assert result.total_tasks == 25
    assert result.total_points == 25.0


def test_unknown_task_id_in_solutions_is_skipped():
    """A stale/invalid task ID in the solution does not cause an error."""
    exam = _make_exam(1, 2)
    solutions = [
        {"id": "g0t0", "state": True},
        {"id": "nonexistent_task", "state": True},
        {"id": "g0t1", "state": False},
    ]
    result = _calculate_user_results("exam1", "pw", "s@x", exam, solutions, _FakeDB())
    assert result.total_tasks == 2  # only the two valid tasks
