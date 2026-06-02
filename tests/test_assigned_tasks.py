"""Tests for _get_assigned_tasks / _get_assigned_groups: helpers that return
only the tasks/groups the student was actually shown, in per-student order."""

from api.solutions.routes import _get_assigned_tasks, _get_assigned_groups


def _make_exam(n_groups: int, tasks_per_group: int, **kwargs) -> dict:
    return {
        "id": "exam1",
        "groups": [
            {
                "id": f"g{g}",
                "tasks": [{"id": f"g{g}t{t}", "state": True} for t in range(tasks_per_group)],
            }
            for g in range(n_groups)
        ],
        **kwargs,
    }


def test_no_limits_returns_all_tasks():
    exam = _make_exam(3, 5)
    tasks = _get_assigned_tasks(exam, "a@x")
    assert len(tasks) == 15


def test_number_of_displayed_tasks_limits_per_group():
    exam = _make_exam(3, 10, numberOfDisplayedTasks=5)
    tasks = _get_assigned_tasks(exam, "a@x")
    # 3 groups × 5 tasks each
    assert len(tasks) == 15


def test_number_of_displayed_groups_limits_groups():
    exam = _make_exam(5, 4, numberOfDisplayedQuestions=2, shuffleQuestions=True)
    tasks = _get_assigned_tasks(exam, "a@x")
    # 2 groups × 4 tasks
    assert len(tasks) == 8


def test_both_limits():
    exam = _make_exam(5, 10, numberOfDisplayedQuestions=3, shuffleQuestions=True, numberOfDisplayedTasks=4)
    tasks = _get_assigned_tasks(exam, "a@x")
    # 3 groups × 4 tasks
    assert len(tasks) == 12


def test_shuffle_deterministic_per_email():
    exam = _make_exam(1, 10, shuffleTasks=True)
    ids_a = [t["id"] for t in _get_assigned_tasks(exam, "alice@x")]
    ids_a2 = [t["id"] for t in _get_assigned_tasks(exam, "alice@x")]
    ids_b = [t["id"] for t in _get_assigned_tasks(exam, "bob@x")]
    assert ids_a == ids_a2, "same email must yield same order"
    # Different emails almost certainly yield different orders (not guaranteed but near-certain)
    assert set(ids_a) == set(ids_b), "same tasks, different order allowed"


def test_denominator_25_not_34():
    """Regression: 5 groups × 5 tasks = 25 denominator, not 34 (full pool)."""
    exam = _make_exam(7, 7, numberOfDisplayedQuestions=5, shuffleQuestions=True, numberOfDisplayedTasks=5)
    tasks = _get_assigned_tasks(exam, "student@unipu.hr")
    assert len(tasks) == 25


def test_no_shuffle_no_email_still_works():
    exam = _make_exam(2, 3)
    tasks = _get_assigned_tasks(exam, "")
    assert len(tasks) == 6


def test_assigned_groups_preserves_shuffle_order():
    """Groups returned by _get_assigned_groups must match the shuffled task order seen by the student."""
    exam = _make_exam(1, 5, shuffleTasks=True)
    groups = _get_assigned_groups(exam, "alice@x")
    flat_from_groups = [t["id"] for t in groups[0]["tasks"]]
    flat_direct = [t["id"] for t in _get_assigned_tasks(exam, "alice@x")]
    assert flat_from_groups == flat_direct


def test_assigned_groups_order_differs_per_student():
    """Two different students get different task orderings from _get_assigned_groups."""
    exam = _make_exam(1, 8, shuffleTasks=True)
    order_a = [t["id"] for t in _get_assigned_groups(exam, "alice@x")[0]["tasks"]]
    order_b = [t["id"] for t in _get_assigned_groups(exam, "bob@x")[0]["tasks"]]
    assert set(order_a) == set(order_b), "same tasks"
    assert order_a != order_b, "different per-student order"
