"""Router-only tests — no LangGraph required.

Three branches, three tests. ``route_after_verify`` is a pure function of
``state``, so we just hand it dicts.
"""

from __future__ import annotations

from agents.graph import route_after_verify


def test_verified_routes_to_report():
    state = {"verdict": "verified", "attempt": 1, "max_attempts": 3}
    assert route_after_verify(state) == "report"


def test_budget_exhausted_routes_to_report():
    state = {"verdict": "still_vulnerable", "attempt": 3, "max_attempts": 3}
    assert route_after_verify(state) == "report"


def test_failure_with_budget_left_routes_to_self_heal():
    state = {"verdict": "compile_error", "attempt": 1, "max_attempts": 3}
    assert route_after_verify(state) == "self_heal"


def test_verified_wins_even_on_last_attempt():
    """A fix that lands on the last allowed attempt is still a success."""
    state = {"verdict": "verified", "attempt": 3, "max_attempts": 3}
    assert route_after_verify(state) == "report"
