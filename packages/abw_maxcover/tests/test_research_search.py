import json
from pathlib import Path

import numpy as np
import pytest

from abw_maxcover import (
    ResearchSearchConfig,
    build_instance_from_facility_map,
    repair_saved_solution,
    run_research_search,
)


def original_score(ins, sites):
    covered = set().union(*(set(map(int, ins.demand_of(j))) for j in sites))
    return sum(int(ins.weights[d]) for d in covered)


def trap():
    rows = {i: list(range(i * 3, i * 3 + 3)) for i in range(3)}
    rows.update({3 + j: [j, j + 3, j + 6, 9 + j] for j in range(3)})
    rows.update({6: [12], 7: []})
    return build_instance_from_facility_map(rows, [1] * 12 + [4], n_facilities=8)


def test_pool_repair_escapes_three_exchange_trap():
    ins = trap()
    sites = [0, 1, 2, 6]
    # Every one-for-one move fails; improvement requires changing three sites.
    before = original_score(ins, sites)
    assert before == 13
    for out in sites:
        for entry in set(range(8)) - set(sites):
            assert original_score(ins, [j for j in sites if j != out] + [entry]) <= before
    result = repair_saved_solution(ins, sites, time_limit_seconds=5)
    assert result.objective == original_score(ins, result.solution) == 16
    assert len(set(result.solution)) == 4 and sites == [0, 1, 2, 6]
    assert result.metadata["pool_search"]["completed"]
    assert result.upper_bound is None


def test_expired_repair_retains_parent(monkeypatch):
    import abw_maxcover.research_search as module

    ticks = iter([0.0, 100.0, 100.0])
    monkeypatch.setattr(module, "perf_counter", lambda: next(ticks))
    result = repair_saved_solution(trap(), [0, 1, 2, 6], time_limit_seconds=0.001)
    assert result.objective == 13 and result.solution == [0, 1, 2, 6]


@pytest.mark.parametrize("time", [True, 0, -1, float("nan"), float("inf")])
def test_repair_rejects_invalid_time(time):
    with pytest.raises(ValueError):
        repair_saved_solution(trap(), [0], time_limit_seconds=time)


@pytest.mark.parametrize("sites", [[0, 0], [-1], [True], [], [8]])
def test_repair_rejects_invalid_sites(sites):
    with pytest.raises(ValueError):
        repair_saved_solution(trap(), sites)


@pytest.mark.parametrize(
    "setting",
    [
        {"seed": True},
        {"seed": 1.5},
        {"fake_clock": 1},
        {"time_limit_seconds": True},
        {"pool_repair": 1},
        {"repair_fraction": float("nan")},
    ],
)
def test_research_rejects_invalid_controls(setting):
    with pytest.raises(ValueError):
        run_research_search(trap(), 4, config=ResearchSearchConfig(**setting))


def test_frozen_record_core_trace():
    fixture = json.loads((Path(__file__).parent / "fixtures/research_trace.json").read_text())
    ins = build_instance_from_facility_map(
        {int(j): row for j, row in fixture["rows"].items()},
        fixture["weights"],
        n_facilities=len(fixture["rows"]),
    )
    for case in fixture["cases"]:
        trace = []
        events = []
        result = run_research_search(
            ins,
            fixture["budget"],
            config=ResearchSearchConfig(**case["config"]),
            facility_coordinates=np.array(fixture["coordinates"]),
            facility_neighbors=[np.array(row) for row in fixture["neighbors"]],
            trace_callback=lambda i, d, trace=trace: trace.append([i, d]),
            incumbent_callback=events.append,
        )
        assert trace == case["trace"]
        assert result.objective == case["objective"] == original_score(ins, result.solution)
        assert result.solution == case["sites"]
        assert all(original_score(ins, e["sites"]) == e["objective"] for e in events)
        assert [e["objective"] for e in events] == sorted({e["objective"] for e in events})


def test_solver_free_import_and_search(monkeypatch):
    import sys

    monkeypatch.setitem(sys.modules, "gurobipy", None)
    ins = trap()
    events = []
    result = run_research_search(
        ins,
        4,
        config=ResearchSearchConfig(
            iterations=2, fake_clock=True, time_limit_seconds=5, pool_repair=True
        ),
        facility_coordinates=np.zeros((8, 2)),
        facility_neighbors=[np.arange(8) for _ in range(8)],
        incumbent_callback=events.append,
    )
    assert result.objective == original_score(ins, result.solution)
    assert len(result.solution) == len(set(result.solution)) == 4
    assert all(e["elapsed_seconds"] <= result.total_seconds for e in events)


def test_numpy_123_subset_count_fallback(monkeypatch):
    import abw_maxcover.component_recombine as module

    module.subset_counts.cache_clear()
    if hasattr(np, "bitwise_count"):
        monkeypatch.delattr(np, "bitwise_count")
    assert module.subset_counts(8).tolist() == [j.bit_count() for j in range(256)]
    module.subset_counts.cache_clear()
