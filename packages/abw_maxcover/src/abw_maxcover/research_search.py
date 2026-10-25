"""Opt-in solver-free research search and protected candidate-pool finishing."""

from dataclasses import dataclass, replace
from math import isfinite
from time import perf_counter

import numpy as np

from .exclusion_neighborhood import exclusion_neighborhood
from .java_ils import JavaILSConfig, State, run_java_ils
from .results import MaxCoverResult


@dataclass
class ResearchSearchConfig(JavaILSConfig):
    """Kings record-producing core; optional finishing shares its time budget.

    This is an experimental preset, not a general benchmark winner. Geography
    guides perturbation; coverage and every accepted score use input incidence.
    """

    faithful_local: bool = True
    initial_starts: int = 8
    pool_repair: bool = False
    repair_fraction: float = 0.1
    repair_extra_cap: int = 24


def repair_saved_solution(instance, sites, *, time_limit_seconds=1.0, extra_cap=24):
    """Improve an exact-cardinality incumbent in a restricted near-neutral pool.

    Retains the supplied solution if preparation/DP expires or no gain is found.
    A completed pool calculation is not a whole-instance optimality certificate.
    Input state construction and validation are paid by the soft wall-clock cap.
    """
    started = perf_counter()
    if (
        type(time_limit_seconds) not in (int, float)
        or not isfinite(time_limit_seconds)
        or time_limit_seconds <= 0
    ):
        raise ValueError("Positive finite repair time required")
    if type(extra_cap) is not int or not 0 <= extra_cap <= 160:
        raise ValueError("extra_cap must be an integer within 0..160")
    sites = list(sites)
    if (
        not sites
        or len(sites) != len(set(sites))
        or any(type(j) is not int or not 0 <= j < instance.n_facilities for j in sites)
    ):
        raise ValueError("Unique nonempty integer sites required")
    if np.any(instance.weights < 0) or sum(map(int, instance.weights)) > np.iinfo(np.int64).max:
        raise ValueError("Nonnegative int64-safe weights required")
    state = State(instance, len(sites))
    state.gains[:] = [
        sum(map(int, instance.weights[instance.demand_of(j)])) for j in range(instance.n_facilities)
    ]
    for pos, j in enumerate(sites):
        state.add(j)
        state.swap_positions(pos, int(state.positions[j]))
    parent = state.f
    child, details = exclusion_neighborhood(
        state, started + time_limit_seconds, extra_cap=extra_cap
    )
    winner = child if child is not None and child.f > parent else state
    selected = list(map(int, winner.order[: winner.p]))
    covered = np.zeros(instance.n_demand, dtype=bool)
    for j in selected:
        covered[instance.demand_of(j)] = True
    actual = sum(map(int, instance.weights[covered]))
    if len(set(selected)) != len(sites) or actual != winner.f or actual < parent:
        raise RuntimeError("Candidate-pool result failed original-incidence validation")
    elapsed = perf_counter() - started
    return MaxCoverResult(
        budget=len(sites),
        method="near_neutral_pool",
        objective=actual,
        solution=selected,
        coverage=winner.coverage.copy(),
        total_seconds=elapsed,
        solve_seconds=elapsed,
        metadata=dict(
            parent_objective=parent, conditional_pool=True, pool_search=details, experimental=True
        ),
    )


def run_research_search(
    instance,
    budget,
    *,
    config=None,
    facility_coordinates=None,
    facility_neighbors=None,
    incumbent_callback=None,
    trace_callback=None,
):
    """Run the eight-start faithful-local/dynamic/component ILS preset.

    Optional near-neutral repair reserves repair_fraction of the same total
    allowance. This protects its own first-phase result; it does not guarantee
    matching the core when the core receives the entire allowance. No solver.
    """
    started = perf_counter()
    cfg = config or ResearchSearchConfig()
    if not isinstance(cfg, ResearchSearchConfig):
        raise TypeError("ResearchSearchConfig required")
    if type(cfg.seed) is not int or type(cfg.fake_clock) is not bool:
        raise ValueError("Integer seed and boolean fake_clock required")
    if type(cfg.pool_repair) is not bool:
        raise ValueError("pool_repair must be bool")
    if (
        type(cfg.repair_fraction) not in (int, float)
        or not isfinite(cfg.repair_fraction)
        or not 0 < cfg.repair_fraction < 1
    ):
        raise ValueError("repair_fraction must lie strictly between 0 and 1")
    if type(cfg.repair_extra_cap) is not int or not 0 <= cfg.repair_extra_cap <= 160:
        raise ValueError("repair_extra_cap must be integer within 0..160")
    if (
        type(cfg.time_limit_seconds) not in (int, float)
        or not isfinite(cfg.time_limit_seconds)
        or cfg.time_limit_seconds <= 0
    ):
        raise ValueError("Positive finite search time required")
    deadline = started + cfg.time_limit_seconds
    core_cfg = replace(
        cfg,
        time_limit_seconds=cfg.time_limit_seconds * (1 - cfg.repair_fraction)
        if cfg.pool_repair
        else cfg.time_limit_seconds,
    )

    def emit(event):
        if incumbent_callback is not None:
            incumbent_callback(dict(event, elapsed_seconds=perf_counter() - started))

    result = run_java_ils(
        instance,
        budget,
        config=core_cfg,
        facility_coordinates=facility_coordinates,
        facility_neighbors=facility_neighbors,
        incumbent_callback=emit,
        trace_callback=trace_callback,
    )
    result.method = "research_ils"
    result.metadata["research_preset"] = True
    remaining = deadline - perf_counter()
    if cfg.pool_repair and remaining > 0:
        repaired = repair_saved_solution(
            instance,
            list(map(int, result.solution)),
            time_limit_seconds=remaining,
            extra_cap=cfg.repair_extra_cap,
        )
        result.metadata["pool_finish"] = repaired.metadata
        if repaired.objective > result.objective:
            result.objective = repaired.objective
            result.solution = repaired.solution
            result.coverage = repaired.coverage
            emit(
                dict(
                    objective=result.objective,
                    sites=result.solution.copy(),
                    phase="near_neutral_pool",
                )
            )
    result.total_seconds = perf_counter() - started
    return result
