"""Opt-in persistent-memory iterated maximum-cover search.

This is our experimental search composition, not a reproduction of AILS-PR.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import asdict, dataclass
from math import isfinite
from time import perf_counter

import numpy as np

from ._incremental_core import (
    SparseSwapLocalSearch,
    budgeted_construct,
    path_relink_fast,
)
from .instance import MaxCoverInstance
from .results import HeuristicResult, MaxCoverResult


@dataclass(slots=True)
class IteratedSearchConfig:
    time_limit_seconds: float = 60.0
    max_iterations: int | None = None
    seed: int = 42
    max_pool: int = 8
    perturb: bool = True
    adaptive: bool = True
    perturbation_fraction: float = 0.05
    target_distance_fraction: float = 0.10
    acceptance_window: int = 20
    restart_interval: int = 5
    path_relinking: bool = True


def _elite_insert(pool: list[HeuristicResult], item: HeuristicResult, capacity: int) -> None:
    """Keep distinct solutions; protect the best and favor site-set diversity."""
    sites = frozenset(item.solution)
    if any(frozenset(p.solution) == sites for p in pool):
        return
    pool.append(item)
    if len(pool) > capacity:
        best = max(pool, key=lambda p: p.objective)
        others = [p for p in pool if p is not best]
        # Remove the worse member among the most redundant non-best solutions.
        victim = min(
            others,
            key=lambda p: (
                min(len(set(p.solution) ^ set(q.solution)) for q in pool if q is not p),
                p.objective,
            ),
        )
        pool[:] = [p for p in pool if p is not victim]


def run_iterated_search(
    instance: MaxCoverInstance,
    budget: int,
    *,
    config: IteratedSearchConfig | None = None,
    callback: Callable[[dict], None] | None = None,
) -> MaxCoverResult:
    """Search one budget, retaining elite memory between iterations.

    Time is checked between iterations; one in-progress iteration may overrun.
    max_iterations permits reproducible fixed-work tests. No reference sites
    or objective values are consumed. Existing run_heuristics is unchanged.
    """
    cfg = config or IteratedSearchConfig()
    if budget < 0 or budget > instance.n_facilities:
        raise ValueError("budget must be between zero and candidate count")
    if not isfinite(cfg.time_limit_seconds) or cfg.time_limit_seconds < 0 or cfg.max_pool < 1:
        raise ValueError("nonnegative time and positive pool size required")
    if cfg.max_iterations is not None and cfg.max_iterations < 0:
        raise ValueError("max_iterations must be nonnegative")
    if not 0 < cfg.perturbation_fraction <= 1 or not 0 < cfg.target_distance_fraction <= 1:
        raise ValueError("perturbation and distance fractions must lie in (0, 1]")
    if cfg.acceptance_window < 1 or cfg.restart_interval < 1:
        raise ValueError("acceptance_window and restart_interval must be positive")
    started = perf_counter()
    rng = np.random.default_rng(cfg.seed)
    engine = SparseSwapLocalSearch.from_instance(instance)
    current = engine.improve(
        budgeted_construct(instance, budget, constructor="greedy", seed=cfg.seed).solution
    )
    best = current
    pool: list[HeuristicResult] = [current]
    history = [current.objective] * cfg.acceptance_window
    strength = max(1, int(round(budget * cfg.perturbation_fraction)))
    iterations = 0
    total = int(instance.weights.sum())
    while (
        budget > 0
        and (cfg.max_iterations is None or iterations < cfg.max_iterations)
        and perf_counter() - started < cfg.time_limit_seconds
        and best.objective < total
    ):
        previous = current
        restart = not cfg.perturb or (iterations + 1) % cfg.restart_interval == 0
        seed = int(rng.integers(0, 2**31))
        if restart:
            constructed = budgeted_construct(
                instance,
                budget,
                constructor=("randomized", "sample", "random_plus")[iterations % 3],
                seed=seed,
            )
        else:
            # Keep the retained facilities in their original order.
            k = min(strength, len(current.solution))
            removed = set(rng.choice(len(current.solution), size=k, replace=False).tolist())
            retained = [j for pos, j in enumerate(current.solution) if pos not in removed]
            constructed = budgeted_construct(
                instance, budget, constructor="randomized", initial_solution=retained, seed=seed
            )
        candidate = engine.improve(
            constructed.solution, constructed.coverage, constructed.objective
        )
        raw_distance = len(set(candidate.solution) ^ set(previous.solution)) / 2
        if cfg.adaptive and cfg.perturb and not restart:
            target = max(1, budget * cfg.target_distance_fraction)
            if raw_distance < target:
                strength = min(max(1, budget), strength + 1)
            elif raw_distance > 2 * target:
                strength = max(1, strength - 1)
        if cfg.path_relinking and pool:
            guides = [p for p in pool if set(p.solution) != set(candidate.solution)]
            if guides:
                guide = guides[int(rng.integers(len(guides)))]
                linked = path_relink_fast(
                    instance,
                    candidate,
                    guide.solution,
                    max_steps=64,
                    candidate_width=16,
                    refresh_interval=8,
                )
                if linked.objective > candidate.objective:
                    candidate = engine.improve(linked.solution, linked.coverage, linked.objective)
        _elite_insert(pool, candidate, cfg.max_pool)
        if candidate.objective > best.objective:
            best = candidate
        # Late acceptance can leave a local optimum without discarding best.
        slot = iterations % cfg.acceptance_window
        accepted = candidate.objective >= current.objective or candidate.objective >= history[slot]
        if accepted:
            current = candidate
        history[slot] = current.objective
        iterations += 1
        if callback is not None:
            callback(
                {
                    "iteration": iterations,
                    "elapsed_seconds": perf_counter() - started,
                    "candidate_objective": int(candidate.objective),
                    "best_objective": int(best.objective),
                    "accepted": bool(accepted),
                    "restart": restart,
                    "strength": strength,
                    "distance": raw_distance,
                    "pool_size": len(pool),
                    "seed": seed,
                }
            )
    elapsed = perf_counter() - started
    return MaxCoverResult(
        budget=budget,
        method="iterated_adaptive"
        if cfg.adaptive and cfg.perturb
        else ("iterated_fixed" if cfg.perturb else "persistent_restart"),
        objective=int(best.objective),
        solution=list(best.solution),
        coverage=best.coverage.copy(),
        status="ok",
        solve_seconds=elapsed,
        total_seconds=elapsed,
        seed=cfg.seed,
        metadata={
            "iterations": iterations,
            "pool_size": len(pool),
            "config": asdict(cfg),
            "final_strength": strength,
            "timing_scope": "construction, local search, memory and callback; iteration boundary time checks",
        },
    )
