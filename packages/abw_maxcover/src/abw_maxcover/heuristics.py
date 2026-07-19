"""Scalable maximum-cover heuristics."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, replace
from typing import Any

from ._budgets import normalise_budget_order
from ._incremental_core import (
    DETERMINISTIC_CONSTRUCTORS,
    RANDOMIZED_CONSTRUCTORS,
    ConstructorName,
    LocalSearchName,
    SparseSwapLocalSearch,
    budgeted_construct,
    drop_redundant_facilities,
    improve_local_search,
    normalise_constructor,
    path_relink_fast,
    prefix_result,
    select_by_marginal_gain,
)
from .instance import MaxCoverInstance
from .results import HeuristicResult, MaxCoverCurve, MaxCoverResult


@dataclass(slots=True)
class HeuristicConfig:
    constructors: tuple[ConstructorName, ...] = (
        "greedy",
        "compact",
        "regreedy",
        "randomized",
        "sample",
        "random_plus",
    )
    randomized_repeats: int = 3
    rcl_size: int = 25
    sample_size: int = 250
    random_plus_fraction: float = 0.15
    local_search: LocalSearchName = "first_sparse"
    local_search_max_moves: int | None = None
    local_search_time_limit_seconds: float | None = None
    use_path_relinking: bool = True
    max_pool: int = 8
    path_relinking_max_steps: int | None = 64
    path_relinking_candidate_width: int | None = 16
    path_relinking_refresh_interval: int = 8
    seed: int = 42


def _heuristic_to_result(
    *,
    budget: int,
    method: str,
    result: HeuristicResult,
    construction: HeuristicResult | None = None,
    seed: int | None = None,
    repeat: int | None = None,
    metadata: dict[str, Any] | None = None,
    extra_seconds: float = 0.0,
    local_search_moves: int | None = None,
) -> MaxCoverResult:
    search_seconds = float(result.total_time + extra_seconds)
    total_seconds = search_seconds
    if construction is not None:
        total_seconds += float(construction.total_time)
    return MaxCoverResult(
        budget=int(budget),
        method=method,
        objective=int(result.objective),
        solution=list(result.solution),
        status="ok",
        coverage=result.coverage.copy(),
        solve_seconds=search_seconds,
        total_seconds=total_seconds,
        construction_objective=None if construction is None else int(construction.objective),
        construction_seconds=None if construction is None else float(construction.total_time),
        local_search_moves=(
            max(0, len(result.objectives) - 1)
            if local_search_moves is None
            else max(0, int(local_search_moves))
        ),
        seed=seed,
        repeat=repeat,
        metadata=dict(metadata or {}),
    )


def _deterministic_budget_results(
    instance: MaxCoverInstance,
    budget: int,
    greedy: HeuristicResult,
    constructors: tuple[ConstructorName, ...],
    config: HeuristicConfig,
    sparse_local_search: SparseSwapLocalSearch | None,
) -> list[MaxCoverResult]:
    results: list[MaxCoverResult] = []
    prefix = prefix_result(instance, greedy, budget)
    improved: HeuristicResult | None = None
    compacted: HeuristicResult | None = None
    if "greedy" in constructors:
        results.append(
            _heuristic_to_result(
                budget=budget,
                method="greedy",
                result=prefix,
                seed=config.seed,
            )
        )

    if "greedy" in constructors or "compact" in constructors or "regreedy" in constructors:
        improved = improve_local_search(
            instance,
            prefix,
            local_search=config.local_search,
            sparse_local_search=sparse_local_search,
            max_moves=config.local_search_max_moves,
            time_limit_seconds=config.local_search_time_limit_seconds,
        )
        if "greedy" in constructors:
            results.append(
                _heuristic_to_result(
                    budget=budget,
                    method=f"greedy_{config.local_search}",
                    result=improved,
                    construction=prefix,
                    seed=config.seed,
                )
            )

    if "compact" in constructors or "regreedy" in constructors:
        if improved is None:
            improved = improve_local_search(
                instance,
                prefix,
                local_search=config.local_search,
                sparse_local_search=sparse_local_search,
                max_moves=config.local_search_max_moves,
                time_limit_seconds=config.local_search_time_limit_seconds,
            )
        compacted = drop_redundant_facilities(
            instance,
            improved.solution,
            coverage=improved.coverage,
            objective=improved.objective,
        )
        if "compact" in constructors:
            results.append(
                _heuristic_to_result(
                    budget=budget,
                    method=f"greedy_{config.local_search}_compact",
                    result=compacted,
                    construction=prefix,
                    seed=config.seed,
                    extra_seconds=improved.total_time,
                    metadata={"compacted_selected_count": len(compacted.solution)},
                )
            )

    if "regreedy" in constructors:
        if compacted is None:
            raise RuntimeError("internal error: regreedy requires compacted solution")
        refilled = select_by_marginal_gain(
            instance,
            budget,
            initial_solution=compacted.solution,
        )
        regreedy_improved = improve_local_search(
            instance,
            refilled,
            local_search=config.local_search,
            sparse_local_search=sparse_local_search,
            max_moves=config.local_search_max_moves,
            time_limit_seconds=config.local_search_time_limit_seconds,
        )
        regreedy_compacted = drop_redundant_facilities(
            instance,
            regreedy_improved.solution,
            coverage=regreedy_improved.coverage,
            objective=regreedy_improved.objective,
        )
        results.append(
            _heuristic_to_result(
                budget=budget,
                method=f"greedy_{config.local_search}_compact_regreedy",
                result=regreedy_compacted,
                construction=prefix,
                seed=config.seed,
                extra_seconds=(
                    (improved.total_time if improved is not None else 0.0)
                    + compacted.total_time
                    + refilled.total_time
                    + regreedy_improved.total_time
                ),
                metadata={
                    "compacted_selected_count": len(compacted.solution),
                    "refilled_selected_count": len(refilled.solution),
                    "regreedy_selected_count": len(regreedy_compacted.solution),
                },
            )
        )
    return results


def run_heuristics(
    instance: MaxCoverInstance,
    budgets: list[int],
    *,
    config: HeuristicConfig | None = None,
    progress: Callable[[Any], Any] = lambda iterable: iterable,
    budget_callback: Callable[[int, list[MaxCoverResult]], Any] | None = None,
) -> MaxCoverCurve:
    """Run the configured heuristic portfolio, optionally checkpointing budgets."""
    cfg = config or HeuristicConfig()
    requested_budgets, execution_budgets = normalise_budget_order(budgets)
    constructors = tuple(dict.fromkeys(normalise_constructor(str(c)) for c in cfg.constructors))
    unsupported = [
        constructor
        for constructor in constructors
        if constructor not in DETERMINISTIC_CONSTRUCTORS
        and constructor not in RANDOMIZED_CONSTRUCTORS
    ]
    if unsupported:
        raise ValueError(f"unsupported constructors: {unsupported}")
    results: list[MaxCoverResult] = []
    sparse_local_search: SparseSwapLocalSearch | None = None
    if cfg.local_search == "first_sparse":
        try:
            sparse_local_search = SparseSwapLocalSearch.from_instance(instance)
        except ModuleNotFoundError:
            cfg = replace(cfg, local_search="first")

    deterministic_requested = any(c in DETERMINISTIC_CONSTRUCTORS for c in constructors)
    greedy = (
        budgeted_construct(instance, max(execution_budgets), constructor="greedy", seed=cfg.seed)
        if deterministic_requested and execution_budgets
        else None
    )

    randomized_constructors = tuple(c for c in constructors if c in RANDOMIZED_CONSTRUCTORS)
    stable_constructor_offset = {"randomized": 101, "sample": 211, "random_plus": 307}
    callback_budgets: set[int] = set()
    for budget in progress(execution_budgets):
        if greedy is not None:
            results.extend(
                _deterministic_budget_results(
                    instance,
                    budget,
                    greedy,
                    constructors,
                    cfg,
                    sparse_local_search,
                )
            )
        pool: list[HeuristicResult] = []
        for constructor in randomized_constructors:
            for repeat in range(max(1, int(cfg.randomized_repeats))):
                seed = (
                    int(cfg.seed)
                    + 1009 * repeat
                    + 9176 * int(budget)
                    + stable_constructor_offset.get(constructor, 0)
                )
                constructed = budgeted_construct(
                    instance,
                    budget,
                    constructor=constructor,
                    rcl_size=cfg.rcl_size,
                    sample_size=cfg.sample_size,
                    random_plus_fraction=cfg.random_plus_fraction,
                    seed=seed,
                )
                improved = improve_local_search(
                    instance,
                    constructed,
                    local_search=cfg.local_search,
                    sparse_local_search=sparse_local_search,
                    max_moves=cfg.local_search_max_moves,
                    time_limit_seconds=cfg.local_search_time_limit_seconds,
                )
                candidate = improved
                path_relinking_attempted = False
                path_relinking_improved = False
                path_relinking_seconds = 0.0
                extra_seconds = 0.0
                if cfg.use_path_relinking and pool:
                    path_relinking_attempted = True
                    guide = max(pool, key=lambda item: item.objective)
                    relinked = path_relink_fast(
                        instance,
                        improved,
                        guide.solution,
                        max_steps=cfg.path_relinking_max_steps,
                        candidate_width=cfg.path_relinking_candidate_width,
                        refresh_interval=cfg.path_relinking_refresh_interval,
                    )
                    path_relinking_seconds = float(relinked.total_time)
                    if relinked.objective > candidate.objective:
                        candidate = relinked
                        path_relinking_improved = True
                        extra_seconds = float(improved.total_time)
                    else:
                        extra_seconds = path_relinking_seconds
                pool.append(candidate)
                pool = sorted(pool, key=lambda item: item.objective, reverse=True)[: cfg.max_pool]
                results.append(
                    _heuristic_to_result(
                        budget=budget,
                        method=(
                            f"{constructor}_{cfg.local_search}_path_relinked"
                            if path_relinking_improved
                            else f"{constructor}_{cfg.local_search}"
                        ),
                        result=candidate,
                        construction=constructed,
                        seed=seed,
                        repeat=repeat,
                        extra_seconds=extra_seconds,
                        local_search_moves=max(0, len(improved.objectives) - 1),
                        metadata={
                            "path_relinking": bool(cfg.use_path_relinking),
                            "path_relinking_attempted": path_relinking_attempted,
                            "path_relinking_improved": path_relinking_improved,
                            "path_relinking_seconds": path_relinking_seconds,
                            "path_relinking_max_steps": cfg.path_relinking_max_steps,
                            "path_relinking_candidate_width": cfg.path_relinking_candidate_width,
                            "path_relinking_refresh_interval": cfg.path_relinking_refresh_interval,
                            "local_search_max_moves": cfg.local_search_max_moves,
                            "local_search_time_limit_seconds": cfg.local_search_time_limit_seconds,
                        },
                    )
                )
        if budget_callback is not None:
            budget_results = [result for result in results if int(result.budget) == int(budget)]
            if budget_results:
                budget_callback(int(budget), budget_results)
                callback_budgets.add(int(budget))

    if budget_callback is not None:
        for budget in execution_budgets:
            if int(budget) in callback_budgets:
                continue
            budget_results = [result for result in results if int(result.budget) == int(budget)]
            if budget_results:
                budget_callback(int(budget), budget_results)

    ordered_results: list[MaxCoverResult] = []
    for budget in requested_budgets:
        ordered_results.extend(result for result in results if int(result.budget) == budget)

    return MaxCoverCurve(
        instance_name=instance.name,
        kind="heuristic",
        results=ordered_results,
        metadata={
            "requested_budgets": requested_budgets,
            "execution_budgets": execution_budgets,
        },
    )
