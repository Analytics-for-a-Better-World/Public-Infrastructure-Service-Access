from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from time import perf_counter as pc
from typing import Callable

import numpy as np

SCRIPT_DIR = Path(__file__).resolve().parent
APPROX_SRC = SCRIPT_DIR.parents[2] / "approximated_tradeoff" / "src"
sys.path.insert(0, str(APPROX_SRC))
sys.path.insert(0, str(SCRIPT_DIR))

import mc_heuristics as mch  # noqa: E402
import vietnam_grasp_heuristics as vgh  # noqa: E402
from vietnam_sparse_local_search import SparseSwapLocalSearch  # noqa: E402

try:
    from line_profiler import LineProfiler
except ImportError as exc:  # pragma: no cover - explicit runtime dependency
    raise SystemExit("Install line_profiler before running this script.") from exc


OUTPUT_ROOT = Path(r"C:\local\Parvathy\Vietnam")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run line_profiler on the Vietnam fresh-data heuristic algorithms."
    )
    parser.add_argument(
        "--instance-npz",
        type=Path,
        default=OUTPUT_ROOT / "optimization" / "vietnam_10kmgrid_20km_threshold.npz",
    )
    parser.add_argument("--budget", type=int, default=20)
    parser.add_argument("--grasp-budget", type=int, default=20)
    parser.add_argument("--grasp-iterations", type=int, default=2)
    parser.add_argument("--rcl-size", type=int, default=25)
    parser.add_argument("--sample-size", type=int, default=250)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_ROOT / "line_profiles")
    parser.add_argument(
        "--skip-full-grasp",
        action="store_true",
        help="Skip profiling run_grasp orchestration if only primitive algorithms are needed.",
    )
    return parser.parse_args()


def load_instance(path: Path) -> tuple[mch.MaxCoverInstance, dict]:
    data = np.load(path, allow_pickle=False)
    instance = mch.MaxCoverInstance(
        w=data["w"],
        ij_indptr=data["ij_indptr"],
        ij_indices=data["ij_indices"],
        ji_indptr=data["ji_indptr"],
        ji_indices=data["ji_indices"],
    )
    metadata = json.loads(str(data["metadata_json"]))
    return instance, metadata


def print_stats(profiler: LineProfiler, path: Path) -> None:
    with path.open("w", encoding="utf-8") as handle:
        try:
            profiler.print_stats(stream=handle, stripzeros=True)
        except TypeError:
            profiler.print_stats(stream=handle)


def profile_case(
    *,
    name: str,
    output_dir: Path,
    functions: list[Callable],
    run: Callable[[], dict],
) -> dict:
    profiler = LineProfiler()
    for func in functions:
        profiler.add_function(func)

    start = pc()
    payload = profiler(run)()
    profiled_seconds = pc() - start
    lprof_path = output_dir / f"{name}.lprof"
    text_path = output_dir / f"{name}.txt"
    profiler.dump_stats(str(lprof_path))
    print_stats(profiler, text_path)
    return {
        "case": name,
        "profiled_seconds": profiled_seconds,
        "lprof_path": str(lprof_path),
        "text_path": str(text_path),
        **payload,
    }


def result_payload(result: mch.HeuristicResult, *, scale: float) -> dict:
    return {
        "objective_weight_units": int(result.objective),
        "incremental_population": float(result.objective / scale),
        "solution_size": int(len(result.solution)),
        "algorithm_reported_seconds": float(result.total_time),
        "trace_length": int(len(result.objectives)),
    }


def core_profile_functions() -> list[Callable]:
    return [
        vgh._initial_gain,
        vgh._top_rcl,
        vgh._choose_facility,
        vgh.budgeted_construct,
        vgh.improve_local_search,
        vgh._swap_delta,
        vgh._apply_swap,
        vgh.path_relink,
        vgh.path_relink_fast,
        vgh.update_pool,
        vgh.run_grasp,
        SparseSwapLocalSearch.from_instance,
        SparseSwapLocalSearch.collect_candidates,
        SparseSwapLocalSearch.improve,
        mch.compute_coverage_and_objective,
        mch.swap_first_improving,
        mch.MaxCoverInstance.facilities_of,
        mch.MaxCoverInstance.households_of,
    ]


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    instance, metadata = load_instance(args.instance_npz)
    scale = float(metadata.get("weight_scale", 1.0))
    manifest = {
        "instance_npz": str(args.instance_npz),
        "threshold_m": metadata.get("threshold_m"),
        "n_population": int(instance.n_households),
        "n_candidates": int(instance.n_facilities),
        "budget": int(args.budget),
        "grasp_budget": int(args.grasp_budget),
        "grasp_iterations": int(args.grasp_iterations),
        "rcl_size": int(args.rcl_size),
        "sample_size": int(args.sample_size),
        "seed": int(args.seed),
        "note": "line_profiler timings include tracing overhead and should be used for line-level hotspots, not wall-clock benchmarking.",
    }

    common = core_profile_functions()
    rows: list[dict] = []

    constructor_cases = [
        ("construct_greedy", {"constructor": "greedy", "seed": args.seed}),
        ("construct_randomized", {"constructor": "randomized", "seed": args.seed + 1}),
        ("construct_sample", {"constructor": "sample", "seed": args.seed + 2}),
        ("construct_random_plus", {"constructor": "random_plus", "seed": args.seed + 3}),
    ]

    for case_name, kwargs in constructor_cases:
        def run_constructor(kwargs: dict = kwargs) -> dict:
            result = vgh.budgeted_construct(
                instance,
                args.budget,
                rcl_size=args.rcl_size,
                sample_size=args.sample_size,
                **kwargs,
            )
            return result_payload(result, scale=scale)

        rows.append(
            profile_case(
                name=case_name,
                output_dir=args.output_dir,
                functions=common,
                run=run_constructor,
            )
        )

    greedy_start = vgh.budgeted_construct(instance, args.budget, constructor="greedy", seed=args.seed)

    def run_local_search() -> dict:
        result = vgh.improve_local_search(instance, greedy_start, local_search="first")
        return result_payload(result, scale=scale)

    rows.append(
        profile_case(
            name="first_swap_local_search",
            output_dir=args.output_dir,
            functions=common,
            run=run_local_search,
        )
    )

    sparse_index = SparseSwapLocalSearch.from_instance(instance)

    def run_sparse_local_search() -> dict:
        result = vgh.improve_local_search(
            instance,
            greedy_start,
            local_search="first_sparse",
            sparse_local_search=sparse_index,
        )
        return result_payload(result, scale=scale)

    rows.append(
        profile_case(
            name="first_sparse_local_search",
            output_dir=args.output_dir,
            functions=common,
            run=run_sparse_local_search,
        )
    )

    randomized_start = vgh.budgeted_construct(
        instance,
        args.budget,
        constructor="randomized",
        rcl_size=args.rcl_size,
        sample_size=args.sample_size,
        seed=args.seed + 10,
    )
    randomized_start = vgh.improve_local_search(instance, randomized_start, local_search="first")
    greedy_improved = vgh.improve_local_search(instance, greedy_start, local_search="first")

    def run_path_relink() -> dict:
        result = vgh.path_relink(instance, randomized_start, greedy_improved.solution)
        return result_payload(result, scale=scale)

    rows.append(
        profile_case(
            name="path_relinking_original",
            output_dir=args.output_dir,
            functions=common,
            run=run_path_relink,
        )
    )

    def run_fast_path_relink() -> dict:
        result = vgh.path_relink_fast(instance, randomized_start, greedy_improved.solution)
        return result_payload(result, scale=scale)

    rows.append(
        profile_case(
            name="path_relinking_fast",
            output_dir=args.output_dir,
            functions=common,
            run=run_fast_path_relink,
        )
    )

    if not args.skip_full_grasp:
        def run_grasp_case() -> dict:
            best, records = vgh.run_grasp(
                instance,
                args.grasp_budget,
                max_iterations=args.grasp_iterations,
                time_limit_seconds=3600.0,
                constructor="randomized",
                rcl_size=args.rcl_size,
                sample_size=args.sample_size,
                local_search="first_sparse",
                path_relinking=True,
                path_relinking_method="fast",
                seed=args.seed,
                max_pool=8,
            )
            payload = result_payload(best, scale=scale)
            payload["iterations"] = int(len(records))
            return payload

        rows.append(
            profile_case(
                name="grasp_with_path_relinking",
                output_dir=args.output_dir,
                functions=common,
                run=run_grasp_case,
            )
        )

    summary_path = args.output_dir / "profile_summary.json"
    manifest_path = args.output_dir / "profile_manifest.json"
    summary_path.write_text(json.dumps(rows, indent=2), encoding="utf-8")
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(json.dumps({"output_dir": str(args.output_dir), "cases": len(rows)}, indent=2))


if __name__ == "__main__":
    main()
