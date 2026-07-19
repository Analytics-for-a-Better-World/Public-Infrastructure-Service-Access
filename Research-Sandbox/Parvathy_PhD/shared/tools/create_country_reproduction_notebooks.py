from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(r"C:\work\codex\sandboxes\Conclude_Parvathy_thesis")
NOTEBOOK_DIR = ROOT / "notebooks"


def md(source: str) -> dict:
    return {
        "cell_type": "markdown",
        "metadata": {},
        "source": [line + "\n" for line in source.strip().splitlines()],
    }


def code(source: str) -> dict:
    return {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [line + "\n" for line in source.strip().splitlines()],
    }


COMMON_CODE = r'''
from __future__ import annotations

import json
import os
import platform
import re
import subprocess
import sys
from pathlib import Path


class CommandResult:
    def __init__(self, args, returncode, stdout):
        self.args = args
        self.returncode = returncode
        self.stdout = stdout


def run_command(args, cwd=None, env=None, check=True, dry_run=False):
    """Run a command with readable streaming output. Pass args as a list."""
    cwd = Path(cwd) if cwd else None
    printable = " ".join(str(a) for a in args)
    print(f"\n$ {printable}", flush=True)
    if cwd:
        print(f"  cwd={cwd}", flush=True)
    if dry_run:
        print("  dry-run: command not executed", flush=True)
        return CommandResult([str(a) for a in args], 0, "")

    process = subprocess.Popen(
        [str(a) for a in args],
        cwd=str(cwd) if cwd else None,
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        encoding="utf-8",
        errors="replace",
    )
    assert process.stdout is not None
    lines: list[str] = []
    for line in process.stdout:
        print(line, end="", flush=True)
        lines.append(line)
    returncode = process.wait()
    output = "".join(lines)
    if check and returncode != 0:
        raise RuntimeError(f"command failed with exit code {returncode}: {printable}")
    return CommandResult([str(a) for a in args], int(returncode), output)


def exists(path):
    path = Path(path)
    print(f"{'OK' if path.exists() else 'MISSING'}: {path}")
    return path.exists()


def env_bool(name, default=False):
    value = os.environ.get(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


def report_singleton_inputs(label, base_root, filenames):
    base_root = Path(base_root)
    print(f"\n{label} singleton data root: {base_root}")
    all_present = True
    for filename in filenames:
        all_present = exists(base_root / filename) and all_present
    if all_present:
        print("All required downloaded inputs are already present. The pipeline should reuse them.")
    else:
        print("At least one downloaded input is missing. The pipeline will acquire the missing file(s) into this same root.")
    return all_present


def show_pipeline_products(label, output_dir):
    output_dir = Path(output_dir)
    print(f"\n{label}: {output_dir}")
    exists(output_dir)
    if not output_dir.exists():
        return
    patterns = {
        "candidate files": "*candidate*.parquet",
        "amenity/source files": "*source*.parquet",
        "population/target files": "*population*.parquet",
        "distance matrices": "*distance_matrix*.parquet",
        "run manifests": "run_manifest*.yaml",
    }
    for name, pattern in patterns.items():
        count = len(list(output_dir.glob(pattern)))
        print(f"  {name}: {count}")


def audit_latex_figures(tex_paths, figure_dirs):
    figure_dirs = [Path(p) for p in figure_dirs]
    pattern = re.compile(r"\\includegraphics(?:\[[^\]]*\])?\{([^}]+)\}")
    seen: list[str] = []
    for tex_path in [Path(p) for p in tex_paths]:
        if not tex_path.exists():
            print("MISSING TEX", tex_path)
            continue
        text = tex_path.read_text(encoding="utf-8", errors="replace")
        for match in pattern.finditer(text):
            name = match.group(1)
            if name not in seen:
                seen.append(name)
    print(f"\nFigure audit: {len(seen)} unique LaTeX references")
    missing = []
    for name in seen:
        candidates = []
        raw = Path(name)
        search_names = [raw]
        if raw.suffix == "":
            search_names.extend([raw.with_suffix(".pdf"), raw.with_suffix(".png")])
        for figure_dir in figure_dirs:
            for search_name in search_names:
                candidates.append(figure_dir / search_name)
        ok = next((path for path in candidates if path.exists()), None)
        print(f"{'OK' if ok else 'MISSING'}: {name}" + (f" -> {ok}" if ok else ""))
        if ok is None:
            missing.append(name)
    print(f"Missing figures: {len(missing)}")
    return missing


print("Python", sys.version)
print("Executable", sys.executable)
print("Platform", platform.platform())
'''


CONFIG_CODE = r'''
REPO_URL = "https://github.com/Analytics-for-a-Better-World/Public-Infrastructure-Service-Access.git"

NOTEBOOK_DIR = Path.cwd().resolve()
DEFAULT_WORK_ROOT = NOTEBOOK_DIR
for parent in [NOTEBOOK_DIR, *NOTEBOOK_DIR.parents]:
    if (parent / "notebooks").exists() or (parent / "articles").exists():
        DEFAULT_WORK_ROOT = parent
        break

WORK_ROOT = Path(os.environ.get("PARVATHY_WORK_ROOT", str(DEFAULT_WORK_ROOT))).resolve()

repo_env = os.environ.get("PISA_REPO_DIR")
if repo_env:
    REPO_DIR = Path(repo_env).resolve()
else:
    repo_candidates = [
        NOTEBOOK_DIR,
        *NOTEBOOK_DIR.parents,
        WORK_ROOT / "Public-Infrastructure-Service-Access",
        WORK_ROOT / "github" / "Public-Infrastructure-Service-Access",
    ]
    REPO_DIR = next((p.resolve() for p in repo_candidates if (p / "Research-Sandbox").exists()), WORK_ROOT / "Public-Infrastructure-Service-Access")

RUN_ROOT = Path(os.environ.get("PARVATHY_RUN_ROOT", str(WORK_ROOT / "runs_reproduction"))).resolve()
RESULT_ROOT = Path(os.environ.get("PARVATHY_RESULT_ROOT", str(WORK_ROOT / "outputs_reproduction"))).resolve()

ROUTING_PYTHON = Path(os.environ.get("ROUTING_PYTHON", str(REPO_DIR / ".venv-routing" / "Scripts" / "python.exe")))
OPT_PYTHON = Path(os.environ.get("OPT_PYTHON", str(REPO_DIR / ".venv-optimization" / "Scripts" / "python.exe")))
if not ROUTING_PYTHON.exists():
    ROUTING_PYTHON = Path(sys.executable)
if not OPT_PYTHON.exists():
    OPT_PYTHON = Path(sys.executable)

DRY_RUN = env_bool("PARVATHY_DRY_RUN", True)
RUN_GIT_CLONE = env_bool("PARVATHY_RUN_GIT_CLONE", False)
RUN_ENV_CHECKS = env_bool("PARVATHY_RUN_ENV_CHECKS", True)
RUN_DATA_PIPELINE = env_bool("PARVATHY_RUN_DATA_PIPELINE", True)
RUN_OPTIMIZATION = env_bool("PARVATHY_RUN_OPTIMIZATION", True)

DISTANCE_PIPELINE = REPO_DIR / "Research-Sandbox" / "general_distances_per_country"
ABW_MAXCOVER_SRC = REPO_DIR / "packages" / "abw_maxcover" / "src"
PARVATHY = REPO_DIR / "Research-Sandbox" / "Parvathy_PhD"

tool_candidates = [
    Path(os.environ["PARVATHY_LOCAL_TOOLS"]).resolve()
    if os.environ.get("PARVATHY_LOCAL_TOOLS")
    else None,
    WORK_ROOT / "tools",
    PARVATHY / "shared" / "tools",
]
LOCAL_TOOLS = next(
    (
        path
        for path in tool_candidates
        if path is not None and (path / "run_pipeline_fresh_root.py").exists()
    ),
    WORK_ROOT / "tools",
)

for p in [DISTANCE_PIPELINE, ABW_MAXCOVER_SRC, PARVATHY]:
    if p.exists() and str(p) not in sys.path:
        sys.path.insert(0, str(p))

print("REPO_DIR       ", REPO_DIR)
print("RUN_ROOT       ", RUN_ROOT)
print("RESULT_ROOT    ", RESULT_ROOT)
print("LOCAL_TOOLS    ", LOCAL_TOOLS)
print("ROUTING_PYTHON ", ROUTING_PYTHON)
print("OPT_PYTHON     ", OPT_PYTHON)
print("DRY_RUN        ", DRY_RUN)
print("RUN_DATA_PIPELINE", RUN_DATA_PIPELINE)
print("RUN_OPTIMIZATION", RUN_OPTIMIZATION)
'''


VERIFY_CODE = r'''
if RUN_GIT_CLONE:
    if REPO_DIR.exists() and (REPO_DIR / ".git").exists():
        run_command(["git", "-C", REPO_DIR, "status", "--short"], dry_run=False)
    else:
        REPO_DIR.parent.mkdir(parents=True, exist_ok=True)
        run_command(["git", "clone", REPO_URL, REPO_DIR], dry_run=False)
else:
    print("RUN_GIT_CLONE is False; verifying local paths only.")

exists(REPO_DIR)
exists(DISTANCE_PIPELINE)
exists(ABW_MAXCOVER_SRC)
exists(PARVATHY)

if (REPO_DIR / ".git").exists():
    run_command(["git", "-C", REPO_DIR, "rev-parse", "HEAD"], dry_run=False)
'''


ENV_CODE = r'''
if RUN_ENV_CHECKS:
    for label, py in [("routing", ROUTING_PYTHON), ("optimization", OPT_PYTHON)]:
        print(f"\n--- {label} environment ---")
        run_command([py, "-c", "import sys; print(sys.version); print(sys.executable)"], dry_run=False)
        run_command([py, "-m", "pip", "check"], check=False, dry_run=False)
        run_command([py, "-c", "import importlib; mods=['numpy','pandas','geopandas','pandana','pyarrow'];\nfor m in mods:\n    try:\n        mod=importlib.import_module(m); print(m, getattr(mod,'__version__','ok'))\n    except Exception as e:\n        print(m, 'MISSING/ERROR', repr(e))"], check=False, dry_run=False)
else:
    print("Environment checks skipped.")
'''


def base_cells(title: str, country: str) -> list[dict]:
    return [
        md(f'''
# {title}

This notebook is the executable runbook for the {country} case. It starts from the open-data pipeline: OSM PBF, WorldPop population raster, OSM amenity extraction, candidate-grid generation, component-aware snapping, and sparse road-distance matrices. Optimization and figures are run only after those data products exist.

Heavy commands are controlled by environment variables. By default the notebook is a dry run. Set `PARVATHY_DRY_RUN=0` to execute.
'''),
        md('''
## Manual Setup

If GitHub access is unavailable inside the notebook, clone the repository manually first:

```powershell
git clone https://github.com/Analytics-for-a-Better-World/Public-Infrastructure-Service-Access.git
```

Use the pipeline in `Research-Sandbox/general_distances_per_country`, the optimization package in `packages/abw_maxcover`, and the reproduction scripts in `Research-Sandbox/Parvathy_PhD`.

For portable Pandana replication use Python <= 3.12, `numpy<2`, and `pandana==0.7.*`. This machine also has a local patched Pandana runtime used by Codex.
'''),
        code(COMMON_CODE),
        md("## Configuration"),
        code(CONFIG_CODE),
        md("## Verify Code and Environment"),
        code(VERIFY_CODE),
        code(ENV_CODE),
    ]


def vietnam_cells() -> list[dict]:
    return base_cells("Vietnam OSM-Health Reproduction Notebook", "Vietnam") + [
        md('''
## Singleton Data Check

The pipeline owns acquisition of the Vietnam OSM PBF and WorldPop raster. The notebook deliberately points repeated runs to the same `vietnam_data` directory. If `vietnam-latest.osm.pbf` and `vnm_ppp_2020.tif` already exist there, the pipeline reuses them; otherwise it downloads the missing input once to that path.
'''),
        code(r'''
VIETNAM_DIR = PARVATHY / "Vietnam"
VIETNAM_SCRIPT_DIR = LOCAL_TOOLS if (LOCAL_TOOLS / "run_vietnam_osm_health_pipeline_batch.py").exists() else VIETNAM_DIR / "scripts"
VIETNAM_RUN_ROOT = Path(os.environ.get("PARVATHY_VIETNAM_RUN_ROOT", str(RUN_ROOT / "vietnam_osm_health"))).resolve()
VIETNAM_DATA_ROOT = VIETNAM_RUN_ROOT / "vietnam_data"
VIETNAM_PIPELINE_OUTPUT = VIETNAM_DATA_ROOT / "outputs"
VIETNAM_RESULT_ROOT = Path(os.environ.get("PARVATHY_VIETNAM_RESULT_ROOT", str(RESULT_ROOT / "vietnam_osm_health"))).resolve()
VIETNAM_APPROX_ROOT = Path(os.environ.get("PARVATHY_VIETNAM_APPROX_ROOT", str(RESULT_ROOT / "vietnam_osm_health_approx_pareto"))).resolve()

print("Vietnam tools:", VIETNAM_SCRIPT_DIR)
print("Pipeline package:", DISTANCE_PIPELINE)
print("Vietnam run root:", VIETNAM_RUN_ROOT)
print("Vietnam result root:", VIETNAM_RESULT_ROOT)
print("Vietnam approximate Pareto root:", VIETNAM_APPROX_ROOT)
report_singleton_inputs(
    "Vietnam",
    VIETNAM_DATA_ROOT,
    ["vietnam-latest.osm.pbf", "vnm_ppp_2020.tif"],
)
'''),
        md('''
## Data Pipeline

This run uses OSM hospital/clinic amenities as existing facilities and generates candidate grids at 10 km, 5 km, and 1 km. Component-aware snapping retains components `0,1`; population is aggregated with factor 5. Candidate matrices are included. The 10 km and 5 km cases run unchunked because they fit comfortably; the 1 km case uses 100,000-target chunks to keep the spatial prefilter and final matrix write memory-safe.
'''),
        code(r'''
vietnam_pipeline_base = [
        ROUTING_PYTHON,
        VIETNAM_SCRIPT_DIR / "run_vietnam_osm_health_pipeline_batch.py",
        "--pipeline-dir", DISTANCE_PIPELINE,
        "--python", ROUTING_PYTHON,
        "--fresh-root", VIETNAM_RUN_ROOT,
        "--output-root", VIETNAM_RESULT_ROOT,
        "--snap-components", "0,1",
        "--aggregate-factor", "5",
        "--include-candidates",
]

VIETNAM_1KM_LOG_DIR = VIETNAM_RESULT_ROOT / "pipeline_logs" / "vietnam_osm_health_candidates_only_driving_unsimplified_1km"
VIETNAM_1KM_LOG_DIR.mkdir(parents=True, exist_ok=True)

vietnam_pipeline_commands = [
    (vietnam_pipeline_base + ["--spacings", "10000", "5000"], WORK_ROOT),
    (
        [
            ROUTING_PYTHON,
            LOCAL_TOOLS / "run_pipeline_fresh_root.py",
            "--pipeline-dir", DISTANCE_PIPELINE,
            "--fresh-base-root", VIETNAM_RUN_ROOT,
            "vietnam",
            "--log-file",
            VIETNAM_1KM_LOG_DIR / "vietnam_osm_health_candidates_only_driving_unsimplified_1km.pipeline.log",
            "--network-backend", "osmium",
            "--simplify-network", "false",
            "--network-profile", "driving",
            "--diagnose-connectivity", "true",
            "--snap-components", "0,1",
            "--sources", "candidates",
            "--destinations", "population",
            "--amenity", "hospital", "clinic",
            "--candidate-grid-spacing-m", "1000",
            "--candidate-max-snap-dist-m", "500",
            "--candidate-exclude-water", "false",
            "--max-total-dist", "20000",
            "--matrix-output-mode", "split",
            "--matrix-shape", "sparse",
            "--aggregate-factor", "5",
            "--sparse-target-chunk-size", "100000",
        ],
        WORK_ROOT,
    ),
]

if RUN_DATA_PIPELINE:
    for cmd, cwd in vietnam_pipeline_commands:
        run_command(cmd, cwd=cwd, dry_run=DRY_RUN)
else:
    print("RUN_DATA_PIPELINE is False. Commands that would run:")
    for cmd, cwd in vietnam_pipeline_commands:
        run_command(cmd, cwd=cwd, dry_run=True)

show_pipeline_products("Vietnam pipeline outputs", VIETNAM_PIPELINE_OUTPUT)
'''),
        md('''
## Context Maps and Data Figures

These map-only pipeline calls regenerate the paper-style context figures from the same data root used above. They include population points, OSM hospital/clinic amenities, candidate grids, and OSM road context. Because `--map-only` is used, the commands stop before distance-matrix computation.
'''),
        code(r'''
VIETNAM_FIGURE_ROOT = VIETNAM_RESULT_ROOT / "figures"
VIETNAM_FIGURE_ROOT.mkdir(parents=True, exist_ok=True)

vietnam_context_map_commands = []
for spacing in [10000, 5000, 1000]:
    vietnam_context_map_commands.append(
        ([
            ROUTING_PYTHON,
            LOCAL_TOOLS / "run_pipeline_fresh_root.py",
            "--pipeline-dir", DISTANCE_PIPELINE,
            "--fresh-base-root", VIETNAM_RUN_ROOT,
            "vietnam",
            "--network-backend", "osmium",
            "--simplify-network", "false",
            "--network-profile", "driving",
            "--sources", "amenities", "candidates",
            "--destinations", "population",
            "--amenity", "hospital", "clinic",
            "--candidate-grid-spacing-m", str(spacing),
            "--candidate-max-snap-dist-m", str(max(1, spacing // 2)),
            "--candidate-exclude-water", "false",
            "--aggregate-factor", "5",
            "--map-only",
            "--map-roads", "true",
            "--map-basemap-alpha", "0.70",
            "--map-dpi", "260",
            "--map-path", VIETNAM_FIGURE_ROOT / f"vietnam_osm_health_context_{spacing // 1000}km.pdf",
        ], DISTANCE_PIPELINE)
    )

if RUN_DATA_PIPELINE:
    for cmd, cwd in vietnam_context_map_commands:
        run_command(cmd, cwd=cwd, dry_run=DRY_RUN)
else:
    print("RUN_DATA_PIPELINE is False. Context-map commands that would run:")
    for cmd, cwd in vietnam_context_map_commands:
        run_command(cmd, cwd=cwd, dry_run=True)
'''),
        md('''
## Approximate Pareto

The approximation script reads the pipeline manifests, extracts candidate-to-population sparse matrices at a 5 km threshold, removes population already covered by existing OSM hospital/clinic amenities, and then runs the greedy / compact / regreedy approximate Pareto routine from `abw_maxcover`.
'''),
        code(r'''
vietnam_analysis_commands = [
    ([
        OPT_PYTHON,
        VIETNAM_SCRIPT_DIR / "run_vietnam_osm_health_approx_pareto.py",
        "--run-output", VIETNAM_PIPELINE_OUTPUT,
        "--output-root", VIETNAM_APPROX_ROOT,
        "--threshold-m", "5000",
        "--abw-maxcover-src", ABW_MAXCOVER_SRC,
    ], WORK_ROOT),
]

if RUN_OPTIMIZATION:
    for cmd, cwd in vietnam_analysis_commands:
        run_command(cmd, cwd=cwd, dry_run=DRY_RUN)
else:
    print("RUN_OPTIMIZATION is False. Commands that would run:")
    for cmd, cwd in vietnam_analysis_commands:
        run_command(cmd, cwd=cwd, dry_run=True)
'''),
        md("## Checks"),
        code(r'''
summary_path = VIETNAM_APPROX_ROOT / "vietnam_osm_health_approx_pareto_summary.json"
if summary_path.exists():
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    print(json.dumps(summary, indent=2)[:6000])
else:
    print("MISSING", summary_path)

print("\nExpected Vietnam context maps:")
for spacing in [10000, 5000, 1000]:
    exists(
        VIETNAM_FIGURE_ROOT
        / f"vietnam_osm_health_context_{spacing // 1000}km_resolution_{spacing}m.pdf"
    )

audit_latex_figures(
    [
        WORK_ROOT / "articles" / "integrated_access_report" / "manuscript.tex",
        WORK_ROOT / "presentations" / "integrated_access_deck" / "story_deck.tex",
    ],
    [
        WORK_ROOT / "articles" / "integrated_access_report" / "figures",
        WORK_ROOT / "presentations" / "integrated_access_deck" / "figures",
        VIETNAM_FIGURE_ROOT,
    ],
)

if (REPO_DIR / ".git").exists():
    run_command(["git", "-C", REPO_DIR, "status", "--short"], dry_run=False)
    run_command(["git", "-C", REPO_DIR, "rev-parse", "HEAD"], dry_run=False)
'''),
    ]


def timor_cells() -> list[dict]:
    return base_cells("Timor-Leste Access Reproduction Notebook", "Timor-Leste") + [
        md('''
## Singleton Data Check

The Timor-Leste pipeline uses the `east-timor_data` root. If the OSM PBF and WorldPop raster already exist there, they are reused. The local reproduction can also seed this root from an archived run to avoid unnecessary downloads.
'''),
        code(r'''
TIMOR_DIR = PARVATHY / "Timor_Leste"
TIMOR_TOOL_DIR = LOCAL_TOOLS if (LOCAL_TOOLS / "identify_timor_components.py").exists() else TIMOR_DIR / "tools"
TIMOR_RUNS = WORK_ROOT / "runs"
TIMOR_OUTPUTS = WORK_ROOT / "outputs"
TIMOR_REPRO_RUN = RUN_ROOT / "timor_network_profile_component012_pandana"
TIMOR_DATA_ROOT = TIMOR_REPRO_RUN / "east-timor_data"
TIMOR_PIPELINE_RUN = TIMOR_DATA_ROOT / "outputs"
TIMOR_REPRO_OUTPUT = RESULT_ROOT / "timor_network_profile_component012_pandana"
TIMOR_REPRO_COMPONENTS = RESULT_ROOT / "timor_component_geography"
TIMOR_REPRO_SATURATION = RESULT_ROOT / "timor_component012_saturation"

print("Timor tools:", TIMOR_TOOL_DIR)
print("Pipeline package:", DISTANCE_PIPELINE)
report_singleton_inputs(
    "Timor-Leste",
    TIMOR_DATA_ROOT,
    ["east-timor-latest.osm.pbf", "tls_ppp_2020.tif"],
)
'''),
        md('''
## Data Pipeline

The Timor-Leste run computes driving and drive-plus-walk profiles, simplified and unsimplified networks, and 10 km, 5 km, and 1 km candidate grids. Component-aware snapping retains components `0,1,2` to keep mainland Timor-Leste, Oecusse-Ambeno, and Atauro.
'''),
        code(r'''
timor_pipeline_commands = [
    ([
        ROUTING_PYTHON,
        TIMOR_TOOL_DIR / "identify_timor_components.py",
        "--cache-dir", TIMOR_RUNS / "timor_network_profile_20260623" / "east-timor_data" / "cache",
        "--output-dir", TIMOR_REPRO_COMPONENTS,
    ], WORK_ROOT),
    ([
        ROUTING_PYTHON,
        TIMOR_TOOL_DIR / "run_timor_network_profile_sensitivity.py",
        "--pipeline-dir", DISTANCE_PIPELINE,
        "--python", ROUTING_PYTHON,
        "--fresh-root", TIMOR_REPRO_RUN,
        "--output-dir", TIMOR_REPRO_OUTPUT,
        "--seed-root", TIMOR_RUNS / "network_only_20260622_1645" / "east-timor_data",
        "--snap-components", "0,1,2",
        "--candidate-exclude-water", "false",
    ], WORK_ROOT),
]

if RUN_DATA_PIPELINE:
    for cmd, cwd in timor_pipeline_commands:
        run_command(cmd, cwd=cwd, dry_run=DRY_RUN)
else:
    print("RUN_DATA_PIPELINE is False. Commands that would run:")
    for cmd, cwd in timor_pipeline_commands:
        run_command(cmd, cwd=cwd, dry_run=True)

show_pipeline_products("Timor-Leste pipeline outputs", TIMOR_PIPELINE_RUN)
'''),
        md('''
## Context Maps and Data Figures

These map-only pipeline calls create the Timor-Leste data-context maps from the same singleton data root: population points, health amenities, candidate grids, and OSM road context. They are separate from optimization figures so the data journey can be audited independently.
'''),
        code(r'''
TIMOR_FIGURE_ROOT = TIMOR_REPRO_OUTPUT / "figures"
TIMOR_FIGURE_ROOT.mkdir(parents=True, exist_ok=True)

timor_context_map_commands = []
for spacing in [10000, 5000, 1000]:
    timor_context_map_commands.append(
        ([
            ROUTING_PYTHON,
            LOCAL_TOOLS / "run_pipeline_fresh_root.py",
            "--pipeline-dir", DISTANCE_PIPELINE,
            "--fresh-base-root", TIMOR_REPRO_RUN,
            "timor_leste",
            "--network-backend", "osmium",
            "--simplify-network", "false",
            "--network-profile", "driving",
            "--sources", "amenities", "candidates",
            "--destinations", "population",
            "--candidate-grid-spacing-m", str(spacing),
            "--candidate-max-snap-dist-m", "5000",
            "--candidate-exclude-water", "false",
            "--no-aggregate",
            "--map-only",
            "--map-roads", "true",
            "--map-basemap-alpha", "0.70",
            "--map-dpi", "260",
            "--map-path", TIMOR_FIGURE_ROOT / f"timor_leste_context_{spacing // 1000}km.pdf",
        ], DISTANCE_PIPELINE)
    )

if RUN_DATA_PIPELINE:
    for cmd, cwd in timor_context_map_commands:
        run_command(cmd, cwd=cwd, dry_run=DRY_RUN)
else:
    print("RUN_DATA_PIPELINE is False. Context-map commands that would run:")
    for cmd, cwd in timor_context_map_commands:
        run_command(cmd, cwd=cwd, dry_run=True)

print("\nExpected Timor-Leste context maps:")
for spacing in [10000, 5000, 1000]:
    exists(
        TIMOR_FIGURE_ROOT
        / f"timor_leste_context_{spacing // 1000}km_resolution_{spacing}m.pdf"
    )

audit_latex_figures(
    [
        WORK_ROOT / "articles" / "integrated_access_report" / "manuscript.tex",
        WORK_ROOT / "presentations" / "integrated_access_deck" / "story_deck.tex",
    ],
    [
        WORK_ROOT / "articles" / "integrated_access_report" / "figures",
        WORK_ROOT / "presentations" / "integrated_access_deck" / "figures",
        TIMOR_FIGURE_ROOT,
    ],
)
'''),
        md("## Optimization, Sensitivity, and Figures"),
        code(r'''
timor_analysis_commands = [
    ([
        ROUTING_PYTHON,
        TIMOR_TOOL_DIR / "compare_timor_component_snapping.py",
        "--old-stats", TIMOR_OUTPUTS / "timor_network_profile_optimization_dense_primary_exact_20260623" / "timor_network_profile_instance_statistics.csv",
        "--old-results", TIMOR_OUTPUTS / "timor_network_profile_optimization_dense_primary_exact_20260623" / "timor_network_profile_results.csv",
        "--new-stats", TIMOR_OUTPUTS / "timor_network_profile_component012_optimization_20260626" / "timor_network_profile_instance_statistics.csv",
        "--new-results", TIMOR_OUTPUTS / "timor_network_profile_component012_optimization_20260626" / "timor_network_profile_results.csv",
        "--component-geography", TIMOR_REPRO_COMPONENTS / "timor_component_geography.csv",
        "--output-dir", RESULT_ROOT / "timor_component_snapping_comparison",
    ], WORK_ROOT),
    ([
        OPT_PYTHON,
        TIMOR_TOOL_DIR / "compute_timor_saturation_curves.py",
        "--outputs-dir", TIMOR_PIPELINE_RUN,
        "--saturation-csv", TIMOR_OUTPUTS / "timor_component012_saturation_20260626" / "timor_primary_saturation_summary.csv",
        "--output-dir", TIMOR_REPRO_SATURATION,
        "--threads", "1",
        "--time-limit", "300",
    ], WORK_ROOT),
    ([OPT_PYTHON, TIMOR_TOOL_DIR / "make_timor_network_profile_results_figures.py"], WORK_ROOT),
    ([ROUTING_PYTHON, TIMOR_TOOL_DIR / "make_timor_sibuni_network_detail.py"], WORK_ROOT),
]

if RUN_OPTIMIZATION:
    for cmd, cwd in timor_analysis_commands:
        run_command(cmd, cwd=cwd, dry_run=DRY_RUN)
else:
    print("RUN_OPTIMIZATION is False. Commands that would run:")
    for cmd, cwd in timor_analysis_commands:
        run_command(cmd, cwd=cwd, dry_run=True)
'''),
    ]


def write_notebook(path: Path, cells: list[dict]) -> None:
    notebook = {
        "cells": cells,
        "metadata": {
            "kernelspec": {
                "display_name": "Python 3",
                "language": "python",
                "name": "python3",
            },
            "language_info": {
                "name": "python",
                "pygments_lexer": "ipython3",
            },
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }
    path.write_text(json.dumps(notebook, indent=1, ensure_ascii=False), encoding="utf-8")
    print(path)


def main() -> None:
    NOTEBOOK_DIR.mkdir(parents=True, exist_ok=True)
    write_notebook(NOTEBOOK_DIR / "vietnam_access_reproduction.ipynb", vietnam_cells())
    write_notebook(NOTEBOOK_DIR / "timor_leste_access_reproduction.ipynb", timor_cells())


if __name__ == "__main__":
    main()
