# Parvathy PhD access-optimization reproducibility code

This folder contains country-specific reproducibility code for the Timor-Leste and Vietnam access-optimization studies used in Parvathy's PhD work.

The code is intentionally stored next to, but separate from, the two reusable project components:

```text
Research-Sandbox/general_distances_per_country
Research-Sandbox/abw_maxcover
```

Use `general_distances_per_country` to generate road-network distance matrices and component-aware snapped source/destination layers. Use `abw_maxcover` for exact and heuristic maximum-covering optimization.

## Layout

```text
Parvathy_PhD/
  README.md
  requirements-reproduction.txt
  shared/
    README.md
    tools/
      shared pipeline helpers and cross-country figure builders
  Timor_Leste/
    README.md
    tools/
      Timor-Leste pipeline, component, optimization, and figure scripts
  Vietnam/
    README.md
    docs/
      existing Vietnam replication notes plus 2026 integrated runbook
    scripts/
      Vietnam pipeline, geocoding audit, optimization, and figure scripts
  report/
    README.md
    tools/
      scripts that assemble cross-country report/deck figures from outputs
```

## What is committed

Committed:

- Python scripts needed to regenerate the country-specific pipeline calls, component diagnostics, optimization instances, Pareto curves, and figures.
- Runbooks describing the intended execution order and local output roots.
- Lightweight documentation and provenance notes.

Not committed:

- OSM PBF files.
- WorldPop rasters and derived population parquet files.
- Distance matrices and parquet-part folders.
- Gurobi logs, solver cache files, and generated `.npz` max-cover instances.
- Report/deck PDFs and rendered figure outputs.

Those files can be regenerated from the scripts and the public data sources, or reused from local archived outputs when auditing a historical run.

## Environment

The 2026 runs were executed on Windows with Python 3.14, the PISA distance pipeline, `pandana`, and the local `abw_maxcover` package. A typical development setup is:

```powershell
cd C:\github\Public-Infrastructure-Service-Access
py -3.14 -m venv Research-Sandbox\general_distances_per_country\.venv
Research-Sandbox\general_distances_per_country\.venv\Scripts\python.exe -m pip install -r Research-Sandbox\Parvathy_PhD\requirements-reproduction.txt
Research-Sandbox\general_distances_per_country\.venv\Scripts\python.exe -m pip install -e Research-Sandbox\abw_maxcover
```

Some exact runs require a licensed Gurobi installation. The heuristic and figure-generation code should still be inspectable without Gurobi; the reusable `abw_maxcover` package delays solver imports to solver-specific modules.

## Local output roots

The scripts write large generated data outside Git. The original sandbox runs used paths such as:

```text
C:\work\codex\sandboxes\Conclude_Parvathy_thesis\outputs
C:\work\codex\sandboxes\Conclude_Parvathy_thesis\runs
```

For a new machine, either edit the constants near the top of the relevant scripts or pass the available CLI options shown by:

```powershell
python <script>.py --help
```

The country READMEs describe the intended execution order.
