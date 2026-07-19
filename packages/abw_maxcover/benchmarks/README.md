# Benchmarks

These entry points exercise the installed `abw-maxcover` package. They do not
modify `sys.path` and therefore test the same import boundary as an external
user.

## Self-contained campaign

```powershell
python -m pip install -e packages\abw_maxcover[benchmark]
python packages\abw_maxcover\benchmarks\run_scaling.py --output benchmark-output
python packages\abw_maxcover\benchmarks\run_small_exact_audit.py --output benchmark-output
```

`run_scaling.py` generates square and asymmetric sparse instances, computes
complete greedy--compress--regreedy frontiers, and records stage times and
instance dimensions. `run_small_exact_audit.py` compares every approximate
budget with a dependency-free brute-force optimum on small random instances.

The scripts create CSV result files and JSON manifests containing environment,
configuration, seed, and elapsed time. They are intended as portable package
checks and as templates for paper-specific adapters. Literature datasets are
not bundled; an MPC artifact should provide immutable download instructions and
hashes when their licenses prevent redistribution.
