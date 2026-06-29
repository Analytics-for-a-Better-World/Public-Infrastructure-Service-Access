# Integrated report and deck reproducibility helpers

This folder contains code used to assemble cross-country figures for the integrated report and Beamer deck.

The report/deck combine outputs produced by the Timor-Leste and Vietnam country folders. The plotting script does not rerun heavy distance or optimization computations. It reads existing CSV/JSON/parquet-derived summaries from local output folders and regenerates the engineering and component-diagnostic figures used in the narrative.

## Main script

```text
tools/make_integrated_expansion_figures.py
```

It generates figures such as:

- Timor-Leste component-policy matrix effect;
- Timor-Leste component-size tail;
- Vietnam sparse-memory funnel;
- Vietnam engineering runtime breakdown;
- Vietnam approximation behavior.

The script expects the dated output folders produced by the country workflows. Update the root constants if reproducing on a different machine.
