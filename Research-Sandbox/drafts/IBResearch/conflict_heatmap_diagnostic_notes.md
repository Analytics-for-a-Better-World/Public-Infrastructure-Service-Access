# Conflict Heatmap Diagnostic Notes

These notes document the diagnostic check behind the conflict-heatmap discussion in the manuscript. The reusable command is:

```powershell
py -B analyze_conflict_heatmap_structure.py
```

It uses the proven optimal toy solution `clean_runs_20260508/toy_antony_verbatim_mip.csv` and the best known full-instance solution `clean_runs_20260508/full_lns_nb34_guarded_6x120.csv`.

## Main conclusion

The strong diagonal appearance originally seen in the full-instance heatmap was mostly a plotting-order artifact. The full exam list uses the column name `FULL_NAME`, while the heatmap helper originally recognized only `Full Name` and `Exam_Name`. Because of that mismatch, the helper fell back to the already chronological timetable order for the full-instance rows. Plotting chronological rows against chronological columns naturally creates a near diagonal.

The helper now recognizes `FULL_NAME`, so rows follow the source exam-list order for both the toy and full-instance heatmaps. With this corrected order, the full-instance assignment pattern is only mildly closer to the diagonal than random placement: mean normalized diagonal distance 0.254 versus a random mean of 0.339. The conflict cells themselves are not strongly diagonal: their mean normalized distance is 0.308 versus a random mean of 0.337, and their row/column correlation is -0.067.

## Toy instance

The toy solution used here is proven optimal at objective 25,190 for the currently available toy data. It has no same-slot conflict mass in the toy pair matrix:

- exams: 20
- date-slot columns: 32
- same-slot conflict exams: 0
- same-slot conflict pairs: 0
- same-slot conflict mass: 0
- assigned-cell row/column correlation: 0.327
- assigned-cell mean normalized diagonal distance: 0.310
- random baseline mean normalized diagonal distance: 0.346

Because there are no same-slot conflicts, any impression of scattered conflict cells in the toy heatmap is not supported by the corrected conflict diagnostic. The toy instance is also too small to reveal stable large-scale community structure in the assignment heatmap.

## Full instance

The full-instance solution is the best solution found so far, not a proven optimum. Its same-slot conflicts are concentrated in small subject-family components, not in a single global diagonal chain:

- exams: 64
- date-slot columns: 68
- same-slot conflict exams: 36
- same-slot conflict pairs: 31
- same-slot conflict mass: 3,420
- assigned-cell row/column correlation: 0.309
- assigned-cell mean normalized diagonal distance: 0.254
- random baseline mean normalized diagonal distance: 0.339
- conflict-cell row/column correlation: -0.067
- conflict-cell mean normalized diagonal distance: 0.308
- random conflict-cell baseline mean normalized diagonal distance: 0.337

The largest conflict components are small and interpretable. Examples include:

- `ENV. AND SOC. PAPER TWO` with `PHYSICS P1 AND P3`, mass 1,006.
- French, generic-language, and Spanish acquisition reading papers, mass 543.
- French, generic-language, and Spanish listening papers, mass 543.
- Geography, global politics, social/cultural anthropology, and world religions papers, mass 434.

These components suggest subject-family overlap and co-enrolment communities. They do not indicate geography, routes, facilities, or spatial locality, because the instance data used in these scripts do not contain geographical locations, facility assignments, routes, or allocation regions.

## Co-enrolment locality

The co-enrolment matrix has more source-order locality in the toy instance than in the full instance:

- toy: 59.4% of positive co-enrolment weight lies within six rows of the source order;
- full: 22.4% of positive co-enrolment weight lies within six rows of the source order.

Therefore the corrected full-instance conflict pattern is not explained by a strongly banded co-enrolment matrix. The visually important structure comes from a combination of source-list subject grouping, the heuristic/MILP tendency to schedule related papers in coherent windows, and small conflict communities that remain after optimization.

## Diagnostic outputs

The script writes:

- `diagnostic_summary.json`
- `toy_same_slot_conflict_pairs.csv`
- `full_same_slot_conflict_pairs.csv`
- `toy_source_order_assignment_diagnostic.png`
- `full_source_order_assignment_diagnostic.png`
- `toy_chronological_row_assignment_diagnostic.png`
- `full_chronological_row_assignment_diagnostic.png`
- `assignment_distance_from_diagonal.png`
- `coenrollment_weight_by_source_row_distance.png`

The chronological-row diagnostic is intentionally retained. It demonstrates how a diagonal can be created by row ordering alone, and is useful as a warning when interpreting assignment heatmaps.
