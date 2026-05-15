# Luxembourg school case comparison: earlier pipeline outputs vs WFP architecture

This comparison uses the local Luxembourg artifacts from `C:\local\Download_Depot\luxembourg_data\outputs` as the earlier-run baseline. The referenced `codex://` thread is not exposed as a readable resource in this session, so the comparison is grounded in the persisted outputs and manifests on disk.

## Use Cases Compared

- Luxembourg schools, aggregate factor 10, population threshold 1, sample 1, max total distance 1000 m, candidates spacing 5000 m, max snap 5000 m.
- Luxembourg schools, full resolution (`aggregate_factor=None`), same remaining parameters.

## Summary Table

| use case | metric | earlier output | current WFP architecture/cache | discrepancy |
|---|---:|---:|---:|---|
| agg_10 | population rows | 4800 | 4800 | none |
| agg_10 | existing school/source rows | 394 | 391 | -3 |
| agg_10 | all source rows | 490 | 487 | -3 |
| agg_10 | distance matrix rows | 885 | 879 | -6 |
| full_resolution | population rows | 102061 | 102061 | none |
| full_resolution | existing school/source rows | 394 | 391 | -3 |
| full_resolution | all source rows | 490 | 487 | -3 |
| full_resolution | distance matrix rows | 65583 | missing | current matrix cache missing / not recomputed |

## Discrepancies Identified

- aggregate-10: existing school/source count differs: earlier output has 394, current architecture has 391. The current path uses `facility_points_*_dedup_v1` and current layer-signature source snapping.
- aggregate-10: total sources differ: earlier output has 490, current architecture has 487. This follows from the school/source count difference while candidate count remains 96.
- aggregate-10: distance matrix row count differs: earlier output has 885, current architecture has 879. The current matrix was computed with source cache signature `['src_amenities-candidates', 'dst_population', 'amenity_school']`.
- full-resolution: existing school/source count differs: earlier output has 394, current architecture has 391. The current path uses `facility_points_*_dedup_v1` and current layer-signature source snapping.
- full-resolution: total sources differ: earlier output has 490, current architecture has 487. This follows from the school/source count difference while candidate count remains 96.
- full-resolution: current architecture-compatible matrix cache is missing at `C:\local\Download_Depot\luxembourg_data\cache\luxembourg-latest.osm_distance_matrix_threshold_300km_max_total_1000m_pop_1_sample_1_agg_none_max_none_amenity_school_dst_population_src_amenities-candidates_candidates_spacing_5000m_max_snap_5000m.pkl`; earlier exported matrix has 65583 rows. Full recomputation was not triggered in this comparison because it would build a new full-resolution matrix cache.

## Identifier and Schema Details

- Matrix schema is unchanged for the aggregate-10 comparison: both earlier and current matrices have `target_id`, `source_id`, `source_nearest_node`, `target_nearest_node`, `target_to_road_dist`, `road_distance`, `source_to_road_dist`, `total_dist`.
- Existing-source schema is intentionally slimmer in the current architecture path. Earlier exported sources carried the full OSM tag payload, while current prepared sources keep the routing contract columns plus `name` and `amenity`: `ID`, `Longitude`, `Latitude`, `nearest_node`, `dist_snap_source`, `source_type`, `name`, `amenity`.
- Source IDs are now role-prefixed in the architecture path, for example `source_amenities_265819709` instead of `265819709`, to avoid collisions when the same table can be used as source and target.
- After stripping the `source_amenities_` prefix, the actual three earlier school IDs missing from the current architecture path are:
  - `185207530` at `(6.190196, 49.515029)`
  - `268229700` at `(6.479845, 49.700772)`
  - `1461344068` at `(6.162993, 49.621714)`
- No new unprefixed school IDs appear in the current architecture path; the difference is a strict reduction from 394 to 391 existing schools.

## Notes

- Current architecture source cache values: `['src_amenities-candidates', 'dst_population', 'amenity_school']`.
- Current architecture candidate count: `96`.
- The aggregate-10 architecture matrix cache exists and was compared end to end.
- The full-resolution architecture-compatible matrix cache does not yet exist, so only lightweight full-resolution stages were compared.
