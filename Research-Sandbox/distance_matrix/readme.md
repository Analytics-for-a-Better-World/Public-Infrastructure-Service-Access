# Distance Matrix Computation

This module computes a population to hospital distance matrix by combining:

1. a spatial prefilter using a KD tree over geographic coordinates
2. road network shortest path distances between nearest network nodes
3. access and egress distances from each point to the road network

The goal is to efficiently generate candidate population hospital pairs within a maximum geographic threshold, then compute road based travel distances only for those relevant pairs.

## What the function does

The `compute_distances` function:

1. checks that both input dataframes are indexed by `ID`
2. extracts population and hospital coordinates
3. builds a `cKDTree` for hospital coordinates
4. finds all hospitals within a geographic radius of each population point
5. maps each population hospital pair to the corresponding nearest road network nodes
6. removes duplicate origin destination node pairs
7. computes shortest path lengths on the road network
8. discards unreachable paths
9. merges back the original population and hospital identifiers
10. adds access distance, road distance, and egress distance into a total distance

It returns a dataframe with one row per valid population hospital pair.

## Dependencies

This code uses:

- `numpy`
- `pandas`
- `scipy`
- a road network object exposing `shortest_path_lengths(o, d)`

It also suppresses a specific warning emitted by `pandana.network`.

## Function signature

```python
def compute_distances(population, all_hospitals, distance_threshold_largest, network):
    ...