import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd
import polars as pl

from distance_pipeline.distance_matrix import (
    compute_distances_polars,
    write_distances_polars_partitioned,
)


class FakeNetwork:
    def shortest_path_lengths(self, target_nodes, source_nodes, imp_name=None):
        return np.abs(
            np.asarray(target_nodes, dtype=np.float64)
            - np.asarray(source_nodes, dtype=np.float64)
        )


class CountingNetwork(FakeNetwork):
    def __init__(self) -> None:
        self.requested_paths = 0

    def shortest_path_lengths(self, target_nodes, source_nodes, imp_name=None):
        self.requested_paths += len(target_nodes)
        return super().shortest_path_lengths(target_nodes, source_nodes, imp_name)


def _sorted_frame(frame: pl.DataFrame) -> pd.DataFrame:
    return (
        frame
        .sort(['target_id', 'source_id'])
        .to_pandas()
        .reset_index(drop=True)
    )


class DistanceMatrixTests(unittest.TestCase):
    def test_partitioned_sparse_matches_single_table_result(self) -> None:
        targets = pd.DataFrame(
            {
                'ID': [10, 11, 12, 13],
                'xcoord': [0.0, 0.01, 0.02, 0.03],
                'ycoord': [0.0, 0.01, 0.02, 0.03],
                'nearest_node': [100, 101, 102, 103],
                'dist_snap_target': [1.0, 2.0, 3.0, 4.0],
                'target_type': ['population'] * 4,
            }
        ).set_index('ID', drop=False)
        sources = pd.DataFrame(
            {
                'ID': [20, 21, 22],
                'Longitude': [0.0, 0.02, 0.04],
                'Latitude': [0.0, 0.02, 0.04],
                'nearest_node': [90, 105, 110],
                'dist_snap_source': [5.0, 6.0, 7.0],
                'source_type': ['amenities'] * 3,
            }
        ).set_index('ID', drop=False)

        expected = compute_distances_polars(
            targets=targets,
            sources=sources,
            distance_threshold_largest=1000,
            network=FakeNetwork(),
            max_total_dist=30,
        )

        with tempfile.TemporaryDirectory() as tmp:
            output_dir = Path(tmp) / 'matrix.parquet_parts'
            summary = write_distances_polars_partitioned(
                targets=targets,
                sources=sources,
                distance_threshold_largest=1000,
                network=FakeNetwork(),
                output_dir=output_dir,
                max_total_dist=30,
                target_chunk_size=2,
            )
            parts = sorted(output_dir.glob('part-*.parquet'))
            actual = pl.concat([pl.read_parquet(path) for path in parts])

            self.assertEqual(summary['row_count'], expected.height)
            self.assertTrue((output_dir / '_SUCCESS.json').exists())
            pd.testing.assert_frame_equal(
                _sorted_frame(actual),
                _sorted_frame(expected),
            )

    def test_sparse_prefilter_prunes_components_and_stitch_only_impossible_pairs(self) -> None:
        targets = pd.DataFrame(
            {
                'ID': [10, 11],
                'xcoord': [0.0, 0.0],
                'ycoord': [0.0, 0.0],
                'nearest_node': [100, 200],
                'dist_snap_target': [2.0, 90.0],
                'target_type': ['population'] * 2,
                'component_id': [0, 1],
            }
        ).set_index('ID', drop=False)
        sources = pd.DataFrame(
            {
                'ID': [20, 21],
                'Longitude': [0.0, 0.0],
                'Latitude': [0.0, 0.0],
                'nearest_node': [90, 210],
                'dist_snap_source': [3.0, 20.0],
                'source_type': ['amenities'] * 2,
                'component_id': [0, 1],
            }
        ).set_index('ID', drop=False)
        network = CountingNetwork()

        result = compute_distances_polars(
            targets=targets,
            sources=sources,
            distance_threshold_largest=1000,
            network=network,
            max_total_dist=100,
        )

        self.assertEqual(network.requested_paths, 1)
        self.assertEqual(result.height, 1)
        self.assertEqual(result['target_id'].to_list(), [10])
        self.assertEqual(result['source_id'].to_list(), [20])

    def test_node_pair_cache_is_reused_from_bucketed_chunks(self) -> None:
        targets = pd.DataFrame(
            {
                'ID': [10, 11],
                'xcoord': [0.0, 0.01],
                'ycoord': [0.0, 0.01],
                'nearest_node': [100, 101],
                'dist_snap_target': [1.0, 1.0],
            }
        ).set_index('ID', drop=False)
        sources = pd.DataFrame(
            {
                'ID': [20, 21],
                'Longitude': [0.0, 0.01],
                'Latitude': [0.0, 0.01],
                'nearest_node': [90, 91],
                'dist_snap_source': [1.0, 1.0],
            }
        ).set_index('ID', drop=False)

        with tempfile.TemporaryDirectory() as tmp:
            cache_dir = Path(tmp) / 'node_pair_cache'
            first_network = CountingNetwork()
            first = compute_distances_polars(
                targets=targets,
                sources=sources,
                distance_threshold_largest=1000,
                network=first_network,
                node_pair_cache_dir=cache_dir,
            )
            self.assertGreater(first_network.requested_paths, 0)
            self.assertTrue(list(cache_dir.glob('bucket=*/node_pairs_*.parquet')))

            second_network = CountingNetwork()
            second = compute_distances_polars(
                targets=targets,
                sources=sources,
                distance_threshold_largest=1000,
                network=second_network,
                node_pair_cache_dir=cache_dir,
            )

            self.assertEqual(second_network.requested_paths, 0)
            pd.testing.assert_frame_equal(
                _sorted_frame(second),
                _sorted_frame(first),
            )


    def test_partitioned_sparse_adapts_requested_chunk_to_pair_cap(self) -> None:
        targets = pd.DataFrame(
            {
                'ID': [10, 11, 12, 13, 14, 15],
                'xcoord': [0.0, 0.001, 0.002, 0.003, 0.004, 0.005],
                'ycoord': [0.0, 0.001, 0.002, 0.003, 0.004, 0.005],
                'nearest_node': [100, 101, 102, 103, 104, 105],
                'dist_snap_target': [1.0] * 6,
                'target_type': ['population'] * 6,
            }
        ).set_index('ID', drop=False)
        sources = pd.DataFrame(
            {
                'ID': [20, 21, 22, 23],
                'Longitude': [0.0, 0.001, 0.002, 0.003],
                'Latitude': [0.0, 0.001, 0.002, 0.003],
                'nearest_node': [90, 91, 92, 93],
                'dist_snap_source': [1.0] * 4,
                'source_type': ['amenities'] * 4,
            }
        ).set_index('ID', drop=False)

        expected = compute_distances_polars(
            targets=targets,
            sources=sources,
            distance_threshold_largest=1000,
            network=FakeNetwork(),
            max_total_dist=30,
        )

        with tempfile.TemporaryDirectory() as tmp:
            output_dir = Path(tmp) / 'matrix.parquet_parts'
            summary = write_distances_polars_partitioned(
                targets=targets,
                sources=sources,
                distance_threshold_largest=1000,
                network=FakeNetwork(),
                output_dir=output_dir,
                max_total_dist=30,
                target_chunk_size=6,
                max_spatial_pairs_per_chunk=8,
            )
            parts = sorted(output_dir.glob('part-*.parquet'))
            actual = pl.concat([pl.read_parquet(path) for path in parts])

            self.assertGreater(summary['part_count'], 1)
            self.assertEqual(summary['max_spatial_pairs_per_chunk'], 8)
            self.assertIn('chunks', summary)
            self.assertEqual(summary['chunking']['max_spatial_pairs_per_chunk'], 8)
            self.assertEqual(summary['chunking']['chunk_count'], len(summary['chunks']))
            self.assertGreater(summary['chunking']['adjusted_chunk_count'], 0)
            self.assertLessEqual(
                summary['chunking']['max_estimated_spatial_candidate_pairs'],
                8,
            )
            pd.testing.assert_frame_equal(
                _sorted_frame(actual),
                _sorted_frame(expected),
            )


if __name__ == '__main__':
    unittest.main()
