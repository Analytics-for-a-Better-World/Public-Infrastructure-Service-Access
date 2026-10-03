import tempfile
import unittest
from pathlib import Path

import pandas as pd

from distance_pipeline.config_loader import load_cfg
from distance_pipeline.manifest import build_run_manifest
from distance_pipeline.settings import PipelineSettings
from run_pipeline import add_component_labels, summarize_layer_retention


class ManifestDiagnosticsTests(unittest.TestCase):
    def test_component_labels_are_attached_from_nearest_nodes(self) -> None:
        table = pd.DataFrame({'nearest_node': [11, 22, 99]})
        component_ids = pd.Series({11: 0, 22: 2}, name='component_id')

        result = add_component_labels(table, component_ids)

        self.assertEqual(result['component_id'].tolist(), [0, 2, -1])

    def test_retention_summary_preserves_full_population_denominator(self) -> None:
        original = pd.DataFrame({'population': [10.0, 20.0, 30.0]})
        retained = original.iloc[:2].copy()

        summary = summarize_layer_retention(
            original,
            retained,
            weight_col='population',
        )

        self.assertEqual(summary['input_rows'], 3)
        self.assertEqual(summary['retained_rows'], 2)
        self.assertEqual(summary['excluded_rows'], 1)
        self.assertEqual(summary['input_weight'], 60.0)
        self.assertEqual(summary['retained_weight'], 30.0)
        self.assertEqual(summary['excluded_weight'], 30.0)
        self.assertEqual(summary['coverage_denominator_weight'], 60.0)

    def test_manifest_records_scientific_policy_and_diagnostics(self) -> None:
        cfg = load_cfg('timor_leste')
        policy = {
            'routing_graph_scope': 'full_loaded_network',
            'snapping_node_scope': 'selected_weak_components',
        }
        diagnostics = {
            'layers': {
                'population': {
                    'input_rows': 3,
                    'retained_rows': 2,
                    'excluded_rows': 1,
                }
            }
        }

        with tempfile.TemporaryDirectory() as tmp_dir:
            output = Path(tmp_dir) / 'output.parquet'
            output.write_bytes(b'test')
            manifest = build_run_manifest(
                cfg=cfg,
                settings=PipelineSettings(snap_components=(0,)),
                aggregate_factor=1,
                amenity_values=['hospital'],
                candidate_grid_spacing_m=None,
                candidate_max_snap_dist_m=None,
                has_candidates=False,
                output_paths={'population': output},
                repo_dir=Path(__file__).resolve().parents[1],
                scientific_policy=policy,
                diagnostics=diagnostics,
            )

        self.assertEqual(
            manifest['parameters']['scientific_policy'],
            policy,
        )
        self.assertEqual(manifest['diagnostics'], diagnostics)


if __name__ == '__main__':
    unittest.main()
