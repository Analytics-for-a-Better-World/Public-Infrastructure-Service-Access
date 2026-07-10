import argparse
import unittest
from pathlib import Path

from distance_pipeline.config_loader import load_cfg
from distance_pipeline.pipeline_support import build_output_run_tag
from distance_pipeline.settings import PipelineSettings
from run_pipeline import (
    build_parser,
    pbf_filename_for_output_tag,
    population_label_for_output_tag,
    resolve_input_config,
    short_output_path,
)


def make_args(**overrides: object) -> argparse.Namespace:
    values: dict[str, object] = {
        'base_root': None,
        'data_root': None,
        'cache_root': None,
        'output_root': None,
        'worldpop_year': None,
        'worldpop_dataset': None,
        'worldpop_release': None,
        'worldpop_version': None,
        'worldpop_resolution': None,
        'worldpop_constrained': None,
        'worldpop_filename': None,
        'worldpop_url': None,
        'worldpop_path': None,
        'population_provider': None,
        'population_format': None,
        'population_filename': None,
        'population_url': None,
        'population_path': None,
        'meta_population_year': None,
        'pbf_filename': None,
        'pbf_url': None,
    }
    values.update(overrides)
    return argparse.Namespace(**values)


class PbfOverrideTests(unittest.TestCase):
    def test_default_timor_leste_pbf_config_is_unchanged(self) -> None:
        cfg = load_cfg('timor_leste')

        resolved = resolve_input_config(cfg, make_args())

        self.assertIs(resolved, cfg)
        self.assertEqual(resolved.resolved_pbf_filename, 'east-timor-latest.osm.pbf')
        self.assertEqual(resolved.population_provider, 'worldpop')
        self.assertEqual(resolved.POPULATION_PATH, resolved.WORLDPOP_PATH)
        self.assertIsNone(population_label_for_output_tag(resolved, cfg))
        self.assertEqual(
            resolved.PBF_URL,
            'https://download.geofabrik.de/asia/east-timor-latest.osm.pbf',
        )
        self.assertIsNone(pbf_filename_for_output_tag(resolved, cfg))

    def test_pbf_filename_override_uses_country_geofabrik_region(self) -> None:
        cfg = load_cfg('timor_leste')

        resolved = resolve_input_config(
            cfg,
            make_args(pbf_filename='east-timor-260613.osm.pbf'),
        )

        self.assertEqual(resolved.resolved_pbf_filename, 'east-timor-260613.osm.pbf')
        self.assertEqual(
            resolved.PBF_URL,
            'https://download.geofabrik.de/asia/east-timor-260613.osm.pbf',
        )
        self.assertEqual(
            pbf_filename_for_output_tag(resolved, cfg),
            'east-timor-260613.osm.pbf',
        )

    def test_pbf_url_derives_cache_filename_from_url_path(self) -> None:
        cfg = load_cfg('timor_leste')

        resolved = resolve_input_config(
            cfg,
            make_args(
                pbf_url='https://example.test/osm/east-timor-260613.osm.pbf',
            ),
        )

        self.assertEqual(resolved.resolved_pbf_filename, 'east-timor-260613.osm.pbf')
        self.assertEqual(
            resolved.PBF_URL,
            'https://example.test/osm/east-timor-260613.osm.pbf',
        )
        self.assertEqual(resolved.PBF_PATH.name, 'east-timor-260613.osm.pbf')

    def test_explicit_pbf_filename_can_pair_with_custom_url(self) -> None:
        cfg = load_cfg('timor_leste')

        resolved = resolve_input_config(
            cfg,
            make_args(
                pbf_filename='timor-leste-mirror.osm.pbf',
                pbf_url='https://example.test/download',
            ),
        )

        self.assertEqual(resolved.resolved_pbf_filename, 'timor-leste-mirror.osm.pbf')
        self.assertEqual(resolved.PBF_URL, 'https://example.test/download')

    def test_output_tag_preserves_default_and_tags_pbf_overrides(self) -> None:
        settings = PipelineSettings(max_points=10, max_total_dist=5000)
        kwargs = {
            'settings': settings,
            'aggregate_factor': None,
            'amenity_values': ['hospital', 'clinic'],
            'candidate_grid_spacing_m': None,
            'candidate_max_snap_dist_m': None,
            'has_candidates': False,
        }

        default_tag = build_output_run_tag(**kwargs)
        pbf_tag = build_output_run_tag(
            **kwargs,
            pbf_filename='east-timor-260613.osm.pbf',
        )

        self.assertEqual(
            default_tag,
            (
                'pop_1_sample_1_seed_42_max_10_agg_none_maxdist_5000_'
                'amenity_clinic-hospital_no_candidates'
            ),
        )
        self.assertTrue(pbf_tag.endswith('_pbf_east-timor-260613'))

    def test_shortened_output_path_preserves_visible_pbf_suffix(self) -> None:
        path = Path('C:/local/Download_Depot/east-timor_data/outputs') / (
            'distance_matrix_src_amenities_dst_population_'
            'pop_1_sample_1_seed_42_max_10_agg_none_maxdist_5000_'
            'amenity_amenity_clinic-doctors-hospital-dst_population-src_amenities_'
            'no_candidates_pbf_east-timor-260613.parquet'
        )

        shortened = short_output_path(path)

        self.assertLessEqual(len(shortened.name), 120)
        self.assertIn('_pbf_east-timor-260613_', shortened.name)

    def test_meta_population_path_override_uses_local_population_input(self) -> None:
        cfg = load_cfg('timor_leste')

        resolved = resolve_input_config(
            cfg,
            make_args(
                population_provider='meta',
                population_format='table',
                population_path='C:/data/meta_tls_population.csv',
                meta_population_year=2020,
            ),
        )

        self.assertEqual(resolved.population_provider, 'meta')
        self.assertEqual(resolved.population_format, 'table')
        self.assertEqual(resolved.meta_population_year, 2020)
        self.assertEqual(
            resolved.POPULATION_PATH,
            Path('C:/data/meta_tls_population.csv'),
        )
        self.assertEqual(
            population_label_for_output_tag(resolved, cfg),
            'meta_meta_tls_population',
        )

    def test_meta_population_url_derives_filename(self) -> None:
        cfg = load_cfg('timor_leste')

        resolved = resolve_input_config(
            cfg,
            make_args(
                population_provider='meta',
                population_format='raster',
                population_url='https://example.test/meta/tls_hrsl_2020.tif',
            ),
        )

        self.assertEqual(resolved.population_provider, 'meta')
        self.assertEqual(resolved.resolved_meta_population_filename, 'tls_hrsl_2020.tif')
        self.assertEqual(resolved.POPULATION_PATH.name, 'tls_hrsl_2020.tif')
        self.assertEqual(resolved.POPULATION_URL, 'https://example.test/meta/tls_hrsl_2020.tif')

    def test_root_defaults_preserve_legacy_layout(self) -> None:
        cfg = load_cfg('timor_leste')

        resolved = resolve_input_config(cfg, make_args())

        self.assertIs(resolved, cfg)
        self.assertEqual(resolved.DATA_ROOT, resolved.base_root)
        self.assertEqual(
            resolved.BASE_DIR,
            resolved.base_root / f'{resolved.country_slug}_data',
        )
        self.assertEqual(resolved.CACHE_DIR, resolved.BASE_DIR / 'cache')
        self.assertEqual(resolved.OUTPUT_DIR, resolved.BASE_DIR / 'outputs')
        self.assertEqual(resolved.FIGURES_DIR, resolved.BASE_DIR / 'figures')

    def test_root_overrides_keep_data_cache_and_outputs_separate(self) -> None:
        cfg = load_cfg('timor_leste')

        resolved = resolve_input_config(
            cfg,
            make_args(
                base_root='C:/legacy_depot',
                data_root='C:/pipeline_data',
                cache_root='D:/pipeline_cache/timor',
                output_root='E:/pipeline_outputs/timor_run_001',
            ),
        )

        self.assertEqual(resolved.base_root, Path('C:/legacy_depot'))
        self.assertEqual(resolved.DATA_ROOT, Path('C:/pipeline_data'))
        self.assertEqual(
            resolved.BASE_DIR,
            Path('C:/pipeline_data') / f'{resolved.country_slug}_data',
        )
        self.assertEqual(resolved.CACHE_DIR, Path('D:/pipeline_cache/timor'))
        self.assertEqual(resolved.OUTPUT_DIR, Path('E:/pipeline_outputs/timor_run_001'))
        self.assertEqual(
            resolved.FIGURES_DIR,
            Path('E:/pipeline_outputs/timor_run_001') / 'figures',
        )
        self.assertEqual(
            resolved.PBF_PATH,
            resolved.BASE_DIR / resolved.resolved_pbf_filename,
        )
        self.assertEqual(
            resolved.POPULATION_PATH,
            resolved.BASE_DIR / resolved.resolved_population_filename,
        )

    def test_parser_accepts_storage_root_options(self) -> None:
        parser = build_parser()

        args = parser.parse_args([
            'timor_leste',
            '--data-root', 'C:/pipeline_data',
            '--cache-root', 'D:/pipeline_cache/timor',
            '--output-root', 'E:/pipeline_outputs/timor_run_001',
        ])

        self.assertEqual(args.data_root, 'C:/pipeline_data')
        self.assertEqual(args.cache_root, 'D:/pipeline_cache/timor')
        self.assertEqual(args.output_root, 'E:/pipeline_outputs/timor_run_001')


if __name__ == '__main__':
    unittest.main()
