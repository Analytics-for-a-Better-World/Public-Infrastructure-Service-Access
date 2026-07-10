import tempfile
import unittest
from pathlib import Path

import pandas as pd

from distance_pipeline.population import population_to_points


class PopulationProviderTests(unittest.TestCase):
    def test_meta_style_population_table_to_points(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            table_path = Path(tmp_dir) / 'meta_population.csv'
            pd.DataFrame(
                {
                    'longitude': [6.13, 6.14, 6.15],
                    'latitude': [49.61, 49.62, 49.63],
                    'population_count': [0.5, 2.0, 3.0],
                }
            ).to_csv(table_path, index=False)

            points = population_to_points(
                table_path,
                population_threshold=1.0,
                data_format='table',
                verbose=False,
            )

        self.assertEqual(len(points), 2)
        self.assertEqual(points.crs.to_epsg(), 4326)
        self.assertEqual(points['population'].tolist(), [2.0, 3.0])
        self.assertEqual(points['ID'].tolist(), [0, 1])


if __name__ == '__main__':
    unittest.main()
