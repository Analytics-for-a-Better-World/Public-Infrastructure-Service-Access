import unittest

import geopandas as gpd
import pandas as pd
from shapely.geometry import LineString

from distance_pipeline.viz import merge_roads_for_plotting


class RoadPlotMergeTests(unittest.TestCase):
    def test_merge_roads_for_plotting_merges_by_road_class(self) -> None:
        roads = gpd.GeoDataFrame(
            {
                'road_class': pd.Categorical(
                    ['primary', 'primary', 'secondary'],
                    categories=['primary', 'secondary'],
                ),
                'geometry': [
                    LineString([(0, 0), (1, 0)]),
                    LineString([(1, 0), (2, 0)]),
                    LineString([(0, 1), (1, 1)]),
                ],
            },
            geometry='geometry',
            crs='EPSG:3857',
        )

        merged = merge_roads_for_plotting(roads, verbose=False)

        self.assertEqual(len(merged), 2)
        self.assertEqual(merged.crs, roads.crs)
        self.assertEqual(
            merged['road_class'].value_counts().to_dict(),
            {'primary': 1, 'secondary': 1},
        )
        primary = merged.loc[merged['road_class'] == 'primary'].geometry.iloc[0]
        self.assertEqual(list(primary.coords), [(0.0, 0.0), (1.0, 0.0), (2.0, 0.0)])


if __name__ == '__main__':
    unittest.main()
