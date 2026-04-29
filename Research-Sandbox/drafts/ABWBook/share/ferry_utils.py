import pandas as pd
import geopandas as gpd

# ferry_utils.py

def connect_ferry_to_road(
    ferryCoords: list[tuple[float, float]],
    ferryIds: list[int],
    roadTree: cKDTree,
    roadCoords: np.ndarray,
    roadNodeIds: np.ndarray,
    maxDistance: float,
    targetCRS: str,
    sourceCRS: str = projectedCrsNL
) -> gpd.GeoDataFrame:
    """
    Connects ferry coordinates to their nearest road nodes using a precomputed cKDTree, 
    transforming all geometries to the target CRS.

    Args:
        ferryCoords: List of (x, y) coordinate tuples in source CRS.
        ferryIds: List of ferry IDs corresponding to each coordinate.
        roadTree: cKDTree built from road node coordinates (in source CRS).
        roadCoords: Nx2 array of (x, y) coordinates corresponding to roadTree (source CRS).
        roadNodeIds: 1D array of node IDs corresponding to roadCoords.
        maxDistance: Maximum snapping distance (in source CRS units).
        targetCRS: CRS to use for the output GeoDataFrame (e.g., 'EPSG:4326').
        sourceCRS: CRS of input coordinates (default: 'EPSG:28992').

    Returns:
        GeoDataFrame with columns:
            - 'ferryId'
            - 'ferryPoint' (in target CRS)
            - 'roadPoint' (in target CRS)
            - 'roadNodeId'
            - 'distance' (in source CRS units)
            - 'geometry' (LineString in target CRS)
    """
    start_total = pc()
    print('⏳ Starting ferry-to-road connection...')

    # Step 1: Nearest-neighbor search (in source CRS)
    t1 = pc()
    ferryArray = np.array(ferryCoords)
    distances, indices = roadTree.query(ferryArray, k=1)
    print(f'✅ Nearest-neighbor query completed in {pc() - t1:.2f}s.')

    # Step 2: Transformer for coordinates
    to_target_crs = pyproj.Transformer.from_crs(sourceCRS, targetCRS, always_xy=True).transform

    # Step 3: Build connector features (with transformed coordinates)
    t2 = pc()
    connectors = []
    for i, dist in enumerate(distances):
        if dist <= maxDistance:
            ferryPt_raw = Point(ferryCoords[i])
            roadPt_raw = Point(roadCoords[indices[i]])

            ferryPt = transform(to_target_crs, ferryPt_raw)
            roadPt = transform(to_target_crs, roadPt_raw)

            connectors.append({
                'ferryId': ferryIds[i],
                'ferryPoint': ferryPt,
                'roadPoint': roadPt,
                'roadNodeId': roadNodeIds[indices[i]],
                'distance': dist,  # remains in source CRS units
                'geometry': LineString([ferryPt, roadPt])
            })

    print(f'✅ Constructed {len(connectors)} connectors (≤ {maxDistance} units) in {pc() - t2:.2f}s.')

    if not connectors:
        print('⚠️ No connectors found within the specified maxDistance.')

    t_total = pc()
    print(f'🏁 Total runtime: {t_total - start_total:.2f}s.\n')

    return gpd.GeoDataFrame(connectors, geometry='geometry', crs=targetCRS)


def group_ferry_ids_by_coordinate(
    ferryCoords: list[tuple[float, float]],
    ferryIds: list[int],
) -> dict[tuple[float, float], list[int]]:
    """
    Groups ferry IDs by their coordinate location.

    Args:
        ferryCoords: List of (x, y) coordinate tuples.
        ferryIds: List of ferry IDs corresponding to each coordinate.

    Returns:
        A dictionary mapping each coordinate to a list of ferry IDs that pass through it.
    """
    coordToIds = defaultdict(list)
    for coord, fid in zip(ferryCoords, ferryIds):
        coordToIds[coord].append(fid)
    return dict(coordToIds)

def extract_ferry_coordinates_and_ids(ferries: gpd.GeoDataFrame) -> tuple[list[tuple[float, float]], list[int]]:
    """
    Extracts all coordinate tuples from LineString or MultiLineString geometries in a GeoDataFrame,
    and associates each coordinate with its corresponding ferry ID.

    Args:
        ferries: A GeoDataFrame with a 'geometry' column containing LineString or MultiLineString geometries
                 and an 'id' attribute per row (as either a column or an attribute).

    Returns:
        A tuple (coordinates, ferry_ids):
            - coordinates: List of (x, y) coordinate tuples.
            - ferry_ids: List of ferry IDs corresponding to each coordinate.
    """
    coordinates_ids = [
        (coord, row.id)
        for row in ferries.itertuples()
        for geom in [row.geometry]
        for coord in (
            geom.coords if geom.geom_type == 'LineString'
            else [pt for line in geom.geoms for pt in line.coords]
        )
    ]

    if not coordinates_ids:
        return [], []

    coordinates, ferry_ids = zip(*coordinates_ids)
    return list(coordinates), list(ferry_ids)

import pandas as pd
import geopandas as gpd

# ferry_utils.py

def connect_ferry_to_road(
    ferryCoords: list[tuple[float, float]],
    ferryIds: list[int],
    roadTree: cKDTree,
    roadCoords: np.ndarray,
    roadNodeIds: np.ndarray,
    maxDistance: float,
    targetCRS: str,
    sourceCRS: str = projectedCrsNL
) -> gpd.GeoDataFrame:
    """
    Connects ferry coordinates to their nearest road nodes using a precomputed cKDTree, 
    transforming all geometries to the target CRS.

    Args:
        ferryCoords: List of (x, y) coordinate tuples in source CRS.
        ferryIds: List of ferry IDs corresponding to each coordinate.
        roadTree: cKDTree built from road node coordinates (in source CRS).
        roadCoords: Nx2 array of (x, y) coordinates corresponding to roadTree (source CRS).
        roadNodeIds: 1D array of node IDs corresponding to roadCoords.
        maxDistance: Maximum snapping distance (in source CRS units).
        targetCRS: CRS to use for the output GeoDataFrame (e.g., 'EPSG:4326').
        sourceCRS: CRS of input coordinates (default: 'EPSG:28992').

    Returns:
        GeoDataFrame with columns:
            - 'ferryId'
            - 'ferryPoint' (in target CRS)
            - 'roadPoint' (in target CRS)
            - 'roadNodeId'
            - 'distance' (in source CRS units)
            - 'geometry' (LineString in target CRS)
    """
    start_total = pc()
    print('⏳ Starting ferry-to-road connection...')

    # Step 1: Nearest-neighbor search (in source CRS)
    t1 = pc()
    ferryArray = np.array(ferryCoords)
    distances, indices = roadTree.query(ferryArray, k=1)
    print(f'✅ Nearest-neighbor query completed in {pc() - t1:.2f}s.')

    # Step 2: Transformer for coordinates
    to_target_crs = pyproj.Transformer.from_crs(sourceCRS, targetCRS, always_xy=True).transform

    # Step 3: Build connector features (with transformed coordinates)
    t2 = pc()
    connectors = []
    for i, dist in enumerate(distances):
        if dist <= maxDistance:
            ferryPt_raw = Point(ferryCoords[i])
            roadPt_raw = Point(roadCoords[indices[i]])

            ferryPt = transform(to_target_crs, ferryPt_raw)
            roadPt = transform(to_target_crs, roadPt_raw)

            connectors.append({
                'ferryId': ferryIds[i],
                'ferryPoint': ferryPt,
                'roadPoint': roadPt,
                'roadNodeId': roadNodeIds[indices[i]],
                'distance': dist,  # remains in source CRS units
                'geometry': LineString([ferryPt, roadPt])
            })

    print(f'✅ Constructed {len(connectors)} connectors (≤ {maxDistance} units) in {pc() - t2:.2f}s.')

    if not connectors:
        print('⚠️ No connectors found within the specified maxDistance.')

    t_total = pc()
    print(f'🏁 Total runtime: {t_total - start_total:.2f}s.\n')

    return gpd.GeoDataFrame(connectors, geometry='geometry', crs=targetCRS)


def group_ferry_ids_by_coordinate(
    ferryCoords: list[tuple[float, float]],
    ferryIds: list[int],
) -> dict[tuple[float, float], list[int]]:
    """
    Groups ferry IDs by their coordinate location.

    Args:
        ferryCoords: List of (x, y) coordinate tuples.
        ferryIds: List of ferry IDs corresponding to each coordinate.

    Returns:
        A dictionary mapping each coordinate to a list of ferry IDs that pass through it.
    """
    coordToIds = defaultdict(list)
    for coord, fid in zip(ferryCoords, ferryIds):
        coordToIds[coord].append(fid)
    return dict(coordToIds)

def extract_ferry_coordinates_and_ids(ferries: gpd.GeoDataFrame) -> tuple[list[tuple[float, float]], list[int]]:
    """
    Extracts all coordinate tuples from LineString or MultiLineString geometries in a GeoDataFrame,
    and associates each coordinate with its corresponding ferry ID.

    Args:
        ferries: A GeoDataFrame with a 'geometry' column containing LineString or MultiLineString geometries
                 and an 'id' attribute per row (as either a column or an attribute).

    Returns:
        A tuple (coordinates, ferry_ids):
            - coordinates: List of (x, y) coordinate tuples.
            - ferry_ids: List of ferry IDs corresponding to each coordinate.
    """
    coordinates_ids = [
        (coord, row.id)
        for row in ferries.itertuples()
        for geom in [row.geometry]
        for coord in (
            geom.coords if geom.geom_type == 'LineString'
            else [pt for line in geom.geoms for pt in line.coords]
        )
    ]

    if not coordinates_ids:
        return [], []

    coordinates, ferry_ids = zip(*coordinates_ids)
    return list(coordinates), list(ferry_ids)