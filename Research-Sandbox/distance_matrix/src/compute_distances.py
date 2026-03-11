
# Standard library imports
from time import perf_counter as pc

# Third-party library imports
import numpy as np
import pandas as pd
from scipy.spatial import cKDTree

# Suppress specific warnings from Pandana
import warnings
warnings.filterwarnings(
    "ignore",
    message="Unsigned integer: shortest path distance is trying to be calculated",
    module="pandana.network"
)

def compute_distances( population, all_hospitals, distance_threshold_largest, network ):
    t = pc()

    # assert that the frames are indexed on the IDs
    assert ( population.index == population.ID ).all()
    assert ( all_hospitals.index == all_hospitals.ID ).all()

    # Convert population and hospital coordinates to NumPy arrays
    pop_coords = population[['xcoord', 'ycoord']].values
    hosp_coords = all_hospitals[['Longitude', 'Latitude']].values

    # Convert degrees to radians for cKDTree (better for spherical distance calculations)
    pop_coords_rad = np.radians(pop_coords)
    hosp_coords_rad = np.radians(hosp_coords)

    print( f'preparing {len(pop_coords_rad):,} x {len(hosp_coords_rad):,} for spatial nearest neighbors bounded by {distance_threshold_largest} km in {pc() - t:.2f} seconds' )
    t = pc()

    # Build a KDTree for hospitals using all hospital data
    tree = cKDTree(hosp_coords_rad)

    # Query the tree to find hospitals within the given threshold for each population point
    indices = tree.query_ball_point(pop_coords_rad, r=np.radians(distance_threshold_largest / 6367.0))  # Convert km to radians

    # Flatten the indices result while keeping track of original indices
    pop_indices = np.repeat(np.arange(len(indices)), [len(n) for n in indices])  # Expand population indices
    hosp_indices = np.concatenate(indices)  # Flatten all hospital index lists

    print( f'finding {len(hosp_indices):,} pairs of spatial nearest neighbors in {pc() - t:.2f} seconds' )
    t = pc()

    # Create DataFrame for mapping population and hospital IDs
    df = pd.DataFrame({'pop_index': pop_indices, 'hosp_index': hosp_indices})

    # Map indices directly to population and hospital IDs
    df['pop_id'] = population.iloc[df['pop_index']]['ID'].values
    df['hosp_id'] = all_hospitals.iloc[df['hosp_index']]['ID'].values

    # Map IDs back to their nearest nodes (matching by ID, not index)
    df['pop_nearest_node'] = population.loc[df['pop_id']]['nearest_node'].values
    df['hosp_nearest_node'] = all_hospitals.loc[df['hosp_id']]['nearest_node'].values

    # Drop duplicates to ensure unique (nearest_node_o, nearest_node_d) pairs
    df.drop_duplicates(subset=['pop_nearest_node', 'hosp_nearest_node'], inplace=True)
    o = df.pop_nearest_node.values
    d = df.hosp_nearest_node.values

    print( f'creating the origins and destinations in {pc() - t:.2f} seconds' )

    t = pc()
    sp = network.shortest_path_lengths(o,d)
    tsp = pc()-t
    t = pc()

    # 4294967.295 = (2^32 − 1)/1000 is pandana's way to tell that no path exists
    # https://github.com/UDST/pandana/issues/168 
    dists_df = pd.DataFrame.from_records(
        [(o, d, p) for o, d, p in zip(o, d, sp) if p < 4294967.295], 
        columns=['pop_nearest_node', 'hosp_nearest_node', 'road_distance']
    )
    print( f'{len(sp):,} shortest paths of which {dists_df.shape[0]:,} exist found in {tsp:.2f} seconds' )

    # Merge with population and hospital data to get IDs and distances
    pop_df = population[['ID', 'nearest_node', 'pop_dist_road_estrada']].rename(
        columns={'nearest_node': 'pop_nearest_node', 'ID': 'pop_id', 'pop_dist_road_estrada': 'pop_to_road_dist'}
    )
    hosp_df = all_hospitals[['ID', 'nearest_node', 'hosp_dist_road_estrada']].rename(
        columns={'nearest_node': 'hosp_nearest_node', 'ID': 'hosp_id', 'hosp_dist_road_estrada': 'hosp_to_road_dist'}
    )

    # Merge to get population and hospital IDs
    matrix_df = (
        dists_df
        .merge(pop_df, on='pop_nearest_node', how='inner')  # Map population IDs
        .merge(hosp_df, on='hosp_nearest_node', how='inner')  # Map hospital IDs
    )

    # Extract relevant columns
    matrix_df = matrix_df[['pop_id', 'hosp_id', 'pop_to_road_dist', 'road_distance', 'hosp_to_road_dist']]

    # Complement 
    matrix_df['total_dist'] = matrix_df['pop_to_road_dist']+matrix_df['road_distance']+matrix_df['hosp_to_road_dist']

    print( f'assembling {matrix_df.shape[0]:,} distances of interest in {pc() - t:.2f} seconds' )
    return matrix_df