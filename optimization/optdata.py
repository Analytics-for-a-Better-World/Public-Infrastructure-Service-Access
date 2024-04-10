"""

Data for geospatial optimization, J. Gromicho 2023.

This module includes functions to manipulate geospatial data to make it
suitable for optimization.

"""

import numpy as np
import pandas as pd


def all_in(list_of_lists: list[list]) -> np.ndarray:
    """
    Returns a numpy array of unique elements from a list of lists

    Parameters:
        list_of_lists (list[list]): A list of lists

    Returns:
        numpy.ndarray: A numpy array of unique elements
    """
    return np.unique(np.concatenate(list_of_lists))


def ExtractOptimizationDataFromTravelDistanceMatrix(
    travel_dist: pd.DataFrame,
    dist_threshold: float,
    col_distance: str = 'distance',
    col_facility_id: str = 'fac_id',
    col_pop_id: str = 'pop_id'
) -> dict[any, any]:
    """
    Extracts optimization data from a travel distance matrix.

    This function filters the input travel distance matrix based on a distance
    threshold and returns a dictionary where facility IDs are keys and the
    lists of associated population IDs within the specified distance threshold
    are the values.

    Args:
        travel_dist (pd.DataFrame): The travel distance matrix containing
            distance information.
        dist_threshold (float): The maximum distance allowed for population
            association.
        col_distance (str, optional): The column name for distance values.
            Defaults to 'distance'.
        col_facility_id (str, optional): The column name for facility IDs.
            Defaults to 'fac_id'.
        col_pop_id (str, optional): The column name for population IDs.
            Defaults to 'pop_id'.

    Returns:
        Dict[Any, Any]: A dictionary where facility IDs are keys and lists of
        associated population IDs within the specified distance threshold are
        values.
    """
    filtered_data = travel_dist[travel_dist[col_distance] <= dist_threshold]
    pivot_table = filtered_data.pivot_table(
        index=col_facility_id,
        values=col_pop_id,
        aggfunc=lambda x: list(set(x))
    )
    result_dict = pivot_table[col_pop_id].to_dict()
    return result_dict


def CreateIndexMapping(
    all_facs: dict,
    household: list,
    covered: set = set()
) -> tuple[np.array, np.array, dict, dict]:
    """
    CreateIndexMapping creates a mapping between the indices of the households
    and the facilities.

    Parameters:
    all_facs (dict): A dictionary of facilities and their associated indices.
    household (list): A list of households.
    covered (set): A set of indices that are already covered.

    Returns:
    I (np.array): An array of indices of households.
    J (np.array): An array of indices of facilities.
    IJ (dict): A dictionary of households to the facilities that they reach.
    JI (dict): A dictionary of facilities to the households in catchment area.
    """
    not_covered = np.setdiff1d(np.arange(len(household)),
                               covered,
                               assume_unique=True)
    JI = {j: np.setdiff1d(i, covered, assume_unique=True)
          for j, i in all_facs.items()}
    JI = {j: i for j, i in JI.items() if len(i)}
    IJ = {i: [] for i in not_covered}
    for j, I in JI.items():
        for i in I:
            if i in IJ.keys():
                IJ[i].append(j)
    IJ = {i: np.unique(j) for i, j in IJ.items() if len(j)}
    I = np.unique(list(IJ.keys()))  # noqa: E741
    J = np.unique(np.concatenate(list(IJ.values())))
    return I, J, IJ, JI


def CheckIndexMapping(
    I: list,  # noqa: E741
    J: list,
    IJ: dict,
    JI: dict,
    w: list
) -> bool:
    """
    Checks if the index mapping is valid.

    Parameters:
    I (list): List of households
    J (list): List of potential facilities
    IJ (dict): Mapping from households to potential facilities
    JI (dict): Mapping from potential facilities to households
    w (list): List of weights

    Returns:
    bool: True if the index mapping is valid, fires an assertion otherwise
    """
    assert set(I) == set(range(len(w))), 'unknown household'
    assert set(I).issuperset(set(IJ.keys())), 'unknown household'
    assert set(J).issuperset(set(JI.keys())), 'unknown facility'
    assert set(J).issuperset(all_in(list(IJ.values()))), 'unknown facility'
    assert set(I).issuperset(all_in(list(JI.values()))), 'unknown household'
    assert set(all_in(list(JI.values()))).issubset(set(range(len(w)))),\
        'unknown household'
    return True
