def CurrentHospitals(current_hospitals, network, nodes):
    from .haversine_vectorize import haversine_vectorize
    import pandas as pd
    
    current_hospitals.columns = ['Hosp_ID','Longitude','Latitude','L_NAME']
    current_hospitals_ID = current_hospitals['Hosp_ID'].unique()

    current_hospitals['nearest_node'] = network.get_node_ids(current_hospitals['Longitude'], current_hospitals['Latitude'], mapping_distance=None)
    current_hospitals = pd.merge(current_hospitals,nodes,right_on='nodeID',left_on='nearest_node')
    current_hospitals['hosp_dist_road_estrada'] = haversine_vectorize(current_hospitals['Longitude'],current_hospitals['Latitude'],current_hospitals['lon'],current_hospitals['lat'])
    return current_hospitals_ID, current_hospitals

def NewHospitalsCSV(current_hospitals, new_hospitals, network, nodes ):
    from .haversine_vectorize import haversine_vectorize
    import numpy as np
    import pandas as pd

    
    new_hospitals=new_hospitals[['id','xcoord','ycoord']]
    new_hospitals.columns = ['Old_Cluster_ID', 'Longitude','Latitude']
    new_hospitals['Cluster_ID'] = np.arange(len(new_hospitals)) + len(current_hospitals)
    new_hospitals_ID = new_hospitals['Cluster_ID'].unique()

    new_hospitals['nearest_node'] = network.get_node_ids(new_hospitals['Longitude'], new_hospitals['Latitude'], mapping_distance=None)
    new_hospitals = pd.merge(new_hospitals,nodes,right_on='nodeID',left_on='nearest_node')
    new_hospitals['hosp_dist_road_estrada'] = haversine_vectorize(new_hospitals['Longitude'],new_hospitals['Latitude'],new_hospitals['lon'],new_hospitals['lat'])
    return (new_hospitals_ID, new_hospitals)

def NewHospitals(current_hospitals, new_hospitals, network, nodes):
    import numpy as np
    import pandas as pd
    from .haversine_vectorize import haversine_vectorize

    new_hospitals['Longitude'] = new_hospitals.geometry.x
    new_hospitals['Latitude'] = new_hospitals.geometry.y
    new_hospitals = new_hospitals[['full_id','Longitude','Latitude']]

    new_hospitals.columns = ['Cluster_ID','Longitude','Latitude']

    # Rename cluster IDs from len(current_hospitals) up to total current+new hosps
    new_hospitals = new_hospitals.assign(
        Name=new_hospitals['Cluster_ID'].apply(lambda x: f'potential_{x}'),
        Cluster_ID=np.arange(len(new_hospitals)) + len(current_hospitals)
    )

    new_hospitals = new_hospitals[['Name','Cluster_ID','Longitude','Latitude']].drop_duplicates()
    new_hospitals_ID = new_hospitals['Cluster_ID'].unique()

    new_hospitals['nearest_node'] = network.get_node_ids(new_hospitals['Longitude'], new_hospitals['Latitude'], mapping_distance=None)
    new_hospitals = pd.merge(new_hospitals,nodes,right_on='nodeID',left_on='nearest_node')
    new_hospitals['hosp_dist_road_estrada'] = haversine_vectorize(new_hospitals['Longitude'],new_hospitals['Latitude'],new_hospitals['lon'],new_hospitals['lat'])
    return new_hospitals_ID, new_hospitals

def NewHospitalsGrid(current_hospitals, new_hospitals, network, nodes):
    import numpy as np
    import pandas as pd
    from .haversine_vectorize import haversine_vectorize

    new_hospitals['Longitude'] = new_hospitals.geometry.x
    new_hospitals['Latitude'] = new_hospitals.geometry.y
    new_hospitals = new_hospitals[['id','Longitude','Latitude']]

    new_hospitals.columns = ['Cluster_ID','Longitude','Latitude']

    # Rename cluster IDs from len(current_hospitals) up to total current+new hosps
    new_hospitals = new_hospitals.assign(
        Name=new_hospitals['Cluster_ID'].apply(lambda x: f'potential_{x}'),
        Cluster_ID=np.arange(len(new_hospitals)) + len(current_hospitals)
    )

    new_hospitals = new_hospitals[['Name','Cluster_ID','Longitude','Latitude']].drop_duplicates()
    new_hospitals_ID = new_hospitals['Cluster_ID'].unique()

    new_hospitals['nearest_node'] = network.get_node_ids(new_hospitals['Longitude'], new_hospitals['Latitude'], mapping_distance=None)
    new_hospitals = pd.merge(new_hospitals,nodes,right_on='nodeID',left_on='nearest_node')
    new_hospitals['hosp_dist_road_estrada'] = haversine_vectorize(new_hospitals['Longitude'],new_hospitals['Latitude'],new_hospitals['lon'],new_hospitals['lat'])
    return new_hospitals_ID, new_hospitals

def Population(digits_rounding, population, network, nodes):
    import pandas as pd
    from .haversine_vectorize import haversine_vectorize
    
    population.columns = ['ID','xcoord','ycoord']
    population['xcoord'] = population['xcoord'].round(digits_rounding)
    population['ycoord'] = population['ycoord'].round(digits_rounding)

    household_count = population.groupby(['xcoord','ycoord'])['ID'].nunique().reset_index()
    household_count.columns = ['xcoord','ycoord','household_count']

    population = population[['xcoord','ycoord']].drop_duplicates().reset_index().reset_index()
    del population['index']
    population.columns = ['ID','xcoord','ycoord']

    population['nearest_node'] = network.get_node_ids(population['xcoord'], population['ycoord'], mapping_distance=None)
    population = pd.merge(population,nodes,right_on='nodeID',left_on='nearest_node')
    population['pop_dist_road_estrada'] = haversine_vectorize(population['xcoord'],population['ycoord'],population['lon'],population['lat'])

    population = pd.merge(population,household_count,on=['xcoord','ycoord'])

    array_household = population.sort_values(by='ID')['household_count'].values
    
    return array_household, population

def PopulationFB(digits_rounding, population, network, nodes):
    import pandas as pd
    from .haversine_vectorize import haversine_vectorize
    
    # Round the coordinates to cluster the population
    population.columns = ['ID','xcoord','ycoord','household_count']
    population['xcoord'] = population['xcoord'].round(digits_rounding) # I think here should be the rounding?
    population['ycoord'] = population['ycoord'].round(digits_rounding) # I think here should be the rounding?
    
    # Might take a lot of time while all you actually do is rounding -- not if you add rounding in previous steps
    household_count = population.groupby(['xcoord','ycoord'])['household_count'].sum().round().reset_index()
    household_count.columns = ['xcoord','ycoord','household_count']

    population = population[['xcoord','ycoord']].drop_duplicates().reset_index().reset_index()
    del population['index']
    population.columns = ['ID','xcoord','ycoord']

    population['nearest_node'] = network.get_node_ids(population['xcoord'], population['ycoord'], mapping_distance=None)
    population = pd.merge(population,nodes,right_on='nodeID',left_on='nearest_node')
    population['pop_dist_road_estrada'] = haversine_vectorize(population['xcoord'],population['ycoord'],population['lon'],population['lat'])

    population = pd.merge(population,household_count,on=['xcoord','ycoord'])

    array_household = population.sort_values(by='ID')['household_count'].values
    
    return array_household, population



