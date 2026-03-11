def get_length_edge(x):
    import geopy.distance
    lon_x = float(x['from_x'])
    lat_x = float(x['from_y'])    
    lon_y = float(x['to_x'])
    lat_y = float(x['to_y'])
    dist = geopy.distance.geodesic((lat_x,lon_x),(lat_y,lon_y))
    return((dist.meters)/1000)

def get_nodes_and_edges(json_file,rounding=5):
    import pandas as pd
    import geopandas as gpd
    import geopy.distance
    import pandana
    """Use geopandas to read line shapefile and compile all paths and nodes in a line file based on a rounding tolerance.
    shp_file:path to polyline file with end to end connectivity
    rounding: tolerance parameter for coordinate precision"""
    edges = gpd.read_file(json_file)#,driver='GeoJSON')
    edges["from_x"]=edges["geometry"].apply(lambda x:round(x.coords[0][0],rounding))
    edges["from_y"]=edges["geometry"].apply(lambda x:round(x.coords[0][1],rounding))
    edges["to_x"]=edges["geometry"].apply(lambda x:round(x.coords[-1][0],rounding))
    edges["to_y"]=edges["geometry"].apply(lambda x:round(x.coords[-1][1],rounding))
    nodes_from = edges[["from_x","from_y"]].rename(index=str,columns={"from_x":"x","from_y":"y"})
    nodes_to = edges[["to_x","to_y"]].rename(index=str,columns={"to_x":"x","to_y":"y"})
    nodes = pd.concat([nodes_from,nodes_to],axis=0)
    nodes["xy"] = list(zip(nodes["x"], nodes["y"]))
    nodes = pd.DataFrame(nodes["xy"].unique(),columns=["xy"])
    nodes["x"] = nodes["xy"].apply(lambda x: x[0])
    nodes["y"] = nodes["xy"].apply(lambda x: x[1])
    nodes = nodes[["x","y"]].copy()
    nodes = nodes.reset_index()
    nodes.columns = ['nodeID','lon','lat']
    edges_attr = pd.merge(edges,nodes,left_on=['from_x','from_y'], right_on=['lon','lat'])
    edges_attr = pd.merge(edges_attr,nodes,left_on=['to_x','to_y'], right_on=['lon','lat'])
    edges_attr.rename(columns= {'nodeID_x':'node_start','nodeID_y':'node_end'},inplace=True)
    edges_attr['len_km'] = edges_attr[['from_x','from_y','to_x','to_y']].apply(get_length_edge,axis=1)

    # Road Network Data in Nodes and Edges nodes as a Pandana Network
    network = pandana.Network(nodes['lon'], nodes['lat'], 
                              edges_attr['node_start'], edges_attr['node_end'], edges_attr[['len_km']],twoway=True)
    return nodes, edges_attr, network