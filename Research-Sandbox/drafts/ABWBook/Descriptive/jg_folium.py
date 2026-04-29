import folium

def GetPoints( data ):
    return data[['FINISH_LATITUDE','FINISH_LONGITUDE']].dropna().drop_duplicates().copy().rename(columns={'FINISH_LATITUDE':'lat','FINISH_LONGITUDE':'lon'})

def AdjustBoundsForPoints( folium_map, points ):
    a,b=points.min()
    c,d=points.max()
    folium_map.fit_bounds( ((a,b), (c,d)) )
    return folium_map

def MapForPoints( points, zoom_start=8 ):
    a,b=points.min()
    c,d=points.max()
    start_coords = ((c-a)/2,(d-b)/2)
    folium_map = folium.Map(location=start_coords, zoom_start=zoom_start)
    folium_map.fit_bounds( ((a,b), (c,d)) )
    return folium_map

default_marker = lambda lat,lon,**kwargs : folium.CircleMarker((lat,lon),**kwargs)
default_describer = lambda lat,lon : str((lat,lon))

def MarkPointsOnMap( points, marker=default_marker, describe=default_describer, **kwargs ):
    folium_map = MapForPoints( points )
    for lat,lon in points.values:
        marker(lat,lon,popup=describe(lat,lon),**kwargs).add_to(folium_map)
    return folium_map

def MarkRouteThroughPointsOnMap( points, folium_map=None, describe=default_describer, color='blue', weight=3, opacity=1, icon_shape='marker', background_color='red',border_width=1,inner_icon_style='font-size:10px', **kwargs ):
    if not folium_map:
        folium_map = MapForPoints( points )
    stops = [ (lat,lon) for lat,lon in points.drop_duplicates().dropna().values ]
    for i,(lat,lon) in enumerate(stops):  
        folium.Marker((lat,lon), icon=folium.plugins.BeautifyIcon(number=i,icon_shape=icon_shape,background_color=background_color,border_width=border_width,inner_icon_style=inner_icon_style,**kwargs),popup=describe(lat,lon)).add_to(folium_map)
    folium.PolyLine(stops, color=color, weight=weight, opacity=opacity, **kwargs).add_to(folium_map)
    return folium_map

def FoliumToPng( folium_map, file_name, rendering_seconds=5 ):
    import io
    from PIL import Image
    Image.open(io.BytesIO(folium_map._to_png(rendering_seconds))).save(file_name+'.png')