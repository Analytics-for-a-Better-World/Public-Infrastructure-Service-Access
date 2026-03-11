# -*- coding: utf-8 -*-
"""
Created on Wed Sep 15 10:35:12 2021

@author: fleur
"""
def haversine_vectorize(lon1, lat1, lon2, lat2):
    
    import numpy as np
    
    lon1, lat1, lon2, lat2 = map(np.radians, [lon1, lat1, lon2, lat2])
    newlon = lon2 - lon1
    newlat = lat2 - lat1
    haver_formula = np.sin(newlat/2.0)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(newlon/2.0)**2
    dist = 2 * np.arcsin(np.sqrt(haver_formula ))
    km = 6367 * dist #6367 for distance in KM for miles use 3958
    return round(km,2)