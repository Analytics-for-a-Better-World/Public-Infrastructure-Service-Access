"""Constants used throughout the PISA package.

This module defines constants that are used across the PISA (Public Infrastructure Service Access) package to ensure 
consistency and avoid duplication. These constants include valid modes of transportation, distance types, and 
OpenStreetMap tags for facilities.

Constants
---------
VALID_MODES_OF_TRANSPORT : set
    Set of valid transportation modes that can be used in isopolygon calculations
    
VALID_DISTANCE_TYPES : set
    Set of valid distance types (e.g., length or travel time) for isopolygon calculations
    
OSM_TAGS : dict
    Dictionary of OpenStreetMap tags used to identify facilities of interest
"""

VALID_MODES_OF_TRANSPORT = {"driving", "walking", "cycling"}

VALID_DISTANCE_TYPES = {"length", "travel_time"}

OSM_TAGS = {"amenity": "hospital", "building": "hospital"}
