"""Irradiance Class"""
from .dependencies import *

class RelativeIrradiance:
    """
    Class that holds the output relative irradiance for the terrain surface, and if the 
    canopy inclusive irradiance model was run, the canopy irradiance as well.

    Attributes
    ----------
    leaf_irradiance: np.ndarray
        Holds the coordinates and relative irradiance for the canopy. 
    terrain_irradiance: np.ndarray
        Holds the terrain coordinates and their relative irradiance. Is a numpy array
        with shape (width, height), where each (x, y) coordiante holds the irradiance
        value for that point on the terrain. 
    
    """
    leaf_irradiance: np.ndarray
    terrain_irradiance: np.ndarray

    def __init__(self, terrain_irradiance: np.ndarray, leaf_irradiance: np.ndarray = None):
        self.leaf_irradiance = leaf_irradiance
        self.terrain_irradiance = terrain_irradiance


