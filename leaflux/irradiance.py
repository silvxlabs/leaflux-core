"""Irradiance Class"""
from .dependencies import *

class RelativeIrradiance:
    """
    Class that holds the output relative irradiance for the terrain surface, and if
    returned from attenuate_all, the canopy irradiance as well.

    Attributes
    ----------
    leaf_irradiance: np.ndarray
        Holds the coordinates and relative irradiance for the canopy. Is a numpy array
        with shape (N, 4) where each row is (x, y, z, irradiance), and y runs south 
        to north.
    terrain_irradiance: np.ndarray
        Holds the coordinates and their relative irradiance for the surface/topography. Is a numpy array
        with shape (height, width) where each (y, x) coordinate holds the irradiance
        value for that point on the terrain, and y runs north to south.
    
    """
    leaf_irradiance: np.ndarray
    terrain_irradiance: np.ndarray

    def __init__(self, terrain_irradiance: np.ndarray, leaf_irradiance: np.ndarray = None):
        self.leaf_irradiance = leaf_irradiance
        self.terrain_irradiance = terrain_irradiance


