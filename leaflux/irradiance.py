"""Irradiance Class"""
from .dependencies import *
from leaflux.environment import Sensor

class RelativeIrradiance:
    """
    Class that holds the output relative irradiance for the terrain surface, and if
    returned from attenuate_all, the canopy irradiance as well.

    Attributes
    ----------
    canopy_irradiance: np.ndarray
        Holds the coordinates and relative irradiance for the canopy. Is a numpy array
        with shape (N, 4) where each row is (x, y, z, irradiance), and y runs south 
        to north.
    terrain_irradiance: np.ndarray
        Holds the coordinates and their relative irradiance for the surface/topography. Is a numpy array
        with shape (height, width) where each (y, x) coordinate holds the irradiance
        value for that point on the terrain, and y runs north to south.
    
    """
    canopy_irradiance: np.ndarray
    terrain_irradiance: np.ndarray
    sensor_irradiance: np.ndarray

    def __init__(self, terrain_irradiance: np.ndarray = None, canopy_irradiance: np.ndarray = None, sensor_irradiance: np.ndarray = None):
        self.canopy_irradiance = canopy_irradiance
        self.terrain_irradiance = terrain_irradiance
        self.sensor_irradiance = sensor_irradiance

    def get_sensor_irradiance(self, sensor: Sensor) -> float:
        """
        Returns the irradiance of a given Sensor.

        Parameters
        -
        sensor: Sensor
            Object of type Sensor.

        
        Returns
        -
        float
            Contains the irradiance of the given Sensor.
        """
        mask = (self.sensor_irradiance[:, 0] == sensor.position[0]) & (self.sensor_irradiance[:, 1] == sensor.position[1]) & (self.sensor_irradiance[:, 2] == sensor.position[2])
        return self.sensor_irradiance[mask, 3]



