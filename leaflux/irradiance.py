"""Irradiance Class"""
from .dependencies import *
from leaflux.environment import Sensor
from leaflux.solar import SolarPosition

class Irradiance:
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
    sensor_irradiance: np.ndarray
        Holds the coordinates and relative irradiance for each sensor. Is an (N, 4) stack where each 
        row is (x, y, z, irradiance), and y runs south to north. Irradiance values for a particular 
        provided Sensor object can be retrieved with `get_sensor_irradiance()`.
    
    """
    canopy_irradiance: np.ndarray
    terrain_irradiance: np.ndarray
    sensor_irradiance: np.ndarray

    solar_position: SolarPosition

    def __init__(self, solar_position: SolarPosition, terrain_irradiance: np.ndarray = None, canopy_irradiance: np.ndarray = None, sensor_irradiance: np.ndarray = None):
        self.solar_position = solar_position
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
        mask = (self.sensor_irradiance[:, 0] == sensor.sensor[0]) & (self.sensor_irradiance[:, 1] == sensor.sensor[1]) & (self.sensor_irradiance[:, 2] == sensor.sensor[2])
        return self.sensor_irradiance[mask, 3].item()
    
    def to_srad(self, dni: float, dhi: float, in_place: bool = False):
        """
        Converts the given Irradiance object from relative irradiance to total short wave radiation, GHI (Global Horizontal Irradiance) in Watts/m^2.

        Parameters
        -
        dni: float
            DNI (Direct Normal Irradiance) for the datetime this `Irradiance` result is from in Watts/m^2.

        dhi: float
            DHI (Diffuse Horizontal Irradiance) for the datetime this `Irradiance` result is from in Watts/m^2.

        in_place: bool
            Bool indicating whether the operation should ocurr in place, modidying the given `Irradiance` object, or if it should 
            return a new result. Default is False. 

        Returns
        -
        Irradiance or None
            If `in_place` was set to False, returns an Irradiance object where all irradiance values have been converted into GHI in Watts/m^2. If
            `in_place` was set to True, returns None. 

        """
        if in_place:
            if self.canopy_irradiance is not None:
                self.canopy_irradiance[:, 3] = dhi + (self.canopy_irradiance[:, 3] * dni * np.cos(self.solar_position.zenith))
            if self.sensor_irradiance is not None:
                self.sensor_irradiance[:, 3] = dhi + (self.sensor_irradiance[:, 3] * dni * np.cos(self.solar_position.zenith))
            if self.terrain_irradiance is not None:
                self.terrain_irradiance = dhi + (self.terrain_irradiance * dni * np.cos(self.solar_position.zenith))
            return None
        else:
            if self.canopy_irradiance is not None:
                ci = np.copy(self.canopy_irradiance)
                ci[:, 3] = dhi + (ci[:, 3] * dni * np.cos(self.solar_position.zenith))
            else:
                ci = None
            if self.sensor_irradiance is not None:
                si = np.copy(self.sensor_irradiance)
                si[:, 3] = dhi + (si[:, 3] * dni * np.cos(self.solar_position.zenith))
            else:
                si = None
            if self.terrain_irradiance is not None:
                ti = dhi + (self.terrain_irradiance * dni * np.cos(self.solar_position.zenith))
            else:
                ti = None
            return Irradiance(
                solar_position=self.solar_position,
                terrain_irradiance=ti,
                canopy_irradiance=ci,
                sensor_irradiance=si
                )
    
    def to_par(self, dni: float, dhi: float, par_ratio: float = 0.5, in_place: bool = False):
        """
        Converts the given `Irradiance` object from relative to PAR (Photosynthetically Active Radiation) in micromoles/m^2s. 

        Parameters
        -
        dni: float
            DNI (Direct Normal Irradiance) for the datetime this `Irradiance` result is from in Watts/m^2. 

        dhi: float
            DHI (Diffuse Horizontal Irradiance) for the datetime this `Irradiance` result is from in Watts/m^2.

        par_ratio: float
            Ratio used to convert GHI to PAR. Default is 0.5. 

        in_place: bool
            Bool indicating whether the operation should ocurr in place, modidying the given `Irradiance` object, or if it should 
            return a new result. Default is False. 

        Returns
        -
        Irradiance or None
            If `in_place` was set to False, returns an Irradiance object where all irradiance values have been converted into PAR in micromoles/m^2s. If
            `in_place` was set to True, returns None.
        """
        cf = 4.57 # Conversion factor from W/m^2 -> micromoles/m^2s
        if in_place:
            if self.canopy_irradiance is not None:
                self.canopy_irradiance[:, 3] = (dhi + (self.canopy_irradiance[:, 3] * dni * np.cos(self.solar_position.zenith)))*par_ratio*cf
            if self.sensor_irradiance is not None:
                self.sensor_irradiance[:, 3] = (dhi + (self.sensor_irradiance[:, 3] * dni * np.cos(self.solar_position.zenith)))*par_ratio*cf
            if self.terrain_irradiance is not None:
                self.terrain_irradiance = (dhi + (self.terrain_irradiance * dni * np.cos(self.solar_position.zenith)))*par_ratio*cf
            return None
        else:
            if self.canopy_irradiance is not None:
                ci = np.copy(self.canopy_irradiance)
                ci[:, 3] = (dhi + (ci[:, 3] * dni * np.cos(self.solar_position.zenith)))*par_ratio*cf
            else:
                ci = None
            if self.sensor_irradiance is not None:
                si = np.copy(self.sensor_irradiance)
                si[:, 3] = (dhi + (si[:, 3] * dni * np.cos(self.solar_position.zenith)))*par_ratio*cf
            else:
                si = None
            if self.terrain_irradiance is not None:
                ti = (dhi + (self.terrain_irradiance * dni * np.cos(self.solar_position.zenith)))*par_ratio*cf
            else:
                ti = None
            return Irradiance(
                solar_position=self.solar_position,
                terrain_irradiance=ti,
                canopy_irradiance=ci,
                sensor_irradiance=si
            )


