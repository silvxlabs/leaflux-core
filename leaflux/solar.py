"""Solar Position Class"""
from .dependencies import *

# Single solar position
class SolarPosition:
    """
    Class that holds information about the position of the sun for a given date, 
    time, and latitude.

    Attributes
    ----------
    timestamp: datetime
        The date and time. By default is in UTC.
    
    latitude: float
        Latitude for where the solar position will be found. Can be any float
        between -90 and 90.

    light_vector: np.array
        Holds the resulting light vector as a numpy array of three floats.
    
    """
    timestamp: datetime
    latitude: float
    light_vector: np.array

    def __init__(self, datetime: datetime, latitude: float):
        """
        Constructor for SolarPosition class.

        Parameters
        ----------
        datetime: datetime
            A Python datetime object representing the date and time in UTC. Year is 
            required but generally does not have much effect on outcomes.

        latitude: float
            Latitide at which to run light model. Must be between -90 and 90.

        Returns
        --------
        SolarPosition
            Instance of SolarPosition class.
        """
        if latitude > 90. or latitude < -90.:
            raise ValueError("Latitude must be between -90 and 90.")
        
        self.timestamp = datetime
        self.latitude = latitude

        solar_position = pvlib.solarposition.get_solarposition(datetime, latitude=latitude, longitude=0.0)

        # Check for sun below horizon
        if solar_position['elevation'].iloc[0] < 0:
            raise ValueError("Datetime and latitute provided result in solar elevation below 0.")
        
        # Convert to radians
        solar_position = solar_position.apply(np.radians)

        # Calculate solar vector
        i = -(np.cos(solar_position['elevation'].iloc[0]) * np.sin(solar_position['azimuth'].iloc[0]))
        j = -(np.cos(solar_position['elevation'].iloc[0]) * np.cos(solar_position['azimuth'].iloc[0]))
        k = -(np.sin(solar_position['elevation'].iloc[0]))

        # Normalize
        vec = np.array([i, j, k])
        norm = np.linalg.norm(vec)
        self.light_vector = np.array([i/norm, j/norm, k/norm])