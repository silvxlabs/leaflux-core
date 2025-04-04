"""Solar Position Class"""
from .dependencies import *

# Single solar position
class SolarPosition:
    """
    Class that holds information about the position of the sun for a given date, 
    time, latitude, and longitude.

    Attributes
    ----------
    timestamp: datetime
        The date and time. Must be in UTC.
    
    latitude: float
        Latitide in degrees at which to run light model. Positive south of equator, negative to south. Must be between -90 and 90.
    
    longitude: float
        Longitude in degrees at which to run the light model. Positive south of prime meridian, negative to the west. Must be between -180
        and 180.

    light_vector: np.array
        Holds the resulting light vector as a numpy array of three floats.

    zenith: float
        The zenith angle in radians.

    azimuth: float
        The azimuth angle in radians.
    
    """
    timestamp: datetime
    latitude: float
    longitude: float
    light_vector: np.array
    zenith: float
    azimuth: float

    def __init__(self, datetime: datetime, latitude: float, longitude: float):
        """
        Constructor for SolarPosition class.

        Parameters
        ----------
        datetime: datetime
            A Python datetime object representing the date and time in UTC.

        latitude: float
            Latitide in degrees at which to run light model. Positive south of equator, negative to south. Must be between -90 and 90.

        longitude: float
            Longitude in degrees at which to run the light model. Positive south of prime meridian, negative to the west. Must be between -180
            and 180.

        Returns
        --------
        SolarPosition
            Instance of SolarPosition class.
        """
        if latitude > 90. or latitude < -90.:
            raise ValueError("Latitude must be between -90 and 90.")
        
        if longitude > 180. or longitude < -180.:
            raise ValueError("Longitude must be between -180 and 180.")
        
        self.timestamp = datetime
        self.latitude = latitude
        self.longitude = longitude

        solar_position = pvlib.solarposition.get_solarposition(datetime, latitude=latitude, longitude=longitude)

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

        # Zenith and azimuth in radians
        self.zenith = solar_position['zenith'].iloc[0]
        self.azimuth = solar_position['azimuth'].iloc[0]
        
