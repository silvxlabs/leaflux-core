"""Environment Classes"""
from .dependencies import *

class LeafArea:
    """
    Class that holds the formatted leaf area array, used as input.

    Attributes
    ----------
    leaf_area: np.ndarray
        Represents a point cloud of the canopy leaf area. A numpy array with shape
        (N, 4) where each row contains (x, y, z, leaf area) and y runs south to north.

    width: int
        Width of the area being described. Should be the same as any Terrain width being used in 
        conjunction with this LeafArea. 
    
    height: int
        Height of the area being described. Should be the same as any Terrain height being used in 
        conjunction with this LeafArea.
    """
    leaf_area: np.ndarray
    width: int
    height: int

    # Onramp constructors
    def __init__(self, leaf_area_point_cloud: np.ndarray, width: int, height: int):
        """
        Initializes LeafArea object from a given point cloud.
        
        Parameters
        ----------
        leaf_area_point_cloud: np.ndarray
            Expected as a sparse numpy array with shape (N, 4) where each row is (x, y, z, leaf area)
            and y runs south to north.
        
        width: int
            Width of the area being described. Should be the same as any Terrain width being used in 
            conjunction with this LeafArea. 
        
        height: int
            Height of the area being described. Should be the same as any Terrain height being used in 
            conjunction with this LeafArea.

        Returns
        --------
        Instance of LeafArea class object.
        """
        # From point cloud
        self.leaf_area = leaf_area_point_cloud
        self.width = width
        self.height = height

    @classmethod
    def from_uniformgrid(cls, leaf_area_uniform_grid: np.ndarray):
        """
        Initializes LeafArea object from a given uniform grid.

        Parameters
        -----------
        leaf_area_uniform_grid: np.ndarray
            Uniform grid representing leaf area coordinates and their leaf area. Assumed to 
            be dense. Expected as a 3D numpy array where each (y, x, z) coordinate represents
            a leaf area value, and where y runs north to south.
        
        Returns
        --------
        LeafArea
            Instance of LeafArea class. 

        """
        s_la = sparse.COO(leaf_area_uniform_grid)
        # Stacking like (x, y, z, area)
        # Flipping y coordinates to go south->north
        leaf_area = np.column_stack((s_la.coords[1], (leaf_area_uniform_grid.shape[0] - s_la.coords[0] - 1), s_la.coords[2], s_la.data))
        leaf_area = leaf_area.astype(np.float32)

        return cls(leaf_area, leaf_area_uniform_grid.shape[1], leaf_area_uniform_grid.shape[0])

class Terrain:
    """
    Class that holds information about the terrain input. 

    Attributes
    ----------
    terrain: np.ndarray
        Represents the terrain. A numpy array with shape (N, 3), where each row is 
        (x, y, z) and y runs south to north.
    width: int
        Width of terrain, from shape of input.
    height: int
        Height of terrain, from shape of input. 
    terrain_grid: np.ndarray
        The original 2.5D provided grid, stored for internal use in later calculations.
    """
    terrain: np.ndarray
    width: int
    height: int
    terrain_grid: np.ndarray

    # 2.5D numpy array
    def __init__(self, terrain: np.ndarray):
        """
        Constructor for Terrain class object.

        Parameters
        ----------
        terrain: np.ndarray
            Assumed to be a 2.5D grid representing the terrain, expected as a 2D numpy array
            with shape (height, width) where each (y, x) coordinate value represents a z value 
            and where y runs north to south.

        Returns
        --------
        Terrain
            Instance of Terrain class.
        """
        self.width = terrain.shape[1] # x
        self.height = terrain.shape[0] # y

        self.terrain_grid = terrain

        terr_x, terr_y = np.meshgrid(np.arange(self.width), np.arange(self.height), indexing='xy')
        terr_y = self.height - terr_y - 1 # Flipping to be south->north
        self.terrain = np.column_stack((terr_x.ravel(), terr_y.ravel(), terrain.flatten())) # Rows of (x, y, z)

class Sensor:
    """
    Class that holds information about sensors in the environment. 

    Attributes
    -
    sensor: np.ndarray
        Holds the (x, y, z) coordinates of the sensor, as well as the pitch and azimuth (if provided). Y coordinates run south to north
        and angles are in radians. Is a (5,) shaped array where the single row is (x, y, z, pitch, azimuth).

    """
    sensor: np.array

    def __init__(self, x: int, y: int, z: int, pitch: float = None, azimuth: float = None):
        """
        Constructor for Sensor object. If pitch **and** azimuth are provided, the returned 
        irradiance will be corrected for the tilt of the sensor relative to the SolarPosition 
        provided to `attenuate_all()`. If either pitch or azimuth are not provided, returned
        irradiance will not be corrected.

        Parameters
        -
        x: int
            x coordinate of sensor. 

        y: int
            y coordinate of sensor. Y coordinates are expected to run south to north.

        z: int
            z coordinate of sensor.
        
        pitch: float
            Pitch of sensor in radians. Default is None.
        
        azimuth: float
            Azimuth angle of sensor in radians. Default is None.

        Returns
        -
        Instance of Sensor class.
        """
        self.sensor = np.array([x, y, z, pitch, azimuth], dtype=np.float32)

class LeafAngle:
    """
    Class that holds the formatted leaf angle array, used as input.

    Attributes
    ----------
    leaf_angle: np.ndarray
        Represents a point cloud of canopy mean leaf angle values (radians). A numpy array with shape
        (N, 4) where each row contains (x, y, z, leaf_angle) and y runs south to north.

    width: int
        Width of the area being described. Should be the same as any Terrain and LeafArea width being used in 
        conjunction with this LeafAngle. 
    
    height: int
        Height of the area being described. Should be the same as any Terrain and LeafArea height being used in 
        conjunction with this LeafAngle.
    """
    leaf_angle: np.ndarray
    width: int
    height: int

    # Onramp constructors
    def __init__(self, leaf_angle_point_cloud: np.ndarray, width: int, height: int):
        """
        Initializes LeafAngle object from a given point cloud.
        
        Parameters
        ----------
        leaf_angle_point_cloud: np.ndarray
            Expected as a sparse numpy array with shape (N, 4) where each row is (x, y, z, leaf angle)
            and y runs south to north.
        
        width: int
            Width of the area being described. Should be the same as any Terrain width being used in 
            conjunction with this LeafAngle. 
        
        height: int
            Height of the area being described. Should be the same as any Terrain height being used in 
            conjunction with this LeafAngle.

        Returns
        --------
        Instance of LeafAngle class object.
        """
        # From point cloud
        self.leaf_angle = leaf_angle_point_cloud
        self.width = width
        self.height = height

    @classmethod
    def from_uniformgrid(cls, leaf_angle_uniform_grid: np.ndarray):
        """
        Initializes LeafAngle object from a given uniform grid.

        Parameters
        -----------
        leaf_angle_uniform_grid: np.ndarray
            Uniform grid representing leaf angle coordinates and their leaf angle. Assumed to 
            be dense. Expected as a 3D numpy array where each (y, x, z) coordinate represents
            a leaf angle value, and where y runs north to south.
        
        Returns
        --------
        LeafAngle
            Instance of LeafAngle class. 

        """
        s_la = sparse.COO(leaf_angle_uniform_grid)
        # Stacking like (x, y, z, area)
        # Flipping y coordinates to go south->north
        leaf_area = np.column_stack((s_la.coords[1], (leaf_angle_uniform_grid.shape[0] - s_la.coords[0] - 1), s_la.coords[2], s_la.data))
        leaf_area = leaf_area.astype(np.float32)

        return cls(leaf_area, leaf_angle_uniform_grid.shape[1], leaf_angle_uniform_grid.shape[0])
    
class Environment:
    """
    Class that holds the leaf area and terrain arrays. 

    Attributes
    ----------
    leaf_area: LeafArea
        Object that holds the coordinates and leaf area for the canopy.
    terrain: Terrain
        Object that holds the coordinates of the terrain.
    sensors: np.ndarray
        A column stack of Sensors present in the environment. Each row of the array is like (x, y, z, pitch, azimuth)
    """
    leaf_area: LeafArea
    terrain: Terrain
    leaf_angle: LeafAngle
    sensors: np.ndarray

    def __init__(self, leaf_area: LeafArea, terrain: Terrain = None, leaf_angle: LeafAngle = None, sensors: list[Sensor] = None):
        """
        Constructor for Environment object.

        Parameters
        ----------
        leaf_area: LeafArea
            A LeafArea class object.
        terrain: Terrain
            (optional) A Terrain class object. Default is None.
        leaf_angle: LeafAngle
            (optional) A LeafAngle class object. Default is None.
        sensors: list[Sensor]
            (optional) A list of Sensor objects. Default is None.

        Returns
        -------
        Environment
            Instance of Environment class.
        """
        if not isinstance(leaf_area, LeafArea):
            raise TypeError(f"Expected an object of type 'LeafArea', but got {type(leaf_area)}.")
    
        # If there is no terrain provided, set terrain to None
        if terrain is None:
            self.leaf_area = leaf_area
            self.terrain = None
        else:
            if not isinstance(terrain, Terrain):
                    raise TypeError(f"Expected an object of type 'Terrain', but got {type(terrain)}.")
            
            # Leaf area dimensions must match terrain dimensions
            if leaf_area.width == terrain.width and leaf_area.height == terrain.height:
                self.leaf_area = leaf_area
                self.terrain = terrain
            else:
                raise ValueError(f"Terrain dimensions must match leaf area grid dimensions. Leaf area is ({leaf_area.width}, {leaf_area.height}) and terrain is ({terrain.width}, {terrain.height})")
        
        if leaf_angle is None:
            self.leaf_angle = None
        else:
            if not isinstance(leaf_angle, LeafAngle):
                raise TypeError(f"Expected an object of type 'LeafAngle', but got (type(leaf_angle).")
            
            if not np.array_equal(leaf_area.leaf_area[:, :3], leaf_angle.leaf_angle[:, :3]):
                raise ValueError(f"Leaf angle point coordinates do not match leaf area point coordinates. Coordinates must match. Leaf area grid shape: {leaf_area.leaf_area.shape} Leaf angle grid shape: {leaf_angle.leaf_angle.shape}")
            
            # Dimensions must match leaf_area
            if leaf_area.width == leaf_angle.width and leaf_area.height == leaf_angle.height:
                self.leaf_angle = leaf_angle
            else:
                raise ValueError(f"Leaf angle dimensions must match leaf area grid dimensions. Leaf area is ({leaf_area.width}, {leaf_area.height}) and leaf angle is ({leaf_angle.width}, {leaf_angle.height})")
            
        if sensors != None:
            # Check types of all sensors
            for i, sensor in enumerate(sensors):
                if not isinstance(sensor, Sensor):
                    raise TypeError(f"Expected a list of objects of type 'Sensor', but got {type(sensor)} at index {i}.")
            self.sensors = np.vstack([sensor.sensor for sensor in sensors]) # Stack sensors
        else:
            self.sensors = None

