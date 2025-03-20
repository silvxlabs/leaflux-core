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
    """
    terrain: np.ndarray
    width: int
    height: int

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

        terr_x, terr_y = np.meshgrid(np.arange(self.width), np.arange(self.height), indexing='xy')
        terr_y = self.height - terr_y - 1 # Flipping to be south->north
        self.terrain = np.column_stack((terr_x.ravel(), terr_y.ravel(), terrain.flatten())) # Rows of (x, y, z)

class Environment:
    """
    Class that holds the leaf area and terrain arrays. 

    Attributes
    ----------
    leaf_area: LeafArea
        Object that holds the coordinates and leaf area for the canopy.
    terrain: Terrain
        Object that holds the coordinates of the terrain.
    """
    leaf_area: LeafArea
    terrain: Terrain

    def __init__(self, leaf_area: LeafArea, terrain: Terrain = None):
        """
        Constructor for Environment object.

        Parameters
        ----------
        leaf_area: LeafArea
            A LeafArea class object.
        terrain: Terrain
            (optional) A Terrain class object. Default is None.

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
                raise ValueError(f"Leaf area grid dimensions must match terrain dimensions. Leaf area is ({leaf_area.width}, {leaf_area.height}) and terrain is ({terrain.width}, {terrain.height})")

