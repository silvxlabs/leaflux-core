"""Environment Classes"""
from .dependencies import *

class LeafArea:
    """
    Class that holds the formatted leaf area array, used as input.

    Attributes
    ----------
    leaf_area: np.ndarray
        Represents a point cloud of the canopy leaf area. A numpy array with shape
        (N, 4), where each row contains (x, y, z, leaf area).
    """
    leaf_area: np.ndarray

    # Onramp constructors
    def __init__(self, leaf_area_point_cloud: np.ndarray):
        """
        Initializes LeafArea object from a given point cloud. Point cloud is assumed
        to be a sparse numpy NDArray with shape (N, 4), where each row is (x, y, z, leaf area).
        
        Parameters
        ----------
        leaf_area_point_cloud: np.ndarray
            Numpy NDArray with shape (N, 4) where each row is (x, y, z, leaf area)

        Returns
        --------
        Instance of LeafArea class object.
        """
        # From point cloud
        self.leaf_area = leaf_area_point_cloud

    @classmethod
    def from_uniformgrid(cls, leaf_area_uniform_grid: np.ndarray):
        """
        Initializes LeafArea object from a given uniform grid.

        Parameters
        -----------
        leaf_area_uniform_grid: np.ndarray
            Uniform grid representing leaf area coordinates and their leaf area. Assumed to 
            be dense. 

        
        Returns
        --------
        LeafArea
            Instance of LeafArea class. 

        """
        s_la = sparse.COO(leaf_area_uniform_grid)
        leaf_area = np.column_stack((s_la.coords[0], s_la.coords[1], s_la.coords[2], s_la.data))
        leaf_area = leaf_area.astype(np.float32)
        return cls(leaf_area)

class Terrain:
    """
    Class that holds information about the terrain input. 

    Attributes
    ----------
    terrain: np.ndarray
        Represents the terrain. A numpy array with shape (N, 3), where each row is 
        (x, y, z).
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
            Assumed to be a 2.5D grid representing the terrain, where at each (x, y) 
            position the value is the height.

        Returns
        --------
        Terrain
            Instance of Terrain class.
        """
        self.width = terrain.shape[0]
        self.height = terrain.shape[1]

        terr_x, terr_y = np.meshgrid(np.arange(terrain.shape[0]), np.arange(terrain.shape[1]))

        # Set terrain to stack of (x, y, z)
        self.terrain = np.column_stack((terr_x.flatten(), terr_y.flatten(), terrain.flatten()))

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
        # If there is no terrain provided, set terrain to None
        if terrain is None:
            self.leaf_area = leaf_area
            self.terrain = None
        # Max leaf area indices must be less than max terrain indices. Ideally shapes should be the same
        # but since the leaf area is a sparse array we must just check that it is smaller.
        elif np.max(leaf_area.leaf_area[:, 0]) <= np.max(terrain.terrain[:, 0]) and np.max(leaf_area.leaf_area[:, 1]) <= np.max(terrain.terrain[:, 1]):
            self.leaf_area = leaf_area
            self.terrain = terrain
        else:
            raise ValueError("Leaf area grid indices must be <= terrain indices.")
