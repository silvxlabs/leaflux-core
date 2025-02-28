"""LeafFlux Core General Functions"""
from .dependencies import *

from .environment import *
from .solar import *
from .irradiance import *

# Helper function that calculates a rotation matrix from a given solar vector
def _get_rot_mat(solar_vector: np.array) -> np.ndarray:

    vec_norm = np.linalg.norm(solar_vector)
    solar_vector = solar_vector / vec_norm

    target = np.array([0.0, 0.0, -1.0]) # Points straight down
    theta = np.arccos(np.dot(solar_vector, target)) # Angle that we are rotating

    # Handling edge cases for rotations of 0 or 180 degrees
    if np.isclose(0.0, theta, atol=1e-6):
        return np.eye(3)
    if np.isclose(np.pi, theta, atol=1e-6):
        return -np.eye(3)
    
    k_cross = np.cross(solar_vector, target) # Axis of rotation

    k_norm = np.linalg.norm(k_cross)
    k = k_cross / k_norm

    # Skew symmetric mat
    k_mat = np.array(
        [[   0,  -k[2], k[1]],
        [ k[2], 0,     -k[0]],
        [-k[1], k[0],  0]]
    )

    i = np.eye(3, dtype=float) # Identity

    # Rodrigues formula
    # print(i + np.sin(theta)*k_mat + (1-np.cos(theta))*(k_mat@k_mat))
    return i + np.sin(theta)*k_mat + (1.0-np.cos(theta))*(k_mat@k_mat)

# Light attenuation algorithm for flat surface
def _attenuate_surface_flat(env: Environment, sol: SolarPosition, extn: float) -> RelativeIrradiance:
    width = np.max(env.leaf_area.leaf_area[:, 0]).astype(int)
    height = np.max(env.leaf_area.leaf_area[:, 1]).astype(int)

    # 1) Project points onto the z=0 plane along the solar vector
    leaf_area_surface_grid = np.zeros((width, height))
    projection_distances = - env.leaf_area.leaf_area[:, 2] / sol.light_vector[2]
    projected_points = (
        env.leaf_area.leaf_area[:, :3] + projection_distances[:, None] * sol.light_vector
    )

    # 2) Convert x and y coordinates to grid indices with periodic boundary conditions
    x_indices = np.mod(projected_points[:, 0].astype(int), width)
    y_indices = np.mod(projected_points[:, 1].astype(int), height)

    # 3) Use np.add.at to accumulate projected leaf area values into the grid
    np.add.at(
        leaf_area_surface_grid, (y_indices.astype(int), x_indices.astype(int)), env.leaf_area.leaf_area[:, 3]
    )

    # 4) Compute irradiance using the Beer-Lambert law
    leaf_area_surface_grid = np.exp(-extn * leaf_area_surface_grid)

    return RelativeIrradiance(leaf_area_surface_grid)

# Light attenuation algorithm for irradiance on terrain surface
def _attenuate_surface_terrain(env: Environment, sol: SolarPosition, extn: float) -> RelativeIrradiance:
    # Create copy
    leaf_area = np.copy(env.leaf_area.leaf_area)
    terrain = np.copy(env.terrain.terrain)

    r = _get_rot_mat(sol.light_vector)
    inverse_r = np.linalg.inv(r)

    # Rotate all coords
    leaf_area[:, :3] = (r @ leaf_area[:, :3].T).T
    terrain[:, :3] = (r @ terrain[:, :3].T).T

    # Get max x and y for grids
    leaf_max_x = np.max(leaf_area[:, 0])
    terrain_max_x = np.max(terrain[:, 0])
    max_x = np.max((leaf_max_x, terrain_max_x)).astype(int)

    leaf_max_y = np.max(leaf_area[:, 1])
    terrain_max_y = np.max(terrain[:, 1])
    max_y = np.max((leaf_max_y, terrain_max_y)).astype(int)

    # Get min x and y for grids
    leaf_min_x = np.min(leaf_area[:, 0])
    terrain_min_x = np.min(terrain[:, 0])
    min_x = np.min((leaf_min_x, terrain_min_x)).astype(int)

    leaf_min_y = np.min(leaf_area[:, 1])
    terrain_min_y = np.min(terrain[:, 1])
    min_y = np.min((leaf_min_y, terrain_min_y)).astype(int)

    # Adjust indices for correct indexing into grids
    leaf_area[:, 0] -= min_x
    leaf_area[:, 1] -= min_y
    terrain[:, 0] -= min_x
    terrain[:, 1] -= min_y

    # Add leaf area into cells
    leaf_grid = np.zeros((max_x - min_x + 1, max_y - min_y + 1))
    np.add.at(
        leaf_grid, (leaf_area[:, 0].astype(int), leaf_area[:, 1].astype(int)), leaf_area[:, 3]
    )
    leaf_grid = np.exp(-extn * leaf_grid)

    # x, y, z, irr (all 1s)
    terrain_stack = np.column_stack((terrain[:, 0], terrain[:, 1], terrain[:, 2], np.ones_like(terrain[:, 0].flatten())))

    # Find max terrain value for each cell
    terrain_max = np.zeros((max_x - min_x + 1, max_y - min_y + 1))
    np.maximum.at(
        terrain_max, (terrain[:, 0].astype(int), terrain[:, 1].astype(int)), np.abs(terrain_stack[:, 2])
    )

    # Make irr 0 if value is not max (is in shadow)
    terrain_stack[:, 3] = np.where(
        np.abs(terrain_stack[:, 2]) >= terrain_max[terrain_stack[:, 0].astype(int), terrain_stack[:, 1].astype(int)],
        1, 
        0
    )

    # Readjust terrain coords
    terrain_stack[:, 0] += min_x
    terrain_stack[:, 1] += min_y

    terrain_stack[:, :3] = (inverse_r @ terrain_stack[:, :3].T).T # Rotate back
    irr_2d = np.zeros((env.terrain.width, env.terrain.height)) # Create 2D array of 0s
    irr_2d[terrain_stack[:, 0].astype(int), terrain_stack[:, 1].astype(int)] = terrain_stack[:, 3] # Fill with appropriate irr values

    # Apply gaussian filter to get rid of hill artifacts
    irr_2d = gaussian_filter(irr_2d, sigma=3)
    irr_2d = (irr_2d + 0.5).astype(int)

    terrain_stack[:, 3] = irr_2d[terrain_stack[:, 0].astype(int), terrain_stack[:, 1].astype(int)] # Put irr values back in terrain stack

    # Multiply the irr stack (which is all 0s and 1s) by irradiance to get real values
    terrain_stack[:, 3] = terrain_stack[:, 3] * leaf_grid[terrain[:, 0].astype(int), terrain[:, 1].astype(int)]

    # Make 2D grid with terrain valuess
    terrain_result_grid = np.zeros((env.terrain.width, env.terrain.height))
    terrain_result_grid[env.terrain.terrain[:, 1].astype(int), env.terrain.terrain[:, 0].astype(int)] = terrain_stack[:, 3]

    return RelativeIrradiance(terrain_result_grid)

def attenuate_surface(env: Environment, sol: SolarPosition, extn: float = 0.5) -> RelativeIrradiance:
    """
    Produces RelativeIrradiance object, containing the irradiance on the 
    terrain surface, for a given Environment and SolarPosition. Runs the irradiance attenuation
    model on either the surface provided, if it was provided, or on a flat surface.

    Parameters
    ----------
    env: Environment 
        Envrironment object which contains the leaf area array and (optionally) 
        the terrain array to be used.

    sol: SolarPosition
        SolarPosition object which describes the date, time, and latitude. 

    extn: float 
        Extinction coefficient for Beer's Law. Default is 0.5.

    Returns
    -------
    RelativeIrradiance
        Class containing the resulting relative irradiance for the terrain surface and optionally 
        the canopy if the canopy was attenuated.
    """
    if env.terrain is None:
        return _attenuate_surface_flat(env, sol, extn)
    else:
        return _attenuate_surface_terrain(env, sol, extn)