"""Reference for all user available classes and functions."""
from .dependencies import *

from .environment import *
from .solar import *
from .irradiance import *

from numba import jit
import pyvista as pv

# Correction for sensor based on tilt and azimuth. Helper function for attenuate_all
def sensor_correction(solar_azimuth: np.float32, solar_zenith: np.float32, sensor_pitch: np.float32, sensor_azimuth: np.float32):
    return np.cos(solar_zenith)*np.cos(sensor_pitch) + np.sin(solar_zenith)*np.sin(np.sin(sensor_pitch))*np.cos(solar_azimuth-sensor_azimuth)

# Function to do hash map plane sweep. Helper function for attenuate_all
@jit
def plane_sweep(leaf_area_stack: np.ndarray, x_min: int, x_max: int, y_min: int, y_max: int) -> np.ndarray:
    # Create hash map with (x, y) for each possible (x, y)
    area_map = {(x, y): 0.0 for x in range(x_min, x_max) for y in range(y_min, y_max)}

    # Go through entire leaf area stack
    for i, row in enumerate(leaf_area_stack):
        x, y, z, leaf_area, cum_leaf_area, x_rot, y_rot, z_rot = row

        # Cumulative leaf area is current leaf area plus what's already
        # in bucket at this (x, y)
        cum_leaf_area = leaf_area + area_map[x_rot, y_rot]

        # Set in stack
        leaf_area_stack[i, 4] = cum_leaf_area

        # Update hash map
        area_map[x_rot, y_rot] = cum_leaf_area
    return leaf_area_stack[:, 4]

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
    return i + np.sin(theta)*k_mat + (1.0-np.cos(theta))*(k_mat@k_mat)

# Light attenuation algorithm for flat surface
def _attenuate_surface_flat(env: Environment, sol: SolarPosition, extn: float) -> RelativeIrradiance:

    # For flooring values
    leaf_area_min = np.min(env.leaf_area.leaf_area[:, 2])

    # 1) Project points onto the z=0 plane along the solar vector
    projection_distances = - (env.leaf_area.leaf_area[:, 2] - leaf_area_min) / sol.light_vector[2]
    projected_points = (
        env.leaf_area.leaf_area[:, :3] + projection_distances[:, None] * sol.light_vector
    )

    # 2) Convert x and y coordinates to grid indices with periodic boundary conditions
    x_indices = np.mod(projected_points[:, 0].astype(int), env.leaf_area.width)
    y_indices = np.mod(projected_points[:, 1].astype(int), env.leaf_area.height)
    y_indices = env.leaf_area.height - y_indices - 1 # Flip to y goes north->south

    # 3) Use np.add.at to accumulate projected leaf area values into the grid
    leaf_area_surface_grid = np.zeros((env.leaf_area.height, env.leaf_area.width))
    np.add.at(
        leaf_area_surface_grid, (y_indices.astype(int), x_indices.astype(int)), env.leaf_area.leaf_area[:, 3]
    )

    # 4) Compute irradiance using the Beer-Lambert law
    leaf_area_surface_grid = np.exp(-extn * leaf_area_surface_grid)

    return RelativeIrradiance(terrain_irradiance=leaf_area_surface_grid)

# Light attenuation algorithm for irradiance on terrain surface
def _attenuate_surface_terrain(env: Environment, sol: SolarPosition, extn: float) -> RelativeIrradiance:
    # round_dec = 7
    # Create copy
    leaf_area = np.copy(env.leaf_area.leaf_area)
    terrain = np.copy(env.terrain.terrain)

    # Floor values
    terrain_min_z = np.min(terrain[:, 2])
    terrain[:, 2] -= terrain_min_z
    leaf_area[:, 2] -= terrain_min_z

    # leaf_area = np.round(leaf_area, round_dec)
    # terrain = np.round(terrain, round_dec)

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
    leaf_grid = np.zeros((max_y - min_y + 1, max_x - min_x + 1))
    np.add.at(
        leaf_grid, (leaf_area[:, 1].astype(int), leaf_area[:, 0].astype(int)), leaf_area[:, 3]
    )
    leaf_grid = np.exp(-extn * leaf_grid)
    # leaf_grid = np.round(leaf_grid, round_dec)

    # x, y, z, irr (all 1s)
    terrain_stack = np.column_stack((terrain[:, 0], terrain[:, 1], terrain[:, 2], np.ones_like(terrain[:, 0].flatten())))
    # terrain_stack[:, 2] = np.round(terrain_stack[:, 2], round_dec)

    # Find max terrain value for each cell
    terrain_max = np.zeros((max_x - min_x + 1, max_y - min_y + 1))
    np.maximum.at(
        terrain_max, (terrain_stack[:, 0].astype(int), terrain_stack[:, 1].astype(int)), np.abs(terrain_stack[:, 2])
    )

    # Make irr 0 if value is not max (is in shadow)
    # epsilon = 1e-6
    terrain_stack[:, 3] = np.where(
        np.abs(terrain_stack[:, 2]) >= terrain_max[terrain_stack[:, 0].astype(int), terrain_stack[:, 1].astype(int)], #-epsilon,
        1., 
        0.
    )

    # Readjust terrain coords
    terrain_stack[:, 0] += min_x
    terrain_stack[:, 1] += min_y

    terrain_stack[:, :3] = (inverse_r @ terrain_stack[:, :3].T).T # Rotate back
    # irr_2d = np.zeros((env.terrain.width, env.terrain.height)) # Create 2D array of 0s
    # irr_2d[terrain_stack[:, 0].astype(int), terrain_stack[:, 1].astype(int)] = terrain_stack[:, 3] # Fill with appropriate irr values

    # # Apply gaussian filter to get rid of hill artifacts
    # irr_2d = gaussian_filter(irr_2d, sigma=3)
    # irr_2d = (irr_2d + 0.5).astype(int)

    # terrain_stack[:, 3] = irr_2d[terrain_stack[:, 0].astype(int), terrain_stack[:, 1].astype(int)] # Put irr values back in terrain stack

    # Multiply the irr stack (which is all 0s and 1s) by irradiance to get real values
    terrain_stack[:, 3] = terrain_stack[:, 3] * leaf_grid[terrain[:, 1].astype(int), terrain[:, 0].astype(int)]

    # Make 2D grid with terrain valuess
    terrain_result_grid = np.zeros((env.terrain.height, env.terrain.width))
    terrain_result_grid[(env.terrain.height - env.terrain.terrain[:, 1].astype(int) - 1), env.terrain.terrain[:, 0].astype(int)] = terrain_stack[:, 3]

    return RelativeIrradiance(terrain_irradiance=terrain_result_grid)

def attenuate_surface(env: Environment, sol: SolarPosition, extn: float = 0.5) -> RelativeIrradiance:
    """
    Produces RelativeIrradiance object, containing the irradiance on the 
    terrain surface, for a given Environment and SolarPosition. Runs the irradiance attenuation
    model on either the surface provided, if it was provided, or on a flat surface. Both algorithms 
    manipulate z values to be in relation to 0, but provided LeafArea and Terrain z values can be 
    absolute. If both LeafArea and Terrain are provided it is expected that sets of coordinates are
    either both absolute or both manipulated to be relative to 0.

    *Note:* When run with both a LeafArea and Terrain, this function can have different results depending
    on platform due to vectorized operations. While this algorithm is faster than `attenuate_all()`, 
    use `attenuate_all()` if consistency is a priority for your use case.
    
    *Note:* While having sensors in the provided `Environment` will not cause any errors, the output
    will not contain any results for sensors, as results for sensors can only be obtained using `attenuate_all()`.

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
        Class containing the resulting relative irradiance for the terrain surface.
    """
    if env.terrain is None:
        return _attenuate_surface_flat(env, sol, extn)
    else:
        return _attenuate_surface_terrain(env, sol, extn)

def attenuate_all(env: Environment, sol: SolarPosition, extn: float = 0.5) -> RelativeIrradiance:
    """
    Produces a RelativeIrradiance object containing the relative irradiance for the canopy, and, if appropriate information 
    was provided, the terrain and sensors.

    Parameters
    -
    env: Environment
        Environment object which may or may not contain a Terrain object. If no Terrain object is provided, surface
        irradiance is still returned, but on a flat plane that is created below the leaf area.
    
    sol: SolarPosition 
        SolarPosition object which describes the date, time, and latitude. 

    extn: float
        Extinction coefficient for Beer's Law. Default is 0.5.
    
    Returns
    -
    RelativeIrradiance
        Class containing the resulting relative irradiance for the canopy, the terrain (if provided), and the sensors (if provided).
    """
    r = _get_rot_mat(sol.light_vector)

    # Will hold (x, y, z, leaf_area, cum_leaf_area)
    leaf_area_stack_rot = (r @ env.leaf_area.leaf_area[:, :3].T).T
    leaf_area_stack = np.column_stack((env.leaf_area.leaf_area[:, 0], env.leaf_area.leaf_area[:, 1], env.leaf_area.leaf_area[:, 2], env.leaf_area.leaf_area[:, 3], np.zeros_like(env.leaf_area.leaf_area[:, 0]), leaf_area_stack_rot))

    # Make terrain area array that will hold giant leaf area values
    if env.terrain != None:
        terrain_leaf_area = np.full_like(env.terrain.terrain[:, 0].flatten(), 2000.0, dtype=np.float32) # Make leaf area very high
        terrain_area_rot_stack = (r @ env.terrain.terrain[:, :3].T).T
        terrain_area_stack = np.column_stack((env.terrain.terrain[:, 0].flatten(), env.terrain.terrain[:, 1].flatten(), env.terrain.terrain[:, 2].flatten(), terrain_leaf_area, np.ones_like(env.terrain.terrain[:, 0].flatten(), dtype=np.float32), terrain_area_rot_stack))
        terrain_area_stack = terrain_area_stack.astype(np.float32)

        # Make dummy terrain area stack that will have projected leaf area on it
        dummy_terrain_area_stack = np.copy(terrain_area_stack)
        dummy_terrain_area_stack[:, 3] = 0.0 # No leaf area this time

        terrain_area_stack[:, 7] -= 1
        leaf_terrain_dummy_stack = np.vstack((leaf_area_stack, terrain_area_stack, dummy_terrain_area_stack))
    else:
        # No terrain provided
        leaf_terrain_dummy_stack = leaf_area_stack
    
    if env.sensors is not None:
        # Rotate
        sensor_rot = (r @ env.sensors[:, :3].T).T
        # (x, y, z, leaf_area = 0.0, cum_leaf_area = 0.0, x_rot, y_rot, z_rot)
        sensor_stack = np.column_stack((env.sensors[:, 0].flatten(), env.sensors[:, 1].flatten(), env.sensors[:, 2].flatten(), np.zeros_like(env.sensors[:, 0]), np.zeros_like(env.sensors[:, 0]), sensor_rot))

        # Add to leaf terrain dummy stack
        leaf_terrain_dummy_stack = np.vstack((leaf_terrain_dummy_stack, sensor_stack))

    # Floor x and y values to "bucket"
    leaf_terrain_dummy_stack[:, 5], x_rem = np.divmod(leaf_terrain_dummy_stack[:, 5], 1)
    leaf_terrain_dummy_stack[:, 6], y_rem = np.divmod(leaf_terrain_dummy_stack[:, 6], 1)

    # Sort by z in descending order
    leaf_terrain_dummy_stack = leaf_terrain_dummy_stack[leaf_terrain_dummy_stack[:, 7].argsort()[::-1]]

    # Find max rotated x and y values, use to create  hash map
    x_max = np.max(leaf_terrain_dummy_stack[:, 5]).astype(int) + 1
    y_max = np.max(leaf_terrain_dummy_stack[:, 6]).astype(int) + 1

    x_min = np.min(leaf_terrain_dummy_stack[:, 5]).astype(int)
    y_min = np.min(leaf_terrain_dummy_stack[:, 6]).astype(int)

    leaf_terrain_dummy_stack[:, 4] = plane_sweep(leaf_terrain_dummy_stack, x_min, x_max, y_min, y_max)

    # Irradiance
    leaf_terrain_dummy_stack[:, 4] = np.exp(-extn * leaf_terrain_dummy_stack[:, 4])

    if(env.sensors is not None):
        # Mask out values that are the sensors
        dtype = [('x', np.int32), ('y', np.int32), ('z', np.int32)]
        structured_ltds = leaf_terrain_dummy_stack[:, :3].astype(np.int32).view(dtype)
        structured_ss = env.sensors[:, :3].astype(np.int32).view(dtype)
        sensor_mask = np.isin(structured_ltds, structured_ss).flatten()
        sensors = leaf_terrain_dummy_stack[sensor_mask] # Get stack of just sensors

        # Applying tilt correction to the sensors that provided pitch and azimuth
        env.sensors = env.sensors[env.sensors[:, 2].argsort()[::-1]] # Sort so consistent w sorted sensors from ll
        nan_mask = ~np.isnan(env.sensors[:, 3]) & ~np.isnan(env.sensors[:, 4]) # Mask for sensors with provided values
        sensors[nan_mask, 4] *= sensor_correction(sol.azimuth, sol.zenith, env.sensors[nan_mask, 3], env.sensors[nan_mask, 4])

        sensor_irr_stack = np.column_stack((sensors[:, :3], sensors[:, 4]))
        sensor_irr_stack = sensor_irr_stack.astype(np.float32)

        # Remove from leaf_terrain_dummy_stack
        leaf_terrain_dummy_stack = leaf_terrain_dummy_stack[~sensor_mask]

    if env.terrain == None:
        canopy_result_stack = np.column_stack((leaf_terrain_dummy_stack[:, :3], leaf_terrain_dummy_stack[:, 4]))
        relative_irradiance = RelativeIrradiance(canopy_irradiance=canopy_result_stack)

    else:
        # Get tilt correction for terrain
        dz_dy, dz_dx = np.gradient(env.terrain.terrain_grid) # Get derivatives
        nx = -dz_dx
        ny = -dz_dy
        nz = np.ones_like(env.terrain.terrain_grid)
        mag = np.sqrt(nx**2 + ny**2 + nz**2)
        nx /= mag
        ny /= mag
        nz /= mag
        normals = np.stack((nx, ny, nz), axis=-1)
        dot = np.einsum("ijk,k->ij", normals, -sol.light_vector)
        irr_scale = np.clip(dot, 0, 1)

        # Isolate terrain surface irradiance
        surface_mask = leaf_terrain_dummy_stack[:, 3] == 0.0
        surface = leaf_terrain_dummy_stack[surface_mask, :]
        surface[:, 4] *= irr_scale[(env.terrain.height - surface[:, 1] - 1).astype(int), surface[:, 0].astype(int)] # Apply tilt correction
        surface_result_grid = np.zeros((env.leaf_area.height, env.leaf_area.width), dtype=np.float32)
        surface_result_grid[(env.leaf_area.height - np.round(surface[:, 1]) - 1).astype(int), np.round(surface[:, 0]).astype(int)] = surface[:, 4]

        # Isolate canopy irradiance
        canopy_mask = (leaf_terrain_dummy_stack[:, 3] != 2000.0) & (leaf_terrain_dummy_stack[:, 3] != 0.0)
        canopy_result_stack = np.column_stack((leaf_terrain_dummy_stack[canopy_mask, 0], leaf_terrain_dummy_stack[canopy_mask, 1], leaf_terrain_dummy_stack[canopy_mask, 2], leaf_terrain_dummy_stack[canopy_mask, 4]))
        canopy_result_stack = canopy_result_stack.astype(np.float32)

        relative_irradiance = RelativeIrradiance(surface_result_grid, canopy_result_stack)
    if(env.sensors is not None):
        relative_irradiance.sensor_irradiance = sensor_irr_stack
    
    return relative_irradiance

def plot_irradiance(relative_irradiance: RelativeIrradiance, terrain_coords: Terrain = None, solar_position: SolarPosition = None, show_sensors = False, show_axes: bool = False):
    """
    Uses pyvista to plot the irradiance for the canopy and terrain (if available), optionally showing the solar direction vector, sensors (if available), and axes. The optional 
    parameters can be helpful for debugging.

    Parameters
    -
    relative_irradiance: RelativeIrradiance
        Object of type RelativeIrradiance that has been returned from `attenuate_all` or `attenuate_surface`
    terrain_coords: Terrain
        (optional) Object of type Terrain. This is used to plot the z values of the terrain. Default is None but is *required* if you 
        provided a `RelativeIrradiance` with terrain.
    solar_position: SolarPosition
        (optional) Object of type SolarPosition. Provide this if you would like an arrow in the direction of the solar vector to be drawn. Default is None.
    show_sensors: bool
        (optional) Bool indicating whether sensors, if they exist, will be shown. Default is False.
    show_axes: bool
        (optional) Bool indicating whether axes should be drawn. Default is False.

    """
    plotter = pv.Plotter()

    # If there is only canopy to plot
    all_irr = relative_irradiance.canopy_irradiance

    # If there is terrain, plot this too
    if relative_irradiance.terrain_irradiance is not None:
        if not isinstance(terrain_coords, Terrain):
            raise TypeError(f"Expected an object of type 'Terrain', but got {type(terrain)}. Please supply terrain coordinates.")
        terrain_stack = Terrain(relative_irradiance.terrain_irradiance)
        # x, y, z, irr
        terrain = np.column_stack((terrain_stack.terrain[:, 0].ravel(), terrain_stack.terrain[:, 1].ravel(), terrain_coords.terrain[:, 2].ravel(), terrain_stack.terrain[:, 2].ravel()))

        # Adding canopy and terrain
        if relative_irradiance.canopy_irradiance is not None:
            all_irr = np.vstack((terrain, relative_irradiance.canopy_irradiance))
        # Unless there is only terrain
        else:
            all_irr = terrain
    
    coords = all_irr[:, :3]
    irr = all_irr[:, 3]

    irr_point_cloud = pv.PolyData(coords)
    irr_point_cloud["Relative Irradiance"] = irr

    plotter.add_mesh(
        irr_point_cloud,
        scalars="Relative Irradiance",
        cmap="viridis",            
        point_size=6,
        render_points_as_spheres=True,
        show_edges=False
    )

    # Adding sensors as large red spheres
    if relative_irradiance.sensor_irradiance is not None and show_sensors:
        sensor_coords = relative_irradiance.sensor_irradiance[:, :3]
        sensor_point_cloud = pv.PolyData(sensor_coords)

        plotter.add_mesh(
            sensor_point_cloud,
            color='red',
            point_size=20,
            render_points_as_spheres=True,
            show_edges=False
        )

    # Adding solar direction arrow in red
    # In (x, y, z)
    if solar_position is not None:
        x_med = np.min(all_irr[:, 0]) + ((np.max(all_irr[:, 0]) - np.min(all_irr[:, 0])) / 2.)
        y_med = np.min(all_irr[:, 1]) + ((np.max(all_irr[:, 1]) - np.min(all_irr[:, 1])) / 2.)
        z_range = np.max(all_irr[:, 2]) - np.min(all_irr[:, 2])
        arrow = pv.Arrow(start=(x_med, y_med, np.max(all_irr[:, 2]) + z_range), direction=solar_position.light_vector, scale=100)
        plotter.add_mesh(arrow, color="red")

    # Adding axes
    if show_axes:
        plotter.show_axes()

    # Showing plot
    plotter.show()
