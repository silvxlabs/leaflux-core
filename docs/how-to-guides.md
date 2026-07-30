## How to use LeafLux to generate a irradiance array
**Load a leaf area grid**

To load from an existing sparse numpy ND Array with shape (N, 4), you can 
use the default constructor. You must also provide the width and height of the
area the leaf area is describing. This should be the same width and height as the
`Terrain` if you are providing one. y coordinates are expected to run south to north.

    
```
my_leaf_area = LeafArea(np.load("path/to/my/leafarea.npy"), my_leaf_area_width, my_leaf_area_height)
```
    

If you are loading from a dense uniform grid, load using the `from_uniformgrid` constructor. This grid 
is expected to have shape (y, x, z) where each coordinate maps to a leaf area value. y coordinates are expected to run 
north to south. Width and height are inferred from grid shape.

    
```
my_leaf_area = LeafArea.from_uniform_grid(np.load("path/to/my/uniformgrid.npy"))
```

**Load a mean leaf angle grid (optional)**

Loading this grid has the same options as the leaf area grid, and if you are supplying a mean leaf angle grid, 
its coordinates and dimensions must match those of your leaf area grid exactly.

Mean leaf angle is expected in radians.

If provided, mean leaf angle is used to calcualte G per voxel using the spherical distribution formulas found in
[Campbell 1989](https://doi.org/10.1017/CBO9780511752308.002) and [Campbell 1990](https://doi.org/10.1016/0168-1923(90)90030-A).


Sparse:

```
my_leaf_angle = LeafAngle(np.load("path/to/my/leafarea.npy"), my_leaf_angle_width, my_leaf_angle_height)
```


Uniform grid:

```
my_leaf_angle = LeafAngle.from_uniform_grid(np.load("path/to/my/uniformgrid.npy"))
```
**Load a terrain grid (optional)**

The default constructor will load from a 2.5D uniform grid where the value
at (y, x) is z. y coordinates are expected to run north to south.

```
my_terrain = Terrain(np.load("path/to/my/terrain.npy"))
```

**Add sensors (optional)**

You can create and add sensors to your environment to generate irradiance results at specific points, which can be useful 
for testing and validation.

(x, y, z) coordinates for sensors are expected to have y coordinates running south to north. 

If you provide a pitch and azimuth for a sensor, the returned irradiance will be corrected for the tilt of the sensor. If either
pitch or azimuth are omitted, returned irradiance will not be corrected. Pitch and azimuth are expected in radians.

Sensors must be provided as a list of sensors. Create a list of sensors like this:

```
my_sensor_0 = Sensor(300, 150, 20) # No pitch or azimuth
my_sensor_1 = Sensor(200, 100, 15, 1.2, 2.2) # Providing pitch and azimuth
my_sensor_2 = Sensor(333, 333, 33, 1.0, 2.5) # Prividing pitch and azimuth
my_sensors = list[Sensor]
my_sensors = [my_sensor_0, my_sensor_1, my_sensor_2]
```

You must ensure that the coordinates of your sensors are located within the domain of your environment.

**Create environment**

Create the environment with the objects you have made. `LeafArea` is the only required argument. 

The `Terrain` and `LeafArea` objects that make up your `Environment` should have z values relative to eachother-- in the surface algorithms z values are 
manipulated to be in relation to 0, but provided values can be absolute. Ensure this is the case before creating your `Environment`. Plotting your 
`LeafArea` and `Terrain` can be helpful, as can checking minimum z values.

Again, `LeafAngle` coordiantes (if you are providing a leaf angle grid) and dimensions must match those of your `LeafArea` exactly.

Voxel dimensions can be optionally specified. If dimensions are not specified, a voxel shape of 1mx1mx1m is assumed. Dimensions 
should be supplied as a tuple of floats describing voxel dimensions in (x, y, z). It is up to you to ensure these dimensions are compatible with your input grids. 

```
my_environment = Envronment(my_leaf_area, terrain=my_terrain, leaf_angle=my_leaf_angle, voxel_dim=(1.0, 1.0, 1.0), sensors=my_sensors)
```

**Create solar position with date, time, latitude, and longitude**

The model needs a solar position, which can be created with a date, time, latitude, and longitude. Time **must** be in UTC, **not** local time. This is because longitude is used with UTC to get accurate solar positions. The datetime required by the `SolarPosition` constructor is a Python `datetime` object, see <https://docs.python.org/3/library/datetime.html#datetime-objects> for more detail about `datetime` objects.

```
# datetime parameters are year, month, day, hour, minute
# This example is for April 11th 2024 at 17:00 (5:00PM) UTC
my_datetime = datetime(2024, 4, 11, 17, 00)
my_latitude = 40.
my_longitude = -120.
my_solar_position = SolarPosition(my_datetime, my_latitude, my_longitude)
```

**Run the model and access results**

`attenuate_surface()` returns irradiance on the surface. `attenuate_surface()` will provide a valid output whether you provide a terrain or not. It returns an object of type `RelativeIrradiance`, which contains separate numpy ND Arrays for the terrain and (if applicable) canopy irradiance. If not applicable, canopy irradiance will be `None`. 

```
my_result = attenuate_surface(my_environment, my_solar_position)
my_terrain_irradiance = my_result.terrain_irradiance
```

`attenuate_all()` returns irradiance on the surface and in the canopy. If no terrain is provided, **only** canopy irradiance is returned. If terrain is
provided, results for the canopy and surface are returned. 

```
my_result = attenuate_all(my_environment, my_solar_position)
my_terrain_irradiance = my_result.terrain_irradiance
my_canopy_irradiance = my_result.canopy_irradiance
```

Sensor results can be obtained with the `get_sensor_irradiance()` function. Sensor results are only generated from `attenuate_all()`.

```
sensor_0_irradiance = my_result.get_sensor_irradiance(my_sensor_0)
```

Sensor results can also be accessed directly via the `sensor_irradiance` attribute of the `RelativeIrradiance` class. `sensor_irradiance` is an 
(N, 4) numpy array where each row is (x, y, z, irradiance). The order of the returned sensors is not guarenteed to be the same as the order
provided. This is how you could access information about the first returned sensor:

```
sensor_0_irradiance = my_result.sensor_irradiance[0, 4] # Irradiance
sensor_0_coordinates = my_result.sensor_irradiance[0, :3] # Coordinates
sensor_0_result = my_result.sensor_irradiance[0, :] # Entire row
```

## How to plot LeafLux results

The `plot_irradiance()` function has been included to make plotting the output of LeafLux easy and flexible. 

`plot_irradiance()` just needs the result of either `attenuate_surface()` or `attenuate_all()`, but there are additional options as well. Terrain 
coordinates *must* be provided if the result contains terrain irradiance. These are used to provide z values.

If a `show_solar_vector` is set to True, an arrow will be drawn in the direction of the solar vector. This can be useful for visualizing how the sun is interacting with your environment. 

If sensors are present in the result and `show_sensors` is set to `True`, sensors will be drawn as large red spheres, which can be useful for 
checking that their coordinates are correct.

If `show_axes` is set to `True`, axes will be drawn, which can be helpful in understanding the output.

If `show_canopy` is set to `True`, and terrain results are present, then the canopy will be drawn. This is the default setting. If `show_canopy` is set
to `False`, the canopy will not be drawn. This can be useful for visualizing the terrain surface by itself. 

The below example plots the entire environment and includes all the optional elements.

```
plot_irradiance(my_result, my_terrain, True, True, True, True)
```

The below example utilizes all of the default settings except the canopy will not be drawn. 

```
plot_irradiance(my_result, my_terrain, show_canopy=False)
```

