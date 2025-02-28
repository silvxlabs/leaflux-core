## How to use LeafLux to generate a irradiance array
**Load a leaf area grid**

To load from an existing sparse numpy ND Array with shape (N, 4), you can 
use the default constructor.

    
```
my_leaf_area = LeafArea(np.load("path/to/my/leafarea.npy"))
```
    

If you are loading from a dense uniform grid, load using the` from_uniformgrid` constructor.

    
```
my_leaf_area = LeafArea.from_uniform_grid(np.load("path/to/my/uniformgrid.npy"))
```
    
**Load a terrain grid (optional)**

The default constructor will load from a 2.5D uniform grid where the value
at (x, y) is z.

```
my_terrain = Terrain(np.load("path/to/my/terrain.npy"))
```

**Create environment**

Create the environment with the leaf area and terrain objects you have made. You can also create an environment without a terrain object. In this case, when you run the model, the irradiance will be calculated on a flat plane.

```
my_environment = Envronment(my_leaf_area, my_terrain)
```

**Create solar position with date, time, and latitude**

The model needs a solar position, which can be created with a date, time, and latitude. The datetime required by the SolarPosition constructor is a Python `datetime` object, see <https://docs.python.org/3/library/datetime.html#datetime-objects> for more detail about `datetime` objects.

```
# datetime parameters are year, month, day, hour, minute
# This example is for April 11th 2024 at 14:00 (2:00PM)
my_datetime = datetime(2024, 8, 11, 14, 00)
my_latitude = 40.
my_solar_position = SolarPosition(my_datetime, my_latitude)
```

**Run the model and access results**

`attenuate_surface()` will provide a valid output whether you provide a terrain or not. It returns an object of type `RelativeIrradiance`, which contains separate numpy ND Arrays for the terrain and (if applicable) canopy irradiance. If not applicable, canopy irradiance will be `None`. 

```
my_result = attenuate_surface(my_environment, my_solar_position)
my_terrain_irradiance = my_result.terrain_irradiance
```

## How to access solar irradiance data from NREL to scale your outputs
- Should I do this?

