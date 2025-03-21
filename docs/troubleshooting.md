# Troubleshooting -- Common Issues

## Coordinate system differences

 - Ensure that the coordinanate systems of the leaf area and terrain data you are supplying to the model is what the model expects. 

 - Make sure your y coordinates are not flipped. There are several standards for y coordinate alignment, and your y coordinates may not be what you expect. Flipping may be ocurring if the output you are getting looks correct but flipped, or if the leaf area and terrain are not lining up. The default LeafArea constructor expects y coordinates to be running south to north. The `.from_uniformgrid()` constructor expects y coordinates to be running north to south.

 - Consult the documentation for `LeafArea` and `Terrain` for full details.

 - If you think this may be an issue, plot your data before and after running LeafLux to ensure the output is aligned how you are expecting.

## Z value discrepancies

 - `LeafArea` and `Terrain` coordinates can either be *both* absolute or *both* relative to 0, but *not* mismatched. The model cannot detect this, and will likely run. If your coordinates are mismatched, your results might look like they have been shifted across the terrain.

 - If you are supplying a `LeafArea` and a `Terrain`, ensure that the z values of both objects are aligned with relation to each other. For example, if your `Terrain` coordinates are absolute and the minimum z value of your `Terrain` is 300, you should check to make sure the minimum z value of your `LeafArea` is also somewhere around 300. You can check these values by doing `min_z_value_leaf_area = np.min(my_leaf_area.leaf_area[:, 2])` and `min_z_value_terrain = np.min(my_terrain.terrain[:, 2])`
 
 - It can be useful to plot your `LeafArea` and `Terrain`, and the data they are created from, *before* running LeafLux. Plot them on the same grid to ensure they align the way you expect. Pyvista is a powerful and easy to use tool for this. 

## Time zones

 - The `SolarPosition` class takes a `datetime` object as input. By default the time is in UTC. Ensure you are correcting for this with the times you are supplying. 

 - You can also supply additional arguments to a `datetime` object to change the timezone. More information on this can be found in the `datetime` documentation: <https://docs.python.org/3/library/datetime.html>
