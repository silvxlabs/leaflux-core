import pytest 

from leaflux_core.dependencies import *
from leaflux_core.general import _get_rot_mat, _attenuate_surface_flat, _attenuate_surface_terrain, attenuate_surface
from leaflux_core.solar import *
from leaflux_core.environment import *

class TestGeneral:
    @pytest.mark.parametrize(
        "vector,expected",
        [
            (np.array([0.6, 0.2, -0.6]), np.array([0.0, 0.0, -1.0])),
            (np.array([0.6, 0.0, 1.0]), np.array([0.0, 0.0, -1.0])),
            (np.array([2.0, 3.0, 1.0]), np.array([0.0, 0.0, -1.0])),
            (np.array([-2.0, -3.0, -1.0]), np.array([0.0, 0.0, -1.0])),
            (np.array([2.0, -4.0, 2.0]), np.array([0.0, 0.0, -1.0])),
            (np.array([0.0, 0.0, 4.0]), np.array([0.0, 0.0, -1.0])),
            (np.array([0.0, 0.0, -4.0]), np.array([0.0, 0.0, -1.0])),
            (np.array([1.0, 0.0, 0.0]), np.array([0.0, 0.0, -1.0])),
        ]
    )
    def test_get_rot_mat(self, vector, expected):
        output = _get_rot_mat(vector) @ vector
        output_norm = np.linalg.norm(output)
        output = output / output_norm
        np.testing.assert_allclose(output, expected, atol=1e-6)

    
    def test_attenuate_surface_flat(self):
        # Test against flat_result_1.npy
        # Which is datetime(2024, 6, 15, 16, 00) and lat = 40.
        my_datetime = datetime(2024, 6, 15, 16, 00)
        my_latitude = 40.
        my_solar_position = SolarPosition(my_datetime, my_latitude)

        leaf_area_grid = np.load("test/data/leaf_area_grid.npy")
        my_leaf_area = LeafArea.from_uniformgrid(leaf_area_grid)

        my_flat_env = Environment(my_leaf_area)

        sf1 = attenuate_surface(my_flat_env, my_solar_position)

        np.testing.assert_allclose(np.load("test/data/flat_result_1.npy"), sf1.terrain_irradiance, atol=1e-6)

    def test_attenuate_surface_terrain(self):
        # Test against terrain_result_1.npy
        # Which is datetime(2024, 6, 15, 16, 00) and lat = 40.
        my_datetime = datetime(2024, 6, 15, 16, 00)
        my_latitude = 40.
        my_solar_position = SolarPosition(my_datetime, my_latitude)

        leaf_area_grid = np.load("test/data/leaf_area_grid.npy")
        my_leaf_area = LeafArea.from_uniformgrid(leaf_area_grid)

        my_terrain = Terrain(np.load("test/data/terrain_input300.npy"))

        my_env = Environment(my_leaf_area, my_terrain)

        st1 = attenuate_surface(my_env, my_solar_position)

        np.testing.assert_allclose(np.load("test/data/terrain_result_1.npy"), st1.terrain_irradiance, atol=1e-6)
    
    def test_attenuate_surface(self):
        my_datetime = datetime(2024, 6, 15, 16, 00)
        my_latitude = 40.
        my_solar_position = SolarPosition(my_datetime, my_latitude)

        leaf_area_grid = np.load("test/data/leaf_area_grid.npy")
        my_leaf_area = LeafArea.from_uniformgrid(leaf_area_grid)

        my_terrain = Terrain(np.load("test/data/terrain_input300.npy"))

        my_env = Environment(my_leaf_area, my_terrain)
        my_flat_env = Environment(my_leaf_area)

        # Assert that leaf irradiance is none and terrain is not None
        surface1 = attenuate_surface(my_env, my_solar_position)
        assert surface1.leaf_irradiance is None
        assert surface1.terrain_irradiance is not None

        surface2 = attenuate_surface(my_flat_env, my_solar_position)
        assert surface2.leaf_irradiance is None
        assert surface2.terrain_irradiance is not None
                                    



    