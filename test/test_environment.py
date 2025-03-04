import pytest
from leaflux.dependencies import *
from leaflux.environment import *

class TestLeafArea:
    def test_init(self):
        input = np.load("test/data/leaf_area_grid.npy")
        expected_output = np.load("test/data/leafarea_from_init_test_1.npy")
        output = LeafArea(input).leaf_area

        np.testing.assert_array_equal(expected_output, output)

    def test_from_uniformgrid(self):
        input = np.load("test/data/leaf_area_grid.npy")
        expected_output = np.load("test/data/leafarea_from_uniformgird_test_1.npy")
        output = LeafArea.from_uniformgrid(input).leaf_area

        np.testing.assert_array_equal(expected_output, output)

class TestTerrain:
    def test_init(self):
        input = np.load("test/data/terrain_input300.npy")
        expected_output = np.load("test/data/terrain_output300.npy")
        output = Terrain(input)

        np.testing.assert_array_equal(expected_output, output.terrain)
        assert output.terrain.shape == expected_output.shape
        assert output.width == 300
        assert output.height == 300

class TestEnvironment:
    def test_init(self):
        leaf_area_grid = np.load("test/data/leaf_area_grid.npy")
        terrain_array_1 = np.load("test/data/terrain_input300.npy")
        terrain_array_2 = np.load("test/data/terrain_input600.npy")

        leaf_area300 = LeafArea.from_uniformgrid(leaf_area_grid)
        terrain300 = Terrain(terrain_array_1)
        terrain600 = Terrain(terrain_array_2)

        # Case with leaf area and terrain, check if not None
        full_env = Environment(leaf_area300, terrain300)
        assert full_env.leaf_area is not None
        assert full_env.terrain is not None
        assert np.max(full_env.leaf_area.leaf_area[:, 0]) <= np.max(full_env.terrain.terrain[:, 0])
        assert np.max(full_env.leaf_area.leaf_area[:, 1]) <= np.max(full_env.terrain.terrain[:, 1])
    
        # Case where there is no terrain, check that there is no terrain
        no_terrain_env = Environment(leaf_area300)
        assert no_terrain_env.leaf_area is not None
        assert no_terrain_env.terrain is None

        # Case where ValueError is raised, check that is raised
        with pytest.raises(ValueError):
            Environment(leaf_area300, terrain600)