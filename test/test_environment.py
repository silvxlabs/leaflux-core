import pytest
from leaflux.dependencies import *
from leaflux.environment import *

class TestLeafArea:
    def test_init(self):
        input = np.load("test/data/leaf_area_grid.npy")
        expected_output = np.load("test/data/leafarea_from_init_test_1.npy")

        output = LeafArea(input, input.shape[1], input.shape[0]).leaf_area

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

class TestSensor:
    def test_init(self):
        my_sensor_0 = Sensor(1.5, 1.6, 1.7)

        assert my_sensor_0.sensor[0] == 1.5
        assert my_sensor_0.sensor[1] == 1.6
        assert my_sensor_0.sensor[2] == 1.7

        assert len(my_sensor_0.sensor) == 5

        dummy_leaf_area_grid = np.ones((100, 100, 100), dtype=np.float32)
        dummy_leaf_area = LeafArea.from_uniformgrid(dummy_leaf_area_grid)

        sensor_list = list[Sensor]
        sensor_list = [my_sensor_0]
        my_env_0 = Environment(dummy_leaf_area, sensors=sensor_list)

        assert my_env_0.sensors.shape[0] == len(sensor_list)

        my_sensor_1 = Sensor(2.1, 2.2, 2.3)
        my_sensor_2 = Sensor(3.1, 3.2, 3.3)
        sensor_list.append(my_sensor_1)
        sensor_list.append(my_sensor_2)

        my_env_1 = Environment(dummy_leaf_area, sensors=sensor_list)

        assert my_env_1.sensors is not None
        assert my_env_1.sensors[0, 0] == 1.5
        assert my_env_1.sensors[2, 2] == 3.3

        assert len(my_env_1.sensors) == 3

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

        # Testing with Sensor class
        my_sensor_0 = Sensor(1.5, 1.6, 1.7)
        my_sensor_1 = Sensor(2.1, 2.2, 2.3)
        my_sensor_2 = Sensor(3.1, 3.2, 3.3)
        sensor_list = list[Sensor]
        sensor_list = [my_sensor_0, my_sensor_1, my_sensor_2]

        with_sensor_env = Environment(leaf_area300, sensors=sensor_list)
        assert with_sensor_env.sensors is not None
        assert with_sensor_env.terrain is None

        with_sensor_terrain_env = Environment(leaf_area300, sensors=sensor_list, terrain=terrain300)
        assert isinstance(with_sensor_terrain_env.terrain, Terrain)
        assert isinstance(with_sensor_terrain_env.sensors[0], np.ndarray)
