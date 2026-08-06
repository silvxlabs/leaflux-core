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

class TestLeafAngle:
    def test_init(self):
        # __init__ stores the point cloud and dims directly
        point_cloud = np.array([
            [0.0, 0.0, 0.0, 0.20],
            [1.0, 2.0, 3.0, 0.75],
            [4.0, 5.0, 6.0, 1.10],
        ], dtype=np.float32)

        la = LeafAngle(point_cloud, width=300, height=300)

        np.testing.assert_array_equal(la.leaf_angle, point_cloud)
        assert la.width == 300
        assert la.height == 300

    def test_from_uniformgrid(self):
        # grid indexed [y, x, z], y north->south; from_uniformgrid flips y to south->north
        grid = np.zeros((2, 3, 2), dtype=np.float32)  # height=2, width=3, depth=2
        grid[0, 0, 0] = 0.5
        grid[0, 2, 1] = 0.7
        grid[1, 1, 0] = 0.9

        # expected (x, y_flipped, z, value); y_flipped = height - y_idx - 1
        expected = np.array([
            [0, 1, 0, 0.5],   # from grid[0,0,0]
            [2, 1, 1, 0.7],   # from grid[0,2,1]
            [1, 0, 0, 0.9],   # from grid[1,1,0]
        ], dtype=np.float32)

        la = LeafAngle.from_uniformgrid(grid)

        # sort both so the test doesn't depend on COO's row ordering
        def sort_rows(a):
            return a[np.lexsort((a[:, 2], a[:, 1], a[:, 0]))]

        np.testing.assert_array_equal(sort_rows(la.leaf_angle), sort_rows(expected))
        assert la.width == 3
        assert la.height == 2

class TestEnvironment:
    @pytest.fixture
    def dummy_leaf_area(self):
        # small (x, y, z, leaf area) point cloud
        coords = np.array([
            [0.0, 0.0, 0.0, 0.3],
            [1.0, 0.0, 0.0, 0.4],
            [0.0, 1.0, 0.0, 0.5],
            [2.0, 3.0, 1.0, 0.6],
        ], dtype=np.float32)
        return LeafArea(coords, width=10, height=10)
    
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

    def test_voxel_dim_default_none(self, dummy_leaf_area):
        env = Environment(dummy_leaf_area)
        assert env.voxel_dim is None

    @pytest.mark.parametrize("bad_voxel_dim", [
        [2.0, 2.0, 1.0],      # list, not tuple
        "221",                # 3-char string
        5,                    # int
        (2, 2, 1),            # tuple of ints
        (2.0, 2.0, "1"),      # tuple with a non-float
        (2.0, 2.0, 2.0, 2.0), # Tuple of wrong dims
        (-2.0, 2.0, 1.0),     # Negative value
    ])
    def test_voxel_dim_wrong_type_raises(self, dummy_leaf_area, bad_voxel_dim):
        with pytest.raises(TypeError):
            Environment(dummy_leaf_area, voxel_dim=bad_voxel_dim)

    def test_voxel_dim_valid_is_retrievable(self, dummy_leaf_area):
        env = Environment(dummy_leaf_area, voxel_dim=(2.0, 2.0, 1.0))
        assert env.voxel_dim == (2.0, 2.0, 1.0)

    def test_leaf_angle_default_none(self, dummy_leaf_area):
        env = Environment(dummy_leaf_area)
        assert env.leaf_angle is None

    @pytest.mark.parametrize("bad_leaf_angle", [
        np.zeros((4, 4), dtype=np.float32),  # raw array, not LeafAngle
        "leaf_angle",
        5,
        [1, 2, 3],
    ])
    def test_leaf_angle_wrong_type_raises(self, dummy_leaf_area, bad_leaf_angle):
        with pytest.raises(TypeError):
            Environment(dummy_leaf_area, leaf_angle=bad_leaf_angle)

    def test_leaf_angle_dimension_mismatch_raises(self, dummy_leaf_area):
        # Same coords so the coordinate check passes; only width/height differ
        coords = dummy_leaf_area.leaf_area.copy()
        mismatched = LeafAngle(coords, width=999, height=999)
        with pytest.raises(ValueError):
            Environment(dummy_leaf_area, leaf_angle=mismatched)

    def test_leaf_area_leaf_angle_coord_mismatch_raises(self, dummy_leaf_area):
        # Same dims, but a coordinate differs -> coordinate check fails
        mismatched = dummy_leaf_area.leaf_area.copy()
        mismatched[0, 0] += 5.0   # move first point in x; col 3 (value) irrelevant to check
        leaf_angle = LeafAngle(mismatched, width=dummy_leaf_area.width, height=dummy_leaf_area.height)
        with pytest.raises(ValueError):
            Environment(dummy_leaf_area, leaf_angle=leaf_angle)

    def test_terrain_leaf_area_leaf_angle_dims_consistent(self, dummy_leaf_area):
        # leaf_angle shares leaf_area coords; terrain matches width/height
        angle_cloud = dummy_leaf_area.leaf_area.copy()
        angle_cloud[:, 3] = 0.5   # angle values; coords unchanged so coord check passes
        leaf_angle = LeafAngle(angle_cloud, width=dummy_leaf_area.width, height=dummy_leaf_area.height)
        terrain = Terrain(np.zeros((dummy_leaf_area.height, dummy_leaf_area.width), dtype=np.float32))

        env = Environment(dummy_leaf_area, terrain=terrain, leaf_angle=leaf_angle)

        assert env.leaf_area is not None
        assert env.terrain is not None
        assert env.leaf_angle is not None
        assert env.leaf_area.width == env.terrain.width == env.leaf_angle.width
        assert env.leaf_area.height == env.terrain.height == env.leaf_angle.height
