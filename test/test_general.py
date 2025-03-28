import pytest 

from leaflux.dependencies import *
from leaflux.general import _get_rot_mat, _attenuate_surface_flat, _attenuate_surface_terrain, attenuate_surface, attenuate_all
from leaflux.solar import *
from leaflux.environment import *

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

        expected = np.load("test/data/flat_result_1.npy")

        my_flat_env = Environment(my_leaf_area)
        sf1 = attenuate_surface(my_flat_env, my_solar_position)
        np.testing.assert_allclose(expected, sf1.terrain_irradiance, atol=1e-6)

        # Raising
        my_leaf_area.leaf_area[:, 2] += 100
        my_flat_env_raised = Environment(my_leaf_area)
        sf2 = attenuate_surface(my_flat_env_raised, my_solar_position)
        np.testing.assert_allclose(expected, sf2.terrain_irradiance, atol=1e-6)

        # Lowering
        my_leaf_area.leaf_area[:, 2] -= 200
        my_flat_env_lowered = Environment(my_leaf_area)
        sf3 = attenuate_surface(my_flat_env_lowered, my_solar_position)
        np.testing.assert_allclose(expected, sf3.terrain_irradiance, atol=1e-6)

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

        my_env.terrain.terrain[:, 2] -= 100
        my_env.leaf_area.leaf_area[:, 2] -= 100
        st2 = attenuate_surface(my_env, my_solar_position)

        my_env.terrain.terrain[:, 2] += 300
        my_env.leaf_area.leaf_area[:, 2] += 300
        st3 = attenuate_surface(my_env, my_solar_position)

        expected = np.load("test/data/terrain_result_1.npy")

        actual = st1.terrain_irradiance

        errors = np.abs(expected - actual)

        error_indices = np.where(errors > 0)[0]

        errors_above_1 = np.sum(errors >= 1.0)

        print("Indices of errors:", error_indices)
        print("Total errors: ", len(error_indices))
        print("Number of errors with difference >= 1.0:", errors_above_1)

        expected_sum = np.sum(expected)
        actual_sum = np.sum(actual)

        print("Expected sum: ", expected_sum)
        print("Actual sum: ", actual_sum)

        # Test against expected
        assert (np.abs(actual_sum - expected_sum) / expected_sum) < 0.25

        # Test against lowered and raised 
        np.testing.assert_equal(st1.terrain_irradiance, st2.terrain_irradiance)
        np.testing.assert_equal(st1.terrain_irradiance, st3.terrain_irradiance)

        # np.testing.assert_allclose(expected, actual, atol=1e-6)
    
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
        assert surface1.canopy_irradiance is None
        assert surface1.terrain_irradiance is not None

        surface2 = attenuate_surface(my_flat_env, my_solar_position)
        assert surface2.canopy_irradiance is None
        assert surface2.terrain_irradiance is not None

    def test_attenuate_all(self):
        my_datetime = datetime(2024, 6, 15, 18, 00)
        my_latitude = 40.
        my_solar_position = SolarPosition(my_datetime, my_latitude)

        my_terrain_grid = np.load("test/data/terrain_input300.npy")
        my_terrain = Terrain(my_terrain_grid)

        leaf_area_grid = np.load("test/data/leaf_area_grid.npy")
        my_leaf_area = LeafArea.from_uniformgrid(leaf_area_grid)
        my_leaf_area.leaf_area[:, 2] = my_leaf_area.leaf_area[:, 2] + my_terrain_grid[(my_leaf_area.height - my_leaf_area.leaf_area[:, 1] - 1).astype(int), my_leaf_area.leaf_area[:, 0].astype(int)]

        my_sensor_0 = Sensor(150, 150, 50)
        my_sensor_1 = Sensor(33, 33, 3, 0.5, 0.5)
        my_sensor_2 = Sensor(77, 77, 7)
        sensor_list = list[Sensor]
        sensor_list = [my_sensor_0, my_sensor_1, my_sensor_2]

        my_env = Environment(my_leaf_area, terrain=my_terrain, sensors=sensor_list)
        my_flat_env = Environment(my_leaf_area)

        result = attenuate_all(my_env, my_solar_position)
        result_flat = attenuate_all(my_flat_env, my_solar_position)

        SAVE_NEW = False
        if SAVE_NEW:
            np.save("test/data/all_result_terr_1.npy", result.terrain_irradiance)
            np.save("test/data/all_result_canopy_1.npy", result.canopy_irradiance)
            np.save("test/data/all_result_sensor.npy", result.sensor_irradiance)

            np.save("test/data/all_result_canopy_flat.npy", result_flat.canopy_irradiance)
            # plot_entire(result.terrain_irradiance, my_terrain.terrain, result.canopy_irradiance, my_solar_position, True)

        ASSERT = True
        if ASSERT:
            # With terrain
            expected_terr = np.load("test/data/all_result_terr_1.npy")
            expected_canopy = np.load("test/data/all_result_canopy_1.npy")

            actual_terr = result.terrain_irradiance
            actual_canopy = result.canopy_irradiance

            errors_terr = np.abs(expected_terr - actual_terr)
            errors_canopy = np.abs(expected_canopy - actual_canopy)

            error_indices_terr = np.where(errors_terr > 0)[0]
            error_indices_canopy = np.where(errors_canopy > 0)[0]

            errors_above_1_terr = np.sum(errors_terr >= 1.0)
            errors_above_1_canopy = np.sum(errors_canopy >= 1.0)

            print("Terrain errors: ", len(error_indices_terr))
            print("Caanopy errors: ", len(error_indices_canopy))
            print("Total errors: ", len(error_indices_terr)+len(error_indices_canopy))
            print("Terrain errors over 1: ", errors_above_1_terr)
            print("Canopy errors over 1: ", errors_above_1_canopy)
            print("Number of errors with difference >= 1.0:", errors_above_1_terr + errors_above_1_canopy)

            expected_sum_terr = np.sum(expected_terr)
            expected_sum_canopy = np.sum(expected_canopy[:, 3])
            actual_sum_terr = np.sum(actual_terr)
            actual_sum_canopy = np.sum(actual_canopy[:, 3])

            print("Expected sum terr: ", expected_sum_terr, " Actual sum terr: ", actual_sum_terr, "Terr sum diff: ", expected_sum_terr-actual_sum_terr)
            print("Expected sum canopy: ", expected_sum_canopy, " Actual sum canopy: ", actual_sum_canopy, " Canopy sum diff: ", expected_sum_canopy-actual_sum_canopy)
            print("Total expected sum: ", expected_sum_terr + expected_sum_canopy)
            print("Total actual sum: ", actual_sum_terr + actual_sum_canopy)

            # No terrain
            expected_canopy_flat = np.load("test/data/all_result_canopy_flat.npy")
            actual_canopy_flat = result_flat.canopy_irradiance
            expected_sum_flat_canopy = np.sum(expected_canopy_flat)
            actual_sum_flat_canopy = np.sum(actual_canopy_flat)
            print("* FLAT * Expected sum canopy: ", expected_sum_flat_canopy, " Actual sum canopy: ", actual_sum_flat_canopy, " Canopy sum diff: ", expected_sum_flat_canopy-actual_sum_flat_canopy)

            # Testing against expected result
            np.testing.assert_allclose(expected_terr, actual_terr, atol=1e-6)
            np.testing.assert_allclose(expected_canopy, actual_canopy, atol=1e-6)
            np.testing.assert_equal(expected_terr, actual_terr)
            np.testing.assert_equal(expected_canopy, actual_canopy)

            # Checking if are same within threshhold
            # thresh = 0.99
            # matches_canopy = np.isclose(expected_canopy, actual_canopy)
            # assert (np.sum(matches_canopy) / len(expected_canopy)) >= thresh

            # matches_terr = np.isclose(expected_terr, actual_terr)
            # assert (np.sum(matches_terr) / len(expected_terr)) >= thresh

            # # Assert that flat result has no terrain and testing 
            # assert result_flat.terrain_irradiance == None
            # matches_flat = np.isclose(expected_canopy_flat, actual_canopy_flat)
            # assert (np.sum(matches_flat) / len(actual_canopy_flat)) >= thresh

            # Testing that sensor output coordinates remain the same and that results are consistent
            assert result.get_sensor_irradiance(my_sensor_1) == 1.
            assert result.get_sensor_irradiance(my_sensor_0) == 0.1445081

            expected_sensor = np.load("test/data/all_result_sensor.npy")
            np.testing.assert_allclose(expected_sensor, result.sensor_irradiance, atol=1e-6)
            np.testing.assert_equal(expected_sensor, result.sensor_irradiance)

    @pytest.mark.skip()          
    def test_surface_vs_all(self):
        my_datetime = datetime(2024, 6, 15, 9, 00)
        my_latitude = 40.
        my_solar_position = SolarPosition(my_datetime, my_latitude)

        target_width = 500
        target_height = 500
        my_terrain_grid = np.load("dev/data/kevin_terrain.npy")
        my_terrain_grid = my_terrain_grid[:target_height, :target_width]
        my_terrain = Terrain(my_terrain_grid)

        my_leaf_area_grid = np.load("dev/data/kevin_leaf_area.npy")
        my_leaf_area_grid = my_leaf_area_grid[:target_height, :target_width]
        my_leaf_area = LeafArea.from_uniformgrid(my_leaf_area_grid)
        my_leaf_area.leaf_area[:, 2] = my_leaf_area.leaf_area[:, 2] + my_terrain_grid[(my_leaf_area.height - my_leaf_area.leaf_area[:, 1].astype(int) - 1), my_leaf_area.leaf_area[:, 0].astype(int)] #- np.min(my_terrain.terrain[:, 2]) - leaf_area_min_z

        my_environment = Environment(my_leaf_area, my_terrain)

        my_terrain_output = attenuate_surface(my_environment, my_solar_position)

        my_3d_output = attenuate_all(my_environment, my_solar_position)

        # Compare terrain to 3d alg
        np.testing.assert_allclose(my_terrain_output.terrain_irradiance, my_3d_output.terrain_irradiance, atol=1e-5)
    
    @pytest.mark.skip()
    def test_flat_vs_all(self):
        my_datetime = datetime(2024, 6, 15, 9, 00)
        my_latitude = 40.
        my_solar_position = SolarPosition(my_datetime, my_latitude)

        target_width = 500
        target_height = 500
        my_terrain_grid = np.load("dev/data/kevin_terrain.npy")
        my_terrain_grid = my_terrain_grid[:target_height, :target_width]
        my_terrain = Terrain(my_terrain_grid)

        my_leaf_area_grid = np.load("dev/data/kevin_leaf_area.npy")
        my_leaf_area_grid = my_leaf_area_grid[:target_height, :target_width]
        my_leaf_area = LeafArea.from_uniformgrid(my_leaf_area_grid)
        my_leaf_area.leaf_area[:, 2] = my_leaf_area.leaf_area[:, 2] + my_terrain_grid[(my_leaf_area.height - my_leaf_area.leaf_area[:, 1].astype(int) - 1), my_leaf_area.leaf_area[:, 0].astype(int)] #- np.min(my_terrain.terrain[:, 2]) - leaf_area_min_z

        my_flat_environment = Environment(my_leaf_area)

        my_flat_output = attenuate_surface(my_flat_environment, my_solar_position)
        my_3d_output = attenuate_all(my_flat_environment, my_solar_position)

        np.testing.assert_allclose(my_flat_output.terrain_irradiance, my_3d_output.terrain_irradiance, atol=1e-5)

        

        

                          



        