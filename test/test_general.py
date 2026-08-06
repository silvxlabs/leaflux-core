import pytest 

from leaflux.dependencies import *
from leaflux.general import _get_rot_mat, _attenuate_surface_flat, _attenuate_surface_terrain, attenuate_surface, attenuate_all, _calculate_g_from_mla, plot_irradiance, _calculate_d
from leaflux.solar import *
from leaflux.environment import *

# MLA (radians) that gives x = 1.0. Plug 1.0 into Campbell 1990 eq 16
SPHERICAL_MLA = 9.65 * 4.0**(-1.65)

# Cross-algorithm comparison tolerances (see notes around _compare_terrain)
SUM_RTOL = 0.085 # relative diff in total terrain irradiance (primary)
ELEMENTWISE_ATOL = 0.1 # per-cell terrain irradiance diff
CELL_FRAC = 0.80 # min fraction of cells that must agree within ELEMENTWISE_ATOL

@pytest.fixture
def test_env():
    # Shared setup matching the env used to generate the stored no-MLA grids (test_attenuate_all)
    sol = SolarPosition(datetime(2024, 6, 15, 20, 00), 40., -120.)

    terrain_grid = np.load("test/data/terrain_input300.npy")
    terrain = Terrain(terrain_grid)

    leaf_area = LeafArea.from_uniformgrid(np.load("test/data/leaf_area_grid.npy"))
    leaf_area.leaf_area[:, 2] += terrain_grid[
        (leaf_area.height - leaf_area.leaf_area[:, 1] - 1).astype(int),
        leaf_area.leaf_area[:, 0].astype(int),
    ]  # z adjust for terrain

    sensors = [
        Sensor(150, 150, 50, 0, sol.azimuth),
        Sensor(33, 33, 50),
        Sensor(44, 44, 55, sol.zenith, sol.azimuth),
    ]
    return sol, leaf_area, terrain, sensors

def _build_compare_env(terrain_grid):
    W, H, Z = 30, 30, 12
    sol = SolarPosition(datetime(2024, 6, 15, 20, 00), 40., -120.)
    terrain = Terrain(terrain_grid)

    rng = np.random.default_rng(0)
    leaf_area_grid = np.zeros((H, W, Z), dtype=np.float32)
    leaf_area_grid[8:22, 8:22, 6:11] = rng.uniform(0.1, 0.6, size=(14, 14, 5)).astype(np.float32)
    leaf_area = LeafArea.from_uniformgrid(leaf_area_grid)

    # Raise canopy onto terrain (adds 0 for flat terrain)
    leaf_area.leaf_area[:, 2] += terrain_grid[
        (leaf_area.height - leaf_area.leaf_area[:, 1] - 1).astype(int),
        leaf_area.leaf_area[:, 0].astype(int),
    ]

    mla_grid = leaf_area.leaf_area.copy()
    mla_grid[:, 3] = np.radians(35.0)  # single arbitrary MLA shared by all three algos
    leaf_angle = LeafAngle(mla_grid, width=W, height=H)

    return sol, leaf_area, terrain, leaf_angle, (1.0, 1.0, 1.0)

@pytest.fixture
def compare_env(): # hilly terrain
    W = H = 30
    yy, xx = np.mgrid[0:H, 0:W]
    terrain_grid = (6.0 * np.exp(-(((xx - W / 2) ** 2 + (yy - H / 2) ** 2)
                                   / (2 * (W / 5) ** 2)))).astype(np.float64)
    return _build_compare_env(terrain_grid)

@pytest.fixture
def compare_env_flat(): # flat terrain at z = 0
    W = H = 30
    return _build_compare_env(np.zeros((H, W), dtype=np.float64))

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
        my_datetime = datetime(2024, 6, 15, 20, 00)
        my_latitude = 40.
        my_longitude = -120.
        my_solar_position = SolarPosition(my_datetime, my_latitude, my_longitude)

        leaf_area_grid = np.load("test/data/leaf_area_grid.npy")
        my_leaf_area = LeafArea.from_uniformgrid(leaf_area_grid)

        my_flat_env = Environment(my_leaf_area)
        sf1 = attenuate_surface(my_flat_env, my_solar_position)

        SAVE_NEW = False
        if SAVE_NEW:
            np.save("test/data/flat_result_1.npy", sf1.terrain_irradiance)

        ASSERT = True
        if ASSERT:
            expected = np.load("test/data/flat_result_1.npy")
            
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
        my_datetime = datetime(2024, 6, 15, 20, 00)
        my_latitude = 40.
        my_longitude = -120.
        my_solar_position = SolarPosition(my_datetime, my_latitude, my_longitude)

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

        SAVE_NEW = False
        if SAVE_NEW:
            np.save("test/data/terrain_result_1.npy", st1.terrain_irradiance)

        ASSERT = True
        if ASSERT:
            expected = np.load("test/data/terrain_result_1.npy")

            actual = st1.terrain_irradiance

            errors = np.abs(expected - actual)

            error_indices = np.where(errors > 0)[0]

            errors_above_1 = np.sum(errors >= 1.0)

            # print("Indices of errors:", error_indices)
            # print("Total errors: ", len(error_indices))
            # print("Number of errors with difference >= 1.0:", errors_above_1)

            expected_sum = np.sum(expected)
            actual_sum = np.sum(actual)

            # print("Expected sum: ", expected_sum)
            # print("Actual sum: ", actual_sum)

            # Test against expected
            assert (np.abs(actual_sum - expected_sum) / expected_sum) < 0.25

            # Test against lowered and raised 
            np.testing.assert_equal(st1.terrain_irradiance, st2.terrain_irradiance)
            np.testing.assert_equal(st1.terrain_irradiance, st3.terrain_irradiance)

            # np.testing.assert_allclose(expected, actual, atol=1e-6)
    
    def test_surface_flat_d_applied(self, test_env):
        # d must enter the exponent as a scalar multiplier, so a voxel result
        # equals the default (d=1) result raised to the power d.
        sol, leaf_area, terrain, sensors = test_env
        sol.light_vector = np.array([0.0, 0.0, -1.0])

        # (1,1,2) straight down: A_perp = |vz|*x*y = 1, V = 2 -> d = 2
        expected_d = 2.0
        voxel_env = Environment(leaf_area=leaf_area, voxel_dim=(1.0, 1.0, 2.0))
        default_env = Environment(leaf_area=leaf_area)

        voxel_irr = _attenuate_surface_flat(voxel_env, sol, extn=0.5).terrain_irradiance
        default_irr = _attenuate_surface_flat(default_env, sol, extn=0.5).terrain_irradiance

        np.testing.assert_allclose(voxel_irr, default_irr ** expected_d, rtol=1e-6)
        # d != 1 should actually change something (guard against a no-op pass)
        assert not np.allclose(voxel_irr, default_irr)

    def test_surface_flat_mla_spherical_matches_default(self, test_env):
        # Spherical leaf distribution gives G = 0.5 for all zenith angles, so
        # supplying spherical MLA must match the default path with extn = 0.5.
        sol, leaf_area, terrain, sensors = test_env
        sol.light_vector = np.array([0.0, 0.0, -1.0])

        mla_grid = leaf_area.leaf_area.copy()
        mla_grid[:, 3] = SPHERICAL_MLA
        leaf_angle = LeafAngle(mla_grid, width=300, height=300)

        # extn deliberately != 0.5 so a broken MLA path would fall back to 0.9 and fail
        mla_env = Environment(leaf_area=leaf_area, leaf_angle=leaf_angle)
        default_env = Environment(leaf_area=leaf_area)

        mla_irr = _attenuate_surface_flat(mla_env, sol, extn=0.9).terrain_irradiance
        default_irr = _attenuate_surface_flat(default_env, sol, extn=0.5).terrain_irradiance

        np.testing.assert_allclose(mla_irr, default_irr, rtol=1e-6)

    def test_surface_terrain_d_applied(self, compare_env_flat):
        # Straight-down sun -> R = I.
        sol, leaf_area, terrain, leaf_angle, voxel_dim = compare_env_flat # Use flat terrain so d holds
        sol.light_vector = np.array([0.0, 0.0, -1.0])

        expected_d = 2.0  # (1,1,2) straight down
        voxel_env = Environment(leaf_area=leaf_area, terrain=terrain, voxel_dim=(1.0, 1.0, 2.0))
        default_env = Environment(leaf_area=leaf_area, terrain=terrain)

        voxel_irr = _attenuate_surface_terrain(voxel_env, sol, extn=0.5).terrain_irradiance
        default_irr = _attenuate_surface_terrain(default_env, sol, extn=0.5).terrain_irradiance

        np.testing.assert_allclose(voxel_irr, default_irr ** expected_d, rtol=1e-6)
        assert not np.allclose(voxel_irr, default_irr)

    def test_surface_terrain_mla_spherical_matches_default(self, test_env):
        # Spherical MLA -> G = 0.5, must match default extn = 0.5, terrain path.
        sol, leaf_area, terrain, sensors = test_env
        sol.light_vector = np.array([0.0, 0.0, -1.0])

        mla_grid = leaf_area.leaf_area.copy()
        mla_grid[:, 3] = SPHERICAL_MLA
        leaf_angle = LeafAngle(mla_grid, width=300, height=300)

        mla_env = Environment(leaf_area=leaf_area, terrain=terrain, leaf_angle=leaf_angle)
        default_env = Environment(leaf_area=leaf_area, terrain=terrain)

        mla_irr = _attenuate_surface_terrain(mla_env, sol, extn=0.9).terrain_irradiance
        default_irr = _attenuate_surface_terrain(default_env, sol, extn=0.5).terrain_irradiance

        np.testing.assert_allclose(mla_irr, default_irr, rtol=1e-6)
        
    def test_attenuate_surface(self):
        my_datetime = datetime(2024, 6, 15, 20, 00)
        my_latitude = 40.
        my_longitude = -120.
        my_solar_position = SolarPosition(my_datetime, my_latitude, my_longitude)

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
        my_datetime = datetime(2024, 6, 15, 20, 00)
        my_latitude = 40.
        my_longitude = -120.
        my_solar_position = SolarPosition(my_datetime, my_latitude, my_longitude)

        my_terrain_grid = np.load("test/data/terrain_input300.npy")
        my_terrain = Terrain(my_terrain_grid)

        leaf_area_grid = np.load("test/data/leaf_area_grid.npy")
        my_leaf_area = LeafArea.from_uniformgrid(leaf_area_grid)
        my_leaf_area.leaf_area[:, 2] = my_leaf_area.leaf_area[:, 2] + my_terrain_grid[(my_leaf_area.height - my_leaf_area.leaf_area[:, 1] - 1).astype(int), my_leaf_area.leaf_area[:, 0].astype(int)]

        my_sensor_0 = Sensor(150, 150, 50, 0, my_solar_position.azimuth)
        my_sensor_1 = Sensor(33, 33, 50)
        my_sensor_2 = Sensor(44, 44, 55, my_solar_position.zenith, my_solar_position.azimuth)
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

            # print("Terrain errors: ", len(error_indices_terr))
            # print("Caanopy errors: ", len(error_indices_canopy))
            # print("Total errors: ", len(error_indices_terr)+len(error_indices_canopy))
            # print("Terrain errors over 1: ", errors_above_1_terr)
            # print("Canopy errors over 1: ", errors_above_1_canopy)
            # print("Number of errors with difference >= 1.0:", errors_above_1_terr + errors_above_1_canopy)

            expected_sum_terr = np.sum(expected_terr)
            expected_sum_canopy = np.sum(expected_canopy[:, 3])
            actual_sum_terr = np.sum(actual_terr)
            actual_sum_canopy = np.sum(actual_canopy[:, 3])

            # print("Expected sum terr: ", expected_sum_terr, " Actual sum terr: ", actual_sum_terr, "Terr sum diff: ", expected_sum_terr-actual_sum_terr)
            # print("Expected sum canopy: ", expected_sum_canopy, " Actual sum canopy: ", actual_sum_canopy, " Canopy sum diff: ", expected_sum_canopy-actual_sum_canopy)
            # print("Total expected sum: ", expected_sum_terr + expected_sum_canopy)
            # print("Total actual sum: ", actual_sum_terr + actual_sum_canopy)

            # No terrain
            expected_canopy_flat = np.load("test/data/all_result_canopy_flat.npy")
            actual_canopy_flat = result_flat.canopy_irradiance
            expected_sum_flat_canopy = np.sum(expected_canopy_flat)
            actual_sum_flat_canopy = np.sum(actual_canopy_flat)
            # print("* FLAT * Expected sum canopy: ", expected_sum_flat_canopy, " Actual sum canopy: ", actual_sum_flat_canopy, " Canopy sum diff: ", expected_sum_flat_canopy-actual_sum_flat_canopy)

            # Testing against expected result
            np.testing.assert_allclose(expected_canopy, actual_canopy, atol=1e-6)
            np.testing.assert_allclose(expected_terr, actual_terr, atol=1e-6)
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
            assert result.get_sensor_irradiance(my_sensor_1) == result.get_sensor_irradiance(my_sensor_2)

            expected_sensor = np.load("test/data/all_result_sensor.npy")
            np.testing.assert_allclose(expected_sensor, result.sensor_irradiance, atol=1e-6)
            np.testing.assert_equal(expected_sensor, result.sensor_irradiance)

    def test_calculate_g_from_mla(self):
        # Unit test against Campbell 1989 Table 1.1 ground truth at zenith 0.29 rad
        zenith = 0.29
        mla = np.array([np.radians(35.0), SPHERICAL_MLA, np.radians(75.0), np.nan])  # rad
        leaf_g = np.full_like(mla, fill_value=0.5, dtype=np.float32)                 # default fill

        _calculate_g_from_mla(mla, leaf_g, zenith)  # mutates leaf_g in place

        assert leaf_g[0] == pytest.approx(0.74065, abs=1e-4)  # planophile 35 deg
        assert leaf_g[1] == pytest.approx(0.5, abs=1e-6)      # spherical -> 0.5
        assert leaf_g[2] == pytest.approx(0.27016, abs=1e-4)  # erectophile 75 deg
        assert leaf_g[3] == 0.5                               # nan left at default

        # Planophile intercepts most, erectophile least, at near-overhead sun
        assert leaf_g[0] > leaf_g[1] > leaf_g[2]

    def test_mla_changes_result(self, test_env):
        # A (non-spherical) MLA grid must diverge from the stored no-MLA result
        sol, leaf_area, terrain, sensors = test_env

        mla_grid = leaf_area.leaf_area.copy()
        mla_grid[:, 3] = np.radians(35.0)  # planophile, rad
        leaf_angle = LeafAngle(mla_grid, width=300, height=300)

        env = Environment(leaf_area, terrain=terrain, sensors=sensors, leaf_angle=leaf_angle)
        result = attenuate_all(env, sol)

        no_mla_canopy = np.load("test/data/all_result_canopy_1.npy")
        no_mla_terr = np.load("test/data/all_result_terr_1.npy")

        assert not np.allclose(result.canopy_irradiance, no_mla_canopy)
        assert not np.allclose(result.terrain_irradiance, no_mla_terr)

    def test_mla_spherical_matches_default(self, test_env):
        # Spherical MLA -> G = 0.5, so it must reproduce the stored no-MLA (default 0.5) result
        sol, leaf_area, terrain, sensors = test_env

        mla_grid = leaf_area.leaf_area.copy()
        mla_grid[:, 3] = SPHERICAL_MLA  # rad
        leaf_angle = LeafAngle(mla_grid, width=300, height=300)

        env = Environment(leaf_area, terrain=terrain, sensors=sensors, leaf_angle=leaf_angle)
        result = attenuate_all(env, sol)

        no_mla_canopy = np.load("test/data/all_result_canopy_1.npy")
        no_mla_terr = np.load("test/data/all_result_terr_1.npy")

        np.testing.assert_allclose(result.canopy_irradiance, no_mla_canopy, atol=1e-6)
        np.testing.assert_allclose(result.terrain_irradiance, no_mla_terr, atol=1e-6)

    def test_mla_against_stored_result(self, test_env):
        sol, leaf_area, terrain, sensors = test_env

        mla_grid = leaf_area.leaf_area.copy()
        mla_grid[:, 3] = SPHERICAL_MLA # Using MLA that will give spherical
        leaf_angle = LeafAngle(mla_grid, width=300, height=300)

        env = Environment(leaf_area, terrain=terrain, sensors=sensors, leaf_angle=leaf_angle)
        result = attenuate_all(env, sol)

        SAVE_NEW = False
        if SAVE_NEW:
            np.save("test/data/mla_spherical_result_terr.npy", result.terrain_irradiance)
            np.save("test/data/mla_spherical_result_canopy.npy", result.canopy_irradiance)
            np.save("test/data/mla_spherical_result_sensor.npy", result.sensor_irradiance)

        ASSERT = True
        if ASSERT:
            expected_terr = np.load("test/data/mla_spherical_result_terr.npy")
            expected_canopy = np.load("test/data/mla_spherical_result_canopy.npy")

            actual_terr = result.terrain_irradiance
            actual_canopy = result.canopy_irradiance

            np.testing.assert_allclose(actual=actual_canopy, desired=expected_canopy, atol=1e-6)
            np.testing.assert_allclose(actual=actual_terr, desired=expected_terr, atol=1e-6)
            np.testing.assert_equal(actual=actual_terr, desired=expected_terr)
            np.testing.assert_equal(actual=actual_canopy, desired=expected_canopy)

    def test_1m_down_cube_against_default(self, test_env):
        # Checking that providing a voxel of 1x1x1 with a solar vector pointing straight
        # down results in the same thing as not providing a voxel dim (path length should
        # be 1.0 in voxel provided case)
        sol, leaf_area, terrain, sensors = test_env

        # Set to straight down
        sol.light_vector = np.array([0, 0, -1])

        mla_grid = leaf_area.leaf_area.copy()
        mla_grid[:, 3] = SPHERICAL_MLA # Using MLA that will give spherical
        leaf_angle = LeafAngle(mla_grid, width=300, height=300)

        # x, y, z 1m^3
        voxel_dim = tuple((1.0, 1.0, 1.0))

        voxel_env = Environment(leaf_area=leaf_area, terrain=terrain, voxel_dim=voxel_dim)
        no_voxel_env = Environment(leaf_area=leaf_area, terrain=terrain)

        voxel_result = attenuate_all(env=voxel_env, sol=sol)
        no_voxel_result = attenuate_all(env=no_voxel_env, sol=sol)

        voxel_terrain_result = voxel_result.terrain_irradiance
        voxel_terrain_sum = np.sum(voxel_terrain_result)
        voxel_canopy_result = voxel_result.canopy_irradiance
        voxel_canopy_sum = np.sum(voxel_canopy_result)

        no_voxel_terrain_result = no_voxel_result.terrain_irradiance
        no_voxel_terrain_sum = np.sum(no_voxel_terrain_result)
        no_voxel_canopy_result = no_voxel_result.canopy_irradiance
        no_voxel_canopy_sum = np.sum(no_voxel_canopy_result)

        # Test that these are same
        np.testing.assert_allclose(actual=voxel_terrain_result, desired=no_voxel_terrain_result, atol=1e-6)
        np.testing.assert_allclose(actual=voxel_canopy_result, desired=no_voxel_canopy_result, atol=1e-6)
        np.testing.assert_equal(actual=voxel_terrain_result, desired=no_voxel_terrain_result)
        np.testing.assert_equal(actual=voxel_canopy_result, desired=no_voxel_canopy_result) 

    def test_path_length_calculator_1m_down(self, test_env):
        # Test that path length through a 1m^3 voxel with a straight
        # down solar vector is 1.0
        sol, leaf_area, terrain, sensors = test_env

        # Solar vector that points straight down
        sol.light_vector = np.array([0, 0, -1])

        # Voxel is 1x1x1
        voxel_dim = tuple((1.0, 1.0, 1.0))

        d = _calculate_d(solar_vector=sol.light_vector, voxel_dim=voxel_dim)

        assert d == 1.0

    def test_path_length_calculator_variety(self, test_env):
        # Tests path length calculator against a couple hand calculated/geometry based results
        sol, leaf_area, terrain, sensors = test_env

        # (voxel_dim, raw solar vector, hand-calculated d = V / A_perp)
        cases = [
            # (2, 2, 1), V = 4
            ((2.0, 2.0, 1.0), (0.0, 0.0, -1.0), 1.0),                # down -> z
            ((2.0, 2.0, 1.0), (1.0, 0.0,  0.0), 2.0),                # along x -> x
            ((2.0, 2.0, 1.0), (1.0, 0.0, -1.0), 2*np.sqrt(2)/3),     # 45deg xz ~0.94281
            ((2.0, 2.0, 1.0), (1.0, 1.0, -1.0), np.sqrt(3)/2),       # diagonal ~0.86603
            # (2, 2, 2), V = 8
            ((2.0, 2.0, 2.0), (0.0, 0.0, -1.0), 2.0),                # down -> z
            ((2.0, 2.0, 2.0), (1.0, 0.0,  0.0), 2.0),                # along x -> x
            ((2.0, 2.0, 2.0), (1.0, 0.0, -1.0), np.sqrt(2)),         # 45deg xz ~1.41421
            ((2.0, 2.0, 2.0), (1.0, 1.0, -1.0), 2/np.sqrt(3)),       # diagonal ~1.15470
            # (3, 3, 1), V = 9
            ((3.0, 3.0, 1.0), (0.0, 0.0, -1.0), 1.0),                # down -> z
            ((3.0, 3.0, 1.0), (1.0, 0.0,  0.0), 3.0),                # along x -> x
            ((3.0, 3.0, 1.0), (1.0, 0.0, -1.0), 3*np.sqrt(2)/4),     # 45deg xz ~1.06066
            ((3.0, 3.0, 1.0), (1.0, 1.0, -1.0), 9/(5*np.sqrt(3))),   # diagonal ~1.03923
        ]

        for voxel_dim, vec, expected in cases:
            v = np.array(vec, dtype=float)
            v /= np.linalg.norm(v)  # _calculate_d expects a normalized vector
            d = _calculate_d(solar_vector=v, voxel_dim=voxel_dim)
            np.testing.assert_allclose(
                d, expected, atol=1e-6, err_msg=f"voxel={voxel_dim}, vec={vec}"
            )

    # The below tests focus on testing the algorithms against each other
    # They do not match exactly, but I want to track difference

    # Latest print (run with pytest test -s to see)
    # .flat vs surface (flat terrain) 
    # sums: 737.5117 vs 683.5469 (rel 0.0789), max cell diff: 0.9581, cells within 0.1: 0.806
    # within: 0.8055555555555556 min frac: 0.8

    # .cos(zenith) = 0.9581
    # flat vs all (flat terrain) 
    # sums: 737.5117 vs 735.3042 (rel 0.0030), max cell diff: 0.7495, cells within 0.1: 0.862
    # within: 0.8622222222222222 min frac: 0.8

    # .cos(zenith) = 0.9581
    # surface vs all (flat terrain) 
    # sums: 683.5469 vs 735.3042 (rel 0.0704), max cell diff: 0.9581, cells within 0.1: 0.933
    # within: 0.9333333333333333 min frac: 0.8

    # .surface vs all (hilly terrain) 
    # sums: 633.0236 vs 691.5184 (rel 0.0846), max cell diff: 0.9964, cells within 0.1: 0.902
    # within: 0.9022222222222223 min frac: 0.8

    @staticmethod
    def _compare_terrain(a, b, label="", sum_rtol=SUM_RTOL,
                         atol=ELEMENTWISE_ATOL, min_frac=CELL_FRAC):
        sum_a, sum_b = np.sum(a), np.sum(b)
        rel = abs(sum_a - sum_b) / sum_b
        within = np.mean(np.abs(a - b) <= atol)
        print(f"{label} \nsums: {sum_a:.4f} vs {sum_b:.4f} (rel {rel:.4f}), "
              f"max cell diff: {np.max(np.abs(a - b)):.4f}, "
              f"cells within {atol}: {within:.3f}\n",
              f"within: {within} min frac: {min_frac}\n")
        np.testing.assert_allclose(sum_a, sum_b, rtol=sum_rtol) # primary
        assert within >= min_frac # robust per-cell

    # Flat vs terrain surface with flat DEM
    def test_flat_vs_terrain_surface_flatterrain(self, compare_env_flat):
        sol, leaf_area, terrain, leaf_angle, voxel_dim = compare_env_flat
        flat_env = Environment(leaf_area, leaf_angle=leaf_angle, voxel_dim=voxel_dim)
        surf_env = Environment(leaf_area, terrain=terrain, leaf_angle=leaf_angle, voxel_dim=voxel_dim)

        flat_irr = _attenuate_surface_flat(flat_env, sol, extn=0.5).terrain_irradiance
        surf_irr = _attenuate_surface_terrain(surf_env, sol, extn=0.5).terrain_irradiance
        self._compare_terrain(flat_irr, surf_irr, label="flat vs surface (flat terrain)")

    # Flat vs all with flat DEM
    def test_flat_vs_all_flatterrain(self, compare_env_flat):
        sol, leaf_area, terrain, leaf_angle, voxel_dim = compare_env_flat
        flat_env = Environment(leaf_area, leaf_angle=leaf_angle, voxel_dim=voxel_dim)
        all_env = Environment(leaf_area, terrain=terrain, leaf_angle=leaf_angle, voxel_dim=voxel_dim)

        flat_irr = _attenuate_surface_flat(flat_env, sol, extn=0.5).terrain_irradiance
        all_irr = attenuate_all(all_env, sol).terrain_irradiance
        print(f"cos(zenith) = {np.cos(sol.zenith):.4f}")
        self._compare_terrain(flat_irr, all_irr, label="flat vs all (flat terrain)")

    # Terrain surface vs all with flat DEM
    def test_terrain_surface_vs_all_flatterrain(self, compare_env_flat):
        sol, leaf_area, terrain, leaf_angle, voxel_dim = compare_env_flat
        env = Environment(leaf_area, terrain=terrain, leaf_angle=leaf_angle, voxel_dim=voxel_dim)

        surf_irr = _attenuate_surface_terrain(env, sol, extn=0.5).terrain_irradiance
        all_irr = attenuate_all(env, sol).terrain_irradiance
        print(f"cos(zenith) = {np.cos(sol.zenith):.4f}")
        self._compare_terrain(surf_irr, all_irr, label="surface vs all (flat terrain)")

    # Terrain surface vs all with hilly generated terrain
    def test_terrain_surface_vs_all_hillyterrain(self, compare_env):
        sol, leaf_area, terrain, leaf_angle, voxel_dim = compare_env
        env = Environment(leaf_area, terrain=terrain, leaf_angle=leaf_angle, voxel_dim=voxel_dim)

        surf_irr = _attenuate_surface_terrain(env, sol, extn=0.5).terrain_irradiance
        all_irr = attenuate_all(env, sol).terrain_irradiance
        self._compare_terrain(surf_irr, all_irr, label="surface vs all (hilly terrain)")