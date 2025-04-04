from leaflux.irradiance import *

# Testing at this datetime/lat/long for all here for consistency:
# my_datetime = datetime(2024, 6, 15, 20, 00)
# my_latitude = 40.
# my_longitude = -120.

class TestIrradiance:
    def test_init(self):
        my_datetime = datetime(2024, 6, 15, 20, 00)
        my_latitude = 40.
        my_longitude = -120.
        my_solar_position = SolarPosition(my_datetime, my_latitude, my_longitude)

        leaf_irr = np.load("test/data/leafarea_from_init_test_1.npy")
        terrain_irr = np.load("test/data/terrain_output300.npy")

        # Normal case
        ri1 = Irradiance(solar_position=my_solar_position, terrain_irradiance=terrain_irr, canopy_irradiance=leaf_irr)
        np.testing.assert_array_equal(ri1.terrain_irradiance, terrain_irr)
        np.testing.assert_array_equal(ri1.canopy_irradiance, leaf_irr)

        # No leaf irradiance
        ri2 = Irradiance(solar_position=my_solar_position, terrain_irradiance=terrain_irr)
        np.testing.assert_array_equal(ri2.terrain_irradiance, terrain_irr)
        assert ri2.canopy_irradiance is None

    def test_to_srad(self):
        my_datetime = datetime(2024, 6, 15, 20, 00)
        my_latitude = 40.
        my_longitude = -120.
        my_solar_position = SolarPosition(my_datetime, my_latitude, my_longitude)

        leaf_irr = np.load("test/data/leafarea_from_init_test_1.npy")
        terrain_irr = np.load("test/data/terrain_output300.npy")
        sensor_irr = np.load("test/data/all_result_sensor.npy")

        ri_og = Irradiance(solar_position=my_solar_position, terrain_irradiance=terrain_irr, canopy_irradiance=leaf_irr, sensor_irradiance=sensor_irr)
        ri_unmod = Irradiance(solar_position=my_solar_position, terrain_irradiance=terrain_irr, canopy_irradiance=leaf_irr, sensor_irradiance=sensor_irr)
        ri_srad = ri_og.to_srad(800, 200, in_place=False)
        
        # Make sure that ri_mod was not modified when running to_srad on it
        np.testing.assert_equal(ri_unmod.canopy_irradiance, ri_og.canopy_irradiance)
        np.testing.assert_equal(ri_unmod.terrain_irradiance, ri_og.terrain_irradiance)
        np.testing.assert_equal(ri_unmod.sensor_irradiance, ri_og.sensor_irradiance)

        # Make sure that original and modded are different
        np.testing.assert_raises(AssertionError, np.testing.assert_equal, ri_og.canopy_irradiance, ri_srad.canopy_irradiance)
        
        # Make sure changing it is same as in place
        ri_og.to_srad(800, 200, in_place=True)
        np.testing.assert_equal(ri_srad.canopy_irradiance, ri_og.canopy_irradiance)
        np.testing.assert_equal(ri_srad.terrain_irradiance, ri_og.terrain_irradiance)
        np.testing.assert_equal(ri_srad.sensor_irradiance, ri_og.sensor_irradiance)


        ri_no_terrain_og = Irradiance(solar_position=my_solar_position, canopy_irradiance=leaf_irr)
        ri_no_terrain_unmod = Irradiance(solar_position=my_solar_position, canopy_irradiance=leaf_irr)
        ri_no_terrain = ri_no_terrain_og.to_srad(800, 200)

        # Make sure that ri_mod was not modified when running to_srad on it
        assert ri_no_terrain.terrain_irradiance is None 
        assert ri_no_terrain.sensor_irradiance is None
        np.testing.assert_equal(ri_no_terrain_unmod.canopy_irradiance, ri_no_terrain_og.canopy_irradiance)
        np.testing.assert_equal(ri_no_terrain_unmod.terrain_irradiance, ri_no_terrain_og.terrain_irradiance)
        np.testing.assert_equal(ri_no_terrain_unmod.sensor_irradiance, ri_no_terrain_og.sensor_irradiance)

        # Make sure changing it is same as in place
        ri_no_terrain_og.to_srad(800, 200, in_place=True)
        np.testing.assert_equal(ri_no_terrain.canopy_irradiance, ri_no_terrain_og.canopy_irradiance)
        np.testing.assert_equal(ri_no_terrain.terrain_irradiance, ri_no_terrain_og.terrain_irradiance)
        np.testing.assert_equal(ri_no_terrain.sensor_irradiance, ri_no_terrain_og.sensor_irradiance)

        # Different inputs produce different results
        ri_0 = Irradiance(solar_position=my_solar_position, terrain_irradiance=terrain_irr, canopy_irradiance=leaf_irr, sensor_irradiance=sensor_irr)
        ri_1 = Irradiance(solar_position=my_solar_position, terrain_irradiance=terrain_irr, canopy_irradiance=leaf_irr, sensor_irradiance=sensor_irr)
        ri_0.to_srad(1000, 150, in_place=True)
        ri_1.to_srad(200, 50, in_place=True)
        np.testing.assert_raises(AssertionError, np.testing.assert_equal, ri_0, ri_1)


    def test_to_par(self):
            my_datetime = datetime(2024, 6, 15, 20, 00)
            my_latitude = 40.
            my_longitude = -120.
            my_solar_position = SolarPosition(my_datetime, my_latitude, my_longitude)

            leaf_irr = np.load("test/data/leafarea_from_init_test_1.npy")
            terrain_irr = np.load("test/data/terrain_output300.npy")
            sensor_irr = np.load("test/data/all_result_sensor.npy")

            ri_og = Irradiance(solar_position=my_solar_position, terrain_irradiance=terrain_irr, canopy_irradiance=leaf_irr, sensor_irradiance=sensor_irr)
            ri_unmod = Irradiance(solar_position=my_solar_position, terrain_irradiance=terrain_irr, canopy_irradiance=leaf_irr, sensor_irradiance=sensor_irr)
            ri_par = ri_og.to_par(800, 200, in_place=False)
            
            # Make sure that ri_mod was not modified when running to_par on it
            np.testing.assert_equal(ri_unmod.canopy_irradiance, ri_og.canopy_irradiance)
            np.testing.assert_equal(ri_unmod.terrain_irradiance, ri_og.terrain_irradiance)
            np.testing.assert_equal(ri_unmod.sensor_irradiance, ri_og.sensor_irradiance)

            # Make sure that original and modded are different
            np.testing.assert_raises(AssertionError, np.testing.assert_equal, ri_og.canopy_irradiance, ri_par.canopy_irradiance)
            
            # Make sure changing it is same as in place
            ri_og.to_par(800, 200, in_place=True)
            np.testing.assert_equal(ri_par.canopy_irradiance, ri_og.canopy_irradiance)
            np.testing.assert_equal(ri_par.terrain_irradiance, ri_og.terrain_irradiance)
            np.testing.assert_equal(ri_par.sensor_irradiance, ri_og.sensor_irradiance)


            ri_no_terrain_og = Irradiance(solar_position=my_solar_position, canopy_irradiance=leaf_irr)
            ri_no_terrain_unmod = Irradiance(solar_position=my_solar_position, canopy_irradiance=leaf_irr)
            ri_no_terrain = ri_no_terrain_og.to_par(800, 200)
            print(ri_no_terrain.canopy_irradiance)

            # Make sure that ri_mod was not modified when running to_par on it
            assert ri_no_terrain.terrain_irradiance is None 
            assert ri_no_terrain.sensor_irradiance is None
            np.testing.assert_equal(ri_no_terrain_unmod.canopy_irradiance, ri_no_terrain_og.canopy_irradiance)
            np.testing.assert_equal(ri_no_terrain_unmod.terrain_irradiance, ri_no_terrain_og.terrain_irradiance)
            np.testing.assert_equal(ri_no_terrain_unmod.sensor_irradiance, ri_no_terrain_og.sensor_irradiance)

            # Make sure changing it is same as in place
            ri_no_terrain_og.to_par(800, 200, in_place=True)
            np.testing.assert_equal(ri_no_terrain.canopy_irradiance, ri_no_terrain_og.canopy_irradiance)
            np.testing.assert_equal(ri_no_terrain.terrain_irradiance, ri_no_terrain_og.terrain_irradiance)
            np.testing.assert_equal(ri_no_terrain.sensor_irradiance, ri_no_terrain_og.sensor_irradiance)

            ri_0 = Irradiance(solar_position=my_solar_position, terrain_irradiance=terrain_irr, canopy_irradiance=leaf_irr, sensor_irradiance=sensor_irr)
            ri_1 = Irradiance(solar_position=my_solar_position, terrain_irradiance=terrain_irr, canopy_irradiance=leaf_irr, sensor_irradiance=sensor_irr)
            ri_0.to_par(700, 200, in_place=True)
            ri_1.to_par(700, 200, par_ratio=0.4, in_place=True)
            np.testing.assert_raises(AssertionError, np.testing.assert_equal, ri_0, ri_1)

