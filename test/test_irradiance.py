from leaflux.irradiance import *

class TestIrradiance:
    def test_init(self):
        leaf_irr = np.load("test/data/leafarea_from_init_test_1.npy")
        terrain_irr = np.load("test/data/terrain_output300.npy")

        # Normal case
        ri1 = RelativeIrradiance(terrain_irr, leaf_irr)
        np.testing.assert_array_equal(ri1.terrain_irradiance, terrain_irr)
        np.testing.assert_array_equal(ri1.leaf_irradiance, leaf_irr)

        # No leaf irradiance
        ri2 = RelativeIrradiance(terrain_irr)
        np.testing.assert_array_equal(ri2.terrain_irradiance, terrain_irr)
        assert ri2.leaf_irradiance is None