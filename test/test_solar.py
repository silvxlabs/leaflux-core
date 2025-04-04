import pytest

from leaflux.dependencies import *
from leaflux.solar import *

class TestSolar:
    def test_init(self):
        # Test if ValueError raised for lat
        with pytest.raises(ValueError):
            SolarPosition(datetime(2024, 7, 15, 12, 00), -91., 100.)
        
        with pytest.raises(ValueError):
            SolarPosition(datetime(2024, 12, 30, 13, 00), 91., 100.)

        # Test if ValueError raised for long
        with pytest.raises(ValueError):
            SolarPosition(datetime(2024, 7, 15, 12, 00), 40., -181.)
        
        with pytest.raises(ValueError):
            SolarPosition(datetime(2024, 12, 30, 13, 00), 40., 181.)

        # Edge of long and lat
        sp1 = SolarPosition(datetime(2020, 7, 1, 19, 00), 90., 180.)
        assert sp1.latitude == 90.
        assert sp1.longitude == 180.

        sp2 = SolarPosition(datetime(2020, 1, 1, 19, 00), -90., -180.)
        assert sp2.latitude == -90.
        assert sp2.longitude == -180.

        # Test if ValueError raised for low solar angles
        with pytest.raises(ValueError):
            SolarPosition(datetime(2024, 12, 1, 4, 00), 40., -120.)

        # Test regular: Light vector normalized, light vector
        # 3 things...
        sp3 = SolarPosition(datetime(2024, 8, 1, 19, 00), 40., -120)
        assert np.linalg.norm(sp3.light_vector) == 1