import pytest

from leaflux_core.dependencies import *
from leaflux_core.solar import *

class TestSolar:
    def test_init(self):
        # Test if ValueError raised for lat
        with pytest.raises(ValueError):
            SolarPosition(datetime(2024, 7, 15, 12, 00), -91.)
        
        with pytest.raises(ValueError):
            SolarPosition(datetime(2024, 12, 30, 13, 00), 91.)
        
        # Edge of latatude
        sp1 = SolarPosition(datetime(2020, 7, 1, 12, 00), 90.)
        assert sp1.latitude == 90.

        sp2 = SolarPosition(datetime(2020, 1, 1, 12, 00), -90.)
        assert sp2.latitude == -90.

        # Test if ValueError raised for low solar angles
        with pytest.raises(ValueError):
            SolarPosition(datetime(2024, 12, 1, 00, 00), 40.)

        # Test regular: Light vector normalized, light vector
        # 3 things...
        sp3 = SolarPosition(datetime(2024, 8, 1, 12, 00), 40.)
        assert np.linalg.norm(sp3.light_vector) == 1