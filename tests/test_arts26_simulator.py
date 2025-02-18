"""
Tests for the ARTS26 simulator.
"""
import pytest

from mwsim.data_loaders import Fascod
from mwsim.sensors import GMI
from mwsim.simulators import ARTS26Simulator


NEEDS_MWSIM_DATA_PATH = pytest.mark.skipif("MWSIM_DATA_PATH" not in os.environ, reason="Required data not available.")


@NEEDS_MWSIM_DATA_PATH
def test_simulator():

    fascod = Fascod(rwc=0e-4, swc=0e-4)
    profile = next(iter(fascod))

    sensor = GMI
    sim = ARTS26Simulator(sensor)
    sim.setup(sensor)

    sim.simulate(profile)
    results = sim.get_results()
    tbs = results.brightness_temperatures.data

    assert (100 < tbs).all()
    assert (tbs < 400).all()
