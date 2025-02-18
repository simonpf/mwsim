from mwsim.data_loaders import Fascod

from mwsim.sensors import GMI
from mwsim.simulators import ARTS26Simulator


def test_simulator():

    #fascod = Fascod(rwc=1e-4, swc=1e-4)
    fascod = Fascod(rwc=0e-4, swc=0e-4)
    profile = next(iter(fascod))

    sensor = GMI
    sim = ARTS26Simulator(sensor)
    sim.setup(sensor)

    #sim.setup(sensor)
    #tbs = sim.simulate_profile(profile)
    sim.simulate(profile)
    results = sim.get_results()
    tbs = results.brightness_temperatures.data

    assert (100 < tbs).all()
    assert (tbs < 400).all()
