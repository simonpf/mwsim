"""
Tests for the mwsim.data_loaders module.
"""
import numpy as np
import xarray as xr

from mwsim.geometry import lla_to_ecef
from mwsim.data_loaders import NetCDFDataLoader


def make_test_dataset():
    """
    Creates a test dataset of dimensions 100 x 100 x 100 with a surface altitude field calculated
    as z_sfc = 0.05 * lat + 0.05 * lon.
    """
    lons = np.arange(100)
    lats = np.arange(100)
    alt = np.arange(100)
    lons_2d, lats_2d = np.meshgrid(lons, lats, indexing="xy")
    alt, _, _ = np.meshgrid(alt, lats, lons, indexing="ij")

    dataset = xr.Dataset({
        "latitude": (("latitude"), lats),
        "longitude": (("longitude"), lons),
        "altitude": (("levels", "latitude", "longitude"), alt),
        "surface_altitude": (("latitude", "longitude"), 0.05 * lons_2d + 0.05 * lats_2d),
    })

    for sfc_var in ["10u", "10v", "2t", "skt"]:
        dataset[sfc_var] = (("latitude", "longitude"), np.ones((100, 100)))

    for atm_var in ["ciwc", "clwc", "crwc", "cswc", "pres", "t", "q"]:
        dataset[atm_var] = (("levels", "latitude", "longitude"), np.ones((100, 100, 100)))

    dataset["valid_time"] = np.datetime64("2020-01-01")

    return dataset


def test_netcdf_loader():
    """
    Test nearest-neighbor interpolation of NetCDF loader.
    """
    dataset = make_test_dataset()
    data_loader = NetCDFDataLoader(dataset)

    coords = np.ones((100, 3))
    coords[:, 0] = 50.0
    coords[:, 1] = 50.0
    coords[:, 2] = np.arange(100)

    profile = data_loader.get_profile(lla_to_ecef(coords))

    assert np.isclose(profile.latitude, 50.0)
    assert np.isclose(profile.longitude, 50.0)
    assert np.isclose(profile.altitude, np.arange(100)).all()
    assert np.isclose(profile.surface_altitude, 5.0)
