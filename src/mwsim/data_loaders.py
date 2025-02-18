"""
mwsim.data_loaders
==================

This module provides data loaders for various input datasets. The data loaders
are responsible for loading the profile data along pencil beams through the
atmosphere.
"""
from dataclasses import dataclass
from functools import cached_property
import os
from typing import List

from pathlib import Path
import xarray as xr

import numpy as np
from pykdtree.kdtree import KDTree
from pyarts.xml import load

from mwsim import config
from mwsim.geometry import lla_to_ecef
from mwsim.utils import calculate_altitude, h2o_vmr2mmr

@dataclass
class ProfileData:
    """
    The ProfileData class holds all data required to simulate brightness temperatures along a pencil beam.
    """
    longitude: float
    latitude: float
    time: np.datetime64

    surface_altitude: float
    skin_temperature: float
    two_meter_temperature: float
    wind_speed: float

    pressure: np.ndarray
    altitude: np.ndarray
    temperature: np.ndarray
    humidity: np.ndarray
    clwc: np.ndarray
    ciwc: np.ndarray
    swc: np.ndarray
    rwc: np.ndarray


class Fascod:
    """
    The Fascod data loader is a very simple data loader for testing purposes.
    """
    atmosphere_path: str = "planets/Earth/Fascod/tropical/"

    def __init__(
            self,
            longitude: float = 0.0,
            latitude: float = 0.0,
            rwc:float = 0.0,
            swc: float = 0.0,
            clwc: float = 0.0
    ):
        """
        Args:
             longitude: The longitude of the profile, defaults to 0.0.
             latitude: The latitude of the profile, defaults to 0.0
             rwc: The rain-water content in the profile.
             swc: The snow-water content in the profile
             clwc: The cloud-liquid water content in the profile.
        """
        if not "ARTS_DATA_PATH" in os.environ:
            raise ValueError(
                "ARTS_DATA_PATH environment variable mustb set to load Fascod data."
            )
        self.time = np.datetime64("2020-01-01T00:00:00")
        self.longitude = longitude
        self.latitude = latitude
        self.rwc = rwc
        self.swc = swc
        self.clwc = clwc

    def __iter__(self):
        """
        Returns a single profile.
        """
        arts_data_path = config.get_data_path() / "arts-xml-data-2.6.14"
        data_path = arts_data_path / self.atmosphere_path

        h2o_vmr = load(str(data_path / "tropical.H2O.xml"))
        pressure = h2o_vmr.get_grid(0)
        h2o_mmr = h2o_vmr2mmr(np.array(h2o_vmr.data))[:, 0, 0]
        alt = np.array(load(str(data_path / "tropical.z.xml")).data).squeeze()
        temperature = np.array(load(str(data_path / "tropical.t.xml")).data).squeeze()
        skin_temperature = temperature[0]
        two_meter_temperature = temperature[0]
        wind_speed = 0.0

        clwc = np.zeros_like(temperature)
        ciwc = np.zeros_like(temperature)
        swc = np.zeros_like(temperature)
        rwc = np.zeros_like(temperature)

        freezing_layer = np.where(temperature < 273.15)[0][0]
        swc[freezing_layer:freezing_layer + 5] = self.swc
        rwc[:freezing_layer] = self.rwc
        clwc[:freezing_layer] = self.clwc

        yield ProfileData(
            longitude=self.longitude,
            latitude=self.latitude,
            time=self.time,
            surface_altitude=0.0,
            skin_temperature=skin_temperature,
            two_meter_temperature=two_meter_temperature,
            wind_speed=wind_speed,
            pressure=pressure,
            altitude=alt,
            temperature=temperature,
            humidity=h2o_mmr,
            clwc=clwc,
            ciwc=ciwc,
            swc=swc,
            rwc=rwc
        )


def find_var_file(path: Path, var: str) -> Path:
    """
    Find variable file in data folder.

    Args:
        path: Path to the directory containing the input file.
        var: The name of the variable file to load.

    Return:
        An xr.Dataset containing the loaded data.

    """
    files = sorted(list(path.glob(f"{var}_*.grib")))
    if len(files) == 0:
        raise ValueError(
            f"Could not find input file for variable {var} in path {path}."
        )
    if len(files) > 1:
        raise ValueError(
            f"Found more than 1 file for variable {var} in path {path}."
        )
    return files[0]


class NetCDFDataLoader:
    """
    Generic data loader that loads profile data from 3D gridded atmosphere data using nearest-neighbor
    interpolation.
    """
    def __init__(self, data: xr.Dataset):

        self.data = data

        lons = data.longitude.data
        lats = data.latitude.data
        lons, lats = np.meshgrid(lons, lats, indexing="xy")
        altitude = data.altitude.data
        lons = np.broadcast_to(lons[None], altitude.shape)
        lats = np.broadcast_to(lats[None], altitude.shape)

        shape = altitude.shape
        coords = lla_to_ecef(np.stack([lons.flatten(), lats.flatten(), altitude.flatten()], -1))

        self.shape = shape
        self.tree = KDTree(coords)

        z_surf = data.surface_altitude.data
        surf_coords = lla_to_ecef(np.stack([lons[-1].flatten(), lats[-1].flatten(), z_surf.flatten()], -1))
        self.surface_tree = KDTree(surf_coords)


    def get_profile(self, beam_coords: np.ndarray, max_dist: float = 3e3) -> ProfileData:
        """
        Calculate atmospheric profile along beam.

        Args:
            beam_coords: An array of shape [n, 3] containing the n coordinates defining the beam.

        Return:
            A profile object containing the atmospheric variables along the beam.
        """
        surf_dists, surf_inds = self.surface_tree.query(beam_coords)
        surf_ind = np.argmin(surf_dists)

        surf_ind = np.unravel_index(surf_inds[surf_ind], self.shape[1:])

        lon = self.data["longitude"].data[surf_ind[1]].item()
        lat = self.data["latitude"].data[surf_ind[0]].item()
        u_10 = self.data["10u"].data[surf_ind].item()
        v_10 = self.data["10v"].data[surf_ind].item()
        t2m = self.data["2t"].data[surf_ind].item()
        skt = self.data["skt"].data[surf_ind].item()
        z_surf = self.data["surface_altitude"].data[surf_ind].item()

        dists, inds = self.tree.query(beam_coords)
        inds = np.unravel_index(inds, self.shape)

        ciwc = self.data["ciwc"].data[inds]
        ciwc[dists > max_dist] = np.nan
        clwc = self.data["clwc"].data[inds]
        clwc[dists > max_dist] = np.nan
        crwc = self.data["crwc"].data[inds]
        crwc[dists > max_dist] = np.nan
        cswc = self.data["cswc"].data[inds]
        cswc[dists > max_dist] = np.nan
        p = np.exp(self.data["pres"].data[inds])
        p[dists > max_dist] = np.nan
        t = self.data["t"].data[inds]
        t[dists > max_dist] = np.nan
        alt = self.data["altitude"].data[inds]
        q = self.data["q"].data[inds]

        return ProfileData(
            longitude=lon,
            latitude=lat,
            time=self.data.valid_time.data,
            surface_altitude=z_surf,
            skin_temperature=skt,
            two_meter_temperature=t2m,
            wind_speed=np.sqrt(u_10 ** 2 + v_10 ** 2),
            pressure=p,
            altitude=alt,
            temperature=t,
            humidity=q,
            clwc=clwc,
            ciwc=ciwc,
            swc=cswc,
            rwc=crwc
        )


class ORNLDataLoader(NetCDFDataLoader):
    """
    Data loader to load atmospheric profile data from ECMWF high-resolution climate run data
    """

    required_fields = [
        "ciwc",
        "clwc",
        "crwc",
        "cswc",
        "pres",
        "q",
        "t",
        "2t",
        "skt",
        "z",
        "10u",
        "10v"
    ]


    def __init__(
            self,
            data_path: Path
    ):
        """
        Args:
            data_path: The directory containing the required variable fields.
        """
        self.data_path = Path(data_path)
        [find_var_file(self.data_path, var) for var in self.required_fields]
        data = self.load_data()
        data["surface_altitude"] = data["z"] / 9.81
        super().__init__(data)


    def load_data(self) -> xr.Dataset:
        """
        The atmospheric data loaded into an xarray.Dataset.
        """
        data = []
        for var in self.required_fields:
            path = find_var_file(self.data_path, var)
            var_data = xr.load_dataset(path)
            if "unknown" in var_data:
                var_data = var_data.rename(unknown=var)
            data.append(var_data)

        data = xr.merge(data)

        surface_altitude = data.z.data / 9.81
        temperature = data.t.data[::-1]
        pressure = np.exp(data.pres.data[::-1])

        alt = calculate_altitude(pressure, temperature, surface_altitude)
        data["altitude"] = (("hybrid", "latitude", "longitude"), alt[::-1])

        lons = data.longitude.data
        lons[lons > 180.0] -= 360.0
        data = data.assign_coords(longitude=lons).sortby("longitude")
        return data
