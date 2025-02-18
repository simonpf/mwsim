"""
mwsim.sensors
=============

Defines the configurations for different PMW sensors.
"""
from dataclasses import dataclass
from typing import List

import numpy as np

from mwsim.geometry import calculate_center_beams


@dataclass
class SensorConfig:
    channel_frequencies: List[float]
    channel_offsets: List[float]
    channel_polarizations: List[str]



@dataclass
class Sensor:

    channel_frequencies: List[float]
    channel_offsets: List[float]
    channel_polarizations: List[str]
    earth_incidence_angle: float
    altitude: float
    n_pixels: int
    scan_range: float
    scan_distance: float


    def __init__(
            self,
            channel_frequencies: List[float],
            channel_offsets: List[float],
            channel_polarizations: List[str],
            earth_incidence_angle: float,
            altitude: float,
            n_pixels: int,
            scan_range: float,
            scan_distance: float
    ):
        self.channel_frequencies = channel_frequencies
        self.channel_offsets = channel_offsets
        self.channel_polarizations = channel_polarizations
        self.earth_incidence_angle = earth_incidence_angle
        self.altitude = altitude
        self.n_pixels = n_pixels
        self.scan_range = scan_range
        self.scan_distance = scan_distance

    def get_swath_profile_coordinates(
            self,
            start,
            end,
            n_levels: int = 60,
            sampling_distance: float = 500
    ):
        """
        Calculate coordinates of beam profiles for a sensor overpass.
        """
        fp_coords, fp_los = calculate_center_beams(
            start=start,
            end=end,
            earth_incidence_angle=self.earth_incidence_angle,
            scan_separation=self.scan_distance,
            scan_range=self.scan_range,
            n_pixels=self.n_pixels,
            sensor_altitude=self.altitude
        )
        fp_los = - fp_los / np.linalg.norm(fp_los, keepdims=True, axis=-1)
        coords = [fp_coords + dist * fp_los for dist in sampling_distance * np.arange(n_levels)]
        return np.stack(coords, 1)


GMI = Sensor(
    channel_frequencies = [10.6e9, 18.7e9, 23.0e9, 37.0e9, 89.0e9, 166.0e9, 183e9, 183e9],
    channel_offsets = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 3.0e9, 7.0e9],
    channel_polarizations = ["V", "V", "V", "V", "V", "V", "V", "V"],
    earth_incidence_angle=52.0,
    altitude=440e3,
    n_pixels=221,
    scan_range=76.3,
    scan_distance=13.5e3
)
