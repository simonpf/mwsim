"""
ornl_simulations.geometry
=========================

This module defines geometric utility functions to manipulate sensor and footprint
positions.
"""
from typing import Tuple

import numpy as np
import xarray as xr

EARTH_RADIUS = 6.371e6


def lla_to_ecef(coords_lla: np.ndarray):
    """
    Converts latitude-longitude-altitude (LLA) coordinates to
    earth-centric earth-fixed coordinates (ECEF)

    Params:
        coords_lla: A numpy.ndarray containing the three coordinates oriented along the last axis.

    Return:
        coords_ecef: An array of the same shape as 'coords_lla' but containing the x, y, and z
             coordinates along the last axis.
    """
    SEM_A = 6_378_137.0
    SEM_B = 6_356_752.0
    ECC2 = 1.0 - (SEM_B ** 2 / SEM_A ** 2)

    lon = np.radians(coords_lla[..., 0])
    lat = np.radians(coords_lla[..., 1])
    alt = coords_lla[..., 2]

    roc = SEM_A / np.sqrt(1 - ECC2 * np.sin(lat)**2)

    x = (roc + alt) * np.cos(lat) * np.cos(lon)
    y = (roc + alt) * np.cos(lat) * np.sin(lon)
    z = (roc * (1 - ECC2) + alt) * np.sin(lat)

    return np.stack((x, y, z), -1)


def ecef_to_lla(coords_ecef):
    """
    Converts ECEF coordinates back to LLA coordinates.

    Params:
        coords_ecef: A numpy.ndarray containing the coordinates along the last axis.

    Return:
        coords_lla: A numpy.ndarray of the same shape as 'coords_ecef' containing
            the longitude, latitude, and altitude along tis last axis.
    """
    SEM_A = 6_378_137.0
    SEM_B = 6_356_752.0
    ECC2 = 1.0 - (SEM_B ** 2 / SEM_A ** 2)

    lon = np.arctan2(coords_ecef[..., 1], coords_ecef[..., 0])
    lon = np.nan_to_num(lon, nan=0.0)
    lon = np.degrees(lon)

    p = np.sqrt(coords_ecef[..., 0]**2 + coords_ecef[..., 1]**2)

    lat = np.arctan2(coords_ecef[..., 2], p * (1 - ECC2))
    lat_prev = lat
    roc = SEM_A / np.sqrt(1 - ECC2 * np.sin(lat)**2)
    alt = p / np.cos(lat) - roc
    lat = np.arctan2(coords_ecef[..., -1], p * (1 - ECC2 * (roc / (roc + alt))))


    while np.max(np.abs(lat - lat_prev)) > 1e-6:
        lat_prev = lat
        roc = SEM_A / np.sqrt(1 - ECC2 * np.sin(lat)**2)
        alt = p / np.cos(lat) - roc
        lat = np.arctan2(coords_ecef[..., 2], p * (1 - ECC2 * (roc / (roc + alt))))

    roc = SEM_A / np.sqrt(1 - ECC2 * np.sin(lat)**2)
    alt = p / np.cos(lat) - roc
    lat = np.degrees(lat)

    if not isinstance(lat, np.ndarray):
        if np.isclose(p, 0.0):
            alt = coords_ecef[..., -1]
            lat = np.sign(alt) * 90
            alt = np.abs(alt) - SEM_B
    else:
        mask = np.isclose(p, 0.0)
        alt[mask] = coords_ecef[mask, -1]
        lat[mask] = np.sign(alt[mask]) * 90
        alt[mask] = np.abs(alt[mask]) - SEM_B

    return np.stack([lon, lat, alt], -1)


def rotate_around(x: np.ndarray, axis: np.ndarray, theta: float):
    """
    Rotate vector x around axis by thate degrees.

    Args:
        x: The vector to rotate with coordinates oriented along the last dimension.
        axis: The axis around with to rotate the vector.
        theta: The number of degrees by which to rotate the vector.

    Return:
        The rotated vector x'

    """
    axis = axis / np.linalg.norm(axis)
    theta = np.deg2rad(theta)

    x_cos = x * np.cos(theta)
    x_cross = np.cross(axis, x, axis=-1) * np.sin(theta)
    x_dot = np.tensordot(x, axis, axes=(-1, -1)) * (1 - np.cos(theta))
    x_axis = x_dot[..., None] * np.broadcast_to(axis, x.shape)

    x_rot = x_cos + x_cross + x_axis
    return x_rot


def calculate_surface_intersection(
        sensor_pos_ecef: np.ndarray,
        sensor_los_ecef: np.ndarray,
):
    """
    Calculate intersection of a line-of-sigh (LOS) with the surface of the Earth.

    Args:
        sensor_pos_ecef: The sensor position in ECEF coordinates
        sensor_los_ecef: The line-of-sight direction in ECEF coordinates

    Return:
        The position of the surface intersection in ECEF coordinates.
    """
    SEM_A = 6_378_137.0
    SEM_B = 6_356_752.0
    ECC2 = 1.0 - (SEM_B ** 2 / SEM_A ** 2)

    coeff_a = (
        sensor_los_ecef[..., 0] ** 2 / SEM_A ** 2 +
        sensor_los_ecef[..., -2] ** 2 / SEM_A ** 2 +
        sensor_los_ecef[..., -1] ** 2 / SEM_B ** 2
    )
    coeff_b = 2.0 * (
        sensor_pos_ecef[..., 0] * sensor_los_ecef[..., 0] / SEM_A ** 2 +
        sensor_pos_ecef[..., 1] * sensor_los_ecef[..., 1] / SEM_A ** 2 +
        sensor_pos_ecef[..., 2] * sensor_los_ecef[..., 2] / SEM_B ** 2
    )
    coeff_c = (
        sensor_pos_ecef[..., 0] ** 2 / SEM_A ** 2 +
        sensor_pos_ecef[..., 1] ** 2 / SEM_A ** 2 + sensor_pos_ecef[..., 2] ** 2 / SEM_B ** 2) - 1.0

    discr = coeff_b ** 2 - 4.0 * coeff_a * coeff_c
    root_1 = (np.sqrt(discr) - coeff_b) / (2.0 * coeff_a)
    root_2 = (-np.sqrt(discr) - coeff_b) / (2.0 * coeff_a)

    fac = np.minimum(root_1, root_2)
    pos = sensor_pos_ecef + fac[..., None] * sensor_los_ecef
    return pos


def incidence_angle_to_viewing_angle(incidence_angle, altitude):
    """
    Calculates viewing angle from incidence angles.

    Args:
        incidence_angle: The earth incidence angles in degree for which to
            calculate the viewing angles.
        altitude: The altitude of the sensor in meters.

    Return:
        The calculated viewing angles in degree.
    """
    sin_alpha = EARTH_RADIUS * np.sin(np.pi - np.deg2rad(incidence_angle)) / (EARTH_RADIUS + altitude)
    alpha = np.arcsin(sin_alpha)
    return np.rad2deg(alpha)


def great_circle_distance(lon_start, lat_start, lon_end, lat_end):
    """
    Calculate the great-circle distance between two points on a sphere using the Haversine formula.

    Parameters:
        lon_start: Longitude coordinate of the start point.
        lat_start: Latitude coordinate of the start point.
        lon_end: A single longitude coordinate or an array of longitude coordinates of the end points.
        lat_end: A single latitude coordinate or an array of latitude coordinates of the end points.

    Returns:
        Distance between the two points on the sphere in the same units as the radius.
    """
    lon_start, lat_start, lon_end, lat_end = map(np.deg2rad, [lon_start, lat_start, lon_end, lat_end])
    dlat = lat_end - lat_start
    dlon = lon_end - lon_start
    a = np.sin(dlat / 2.0)**2 + np.cos(lat_start) * np.cos(lat_end) * np.sin(dlon / 2.0)**2
    c = 2.0 * np.arctan2(np.sqrt(a), np.sqrt(1.0 - a))
    distance = EARTH_RADIUS * c
    return distance


def calculate_center_beams(
        start: Tuple[float, float],
        end: Tuple[float, float],
        earth_incidence_angle: float,
        scan_separation: float,
        scan_range: float,
        n_pixels: int,
        sensor_altitude: float
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Calculate surface intersections and directions of the center beams of a conical scanning sensor.

    Args:
        start: A tuple ``(lon, lat)`` containing the longitude and latitude coordinates of the center
            point of the first swath.
        end: A tuple ``(lon, lat)`` containign the longitude and latitude coordinates of the center
            point of the last swath.
        earth_incidence_angle: The earth incidence angle of the sensor.
        scan_separation: The distance between consecutive scan lines at the Earth surface.
        scan_range: The limit of the scan range of the conical scan.
        n_pixels: The number of pixels per scan.

    Return:
        A tuple ``[coords, los]`` containing the coordinates of the surface intersections of the center beams
        in ``coords`` and the satellite-pointing directions in ``los``.
    """
    f_dist = np.linspace(0, 1, 100)[:, None]
    swath = (1.0 - f_dist) * start + f_dist * end
    swath_dist = great_circle_distance(swath[0, 0], swath[0, 1], swath[:, 0], swath[:, 1])
    centers_lon = np.interp(np.arange(swath_dist[0], swath_dist[-1], scan_separation), swath_dist, swath[:, 0])
    centers_lat = np.interp(np.arange(swath_dist[0], swath_dist[-1], scan_separation), swath_dist, swath[:, 1])
    centers = np.stack([centers_lon, centers_lat], axis=-1)

    beta = 180.0 - earth_incidence_angle
    alpha = incidence_angle_to_viewing_angle(earth_incidence_angle, sensor_altitude)
    gamma = 180.0 - alpha - beta
    view_length = np.sin(np.deg2rad(gamma)) * (sensor_altitude + EARTH_RADIUS) / np.sin(np.deg2rad(beta))

    centers_lla = np.concatenate(
        [centers, np.zeros((centers.shape[0], 1), dtype=np.float32)],
        1,
    )
    centers_ecef = lla_to_ecef(centers_lla)
    up = centers_lla.copy()
    up[..., 2] = 1.0
    up = lla_to_ecef(up) - centers_ecef

    forward = -centers_ecef.copy()
    forward[:-1] += centers_ecef[1:]
    forward[-1] = forward[-2]

    rot_axis = np.cross(forward, up)
    center_los = np.stack([rotate_around(up_i, ax_i, earth_incidence_angle) for up_i, ax_i in zip(up, rot_axis)])
    center_los = center_los / np.linalg.norm(center_los, axis=-1, keepdims=True)

    sat_pos_ecef = centers_ecef + center_los * view_length

    sat_pos_lla = ecef_to_lla(sat_pos_ecef)
    sat_pos_lla_1 = sat_pos_lla.copy()
    sat_pos_lla_1[..., -1] += 1.0

    sat_up = sat_pos_ecef / np.linalg.norm(sat_pos_ecef, axis=-1, keepdims=True)
    angs = np.linspace(-scan_range, scan_range, n_pixels)

    coords = []
    views = []

    for sat_pos, up, los in zip(sat_pos_ecef, sat_up, center_los):
        scan_los = np.stack([rotate_around(-los, up, ang) for ang in angs])
        footprints = calculate_surface_intersection(sat_pos, scan_los)
        coords.append(footprints)
        views.append(scan_los)

    return (
        np.concatenate(coords),
        np.concatenate(views)
    )
