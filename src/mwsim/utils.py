"""
mw_sim.utils
=====================

Utility functions.
"""
import numpy as np


def calculate_altitude(
        pressure: np.ndarray,
        temperature: np.ndarray,
        surface_altitude: np.ndarray,
) -> np.ndarray:
    """
    Calculate altitude field using the hydrostatic equation.

    Args:
        pressure: A 3D pressure field in units hPa with the vertical dimension alon axis 0
        temperature: The temperature field corresponding to pressure.
        surface_altitude: The 2D surface altitude field.

    Return:
        The 3D altitude field.
    """
    n_levels = pressure.shape[0]

    altitude = [surface_altitude]
    for ind in range(n_levels - 1):
        t_0 = temperature[ind]
        t_1 = temperature[ind + 1]
        t_c = 0.5 * (t_0 + t_1)
        p_0 = pressure[ind]
        p_1 = pressure[ind + 1]
        p_c = 0.5 * (p_0 + p_1)

        d_z = - t_c * 287.0 / 9.81 * np.log(p_1 / p_0)
        altitude.append(altitude[-1] + d_z)

    altitude = np.stack(altitude)
    return altitude


M_D = 29.0
M_W = 18.0


def h2o_vmr2mmr(h2o: np.ndarray) -> np.ndarray:
    """
    Convert humidity VMR to mass-mixim ratio.
    """
    return h2o / (1.0 - h2o) * M_W / M_D


def h2o_mmr2vmr(h2o: np.ndarray) -> np.ndarray:
    """
    Convert humidity VMR to mass-mixim ratio.
    """
    return h2o / (h2o + M_W / M_D)
