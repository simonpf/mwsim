"""
mwsim.simulators.arts26
=======================

This sub-module implements a brightness-temperature simulator based on ARTS version 2.6.
"""
from dataclasses import dataclass
import gzip
import os
from pathlib import Path
import shutil
from tempfile import TemporaryDirectory
from typing import List

import numpy as np
from pyarts import Workspace
from pyarts.arts import ArrayOfAbsorptionLines
from pyarts.workspace import arts_agenda
from pyarts.xml import load

from mwsim import config
from mwsim.sensors import Sensor
from mwsim.utils import h2o_mmr2vmr
from mwsim.data_loaders import ProfileData
from .base import ParallelSimulator


# Assumed VMR for N2 and O2.
VMR_N2 = 0.78084
VMR_O2 = 0.20946


def setup_ppath(ws):
    """
    Sets up propagation path calculations.
    """
    ws.ppath_step_agendaSet( option="GeometricPath" )
    ws.ppath_agendaSet( option="FollowSensorLosPath" )


def setup_absorption(ws):
    """
    Sets up gas absorption on the ARTS workspace. Current setup taken from easy_arts RTTOV setting omitting O3 for now.
    """
    # Absorption species
    ws.abs_speciesSet(species=[
        "N2-SelfContStandardType",
        "O2-PWR98",
        "H2O-161, H2O-SelfContCKDMT350, H2O-ForeignContCKDMT350",
        "liquidcloud-ELL07"
    ] )
    ## No line data needed here
    ws.abs_lines = ArrayOfAbsorptionLines()
    ws.ArrayOfAbsorptionLinesCreate("temp_lines")
    lines_file = Path(__file__).parent / "files" / "AER_fast_3.8.H2O.xml.gz"
    ws.ReadARTSCAT(
        abs_lines=ws.temp_lines,
        filename=str(lines_file),
        normalization_option="VVH",
        mirroring_option="None",
        lineshapetype_option="VP",
        cutoff_option="ByLine",
        cutoff_value=750e9,
    )
    ws.Append(ws.abs_lines, ws.temp_lines)
    ws.abs_lines_per_speciesCreateFromLines()
    ws.abs_lines_per_speciesManualMirroringSpecies(
        species="H2O-161, H2O-SelfContCKDMT350, H2O-ForeignContCKDMT350"
    )
    ws.lbl_checkedCalc()

def setup_scattering(ws, scattering_data_path):
    """
    Sets up scattering for liquid and frozen particles. Currently uses liquid spheres for rain and LargePlateAggregate
    for frozen particles with Wang et al. (2016) and McFarquahar & Heymsfield (1997) PSDs, respectively.
    """
    scattering_data_path = config.get_data_path() / "StandardHabits"
    scat_data_rain = load(str(scattering_data_path / "FullSet" / "LiquidSphere.xml"))
    scat_meta_rain = load(str(scattering_data_path / "FullSet" / "LiquidSphere.meta.xml"))
    scat_data_ice = load(str(scattering_data_path / "FullSet" / "LargePlateAggregate.xml"))
    scat_meta_ice = load(str(scattering_data_path / "FullSet" / "LargePlateAggregate.meta.xml"))

    ws.scat_data_raw = [scat_data_rain, scat_data_ice]
    ws.scat_meta = [scat_meta_rain, scat_meta_ice]

    ws.pnd_agenda_array = []
    ws.pnd_agenda_array_input_names = []

    ws.ArrayOfStringSet(ws.pnd_agenda_input_names, ["rwc"])
    ws.pnd_agenda_array = []

    @arts_agenda(allow_callbacks=True)
    def pnd_agenda_rain(ws):
        ws.ScatSpeciesSizeMassInfo(species_index=ws.agenda_array_index, x_unit="dveq")
        ws.Copy(ws.psd_size_grid, ws.scat_species_x)
        ws.Copy(ws.pnd_size_grid, ws.scat_species_x)
        ws.psdWangEtAl16(t_min = 273, t_max = 999)
        ws.pndFromPsdBasic()

    ws.pnd_agenda_rain = pnd_agenda_rain
    ws.Append(ws.pnd_agenda_array, ws.pnd_agenda_rain)

    @arts_agenda(allow_callbacks=True)
    def pnd_agenda_ice(ws):
        ws.ScatSpeciesSizeMassInfo(species_index=ws.agenda_array_index, x_unit="dveq", x_fit_start=100e-6)
        ws.Copy(ws.psd_size_grid, ws.scat_species_x)
        ws.Copy(ws.pnd_size_grid, ws.scat_species_x)
        ws.psdMcFarquaharHeymsfield97( t_min = 10, t_max = 273, t_min_psd = 210 )
        ws.pndFromPsdBasic()

    ws.pnd_agenda_ice = pnd_agenda_ice
    ws.Append(ws.pnd_agenda_array, ws.pnd_agenda_ice)
    ws.pnd_agenda_array_input_names = [["rwc"], ["iwc"]]
    ws.particle_bulkprop_names = ["rwc", "iwc"]
    ws.scat_species = ["rain", "ice"]
    ws.scat_dataCalc()


def setup_workspace(sensor: Sensor):

    ws = Workspace()
    # Basic configuration
    ws.water_p_eq_agendaSet()
    ws.gas_scattering_agendaSet()
    ws.PlanetSet(option="Earth")
    ws.iy_main_agendaSet(option="Emission")
    ws.iy_space_agendaSet(option="CosmicBackground")
    ws.iy_surface_agendaSet(option="UseSurfaceRtprop")
    ws.iy_cloudbox_agendaSet(option="QuarticInterpField")

    setup_ppath(ws)
    setup_absorption(ws)

    ws.propmat_clearsky_agendaAuto()
    ## Dimensionality of the atmosphere
    ws.AtmosphereSet1D()
    # Brigtness temperatures used
    ws.StringSet(ws.iy_unit, "PlanckBT")
    # Various things not used
    ws.ArrayOfStringSet(ws.iy_aux_vars, [])
    ws.jacobianOff()

    ws.f_grid = [freq + offs for freq, offs in zip(sensor.channel_frequencies, sensor.channel_offsets)]
    setup_scattering(ws, "")

    ws.stokes_dim = 1
    ws.sensorOff()
    ws.DOAngularGridsSet(N_za_grid=38, N_aa_grid=37)

    data_path = Path(__file__).parent / "files"
    ws.TessemNNReadAscii(ws.tessem_neth, str(data_path / "tessem_sav_net_H.txt"))
    ws.TessemNNReadAscii(ws.tessem_netv, str(data_path / "tessem_sav_net_V.txt"))

    # Looks like telsem in ARTS is currently broken :(
    #ws.telsem_atlasesReadAscii(
    #    directory="/home/simon/data/telsem",
    #    filename_pattern="ssmi_mean_emis_climato_@MM@_cov_interpol"
    #)


    return ws


class ARTS26Simulator(ParallelSimulator):
    """
    The ARTS26Simulator implements sing-column brightness-temperatures simulations using the DISORT scattering
    solver in ARTS v.2.6.
    """
    def __init__(
            self,
            sensor: Sensor,
            n_processes: int = 1
    ):
        """
        Args:
            sensor: A config object describing the sensor configuration.
        """
        super().__init__(sensor, n_processes=n_processes)

    def setup(self, sensor: Sensor) -> None:
        self.ws = setup_workspace(sensor)

    @property
    def has_scattering(self) -> bool:
        """
        Indicates whether the simulatios are setup for scattering.
        """
        return len(self.ws.pnd_agenda_array_input_names.value) > 0


    def load_data(self, profile: ProfileData) -> None:
        """
        Load data from profile into ARTS workspace.

        Args:
            profile: A ProfileData object defining the atmosphere along the pencil beam.
        """
        p_inds = np.isfinite(profile.pressure)

        p_grid = profile.pressure[p_inds]
        delta_p = np.minimum(np.diff(p_grid), -1e-2)
        p_grid_new = p_grid[0] * np.ones_like(p_grid)
        p_grid_new[1:] += np.cumsum(delta_p)
        p_grid = p_grid_new

        z_grid = profile.altitude.copy()[p_inds]
        delta_z = np.maximum(np.diff(z_grid), 1)
        z_grid_new = z_grid[0] * np.ones(z_grid.shape)
        z_grid_new[1:] += np.cumsum(delta_z)
        z_grid = z_grid_new

        h2o = profile.humidity
        h2o = h2o_mmr2vmr(h2o)

        vmr_field = np.zeros((4,) + h2o.shape)
        vmr_field[0] = VMR_N2 * (1.0 - h2o)
        vmr_field[1] = VMR_O2 * (1.0 - h2o)
        vmr_field[2] = h2o
        vmr_field[3] = profile.clwc

        self.ws.p_grid = p_grid
        self.ws.z_field = z_grid[:, None, None]
        self.ws.t_field = profile.temperature[p_inds, None, None]
        self.ws.vmr_field = vmr_field[:, p_inds, None, None]
        self.ws.z_surface = np.array([[max(profile.surface_altitude, z_grid[0])]])

        tiwc = profile.swc + profile.ciwc
        rwc = profile.rwc
        pbf_field = np.stack([rwc, tiwc])
        self.ws.particle_bulkprop_field = pbf_field[:, p_inds, None, None]

        self.ws.surface_skin_t = profile.skin_temperature
        self.ws.atmfields_checkedCalc()
        self.ws.sensor_pos = [[440e3]]
        self.ws.sensor_los = [[180.0]]

        # Calculate surface emissivity
        time = profile.time.astype("datetime64[s]").item()
        self.ws.lat_grid = []
        self.ws.lon_true = [profile.longitude]
        self.ws.lat_true = [profile.latitude]
        self.ws.rtp_pos = [0.0]
        self.ws.rtp_los = [180.0 - self.sensor.earth_incidence_angle]
        #self.ws.telsemSurfaceTypeLandSea(atlas=sel.ws.telsem_atlases[time.month - 1])
        #land = ws.surface_type.value
        #if land:
        #    ws.surfaceTelsem()
        #else:
        self.ws.surfaceTessem(wind_speed=profile.wind_speed)
        self.ws.surface_scalar_reflectivityFromSurface_rmatrix()


    def simulate_profile(self, profile: ProfileData) -> None:
        """
        Simulate brightness temperatures for profile

        Args:
            profile_data: A ProfileData object defining the atmosphere along the beam to simulate.
        """
        self.load_data(profile)

        self.ws.atmgeom_checkedCalc()
        self.ws.sensor_checkedCalc()
        self.ws.scat_data_checkedCalc(sca_mat_threshold=0.25)

        self.ws.cloudboxSetFullAtm()
        self.ws.pnd_fieldZero()
        self.ws.cloudbox_checkedCalc()
        self.ws.gas_scatteringOff()
        self.ws.pnd_fieldCalcFromParticleBulkProps()
        self.ws.cloudbox_checkedCalc()
        self.ws.cloudbox_fieldDisort()
        self.ws.yCalc()

        return np.array(self.ws.y.value).copy()
