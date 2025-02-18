"""
mwsim.simulators.base
=====================

Provides a base class for running simulations in parallel.
"""
import logging
import multiprocessing as mp
import signal
import time

import numpy as np
from tqdm import tqdm
import xarray as xr

from mwsim.data_loaders import ProfileData
from mwsim.sensors import Sensor


LOGGER = logging.getLogger(__name__)


def get_progress_bar(iterable, enabled=True, **tqdm_kwargs):
    """
    Returns a tqdm progress bar if enabled, otherwise returns the plain iterable.

    Params:
        iterable: The iterable to wrap.
        enabled: Whether to display the progress bar.
        tqdm_kwargs: Additional keyword arguments for tqdm.

    Return:
        An iterator with or without tqdm progress bar.
    """
    return tqdm(iterable, **tqdm_kwargs) if enabled else iterable


class ParallelSimulator:
    def __init__(
            self,
            sensor: Sensor,
            n_processes: int
    ):
        """
        Initializes the parallel simulator.

        Args:
            sensor: A sensor object defining the sensor to simulate.
        """
        self.sensor = sensor
        self.input_queue = mp.Queue()
        self.output_queue = mp.Queue()
        self.processes = []

        self.start_workers(n_processes)
        self.n_profiles = 0

        self.results = []
        self.coords_lon = []
        self.coords_lat = []

    def _worker(self, input_queue, output_queue, sensor):
        """
        Worker function that initializes the simulator and processes simulation tasks.

        :param input_queue: Queue to receive profile data objects.
        :param output_queue: Queue to send simulation results.
        :param sensor: Sensor object for simulator setup.
        """
        # Initialize the simulator
        self.setup(sensor)

        while True:
            try:
                inpt = input_queue.get()
                if inpt is None:  # Sentinel value to terminate worker
                    break
                profile_index, profile_data = inpt
                result = self.simulate_profile(profile_data)

                output_queue.put((profile_index, result))
            except Exception as e:
                LOGGER.warning(f"Worker encountered an error: {e}")

    def start_workers(self, n_processes: int):
        """Starts worker processes."""
        self._n_processes = n_processes
        for _ in range(n_processes):
            p = mp.Process(target=self._worker, args=(self.input_queue, self.output_queue, self.sensor))
            p.start()
            self.processes.append(p)

    def stop_workers(self):
        """Stops worker processes gracefully."""
        for _ in range(self._n_processes):
            self.input_queue.put(None)
        for p in self.processes:
            p.join()

    def simulate(self, profile_data: ProfileData) -> None:
        """
        Runs simulation for given profile asynchronously.

        Args:
            profile_data: A ProfileData object defining the atmosphere along the beam to simulate.
        """
        try:
            self.coords_lon.append(profile_data.longitude)
            self.coords_lat.append(profile_data.latitude)
            self.input_queue.put((self.n_profiles, profile_data))
            self.n_profiles += 1
        except KeyboardInterrupt:
            LOGGER.info("Simulation interrupted. Terminating workers.")
            self.stop_workers()
            raise  # Re-raise exception for proper handling

    def get_results(self, progress_bar: bool = False) -> xr.Dataset:
        """
        This function collects all result from the simulator into an xarray.Dataset.

        Return:
            A xarray.Dataset containing the simulated brightness temperatures.
        """
        results = []
        profile_indices = []
        try:
            for ind in get_progress_bar(range(self.n_profiles), progress_bar, desc="Simulation progress:"):
                profile_ind, tbs = self.output_queue.get()
                results.append(tbs)
                profile_indices.append(profile_ind)

        except KeyboardInterrupt:
            LOGGER.info("Simulation interrupted. Terminating workers.")
            self.stop_workers()
            raise  # Re-raise exception for proper handling
        self.stop_workers()

        tbs = np.stack(results)
        profile_indices = np.array(profile_indices)
        order = np.argsort(profile_indices)

        results = xr.Dataset({
            "longitude": (("beams"), np.stack(self.coords_lon)),
            "latitude": (("beams"), np.stack(self.coords_lat)),
            "brightness_temperatures": (("beams", "channels"), tbs[order]),
        })
        return results


