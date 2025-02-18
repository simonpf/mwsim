"""
mwsim.config
============

This module manages the configuration of the ``mwsim`` package.
"""
from pathlib import Path
import os

def get_data_path():
    """
    Return the mwsim data path.
    """
    if not "MWSIM_DATA_PATH" in os.environ:
        raise ValueError(
            "mwsim requires the MWSIM_DATA_PATH environment variable to be set."
        )
    mwsim_data_path = Path(os.environ["MWSIM_DATA_PATH"])
    if not mwsim_data_path.exists():
        raise ValueError(
            "'MWSIM_DATA_PATH' points to a non-existing directory."
        )
    return mwsim_data_path
