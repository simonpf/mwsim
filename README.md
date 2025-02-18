# mwsim

This repository contains the implementation of the ``mwsim`` Python package, a framework for simulating brightness temperature from high-resolution model output.

## Installation

### Software

The ``mwsim.yaml`` file defines a conda environment containing all required software to use ``mwsim``. Install it using

``` shellsession
cond env create --file mwsim.yml
```

Then install the ``mwsim`` package:

``` shellsession
pip install -e .
```

### Data

In order to properly simulate brightness temperatures affected by clouds and precipitation, the ``mwsim`` package makes use of single-scattering data from the ARTS single-scattering database (SSDB). Please download the ``StandardHabit.tar.gz`` from [Zenodo](https://zenodo.org/records/1175573) and place it into a **mwsim data folder**. The ``mwsim`` package expects an evinronment variable ``MWSIM_DATA_PATH`` pointing to the folder containing the untarred ``StandardHabit`` folder.


### Example simulation

An example notebook demonstrating the simulation workflow is provided in ``notebooks/gmi_simulations.ipynb``.
