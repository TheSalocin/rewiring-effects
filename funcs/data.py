#!/usr/bin/env python
# coding: utf-8

import pynwb
import os
import numpy as np


def read_data(path):
    """
    Reads data from file; for .nwb files from Buzsaki-lab
    
    Input:
    
    path: str
        path to file
        
    Output:
    
    excitatory_spikes: list of np.arrays
        each array corresponds to a neuron and contains spiketimes in [s]
    inhibitory_spikes: list of np.arrays
        each array corresponds to a neuron and contains spiketimes in [s]
    starts: np.array
        array of state start times in [s]
    stops: np.array
        array of state end times in [s]
    labels: np.array
        array of state labels as strings ("Awake", "Non-REM", or "REM")
    """
    
    spikes = pynwb.NWBHDF5IO(path, "r").read().units["spike_times"][:]
    cell_types = pynwb.NWBHDF5IO(path, "r").read().units["cell_type"][:]
    excitatory_spikes = [spikes[i] for i in np.where(cell_types == "excitatory")[0]]
    inhibitory_spikes = [spikes[i] for i in np.where(cell_types == "inhibitory")[0]]

    states = pynwb.NWBHDF5IO(path, "r").read().processing["behavior"].data_interfaces["states"].to_dataframe()
    starts = np.array(states.start_time.to_list())
    stops = np.array(states.stop_time.to_list())
    labels = np.array(states.label.to_list())
    
    return excitatory_spikes, inhibitory_spikes, starts, stops, labels





