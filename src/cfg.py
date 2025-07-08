"""
cfg.py

Simulation Configuration
"""
from netpyne import specs
import os
cfg = specs.SimConfig() # Object of class SimConfig to store simulation configuration

#########################################################################################
#
# SIMULATION CONFIGURATION
#
#########################################################################################

#------------------------------------------------------------------------------
#  Simulation Parameters
#------------------------------------------------------------------------------
cfg.duration = 1000 # Duration of simulation in ms
cfg.dt = 0.025 # Internal integration time step (ms)
cfg.hParams = {'celsius': 34, 'v_init':-80}
cfg.verbose = False # Show detailed messages
cfg.printPopAvgRates = True

cfg.createNEURONObj = 1
cfg.createPyStruct = 1

#------------------------------------------------------------------------------
# Recording
#------------------------------------------------------------------------------
cfg.recordStim = True
cfg.recordCells = [0]
cfg.recordTraces = {'V_soma':{'sec': 'soma', 'loc': 0.5, 'var': 'v'}}
cfg.recordStep = 0.1 #step size to save data (e.g. voltage traces, LFP, et.)


#------------------------------------------------------------------------------
# Saving
#------------------------------------------------------------------------------
cfg.filename = 'data/Na12HH16HH_TF_test3' #File output name
cfg.saveJson = True
cfg.savePickle = False
cfg.saveDataInclude = ['simData', 'simConfig', 'netParams', 'net']

#------------------------------------------------------------------------------
# Analysis and Plotting
#------------------------------------------------------------------------------
cfg.analysis['plotRaster'] = {'saveFig': True} # Plot Raster
cfg.analysis['plotTraces'] = {'saveFig': True} # Plot Traces

#------------------------------------------------------------------------------
# Weight Normalization
#------------------------------------------------------------------------------
cfg.weightNorm = 1  # use weight normalization
cfg.weightNormThreshold = 4.0  # weight normalization factor threshold

#------------------------------------------------------------------------------
# Current Inputs
#------------------------------------------------------------------------------
cfg.addIClamp = 0 # change to 1 to add IClamps (can add multiple)

cfg.IClamp1 = {'pop': 'PT5B' ,'sec': 'soma', 'loc': 0.5, 'start': 100, 'dur': 500, 'amp': 0.4}


#------------------------------------------------------------------------------
# NetStim Inputs
#------------------------------------------------------------------------------
cfg.addNetStim = 2   # change to 1 to add NetStims (can add multiple)

"""
cfg.NetStim1 = {'pop': 'PT5B', 'sec': 'soma', 'loc': 0.5, 'synMech': ['AMPA', 'NMDA'], 'synMechWeightFactor': [0.5, 0.5],
				'start': 0, 'interval': 1000.0/40.0, 'noise': 0.0, 'number': 1000.0, 'weight': 0.5, 'delay': 0}
cfg.NetStim1 = {
    'pop': 'PT5B',
    'sec': 'dend_73',
    'loc': 0.5,
    'synMech': ['AMPA', 'NMDA'], 
    'synMechWeightFactor': [0, 1],
    'start': 250,
    'interval': 1.0,
    'noise': 0.0,
    'number': 2,
    'weight': 1,
    'delay': 0.01
}
"""
def get_env_or_default(key, default):
    val_str = os.environ.get(key, default)
    if isinstance(default, int):
        return int(float(val_str))  # Handles strings like "2.0"
    return float(val_str)


cfg.NetStim1 = {
    'pop': 'PT5B',
    'sec': 'dend_73',
    'loc': 0.5,
    'synMech': ['AMPA', 'NMDA'], 
    'synMechWeightFactor': [0.5,0.5],
    'start': get_env_or_default('NSTIM_START', 300.0),
    'interval': get_env_or_default('NSTIM_INTERVAL', 1),
    'noise': get_env_or_default('NSTIM_NOISE', 0.0),
    'number': get_env_or_default('NSTIM_NUMBER', 5.0),
    'weight': get_env_or_default('NSTIM_WEIGHT', 1.0),
    'delay': get_env_or_default('NSTIM_DELAY', 0.01)
}
cfg.NetStim2 = {
    'pop': 'PT5B',                 # Same or different population
    'sec': 'dend_85',             # Different section name
    'loc': 0.7,                   # Different location along the section
    'synMech': ['AMPA', 'NMDA'],
    'synMechWeightFactor': [0.5, 0.5],
    'start': get_env_or_default('NSTIM2_START', 500.0),   # Custom start time
    'interval': get_env_or_default('NSTIM2_INTERVAL', 1),
    'noise': get_env_or_default('NSTIM2_NOISE', 0.0),
    'number': get_env_or_default('NSTIM2_NUMBER', 5.0),
    'weight': get_env_or_default('NSTIM2_WEIGHT', 1.0),
    'delay': get_env_or_default('NSTIM2_DELAY', 0.01)
}

cfg.filename = os.environ.get('OUT_FILENAME', 'data/Na12HH16HH_TF_test3')

#------------------------------------------------------------------------------
# Synaptic Mechs
#------------------------------------------------------------------------------
cfg.nmdaTau1 = 15
