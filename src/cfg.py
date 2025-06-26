"""
cfg.py

Simulation Configuration
"""
from netpyne import specs

cfg = specs.SimConfig() # Object of class SimConfig to store simulation configuration

#########################################################################################
#
# SIMULATION CONFIGURATION
#
#########################################################################################

#------------------------------------------------------------------------------
#  Simulation Parameters
#------------------------------------------------------------------------------
# fg.duration = 1*1e3 # Duration of simulation in ms
cfg.dt = 0.025 # Internal integration time step (ms)
cfg.hParams = {'celsius': 34, 'v_init':-80}
cfg.verbose = False # Show detailed messages
cfg.printPopAvgRates = True


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
cfg.filename = 'data/Na12HH16HH_TF_test' #File output name
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

cfg.IClamp1 = {'pop': 'PT5B' ,'sec': 'dend_20', 'loc': 0.5, 'start': 100, 'dur': 500, 'amp': 0.05}


#------------------------------------------------------------------------------
# NetStim Inputs
#------------------------------------------------------------------------------
cfg.addNetStim = 1     # change to 1 to add NetStims (can add multiple)

cfg.NetStim1 = {'pop': 'PT5B', 'sec': 'soma', 'loc': 0.5, 'synMech': ['AMPA', 'NMDA'], 'synMechWeightFactor': [0.5, 0.5],
				'start': 0, 'interval': 1000.0/40.0, 'noise': 0.0, 'number': 1000.0, 'weight': 0.5, 'delay': 0}

#------------------------------------------------------------------------------
# Synaptic Mechs
#------------------------------------------------------------------------------
cfg.nmdaTau1 = 15
