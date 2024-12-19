"""
init.py
Starting script to run NetPyNE-based models
"""
from netpyne import sim
cfg, netParams = sim.readCmdLineArgs(simConfigDefault='cfg.py', netParamsDefault='netParams.py')
sim.createSimulateAnalyze(netParams, cfg)
