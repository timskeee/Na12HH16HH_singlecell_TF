from netpyne import specs, sim
from cfg import cfg
from netParams import netParams


cfg = specs.SimConfig()  # create a SimConfig object, can be provided with a dictionary on initial call to set initial values
netParams = specs.NetParams()  # create a netParams object

sim.createSimulate(netParams=netParams, simConfig=cfg)
sim.createSimulateAnalyze(netParams, cfg)