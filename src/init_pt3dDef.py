"""
init.py

Starting script to run NetPyNE-based models

"""
from netpyne import sim
from src.cfg import cfg
from netParams import netParams

sim.initialize(
    simConfig = cfg,
    netParams = netParams)  				# create network object and set cfg and net params
sim.net.createPops()               			# instantiate network populations
sim.net.createCells()              			# instantiate network cells based on defined populations

sim.net.defineCellShapes()

print("\n\n axon_0 pt3d")
print(sim.net.cells[0].secs['axon_0'].geom.pt3d)

print("\n\n axon_1 pt3d")
print(sim.net.cells[0].secs['axon_1'].geom.pt3d)