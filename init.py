"""
init.py

Starting script to run NetPyNE-based models

"""
from netpyne import sim

cfg, netParams = sim.readCmdLineArgs(simConfigDefault='cfg.py', netParamsDefault='netParams.py')

# sim.createSimulateAnalyze(netParams, cfg)

sim.initialize(
    simConfig = cfg,
    netParams = netParams)  				# create network object and set cfg and net params
sim.net.createCells()         			# instantiate network cells based on defined populations           			# create connections between cells based on params
sim.net.addStims() 							# add network stimulation
sim.setupRecording()              			# setup variables to record for each cell (spikes, V traces, etc)
sim.runSim()                      			# run parallel Neuron simulation
sim.gatherData()                  			# gather spiking data and cell info from each node


sim.saveData()                  			# gather spiking data and cell info from each node
sim.analysis.plotData()

