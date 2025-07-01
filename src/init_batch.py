<<<<<<< HEAD
from netpyne import specs, sim

cfg = specs.SimConfig()  # create a SimConfig object, can be provided with a dictionary on initial call to set initial values
netParams = specs.NetParams()  # create a netParams object

sim.createSimulateAnalyze(netParams, cfg)
=======
from netpyne.batchtools import specs, comm
from netpyne import sim
from netParams import netParams, cfg
import json

comm.initialize()

sim.createSimulate(netParams=netParams, simConfig=cfg)
print('completed simulation...')

#sim.gatherData()
if comm.is_host():
    netParams.save("{}/{}_params.json".format(cfg.saveFolder, cfg.simLabel))
    print('transmitting data...')
    results = sim.analysis.plotData()
    out_json = json.dump(results)



#TODO put all of this in a single function (James)
    comm.send(out_json)
    comm.close()
>>>>>>> 8369f25d633b5871e67604248cd807af194ce57d
