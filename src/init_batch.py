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