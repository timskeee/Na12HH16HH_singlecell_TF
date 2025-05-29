"""
netParams.py

High-level specifications for M1 network model using NetPyNE

"""

from netpyne import specs
from cfg import cfg

cfg.update()
#try:
#    from __main__ import cfg # import SimConfig object with params from parent module
#except:
#    from src.cfg import cfg

#########################################################################################
#
#NETWORK PARAMETERS
#
#########################################################################################
netParams = specs.NetParams() # Object class NetParams to store network parameters


###############################################################################
# Cell parameters
###############################################################################

#------------------------------------------------------------------------------
# Load cell rules previously saved using netpyne format
#------------------------------------------------------------------------------
loadCellParams = True
saveCellParams = False

if loadCellParams:
   netParams.loadCellParamsRule(label='PT5B_full', fileName='../cells/Na12HH16HH_TF.json')

#------------------------------------------------------------------------------
# Includes importing from hoc template or python class, and setting additional params
#------------------------------------------------------------------------------

if not loadCellParams:
    # import cell model from NEURON/Python code
    netParams.importCellParams('PT5B_full', '../cells/Neuron_Model_12HH16HH/Na12HH16HHModel_TF.py', 'Na12Model_TF' )

    # rename soma to conform to netpyne standard
    netParams.renameCellParamsSec(label='PT5B_full', oldSec='soma_0', newSec='soma')

    # set variable so easier to work with below
    cellRule = netParams.cellParams['PT5B_full']

    # set the spike generation location to the axon (default in NEURON is the soma)
    cellRule['secs']['axon_0']['spikeGenLoc'] = 0.5

    # add pt3d for axon sections so SecList does not break
    cellRule['secs']['axon_0']['geom']['pt3d'] = [[1e30, 1e30, 1e30, 1.6440753755318644]]
    cellRule['secs']['axon_1']['geom']['pt3d'] = [[1e30, 1e30, 1e30, 1.6440753755318644]]

    # define cell conds
    netParams.cellParams['PT5B_full']['conds'] = {'cellModel': 'HH_full', 'cellType': 'PT', 'numCells': 1}

    # create lists useful to define location of synapses
    nonSpiny = ['apic_0', 'apic_1']
    netParams.addCellParamsSecList(label='PT5B_full', secListName='perisom',
                                   somaDist=[0, 50])  # sections within 50 um of soma
    netParams.addCellParamsSecList(label='PT5B_full', secListName='below_soma',
                                   somaDistY=[-600, 0])  # sections within 0-300 um of soma
    cellRule['secLists']['alldend'] = [sec for sec in cellRule.secs if ('dend' in sec or 'apic' in sec)]  # basal+apical
    cellRule['secLists']['apicdend'] = [sec for sec in cellRule.secs if ('apic' in sec)]  # apical
    cellRule['secLists']['spiny'] = [sec for sec in cellRule['secLists']['alldend'] if sec not in nonSpiny]

    for sec in nonSpiny:  # N.B. apic_1 not in `perisom` . `apic_0` and `apic_114` are
        if sec in cellRule['secLists']['perisom']:  # fixed logic
            cellRule['secLists']['perisom'].remove(sec)

    del netParams.cellParams['PT5B_full']['secs']['soma']['pointps']
    del netParams.cellParams['PT5B_full']['secs']['dend_0']['pointps']

    # Decrease dendritic Na
    #for secName in netParams.cellParams['PT5B_full']['secs']:
       # if secName.startswith('apic'):
            #print(secName)
            #print(netParams.cellParams['PT5B_full']['secs'][secName]['mechs']['na12mut'])
            #print(netParams.cellParams['PT5B_full']['secs'][secName]['mechs']['na12'])
            #netParams.cellParams['PT5B_full']['secs'][secName]['mechs']['na12'] *= cfg.dendNa
            #netParams.cellParams['PT5B_full']['secs'][secName]['mechs']['na12mut'] *= cfg.dendNa

    #set weight normalization
    netParams.addCellParamsWeightNorm('PT5B_full', '../conn/PT5B_full_weightNorm.pkl',
                                     threshold=cfg.weightNormThreshold)

    # save to json with all the above modifications so easier/faster to load
    if saveCellParams: netParams.saveCellParamsRule(label='PT5B_full', fileName='../cells/Na12HH16HH_TF.json')


###############################################################################
# Population parameters
###############################################################################

netParams.popParams['PT5B'] =	{'cellType': 'PT5B_full', 'numCells': 1}


###############################################################################
# Synaptic Mechanism parameters
###############################################################################

netParams.synMechParams['AMPA'] = {'mod':'MyExp2SynBB', 'tau1': 0.05, 'tau2': 5.3, 'e': 0}
netParams.synMechParams['NMDA'] = {'mod': 'MyExp2SynNMDABB', 'tau1NMDA': 15, 'tau2NMDA': 150, 'e': 0}


###############################################################################
# Stimulation parameters
###############################################################################

# ------------------------------------------------------------------------------
# Current inputs (IClamp)
# ------------------------------------------------------------------------------
if cfg.addIClamp:
    for key in [k for k in dir(cfg) if k.startswith('IClamp')]:
        params = getattr(cfg, key, None)
        [pop, sec, loc, start, dur, amp] = [params[s] for s in ['pop', 'sec', 'loc', 'start', 'dur', 'amp']]

        # add stim source
        netParams.stimSourceParams[key] = {
            'type': 'IClamp', 
            'delay': start, 
            'dur': dur, 
            'amp': amp}

        # connect stim source to target
        netParams.stimTargetParams[key + '_' + pop] = {
            'source': key,
            'conds': {'pop': pop},
            'sec': sec,
            'loc': loc}

# ------------------------------------------------------------------------------
# NetStim inputs
# ------------------------------------------------------------------------------
if cfg.addNetStim:
    for key in [k for k in dir(cfg) if k.startswith('NetStim')]:
        params = getattr(cfg, key, None)
        [pop, sec, loc, synMech, synMechWeightFactor, start, interval, noise, number, weight, delay] = \
            [params[s] for s in
             ['pop', 'sec', 'loc', 'synMech', 'synMechWeightFactor', 'start', 'interval', 'noise', 'number',
              'weight', 'delay']]

        # add stim source
        netParams.stimSourceParams[key] = {
            'type': 'NetStim', 
            'start': start, 
            'interval': interval, 
            'noise': noise,
            'number': number}

        # connect stim source to target
        netParams.stimTargetParams[key + '_' + pop] = {
            'source': key,
            'conds': {'pop': pop},
            'sec': sec,
            'loc': loc,
            'synMech': synMech,
            'weight': weight,
            'synMechWeightFactor': synMechWeightFactor,
            'delay': delay}