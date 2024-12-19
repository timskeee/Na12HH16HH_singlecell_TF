"""
netParams.py

High-level specifications for M1 network model using NetPyNE

"""

from netpyne import specs, sim
import pickle, json

try:
    from __main__ import cfg # import SimConfig object with params from parent module
except:
    from cfg import cfg

#########################################################################################
#
#NETWORK PARAMETERS
#
#########################################################################################
netParams = specs.NetParams() # Object class NetParams to store network parameters
#netParams.version = 1  # Can specify version when tuning or making changes to keep track

#########################################################################################
# Load cell rules previously saved using netpyne format
#########################################################################################
cellParamLabels = ['PT5B']
loadCellParams = cellParamLabels
saveCellParams = False

for ruleLabel in loadCellParams:
    netParams.loadCellParamsRule(label=ruleLabel, fileName='Na12HH16HH_TF.json')

#########################################################################################
# Specfication of cell rules not previously loaded using netpyne
# Includes importing from hoc template or python class, and setting additional params
#########################################################################################
if 'PT5B' not in loadCellParams:
    netParams.importCellParams('PT5B_full', 'Na12HHMModel_TF.py', 'Na12Model_TF')
    netParams.renameCellParamsSec(label='PT5B_full', oldSec='soma_0', newSec='soma')

    cellRule = netParams.cellParams['PT5B_full']

    cellRule['secs']['axon_0']['geom']['pt3d'] = [[0, 0, 0, 0]]  # stupid workaround that should be fixed in NetPyNE
    cellRule['secs']['axon_1']['geom']['pt3d'] = [
            [1e30, 1e30, 1e30, 1e30]]  # breaks in simulations btw. Just used for the perisom and below_soma sections

    nonSpiny = ['apic_0', 'apic_1']

    netParams.addCellParamsSecList(label='PT5B_full', secListName='perisom',
                                       somaDist=[0, 50])  # sections within 50 um of soma
    netParams.addCellParamsSecList(label='PT5B_full', secListName='below_soma',
                                       somaDistY=[-600, 0])  # sections within 0-300 um of soma

    for sec in nonSpiny:  # N.B. apic_1 not in `perisom` . `apic_0` and `apic_114` are
        if sec in cellRule['secLists']['perisom']:  # fixed logic
            cellRule['secLists']['perisom'].remove(sec)
    cellRule['secLists']['alldend'] = [sec for sec in cellRule['secs'] if
                                           ('dend' in sec or 'apic' in sec)]  # basal+apical
    cellRule['secLists']['apicdend'] = [sec for sec in cellRule['secs'] if ('apic' in sec)]  # apical
    cellRule['secLists']['spiny'] = [sec for sec in cellRule['secLists']['alldend'] if sec not in nonSpiny]


    cellRule['secs']['axon_0']['spikeGenLoc'] = 0.5
    cellRule['secs']['soma'][
        'threshold'] = 0.  # Lowering since it looks like v in soma is not reaching high voltages when spike occurs
    cellRule['secs']['axon_0'][
            'threshold'] = 0.  # Lowering since it looks like v in soma is not reaching high voltages when spike occurs

   # cellRule['secs']['soma']['mechs']['na12']['gbar'] *= cfg.PTNaFactor
   # for secName in cellRule['secLists']['apicdend']:
      #  cellRule['secs'][secName]['mechs']['na12']['gbar'] *= cfg.PTNaFactor

   # for i in range(len(cellRule['secs']['axon_0']['mechs']['na12']['gbar'])):
       # cellRule['secs']['axon_0']['mechs']['na12']['gbar'][i] *= cfg.PTNaFactor

    del netParams.cellParams['PT5B_full']['secs']['axon_0']['geom']['pt3d']
    del netParams.cellParams['PT5B_full']['secs']['axon_1']['geom']['pt3d']

    netParams.cellParams['PT5B_full']['conds'] = {'cellModel': 'HH_full', 'cellType': 'PT'}
    if saveCellParams: netParams.saveCellParamsRule(label='PT5B_full', fileName='Na12HH16HH_TF.json')

# ------------------------------------------------------------------------------
# Current inputs (IClamp)
# ------------------------------------------------------------------------------
if cfg.addIClamp:
    for key in [k for k in dir(cfg) if k.startswith('IClamp')]:
        params = getattr(cfg, key, None)
        [pop, sec, loc, start, dur, amp] = [params[s] for s in ['pop', 'sec', 'loc', 'start', 'dur', 'amp']]

        # cfg.analysis['plotTraces']['include'].append((pop,0))  # record that pop

        # add stim source
        netParams.stimSourceParams[key] = {'type': 'IClamp', 'delay': start, 'dur': dur, 'amp': amp}

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
        [pop, ynorm, sec, loc, synMech, synMechWeightFactor, start, interval, noise, number, weight, delay] = \
            [params[s] for s in
             ['pop', 'ynorm', 'sec', 'loc', 'synMech', 'synMechWeightFactor', 'start', 'interval', 'noise', 'number',
              'weight', 'delay']]

        # cfg.analysis['plotTraces']['include'] = [(pop,0)]

        if synMech == ESynMech:
            wfrac = cfg.synWeightFractionEE
        elif synMech == SOMESynMech:
            wfrac = cfg.synWeightFractionSOME
        else:
            wfrac = [1.0]

        # add stim source
        netParams.stimSourceParams[key] = {'type': 'NetStim', 'start': start, 'interval': interval, 'noise': noise,
                                           'number': number}

        # connect stim source to target
        # for i, syn in enumerate(synMech):
        netParams.stimTargetParams[key + '_' + pop] = {
            'source': key,
            'conds': {'pop': pop, 'ynorm': ynorm},
            'sec': sec,
            'loc': loc,
            'synMech': synMech,
            'weight': weight,
            'synMechWeightFactor': synMechWeightFactor,
            'delay': delay}
