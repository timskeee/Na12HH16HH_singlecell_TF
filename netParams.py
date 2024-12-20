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

###############################################################################
# Cell parameters
###############################################################################

#------------------------------------------------------------------------------
# Load cell rules previously saved using netpyne format
#------------------------------------------------------------------------------
loadCellParams = True
saveCellParams = True

if loadCellParams:
   netParams.loadCellParamsRule(label='PT5B_full', fileName='Na12HH16HH.json')

#------------------------------------------------------------------------------
# Includes importing from hoc template or python class, and setting additional params
#------------------------------------------------------------------------------

if not loadCellParams:
    # import cell model from NEURON/Python code
    netParams.importCellParams('PT5B_full', 'Na12HH16HHModel_TF.py', 'Na12Model_TF')

    # rename soma to conform to netpyne standard
    netParams.renameCellParamsSec(label='PT5B_full', oldSec='soma_0', newSec='soma')

    # set variable so easier to work with below
    cellRule = netParams.cellParams['PT5B_full']

    # create some section lists useful to define the locations of synapses
    cellRule['secLists']['alldend'] = [sec for sec in cellRule['secs'] if ('dend' in sec or 'apic' in sec)]  # basal+apical
    cellRule['secLists']['apicdend'] = [sec for sec in cellRule['secs'] if ('apic' in sec)]  # apical
    nonSpiny = ['apic_0', 'apic_1']
    cellRule['secLists']['spiny'] = [sec for sec in cellRule['secLists']['alldend'] if sec not in nonSpiny]

    # set the spike generation location to the axon (default in NEURON is the soma)
    cellRule['secs']['axon_0']['spikeGenLoc'] = 0.5

    # Lowering V threshold since it looks like v in soma is not reaching high voltages when spike occurs
    cellRule['secs']['soma']['threshold'] = 0.
    cellRule['secs']['axon_0']['threshold'] = 0.

    # svae to json with all the above modifications so easier/faster to load
    if saveCellParams: netParams.saveCellParamsRule(label='PT5B_full', fileName='Na12HH16HH_TF.json')


###############################################################################
# Population parameters
###############################################################################

netParams.popParams['PT5B'] =	{'cellType': 'PT5B_full', 'numCells': 1}


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