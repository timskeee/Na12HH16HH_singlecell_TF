# run_batch.py
from netpyne import sim
from cfg import cfg
from netParams import netParams
import copy

# Base NetStim1 dictionary
base_params = cfg.NetStim1

# Keys to modify and their original values
param_keys =["weight"]
# [k for k in base_params if isinstance(base_params[k], (int, float))]

# Multipliers for parameter sweeping
multipliers = [0.01, 1.0, 10.0]

for key in param_keys:
    original_value = base_params[key]
    for multiplier in multipliers:
        cfg.NetStim1 = copy.deepcopy(base_params)  # reset to original
        cfg.NetStim1[key] = original_value * multiplier
        
        safe_key = key.replace('_', '')  
        safe_val = f"{original_value * multiplier:.6g}" 
        
        cfg.filename = f"data/GY/netstim_{safe_key}_{safe_val}"

        print(f"\n🚀 Running simulation with {key} = {cfg.NetStim1[key]}\n", flush=True)

        sim.createSimulateAnalyze(netParams, cfg)

        print(f"✅ Finished simulation with {key} = {cfg.NetStim1[key]}\n", flush=True)
