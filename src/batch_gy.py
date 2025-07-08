# run_batch.py
from netpyne import sim
from cfg import cfg
from netParams import netParams
import copy
import os
import subprocess
import numpy as np


# Base NetStim1 parameters
base_params = {
    #'NSTIM_DELAY': 0.01,
    #'NSTIM_NUMBER': 20.0,
    #'NSTIM_INTERVAL': 1,
    'NSTIM_WEIGHT':1
}

#factors = [20,50]
factors = np.linspace(10, 20, 20)  # More granular range from 0.1 to 10

for key, base_val in base_params.items():
    for factor in factors:
        new_val = base_val * factor
        env = copy.deepcopy(os.environ)
        env[key] = str(new_val)
        env['OUT_FILENAME'] = f"data/GY/"
        #env['OUT_FILENAME'] = f"data/GY/{key.lower()}_{new_val:.4g}"

        print(f"🔁 Running {key} = {new_val}", flush=True)
        subprocess.run(["python3", "init.py"], env=env)
        print(f"✅ Done with {key} = {new_val}\n", flush=True)
