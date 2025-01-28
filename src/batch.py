from netpyne.batchtools.search import search
import os
import json
import numpy as np

# grid search
params = {'nmdaTau1': [10, 15, 20]}

# use batch_shell_config if running directly on the machine
shell_config = {'command': 'mpiexec -np 4 nrniv -python -mpi init.py',}

# use batch_shell_config if running directly on the machine
shell_config = {'command': 'mpiexec -np 4 nrniv -python -mpi init.py',}


run_config = shell_config
search(job_type = 'sge',
       comm_type       = "socket",
       params          = params,
       run_config      = shell_config,
       label           = "grid_search",
       output_path     = "./grid_batch",
       checkpoint_path = "./ray",
       num_samples     = 1,
       metric          = 'epsp',
       mode            = 'min',
       algorithm       = "variant_generator",
       max_concurrent  = 9)
