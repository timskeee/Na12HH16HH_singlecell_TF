from netpyne.batchtools.search import search
<<<<<<< HEAD
import numpy


# Create parameter grid for search
params = {'IClamp1.amp': [0.3, 0.4, 0.5, 0.6]}

# use batch_sge_config if running on a
shell_config = {'command': 'python init.py',}

search(job_type = 'sh',
=======
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
>>>>>>> 8369f25d633b5871e67604248cd807af194ce57d
       params          = params,
       run_config      = shell_config,
       label           = "grid_search",
       output_path     = "./grid_batch",
       checkpoint_path = "./ray",
<<<<<<< HEAD
       algorithm       = "grid",
=======
       num_samples     = 1,
       metric          = 'epsp',
       mode            = 'min',
       algorithm       = "variant_generator",
>>>>>>> 8369f25d633b5871e67604248cd807af194ce57d
       max_concurrent  = 9)
