from netpyne.batchtools.search import search
import numpy
import plotly.express as px

# Create parameter grid for search
params = {'IClamp1.amp': [0.3, 0.4, 0.5, 0.6]}

# use batch_sge_config if running on a
shell_config = {'command': 'python init_fi.py',}

results = search(job_type = 'sh',
       comm_type = 'socket',
       params          = params,
       run_config      = shell_config,
       label           = "grid_search",
       output_path     = "./grid_batch",
       checkpoint_path = "./ray",
       algorithm       = "grid",
       metric          = 'rate',
       mode            = 'max',
       max_concurrent  = 4)

df = results.data

fig = px.scatter(df, x='amp', y='rate')

fig.write_html('grid_fi.html')
