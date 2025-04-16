import numpy as np
import matplotlib.pyplot as plt
from netpyne import sim

from batch import params



# Analyze spike data and compute firing rate
spike_times = sim.analysis.popAvgRates() # Get spike times
value_list = spike_times['PT5B']
# Store firing rate for current



#  Plot the F-I curve
plt.figure(figsize=(8, 6))
plt.plot(params, value_list, marker='o', linestyle='-', color='b')
plt.xlabel('Injected Current (nA)')
plt.ylabel('Firing Rate (Hz)')
plt.title('F-I Curve')
plt.show()
plt.savefig('data/F-I.png')