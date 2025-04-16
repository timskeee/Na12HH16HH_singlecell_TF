import numpy as np
import matplotlib.pyplot as plt
from netpyne import sim

from batch import params

firing_rates = []  # List to store firing rates for each current value

# Analyze spike data and compute firing rate
spike_times = sim.analysis.popAvgRates() # Get spike times
value_list = spike_times['PT5B']
# Store firing rate for current
firing_rates.append(value_list)


#  Plot the F-I curve
plt.figure(figsize=(8, 6))
plt.plot(params, firing_rates, marker='o', linestyle='-', color='b')
plt.xlabel('Injected Current (nA)')
plt.ylabel('Firing Rate (Hz)')
plt.title('F-I Curve')
plt.show()
plt.savefig('data/F-I.png')