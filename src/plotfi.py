import numpy as np
import matplotlib.pyplot as plt
from netpyne import specs, sim
from netParams import netParams
from cfg import cfg


# Step 3: Define stimulation protocol (current injection range)
currents = np.linspace(0.1, 2.0, 20)  # Range of injected currents (nA)
firing_rates = []  # List to store firing rates for each current value


for current in currents:
    # Step 4.1: Define and add current clamp stimulus
    cfg.addIClamp = 1
    cfg.IClamp1 = {'pop': 'PT5B', 'sec': 'soma', 'loc': 0.5, 'start': 100, 'dur': 1000, 'amp': currents}

    # Step 4.2: Run the simulation
    sim.createSimulateAnalyze(netParams, cfg)

    # Step 4.3: Analyze spike data and compute firing rate
    spike_times = sim.analysis.popAvgRates() # Get spike times
    value_list = spike_times['PT5B']
    # Store firing rate for current
    firing_rates.append(value_list)


# Step 5: Plot the F-I curve
plt.figure(figsize=(8, 6))
plt.plot(currents, firing_rates, marker='o', linestyle='-', color='b')
plt.xlabel('Injected Current (nA)')
plt.ylabel('Firing Rate (Hz)')
plt.title('F-I Curve')
plt.show()
plt.savefig('data/F-I.png')