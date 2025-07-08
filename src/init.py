from netpyne import sim
from cfg import cfg
from netParams import netParams

print(f"🚀 Starting sim with nmdaTau1 = {cfg.nmdaTau1}", flush=True)

sim.createSimulateAnalyze(netParams, cfg)

print("✅ Simulation complete", flush=True)
