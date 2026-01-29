import json
import numpy as np
from sim_engine import make_workload_from_parquet, simulate

tasks = make_workload_from_parquet("data/processed/google_tasks_2h.parquet")

k_min, k_max, delta = 186, 1200, 60
ks = [200, 300, 400, 500, 600, 700, 800, 900, 1000, 1100]

rows = []
for k in ks:
    r = simulate(
        tasks=tasks,
        policy_name="static",
        k_min=k_min,
        k_max=k_max,
        static_k=k,
        delta=delta,
    )
    rows.append(
        {
            "static_k": k,
            "mean_wait_s": r["mean_wait_s"],
            "p95_wait_s": r["p95_wait_s"],
            "vm_hours": r["vm_seconds"] / 3600,
        }
    )
print(json.dumps(rows, indent=2))