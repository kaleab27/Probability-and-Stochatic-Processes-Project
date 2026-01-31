import json
import pickle
import sys

sys.path.append("src")
from sim_engine import make_workload_from_parquet, simulate


def main():
    tasks = make_workload_from_parquet(
        "data/processed/alibaba/alibaba_tasks_24h.parquet"
    )

    mdp = pickle.load(open("results/alibaba_mdp_policy.pkl", "rb"))
    k_min = int(mdp["k_min"])
    k_max = int(mdp["k_max"])
    delta = int(mdp["delta"])

    # Alibaba window is sparse; keep sweep small/realistic.
    # Start from k_min and go up to 60 (adjust if needed).
    ks = list(range(k_min, 61, 3))

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
                "vm_hours": r["vm_seconds"] / 3600.0,
                "sla60": r["sla60_violation"],
                "sla120": r["sla120_violation"],
                "p99_wait_s": r["p99_wait_s"],
            }
        )

    print(json.dumps(rows, indent=2))


if __name__ == "__main__":
    main()
