import json
import pickle

from sim_engine import make_workload_from_parquet, simulate


def main():
    tasks = make_workload_from_parquet("data/processed/google_tasks_2h.parquet")

    with open("results/mdp_policy.pkl", "rb") as f:
        mdp = pickle.load(f)

    k_min = int(mdp["k_min"])
    k_max = int(mdp["k_max"])
    delta = int(mdp["delta"])

    ks = list(range(800, 901, 10))

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
                "p95_wait_s": r["p95_wait_s"],
                "sla60": r["sla60_violation"],
            }
        )

    print(json.dumps(rows, indent=2))


if __name__ == "__main__":
    main()
