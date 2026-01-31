import json
import pickle
import numpy as np
import matplotlib.pyplot as plt

from sim_engine import make_workload_from_parquet, simulate


def save_plot(res, out_prefix: str):
    t = np.array(res["ts"]["t"])
    k = np.array(res["ts"]["k"])
    q_tasks = np.array(res["ts"]["q_tasks"])
    q_work = np.array(res["ts"]["q_work"])

    fig, ax = plt.subplots(3, 1, figsize=(10, 8), sharex=True)
    ax[0].plot(t / 60.0, k)
    ax[0].set_ylabel("Active VMs")
    ax[0].grid(True, alpha=0.3)

    ax[1].plot(t / 60.0, q_tasks)
    ax[1].set_ylabel("Queue tasks")
    ax[1].grid(True, alpha=0.3)

    ax[2].plot(t / 60.0, q_work)
    ax[2].set_ylabel("Queued work\n(dom*runtime)")
    ax[2].set_xlabel("Time (minutes)")
    ax[2].grid(True, alpha=0.3)

    for a in ax:
        a.axvline(120, linestyle="--", linewidth=1, color="k", alpha=0.5)

    fig.tight_layout()
    fig.savefig(f"{out_prefix}.png", dpi=160)
    plt.close(fig)


def main():
    workload_path = "data/processed/alibaba/alibaba_tasks_24h.parquet"
    tasks = make_workload_from_parquet(workload_path)

    with open("results/alibaba_mdp_policy.pkl", "rb") as f:
        mdp = pickle.load(f)
    q_bins = np.load("results/alibaba_mdp_q_bins.npy")

    k_min = int(mdp["k_min"])
    k_max = int(mdp["k_max"])
    delta = int(mdp["delta"])
    mdp_policy = mdp["policy"]

    print("Using k_min,k_max:", k_min, k_max)

    # Choose a reasonable static_k inside bounds
    static_k = int(np.clip((k_min + k_max) // 2, k_min, k_max))

    results = []

    # 1) static
    r_static = simulate(
        tasks=tasks,
        policy_name="static",
        k_min=k_min,
        k_max=k_max,
        static_k=static_k,
        delta=delta,
    )
    results.append(r_static)

    # 2) threshold (tune thresholds quickly if needed)

    print("THRESH PARAMS:", "up_th=", 10000.0, "down_th=", 2000.0, "step_up=", 20, "step_down=", 10)
    
    r_thr = simulate(
        tasks=tasks,
        policy_name="threshold",
        k_min=k_min,
        k_max=k_max,
        static_k=k_min,
        delta=delta,
        up_th=1000.0,
        down_th=2000.0,
        step_up=20,
        step_down=10,
    )
    results.append(r_thr)

    # 3) mdp
    r_mdp = simulate(
        tasks=tasks,
        policy_name="mdp",
        k_min=k_min,
        k_max=k_max,
        static_k=k_min,
        delta=delta,
        mdp_policy=mdp_policy,
        q_bins=q_bins,
    )
    results.append(r_mdp)

    # save json (summary only)
    summary = [
        {
            "policy": r["policy"],
            "tasks": r["tasks"],
            "mean_wait_s": r["mean_wait_s"],
            "p95_wait_s": r["p95_wait_s"],
            "p99_wait_s": r["p99_wait_s"],
            "sla60_violation": r["sla60_violation"],
            "sla120_violation": r["sla120_violation"],
            "vm_seconds": r["vm_seconds"],
        }
        for r in results
    ]

    with open("results/alibaba_summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    print("\nSummary:")
    for s in summary:
        print(s)

    # plots
    import os

    os.makedirs("figures", exist_ok=True)
    save_plot(r_static, "figures/alibaba_static")
    save_plot(r_thr, "figures/alibaba_threshold")
    save_plot(r_mdp, "figures/alibaba_mdp")


if __name__ == "__main__":
    main()
