import json
import os
import matplotlib.pyplot as plt


def main():
    static_path = "results/alibaba_static_sweep.json"
    summary_path = "results/alibaba_summary.json"
    out_path = "figures/alibaba_cost_vs_sla.png"

    with open(static_path) as f:
        static_rows = json.load(f)

    with open(summary_path) as f:
        summary_rows = json.load(f)

    x_static = [r["vm_hours"] for r in static_rows]
    y_static = [r["sla60"] for r in static_rows]
    static_sorted = sorted(zip(x_static, y_static), key=lambda t: t[0])
    x_static = [a for a, _ in static_sorted]
    y_static = [b for _, b in static_sorted]

    points = []
    for r in summary_rows:
        points.append(
            {
                "label": r["policy"],
                "vm_hours": r["vm_seconds"] / 3600.0,
                "sla60": r["sla60_violation"],
            }
        )

    os.makedirs("figures", exist_ok=True)

    plt.figure(figsize=(9, 6))
    plt.plot(
        x_static,
        y_static,
        marker="o",
        linewidth=2,
        markersize=5,
        label="Static provisioning (sweep)",
    )

    colors = {"threshold": "tab:orange", "mdp": "tab:green", "static": "tab:blue"}

    for p in points:
        c = colors.get(p["label"], "tab:red")
        plt.scatter(p["vm_hours"], p["sla60"], s=120, color=c, zorder=5)
        plt.annotate(
            p["label"],
            (p["vm_hours"], p["sla60"]),
            textcoords="offset points",
            xytext=(8, 6),
            ha="left",
            fontsize=10,
        )

    plt.xlabel("Cost (VM-hours)")
    plt.ylabel("SLA60 violation rate  P(wait > 60s)")
    plt.title("Cost vs SLA (Alibaba OpenB pods, busiest 24-hour window)")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=180)
    print("Wrote:", out_path)


if __name__ == "__main__":
    main()
