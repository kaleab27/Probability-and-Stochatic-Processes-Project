from __future__ import annotations

from typing import Dict, Tuple
import numpy as np
import pandas as pd


def make_q_bins(values: np.ndarray, n_bins: int = 10) -> np.ndarray:
    qs = np.quantile(values, np.linspace(0, 1, n_bins + 1))
    qs[0] = 0.0
    # ensure strictly increasing edges
    qs = np.unique(qs)
    if len(qs) < 3:
        # fallback
        qs = np.array([0.0, float(values.max() + 1.0), float(values.max() + 2.0)])
        # ensure the last edge is strictly above any observed value (helps digitize)
        vmax = float(values.max()) if len(values) else 0.0
        if qs[-1] <= vmax:
            qs[-1] = vmax + 1e-6
    return qs


def train_mdp_policy(
    arrivals_work: np.ndarray,
    k_min: int,
    k_max: int,
    delta: int = 60,
    episodes: int = 300,
    gamma: float = 0.95,
    alpha: float = 0.15,
    eps: float = 0.1,
    w_k: float = 1.0,
    w_q: float = 1e-4,
    w_a: float = 0.1,
) -> Tuple[Dict[Tuple[int, int], int], np.ndarray]:
    """
    Aggregated MDP:
      q_{t+1} = max(0, q_t + w_in - k * delta)
    state: (k, q_bin), action: -100/-50/0/50/100
    reward: -(w_k*k + w_q*q_next + w_a*|a|)
    """
    rng = np.random.default_rng(7)
    q_bins = make_q_bins(arrivals_work, n_bins=12)
    n_q = len(q_bins) - 1
    actions = np.array([-100, -50, 0, 50, 100], dtype=int)

    Q = np.zeros((k_max + 1, n_q, len(actions)), dtype=float)

    def qbin(x: float) -> int:
        idx = int(np.digitize([x], q_bins)[0] - 1)
        if idx < 0:
            return 0
        if idx >= n_q:
            return n_q - 1
        return idx

    for _ in range(episodes):
        k = rng.integers(k_min, k_max + 1)
        q = 0.0
        for _t in range(120):  # ~2 hours at 60s per step
            qb = qbin(q)
            if rng.random() < eps:
                ai = rng.integers(0, len(actions))
            else:
                ai = int(np.argmax(Q[k, qb]))

            a = int(actions[ai])
            k2 = int(np.clip(k + a, k_min, k_max))

            w_in = float(rng.choice(arrivals_work))
            q2 = max(0.0, q + w_in - k2 * delta)

            r = -(w_k * k2 + w_q * q2 + w_a * abs(a))

            qb2 = qbin(q2)
            td = r + gamma * float(np.max(Q[k2, qb2])) - float(Q[k, qb, ai])
            Q[k, qb, ai] += alpha * td

            k, q = k2, q2

    policy: Dict[Tuple[int, int], int] = {}
    for k in range(k_min, k_max + 1):
        for qb in range(n_q):
            best = int(actions[int(np.argmax(Q[k, qb]))])
            policy[(k, qb)] = best

    return policy, q_bins


def main():
    inp = "data/processed/google_tasks_2h.parquet"
    df = pd.read_parquet(inp).sort_values("arrival_time_s").reset_index(drop=True)

    # convert to relative time
    t0 = float(df["arrival_time_s"].iloc[0])
    arr = df["arrival_time_s"].astype(float) - t0

    dom = np.maximum(df["cpu_req"].astype(float), df["mem_req"].astype(float))
    work = dom * df["runtime_s"].astype(float)  # resource-seconds

    delta = 60
    minute = np.floor(arr / delta).astype(int)
    arrivals_work = work.groupby(minute).sum().to_numpy()
    arrivals_work = arrivals_work[arrivals_work > 0]

    # basic sizing from arrival work
    mean_in = float(arrivals_work.mean())
    p99_in = float(np.quantile(arrivals_work, 0.99))
    k_min = max(1, int(np.ceil(0.3 * mean_in / delta)))
    k_max = max(k_min + 5, int(np.ceil(1.5 * p99_in / delta)))

    # cap for project feasibility (still trace-driven; just keeps state small)
    k_max = min(k_max, 1200)

    print("Estimated k_min,k_max:", k_min, k_max)

    policy, q_bins = train_mdp_policy(
        arrivals_work=arrivals_work,
        k_min=k_min,
        k_max=k_max,
        delta=delta,
        episodes=600,
        w_k=0.5,
        w_q=1e-2,
        w_a=1.0,
    )

    np.save("results/mdp_q_bins.npy", q_bins)
    import pickle

    with open("results/mdp_policy.pkl", "wb") as f:
        pickle.dump(
            {"policy": policy, "k_min": k_min, "k_max": k_max, "delta": delta}, f
        )

    print("Wrote results/mdp_policy.pkl and results/mdp_q_bins.npy")


if __name__ == "__main__":
    main()
