from __future__ import annotations

from dataclasses import dataclass
from typing import Deque, Dict, List, Tuple
from collections import deque
import heapq
import numpy as np


@dataclass
class Task:
    arrival: float
    runtime: float
    cpu: float
    mem: float


@dataclass
class VM:
    used_cpu: float = 0.0
    used_mem: float = 0.0


def dominant(task: Task) -> float:
    return max(task.cpu, task.mem)


def queued_work(queue: Deque[Task]) -> float:
    # "resource-seconds" backlog approximation for MDP and scaling signals
    return float(sum(dominant(t) * t.runtime for t in queue))


def make_workload_from_parquet(path: str) -> List[Task]:
    import pandas as pd

    df = pd.read_parquet(path).sort_values("arrival_time_s").reset_index(drop=True)

    # shift time so simulation starts at 0
    t0 = float(df["arrival_time_s"].iloc[0])
    arrivals = (df["arrival_time_s"].astype(float) - t0).to_numpy()
    runtimes = df["runtime_s"].astype(float).to_numpy()
    cpu = df["cpu_req"].astype(float).to_numpy()
    mem = df["mem_req"].astype(float).to_numpy()

    tasks = [
        Task(float(a), float(r), float(c), float(m))
        for a, r, c, m in zip(arrivals, runtimes, cpu, mem)
    ]
    return tasks


def simulate(
    tasks: List[Task],
    policy_name: str,
    k_min: int,
    k_max: int,
    static_k: int = 30,
    delta: int = 60,
    up_th: float = 3000.0,
    down_th: float = 500.0,
    mdp_policy: Dict[Tuple[int, int], int] | None = None,
    q_bins: np.ndarray | None = None,
    step_up: int = 50,
    step_down: int = 20,
) -> Dict:
    """
    Event-driven simulation with:
    - FIFO queue
    - First-Fit placement across homogeneous VMs (cpu=1, mem=1)
    - Scaling decisions every `delta` seconds
    """
    assert tasks, "No tasks provided."

    # state
    now = 0.0
    i = 0  # next task index
    n = len(tasks)

    vms: List[VM] = [VM() for _ in range(static_k)]
    k = static_k

    queue: Deque[Task] = deque()
    completions: List[Tuple[float, int, float, float]] = []
    # heap items: (end_time, vm_id, cpu, mem)

    # metrics
    waits: List[float] = []
    vm_time = 0.0  # integral of k over time
    last_t = 0.0

    # time series snapshots at control ticks
    ts_t = []
    ts_k = []
    ts_q_tasks = []
    ts_q_work = []

    def integrate(to_t: float) -> None:
        nonlocal vm_time, last_t
        dt = to_t - last_t
        if dt > 0:
            vm_time += k * dt
            last_t = to_t

    def try_schedule() -> None:
        nonlocal completions
        # FIFO: try to place the head; if it can't fit anywhere, stop
        while queue:
            t = queue[0]
            placed = False
            for vm_id, vm in enumerate(vms[:k]):
                if vm.used_cpu + t.cpu <= 1.0 and vm.used_mem + t.mem <= 1.0:
                    queue.popleft()
                    vm.used_cpu += t.cpu
                    vm.used_mem += t.mem
                    end_t = now + t.runtime
                    heapq.heappush(completions, (end_t, vm_id, t.cpu, t.mem))
                    waits.append(now - t.arrival)
                    placed = True
                    break
            if not placed:
                break

    def scale_to(new_k: int) -> None:
        nonlocal k, vms
        new_k = int(max(k_min, min(k_max, new_k)))
        if new_k == k:
            return

        if new_k > k:
            for _ in range(new_k - k):
                vms.append(VM())
            k = new_k
            return

        # Scale down by "deactivating" VMs from the end only if they are idle.
        # We do NOT delete entries from `vms`, to keep completion vm_id indices valid.
        while k > new_k:
            vm = vms[k - 1]
            if vm.used_cpu == 0 and vm.used_mem == 0:
                k -= 1
            else:
                break

    def qbin(x: float) -> int:
        assert q_bins is not None
        n_q = len(q_bins) - 1
        idx = int(np.digitize([x], q_bins)[0] - 1)
        if idx < 0:
            return 0
        if idx >= n_q:
            return n_q - 1
        return idx

    next_control = 0.0

    while True:
        next_arrival = tasks[i].arrival if i < n else float("inf")
        next_finish = completions[0][0] if completions else float("inf")
        next_event = min(next_arrival, next_finish, next_control)

        if next_event == float("inf"):
            break

        integrate(next_event)
        now = next_event

        # process all finishes at this time
        while completions and completions[0][0] <= now + 1e-9:
            _, vm_id, cpu, mem = heapq.heappop(completions)
            vms[vm_id].used_cpu -= cpu
            vms[vm_id].used_mem -= mem

        # process all arrivals at this time
        while i < n and tasks[i].arrival <= now + 1e-9:
            queue.append(tasks[i])
            i += 1

        # schedule if possible
        try_schedule()

        # control tick
        if abs(now - next_control) <= 1e-9:
            qw = queued_work(queue)
            ts_t.append(now)
            ts_k.append(k)
            ts_q_tasks.append(len(queue))
            ts_q_work.append(qw)

            if policy_name == "static":
                pass
            elif policy_name == "threshold":
                if qw > up_th:
                    scale_to(k + step_up)
                elif qw < down_th:
                    scale_to(k - step_down)
            elif policy_name == "mdp":
                if mdp_policy is None or q_bins is None:
                    raise ValueError("mdp_policy and q_bins required for mdp.")
                s = (k, qbin(qw))
                a = mdp_policy.get(s, 0)  # default 'do nothing'
                scale_to(k + a)
            else:
                raise ValueError(f"Unknown policy {policy_name}")

            next_control += delta

        # stopping condition: all tasks arrived and queue empty and no running tasks
        if i >= n and not queue and not completions:
            break

    waits_arr = np.array(waits) if waits else np.array([0.0])

    p95 = float(np.quantile(waits_arr, 0.95))
    p99 = float(np.quantile(waits_arr, 0.99))
    sla60 = float(np.mean(waits_arr > 60.0))
    sla120 = float(np.mean(waits_arr > 120.0))

    return {
        "policy": policy_name,
        "tasks": n,
        "mean_wait_s": float(waits_arr.mean()),
        "p95_wait_s": p95,
        "p99_wait_s": p99,
        "sla60_violation": sla60,
        "sla120_violation": sla120,
        "vm_seconds": float(vm_time),
        "ts": {
            "t": ts_t,
            "k": ts_k,
            "q_tasks": ts_q_tasks,
            "q_work": ts_q_work,
        },
    }
