# evaluation/metrics.py
from __future__ import annotations
import numpy as np
import pandas as pd

def p95(x): return float(np.percentile(x, 95)) if len(x) else 0.0
def p99(x): return float(np.percentile(x, 99)) if len(x) else 0.0

def jain_fairness(values):
    # Jain's fairness index
    arr = np.asarray(values, dtype=float)
    if len(arr) == 0: return 1.0
    num = (arr.sum())**2
    den = len(arr) * (arr**2).sum()
    return float(num/den) if den > 0 else 1.0

def summarize_episode(log):
    """
    log: dict with lists
      - completion_times
      - finished_on_time (bool list)
      - queue_sizes
      - slot_utilizations  (per-step fraction)
      - ages_on_finish
    """
    comp = np.array(log["completion_times"])
    ontime = np.array(log["finished_on_time"], dtype=bool)
    q = np.array(log["queue_sizes"])
    util = np.array(log["slot_utilizations"])
    ages = np.array(log["ages_on_finish"]) if len(log["ages_on_finish"]) else np.array([0])

    miss_rate = 1.0 - (ontime.mean() if len(ontime) else 0.0)
    throughput = float(len(comp))
    return {
        "miss_rate": float(miss_rate),
        "throughput": throughput,
        "mean_ct": float(comp.mean() if len(comp) else 0.0),
        "p95_ct": p95(comp),
        "p99_ct": p99(comp),
        "avg_queue": float(q.mean() if len(q) else 0.0),
        "max_queue": float(q.max() if len(q) else 0.0),
        "utilization": float(util.mean() if len(util) else 0.0),
        "fairness_age": jain_fairness(ages if len(ages) else [1]),
    }

def to_dataframe(results_dict):
    # results_dict: {agent_name: [summary_rows ...]}
    rows = []
    for agent, stats_list in results_dict.items():
        for s in stats_list:
            s2 = {"agent": agent}
            s2.update(s)
            rows.append(s2)
    return pd.DataFrame(rows)