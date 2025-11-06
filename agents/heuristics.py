# agents/heuristics.py
from __future__ import annotations
import numpy as np
from typing import List, Optional, Tuple

def pick_fifo(obs: np.ndarray, max_jobs: int) -> int:
    # FIFO: pick the earliest in the buffer (lowest index that is non-empty)
    table = obs[:-2].reshape(max_jobs, 3)  # [remaining, deadline_left, age]
    for i in range(max_jobs):
        if table[i,0] > 0:  # job exists
            return i
    return max_jobs  # NOOP

def pick_sjf(obs: np.ndarray, max_jobs: int) -> int:
    table = obs[:-2].reshape(max_jobs, 3)
    candidates = [(i, table[i,0]) for i in range(max_jobs) if table[i,0] > 0]
    if not candidates: return max_jobs
    return min(candidates, key=lambda x: x[1])[0]

def pick_srpt(obs: np.ndarray, max_jobs: int) -> int:
    # shortest remaining processing time = same as SJF here because remaining==duration for new jobs
    return pick_sjf(obs, max_jobs)

def pick_edf(obs: np.ndarray, max_jobs: int) -> int:
    table = obs[:-2].reshape(max_jobs, 3)
    # pick smallest positive deadline_left; if none have deadlines, fallback to FIFO
    candidates = [(i, table[i,1]) for i in range(max_jobs) if table[i,1] > 0 and table[i,0] > 0]
    if candidates:
        return min(candidates, key=lambda x: x[1])[0]
    return pick_fifo(obs, max_jobs)

def pick_deadline_priority(obs: np.ndarray, max_jobs: int) -> int:
    # simple score = w1*(deadline urgency) + w2*(1/remaining) + w3*(age)
    table = obs[:-2].reshape(max_jobs, 3)
    scores: List[Tuple[int, float]] = []
    for i in range(max_jobs):
        rem, ddl, age = table[i]
        if rem <= 0: continue
        urgency = 1.0 / (ddl + 1.0) if ddl > 0 else 0.1
        scores.append((i, 1.5*urgency + 1.0/(rem+1.0) + 0.1*age))
    if not scores: return max_jobs
    scores.sort(key=lambda x: -x[1])
    return scores[0][0]