# envs/jobshop_env.py
from __future__ import annotations
import gymnasium as gym
from gymnasium import spaces
import numpy as np
from dataclasses import dataclass
from typing import Dict, Any, Optional, Tuple, List

@dataclass
class Job:
    duration: int
    deadline: Optional[int]  # None means no deadline
    age: int = 0             # how long it has waited
    remaining: Optional[int] = None  # for preemption/SRPT

    def __post_init__(self):
        if self.remaining is None:
            self.remaining = self.duration

class JobShopEnv(gym.Env):
    """
    Simple queue + multi-slot scheduler.
    Time is discrete. At each step you can choose ONE job to start or preemptively continue,
    and there are `num_slots` parallel machines.
    For simplicity: we fill idle slots by repeatedly asking the agent to pick a job (K picks).
    """
    metadata = {"render.modes": ["human"]}

    def __init__(
        self,
        max_jobs: int = 20,
        num_slots: int = 2,
        arrival_rate: float = 0.5,   # prob of 1 new job per step (Bernoulli)
        duration_low: int = 1,
        duration_high: int = 10,
        deadline_prob: float = 0.6,  # chance job has deadline
        deadline_slack_low: int = 5,
        deadline_slack_high: int = 20,
        preemption: bool = True,
        seed: Optional[int] = None,
    ):
        super().__init__()
        self.max_jobs = max_jobs
        self.num_slots = num_slots
        self.arrival_rate = arrival_rate
        self.duration_low = duration_low
        self.duration_high = duration_high
        self.deadline_prob = deadline_prob
        self.deadline_slack_low = deadline_slack_low
        self.deadline_slack_high = deadline_slack_high
        self.preemption = preemption

        self.np_random, _ = gym.utils.seeding.np_random(seed)
        self.t = 0

        # State representation (child-simple):
        # For each of max_jobs we store [remaining, deadline_left, age]
        # If slot is empty: array row = zeros
        low = np.array([[0, 0, 0]] * self.max_jobs, dtype=np.float32).flatten()
        high = np.array([[self.duration_high, 1000, 1000]] * self.max_jobs, dtype=np.float32).flatten()

        # Observation: jobs table + time + free_slots
        self.observation_space = spaces.Box(
            low=np.concatenate([low, np.array([0, 0], dtype=np.float32)]),
            high=np.concatenate([high, np.array([10000, self.num_slots], dtype=np.float32)]),
            dtype=np.float32
        )
        # Action: pick an index [0..max_jobs-1] OR "noop" (do nothing) via index = max_jobs
        self.action_space = spaces.Discrete(self.max_jobs + 1)

        self.jobs: List[Optional[Job]] = []
        self.running: List[Optional[Tuple[int, Job]]] = [None] * self.num_slots  # (job_index, job)
        self.reset()

    def legal_action_mask(self) -> np.ndarray:
        # Return a boolean mask of length (max_jobs + 1), True = selectable
        # Reuse the internal _valid_mask() but make a public helper for wrappers.
        return self._valid_mask()

    def _spawn_jobs(self):
        # with probability arrival_rate, 1 job arrives
        if self.np_random.random() < self.arrival_rate and len(self.jobs) < self.max_jobs:
            duration = int(self.np_random.integers(self.duration_low, self.duration_high + 1))
            if self.np_random.random() < self.deadline_prob:
                slack = int(self.np_random.integers(self.deadline_slack_low, self.deadline_slack_high + 1))
                deadline = self.t + slack
            else:
                deadline = None
            self.jobs.append(Job(duration=duration, deadline=deadline))

    def _valid_mask(self):
        # You can pick any job that exists and is not already running, or choose NOOP
        mask = np.zeros(self.max_jobs + 1, dtype=bool)
        existing_indices = [i for i, j in enumerate(self.jobs) if j is not None]
        for i in existing_indices:
            mask[i] = True
        mask[self.max_jobs] = True  # NOOP allowed
        return mask

    def _pack_obs(self):
        table = np.zeros((self.max_jobs, 3), dtype=np.float32)
        for i, j in enumerate(self.jobs[: self.max_jobs]):
            if j:
                deadline_left = (j.deadline - self.t) if j.deadline is not None else 0
                table[i] = [j.remaining, max(0, deadline_left), j.age]
        free_slots = sum(1 for x in self.running if x is None)
        obs = np.concatenate([table.flatten(), np.array([self.t, free_slots], dtype=np.float32)], axis=0)
        return obs

    def reset(self, seed: Optional[int] = None, options: Optional[Dict[str, Any]] = None):
        if seed is not None:
            self.np_random, _ = gym.utils.seeding.np_random(seed)
        self.t = 0
        self.jobs = []
        self.running = [None] * self.num_slots
        for _ in range(min(3, self.max_jobs)):  # start with a few jobs
            self._spawn_jobs()
        obs = self._pack_obs()
        info = {"action_mask": self._valid_mask()}
        return obs, info

    def step(self, action: int):
        # 1) Place job if a slot is free and action picks a job
        reward = 0.0
        done = False

        # age everyone
        for j in self.jobs:
            if j: j.age += 1

        free_slot_ids = [i for i, s in enumerate(self.running) if s is None]
        if action < self.max_jobs and free_slot_ids and action < len(self.jobs) and self.jobs[action] is not None:
            job = self.jobs[action]
            # start running this job
            slot = free_slot_ids[0]
            self.running[slot] = (action, job)

        # 2) Advance time by 1 step
        self.t += 1

        # 3) Execute running jobs
        finished_idxs = []
        for i, pair in enumerate(self.running):
            if pair is not None:
                idx, job = pair
                job.remaining -= 1
                if job.remaining <= 0:
                    # finished
                    finished_idxs.append(i)

        # 4) compute reward & remove finished
        for i in finished_idxs:
            idx, job = self.running[i]
            self.running[i] = None
            # Reward: +1 for finishing; penalty if deadline missed
            if job.deadline is not None and self.t > job.deadline:
                reward += 0.0  # no bonus if late
            else:
                reward += 1.0

            # remove job from queue
            if 0 <= idx < len(self.jobs):
                self.jobs[idx] = None

        # 5) deadlines expiring cause negative event (miss)
        miss_penalty = 0.0
        for i, j in enumerate(self.jobs):
            if j and j.deadline is not None and self.t > j.deadline and j.remaining > 0:
                # missed deadline; keep job but count penalty
                miss_penalty -= 0.2
        reward += miss_penalty

        # 6) Add new jobs
        self._spawn_jobs()

        # 7) Clean None holes to keep list compact
        self.jobs = [j for j in self.jobs if j is not None]

        # 8) Done?
        done = self.t >= 500  # fixed episode length to make life simple

        obs = self._pack_obs()
        info = {"action_mask": self._valid_mask()}
        return obs, reward, done, False, info

    def render(self):
        print(f"t={self.t} | running={[(i, j.remaining if j else None) for i,(i2,j) in enumerate(self.running) if j]} | queue={len(self.jobs)}")