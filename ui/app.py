# ui/app.py
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]  # ...\cloud-rl-scheduler
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
    
import streamlit as st
import numpy as np
import pandas as pd
from envs.jobshop_env import JobShopEnv
from agents.heuristics import pick_fifo, pick_sjf, pick_srpt, pick_edf, pick_deadline_priority
from agents.rl_policy import RLPolicy

st.set_page_config(page_title="RL Scheduler Demo", layout="wide")

st.title("🧠 RL vs Heuristics: Job Scheduler")
st.caption("Compare simple rules with a learned policy on a synthetic cluster.")

col1, col2, col3 = st.columns(3)
max_jobs = col1.slider("Max jobs", 10, 50, 20, step=1)
num_slots = col2.slider("Num slots (machines)", 1, 8, 2, step=1)
arrival_rate = col3.slider("Arrival rate (0..1)", 0.0, 1.0, 0.5, step=0.05)

deadline_prob = st.slider("Deadline probability", 0.0, 1.0, 0.6, step=0.05)
duration_low, duration_high = st.slider("Job duration range", 1, 50, (1, 10))
episodes = st.slider("Episodes per agent", 1, 20, 5)

model_path = st.text_input("RL model path (optional)", value="artifacts/maskable_ppo.zip")

env = JobShopEnv(max_jobs=max_jobs, num_slots=num_slots, arrival_rate=arrival_rate,
                 duration_low=duration_low, duration_high=duration_high,
                 deadline_prob=deadline_prob)

# Small inline evaluator (quick)
def quick_score(policy_fn):
    obs, info = env.reset()
    done = False
    total_reward = 0.0
    while not done:
        mask = info.get("action_mask", None)
        action = policy_fn(obs, mask)
        obs, r, done, trunc, info = env.step(action)
        total_reward += r
    return total_reward

# Heuristic wrappers
def make_h(fn):
    def _f(obs, mask):  # same signature
        return fn(obs, env.max_jobs)
    return _f

agents = {
    "fifo": make_h(pick_fifo),
    "sjf": make_h(pick_sjf),
    "srpt": make_h(pick_srpt),
    "edf": make_h(pick_edf),
    "deadline": make_h(pick_deadline_priority),
}

if model_path and len(model_path.strip()) > 0:
    try:
        rl = RLPolicy(model_path)
        def rl_fn(obs, mask): return rl.act(obs, mask)
        agents["maskable_ppo"] = rl_fn
    except Exception as e:
        st.warning(f"Could not load RL model: {e}")

if st.button("Run Comparison"):
    rows = []
    for name, fn in agents.items():
        scores = []
        for ep in range(episodes):
            scores.append(quick_score(fn))
        rows.append({"agent": name, "mean_reward": float(np.mean(scores)), "std": float(np.std(scores))})
    df = pd.DataFrame(rows).sort_values("mean_reward", ascending=False)
    st.bar_chart(df.set_index("agent")["mean_reward"])
    st.dataframe(df, use_container_width=True)

st.divider()
st.markdown("**How to use:** Train an RL model with `python -m agents.rl_train`, then paste `artifacts/maskable_ppo.zip` above and click **Run Comparison**.")
print("DEBUG repo root in sys.path?", str(REPO_ROOT) in sys.path)
print("DEBUG file exists?", (REPO_ROOT / "envs" / "jobshop_env.py").exists())
