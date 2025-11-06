# tests/test_env.py
from envs.jobshop_env import JobShopEnv

def test_env_basic_step():
    env = JobShopEnv(seed=42)
    obs, info = env.reset()
    assert obs.shape[-1] == env.max_jobs*3 + 2
    for _ in range(5):
        action = env.max_jobs  # NOOP
        obs, r, done, trunc, info = env.step(action)
    assert "action_mask" in info