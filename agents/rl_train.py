# agents/rl_train.py
from __future__ import annotations
import os
import gymnasium as gym
import mlflow
from typing import Optional
from sb3_contrib import MaskablePPO
from sb3_contrib.common.wrappers import ActionMasker
from stable_baselines3.common.vec_env import DummyVecEnv

from envs.jobshop_env import JobShopEnv

# ---- mask function for the wrapper ----
def mask_fn(env: gym.Env):
    # We ignore the observation; we compute from env state.
    # ActionMasker will call this automatically.
    return env.unwrapped.legal_action_mask()

def make_env(seed: Optional[int] = 0) -> gym.Env:
    base = JobShopEnv(seed=seed)
    masked = ActionMasker(base, mask_fn)  # <<< IMPORTANT
    return masked

def train(total_timesteps: int = 100_000, run_name: str = "maskable_ppo"):
    mlflow.set_experiment("rl_scheduler")
    with mlflow.start_run(run_name=run_name):
        env = DummyVecEnv([lambda: make_env(seed=0)])
        model = MaskablePPO("MlpPolicy", env, verbose=1)
        model.learn(total_timesteps=total_timesteps)

        os.makedirs("artifacts", exist_ok=True)
        save_path = "artifacts/maskable_ppo.zip"
        model.save(save_path)
        mlflow.log_artifact(save_path)
        print(f"Saved model to {save_path}")
        return save_path

if __name__ == "__main__":
    train()