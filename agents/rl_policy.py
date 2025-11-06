# agents/rl_policy.py
from __future__ import annotations
import os
os.environ["CUDA_VISIBLE_DEVICES"] = ""  # force CPU everywhere

import numpy as np
from sb3_contrib import MaskablePPO

class RLPolicy:
    def __init__(self, model_path: str):
        # Load on CPU regardless of how it was trained
        self.model = MaskablePPO.load(model_path, device="cpu")

    def act(self, obs: np.ndarray, mask: np.ndarray | None) -> int:
        # We DON'T try to set masks here (sb3-contrib no longer exposes set_action_masks).
        # It's okay because your env handles "suboptimal" actions gracefully.
        action, _ = self.model.predict(obs, deterministic=True)
        return int(action)
