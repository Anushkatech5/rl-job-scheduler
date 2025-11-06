# tests/test_metrics.py
import numpy as np
from evaluation.metrics import summarize_episode

def test_metrics_empty():
    log = {"completion_times":[], "finished_on_time":[], "queue_sizes":[], "slot_utilizations":[], "ages_on_finish":[]}
    s = summarize_episode(log)
    assert "miss_rate" in s and "throughput" in s# services/inference.py
from fastapi import FastAPI
from pydantic import BaseModel
import numpy as np
from agents.rl_policy import RLPolicy

app = FastAPI(title="RL Scheduler Inference")
policy = RLPolicy("artifacts/maskable_ppo.zip")  # adjust path as needed

class ActRequest(BaseModel):
    obs: list[float]
    mask: list[bool] | None = None

class ActResponse(BaseModel):
    action: int

@app.post("/act", response_model=ActResponse)
def act(req: ActRequest):
    obs = np.array(req.obs, dtype=np.float32)
    mask = np.array(req.mask, dtype=bool) if req.mask is not None else None
    a = policy.act(obs, mask)
    return ActResponse(action=int(a))
