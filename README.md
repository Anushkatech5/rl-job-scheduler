# Cloud RL Scheduler

> A **Reinforcement Learning** job scheduler with **heuristics**, **action masking**, **Streamlit UI**, **MLflow** tracking, **FastAPI** inference, **Docker** image, and **GitHub Actions** CI. CPU-only friendly on Windows.

---

## Table of Contents
- [Overview](#overview)
- [Features](#features)
- [Repository Structure](#repository-structure)
- [Prerequisites](#prerequisites)
- [Installation](#installation)
  - [Windows: CPU-Only PyTorch](#windows-cpu-only-pytorch)
- [Quickstart](#quickstart)
- [Usage](#usage)
  - [Train RL](#train-rl)
  - [Evaluate & Metrics](#evaluate--metrics)
  - [Launch Streamlit UI](#launch-streamlit-ui)
  - [FastAPI Inference](#fastapi-inference)
  - [MLflow Tracking UI](#mlflow-tracking-ui)
- [Environment Design](#environment-design)
- [Heuristic Baselines](#heuristic-baselines)
- [Configuration](#configuration)
- [Testing](#testing)
- [Docker](#docker)
- [Continuous Integration](#continuous-integration)
- [Contributing](#contributing)

---

## Overview
We simulate a small cluster where jobs arrive over time. Each job has a **duration** and optionally a **deadline**. An agent decides which job to run on available machines (slots).  
We compare classic **heuristics** (FIFO, SJF, SRPT, EDF, Deadline-Priority) against an **RL policy** (MaskablePPO) that uses **action masks** to avoid invalid actions.

**What you get out of the box**
- Streamlit dashboard for quick comparisons
- MLflow experiment tracking
- FastAPI microservice for serving actions
- CPU-friendly setup (no CUDA required)
- Dockerfile + GitHub Actions CI + basic tests

  <img width="1899" height="865" alt="image" src="https://github.com/user-attachments/assets/96069e7c-9aa4-4468-8dd5-29640cc6f264" />
  <img width="1741" height="846" alt="image" src="https://github.com/user-attachments/assets/fad6ce36-86d8-4911-b000-80172f3273b0" />

---

## Features
- **Gymnasium** environment with `ActionMasker` (valid-action masking)
- **Baselines**: FIFO / SJF / SRPT / EDF / Deadline-Priority
- **RL**: MaskablePPO (sb3-contrib) trained on **CPU**
- **Metrics**: miss rate, throughput, mean/P95/P99 completion times*, avg/max queue, utilization, Jain fairness
- **UI**: Streamlit with sliders & agent comparison chart
- **Serving**: FastAPI `/act` endpoint (JSON in, action out)
- **Tooling**: pytest, ruff, mypy, CI, Docker

\* *Exact completion timestamps can be enabled by extending the env to emit finish events (see Roadmap).*

---

## Repository Structure
```text
cloud-rl-scheduler/
├─ envs/
│  ├─ __init__.py
│  └─ jobshop_env.py
├─ agents/
│  ├─ __init__.py
│  ├─ heuristics.py
│  ├─ rl_train.py
│  └─ rl_policy.py
├─ evaluation/
│  ├─ __init__.py
│  ├─ metrics.py
│  └─ evaluate.py
├─ ui/
│  ├─ __init__.py
│  └─ app.py
├─ services/
│  ├─ __init__.py
│  └─ inference.py
├─ configs/
│  ├─ env.yaml
│  └─ train.yaml
├─ scripts/
│  ├─ bootstrap.sh
│  ├─ bootstrap.ps1
│  └─ run_demo.sh
├─ docker/
│  └─ Dockerfile
├─ .github/workflows/
│  └─ ci.yml
├─ artifacts/            # created at runtime (models, CSVs)
├─ requirements.txt
├─ README.md
└─ LICENSE
```
---
## Prerequisites

* **Python** 3.10 or 3.11
* **Git**
* (Optional) **Docker Desktop**
* On Windows: **Microsoft Visual C++ 2015–2022 Redistributable (x64)**

---

## Installation

Create a virtual environment and install dependencies.

```bash
# macOS / Linux
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

```powershell
# Windows PowerShell
py -3 -m venv .venv; .\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

### Windows: CPU-Only PyTorch

If you’re on Windows without CUDA, install CPU PyTorch:

```powershell
pip uninstall -y torch torchvision torchaudio
pip cache purge
pip install --no-cache-dir --index-url https://download.pytorch.org/whl/cpu `
  torch==2.4.1 torchvision==0.19.1 torchaudio==2.4.1

# sanity check
python - << 'PY'
import torch
print("torch", torch.__version__, "cuda?", torch.cuda.is_available())
PY
```

If `import torch` fails with a DLL error, repair/install the **MSVC 2015–2022 x64** redistributable and retry.

---

## Quickstart

```bash
# 1) Train RL (saves model to artifacts/maskable_ppo.zip)
python -m agents.rl_train

# 2) Evaluate (writes artifacts/eval_results.csv)
python -m evaluation.evaluate

# 3) Launch Streamlit UI
streamlit run ui/app.py
```

Optional:

```bash
# FastAPI inference
uvicorn services.inference:app --reload --port 8080

# MLflow UI
mlflow ui
```

---

## Usage

### Train RL

```bash
python -m agents.rl_train
```

* Wraps env with `ActionMasker`
* Trains `MaskablePPO` on **CPU**
* Saves model: `artifacts/maskable_ppo.zip`
* Logs artifact to MLflow experiment `rl_scheduler`

### Evaluate & Metrics

```bash
python -m evaluation.evaluate
```

Outputs:

* Console summary of mean metrics per agent
* CSV: `artifacts/eval_results.csv`

**Metrics**: miss rate, throughput, mean/P95/P99 completion time*, avg/max queue, utilization, Jain fairness.

*To compute exact completion times, enable “finish events” in the env (see Roadmap).

### Launch Streamlit UI

```bash
streamlit run ui/app.py
```

What you can do:

* Adjust `max_jobs`, `num_slots`, `arrival_rate`, `deadline_prob`, and `duration` range
* Compare heuristics against the trained RL model (set model path to `artifacts/maskable_ppo.zip`)
* View a bar chart & table of mean rewards

### FastAPI Inference

```bash
uvicorn services.inference:app --reload --port 8080
```

**POST** `/act`
Request:

```json
{
  "obs": [ ... ],
  "mask": [true, false, ...]   // optional
}
```

Response:

```json
{ "action": 7 }
```

### MLflow Tracking UI

```bash
mlflow ui
```

* Compare runs, parameters, and artifacts

---

## Environment Design

* **Observation**: `[max_jobs * (remaining, deadline_left, age)] + [time, free_slots]`
* **Action**: `Discrete(max_jobs + 1)` → pick job index or NOOP
* **Reward**: +1 if job finishes on/before deadline, 0 if late. Small penalty per tick for late, unfinished jobs.
* **Termination**: fixed horizon (e.g., 500 steps)
* **Action Masking**: `JobShopEnv.legal_action_mask()` + `ActionMasker`

---

## Heuristic Baselines

* **FIFO** — First In, First Out
* **SJF** — Shortest Job First
* **SRPT** — Shortest Remaining Processing Time
* **EDF** — Earliest Deadline First
* **Deadline-Priority** — Weighted mix of urgency, 1/remaining, and age

All implemented as pure functions mapping `(obs, max_jobs) → action`.

---

## Configuration

Optional YAMLs are provided in `configs/`.
Pattern to use in code:

```python
import yaml
from envs.jobshop_env import JobShopEnv

with open("configs/env.yaml") as f:
    env_cfg = yaml.safe_load(f)

env = JobShopEnv(**env_cfg)
```

---

## Testing

```bash
pytest -q
```

* Env shape & stepping sanity
* Metrics function sanity

Extend with:

* Invariants (no negative remaining, deadline monotonicity)
* Seed determinism
* Finish-event emission tests (see Roadmap)

---

## Docker

Build and run the Streamlit UI:

```bash
docker build -t rl-scheduler -f docker/Dockerfile .
docker run -p 8501:8501 rl-scheduler
```

Train inside the container:

```bash
docker run --rm rl-scheduler python -m agents.rl_train
```

---

## Continuous Integration

GitHub Actions workflow: `.github/workflows/ci.yml`
On each push/PR:

* Setup Python
* Install deps
* Lint (`ruff`), type-check (`mypy`)
* Run tests (`pytest`)
* Smoke-train (e.g., 1k timesteps) to catch runtime issues

---

## Contributing

Issues and PRs are welcome!
