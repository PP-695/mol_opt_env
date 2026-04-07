---
title: MolOpt Environment
colorFrom: blue
colorTo: green
sdk: docker
app_port: 8000
pinned: false
suggested_hardware: cpu-basic
short_description: OpenEnv environment for iterative molecule optimization and medicinal chemistry benchmarking.
tags:
  - openenv
  - chemistry
  - reinforcement-learning
  - drug-discovery
---

# MolOpt Environment

MolOpt is an OpenEnv-compatible environment for iterative molecule optimization. It models a real medicinal chemistry workflow: start from a lead compound, make a structural change, recompute drug-relevant properties, and continue improving toward a task objective.

This is a benchmark environment, not a chat assistant. The goal is repeated `reset()` and `step()` interaction with dense rewards, deterministic grading, and reproducible trajectories.

## Why This Is Real-World

Medicinal chemists routinely optimize molecules for multiple competing objectives:

- QED: a normalized estimate of overall drug-likeness from `0.0` to `1.0`
- logP: a lipophilicity measure that strongly affects permeability and exposure
- SA score: an estimate of synthetic accessibility from `1` (easy) to `10` (hard)
- Lipinski violations: simple oral-drug heuristics around mass, lipophilicity, and hydrogen bonding

This environment simulates that loop deterministically using RDKit only.

## Tasks

1. `logp_targeting` (easy)
Hit the target logP window `[2.0, 3.0]` starting from benzene.

2. `qed_maximization` (medium)
Improve the QED score starting from aspirin.

3. `multi_objective` (hard)
Balance QED, SA score, and Lipinski compliance starting from a more complex lead.

## Action And Observation Spaces

### Action

The primary action is:

- `modify_molecule(new_smiles: str)`

Read-only tools:

- `get_properties()`
- `get_task_info()`

Typed Pydantic models live in [models.py](/c:/Code/MetaHack/models.py).

### Observation

Each observation includes:

- current task and difficulty
- current step and remaining budget
- canonical SMILES
- computed RDKit properties
- reward breakdown
- optional terminal `final_score`

Core environment logic is in [env.py](/c:/Code/MetaHack/env.py) and deterministic scoring is in [rubrics.py](/c:/Code/MetaHack/rubrics.py).

## Reward Design

The reward is dense and deterministic.

- valid progress is rewarded every step
- improvements earn a bonus
- invalid SMILES gets `-0.5`
- repeated or unchanged molecules get `-0.1`
- regressions reduce reward

Episode graders always return a normalized score in `[0.0, 1.0]`.

## Project Layout

```text
.
|-- openenv.yaml
|-- pyproject.toml
|-- Dockerfile
|-- inference.py
|-- .env.example
|-- models.py
|-- rubrics.py
|-- env.py
|-- client.py
|-- __init__.py
`-- server/
    |-- app.py
    |-- molopt_environment.py
    |-- sascorer.py
    `-- Dockerfile
```

## Local Run

```bash
python -m venv .venv
. .venv/bin/activate
pip install -e .
uvicorn server.app:app --host 0.0.0.0 --port 8000
```

Open:

- `http://localhost:8000/`
- `http://localhost:8000/health`

## Docker

```bash
docker build -t molopt-env:latest .
docker run --rm -p 7860:8000 molopt-env:latest
```

For local testing, open:

- `http://localhost:7860/`
- `http://localhost:7860/health`

## Inference

The root-level [inference.py](/c:/Code/MetaHack/inference.py) uses the OpenAI client only and reads:

- `API_BASE_URL` default: `https://router.huggingface.co/v1`
- `MODEL_NAME` default: `Qwen/Qwen2.5-72B-Instruct:novita`
- `HF_TOKEN` required
- `LOCAL_IMAGE_NAME` optional for local Docker-backed evaluation

You can store these in a local `.env` file.

```bash
cp .env.example .env
python inference.py
```

## Baseline Scores

Replace these placeholders with measured values before submission:

- `logp_targeting`: pending local run
- `qed_maximization`: pending local run
- `multi_objective`: pending local run

## Validation Checklist

1. `docker build` succeeds
2. `docker run` exposes the service on port `7860`
3. `openenv validate` passes
4. `python inference.py` runs with `HF_TOKEN` set
5. The Hugging Face Space is running before submission
