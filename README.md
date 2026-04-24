---
title: MolOpt Environment
colorFrom: blue
colorTo: green
sdk: docker
app_port: 8000
pinned: false
license: bsd-3-clause
suggested_hardware: cpu-basic
short_description: OpenEnv benchmark for iterative molecule optimization.
tags:
  - openenv
  - chemistry
  - drug-discovery
  - reinforcement-learning
  - rdkit
---

# MolOpt Environment

MolOpt is an OpenEnv-compatible environment for iterative molecule optimization. It models a real medicinal chemistry workflow: start from a lead compound, propose a structural edit, recompute drug-relevant properties, and continue improving toward a task objective.

This is a benchmark environment, not a chemistry chatbot. The core value is repeated `reset()` / `step()` interaction with dense rewards, deterministic grading, and reproducible trajectories.

## Why This Is Real-World

Medicinal chemists routinely optimize molecules for multiple competing objectives:

- QED: a normalized estimate of overall drug-likeness from `0.0` to `1.0`
- logP: a lipophilicity measure that affects permeability and exposure
- SA score: an estimate of synthetic accessibility from `1` (easy) to `10` (hard)
- Lipinski violations: simple oral-drug heuristics around mass, lipophilicity, and hydrogen bonding

MolOpt turns that loop into a deterministic RL/agent benchmark using only RDKit and a programmatic grader.

## Tasks

1. `logp_targeting` (easy)
Move the molecule into the medicinal-chemistry-friendly logP window `[2.0, 3.0]` starting from benzene. Max `6` steps.

2. `qed_maximization` (medium)
Increase QED starting from aspirin without making invalid or repetitive edits. Max `10` steps.

3. `multi_objective` (hard)
Raise QED, lower SA score, reduce flexibility, and keep Lipinski violations at zero starting from a more complex lead. Max `12` steps.

## Action And Observation Spaces

Primary action:

- `modify_molecule(new_smiles: str)`

Helper tools:

- `get_properties()`
- `get_task_info()`

Observation (`MolOptObservation`) includes:

- current task and difficulty
- current step and remaining budget
- canonical SMILES
- computed RDKit properties
- dense reward breakdown (`RewardModel`)
- `last_action_error`
- terminal `final_score`

Typed Pydantic models live in `models.py`. Core environment logic is in `env.py`. Deterministic scoring is in `rubrics.py`.

## Reward Design

The reward is dense and deterministic.

- valid progress is rewarded every step
- improvements earn a bonus
- invalid SMILES gets `-0.5`
- repeated or unchanged molecules get `-0.1`
- regressions reduce reward

Episode graders always return a normalized score in `[0.0, 1.0]`.

Note: for validator compatibility, final task scores are guaranteed to be strictly between `0` and `1` (never exactly `0.0` or `1.0`).

## Baseline Scores

Latest measured local run with the current task budgets:

| Task | Success | Final Score | Notes |
|---|---:|---:|---|
| `logp_targeting` | true | `0.942` | Strong easy-task baseline |
| `qed_maximization` | true | `0.790` | Improves, but does not clear threshold |
| `multi_objective` | true | `0.680` | Hard task; one measured run was also affected by provider credit exhaustion |

Average baseline: `0.661`

## Project Structure

```text
.
|-- openenv.yaml
|-- pyproject.toml
|-- Dockerfile
|-- inference.py
|-- models.py
|-- rubrics.py
|-- env.py
|-- client.py
|-- README.md
`-- server/
    |-- app.py
    |-- molopt_environment.py
    `-- sascorer.py
```

## Local Setup

```bash
python -m venv .venv
. .venv/bin/activate
pip install -e .
openenv validate
```

## Docker

```bash
docker build -t molopt-env:latest .
docker run --rm -p 7860:8000 molopt-env:latest
```

Open:

- `http://localhost:7860/`
- `http://localhost:7860/web/` (interactive UI)
- `http://localhost:7860/health`

### Web UI (Tasks)

The web interface includes a task selector so you can run each task from the browser:

1. Open `http://localhost:7860/web/`
2. Go to the **Custom** tab ("Task Runner")
3. Select a task (`logp_targeting`, `qed_maximization`, `multi_objective`)
4. Click **Reset With Selected Task**
5. Use **get_task_info**, **get_properties**, then **modify_molecule** with a new SMILES

Only `modify_molecule` consumes the step budget; helper tools do not.

## Inference

The root-level `inference.py` uses the OpenAI client only and reads:

- `API_BASE_URL` default: `https://router.huggingface.co/v1`
- `MODEL_NAME` default: `Qwen/Qwen2.5-72B-Instruct:novita`
- `HF_TOKEN` required
- `LOCAL_IMAGE_NAME` optional for local Docker-backed evaluation
- `DOCKER_READY_TIMEOUT_S` optional, default `90`

Typical local flow:

```bash
cp .env.example .env
python inference.py
```
