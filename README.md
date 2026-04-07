---
title: MolOpt Environment
colorFrom: blue
colorTo: green
sdk: docker
app_port: 8000
pinned: false
suggested_hardware: cpu-basic
short_description: OpenEnv environment for real-world iterative molecule optimization and medicinal chemistry lead optimization.
tags:
  - openenv
  - chemistry
  - drug-discovery
  - reinforcement-learning
  - rdkit
---

# MolOpt Environment

**MolOpt** is a complete OpenEnv-compatible environment that simulates **real medicinal chemistry lead optimization** — the exact workflow used daily by chemists at Pfizer, Novartis, Exscientia, and biotech startups.

An agent starts with a lead molecule (SMILES) and iteratively proposes small structural changes while receiving dense, deterministic feedback on drug-likeness (QED), lipophilicity (logP), synthetic accessibility (SA Score), and Lipinski compliance.

## Why This Matters (Real-World Utility)

Medicinal chemists spend hours optimizing molecules for multiple competing objectives. This environment turns that process into a clean, reproducible RL/agent benchmark using only RDKit — no external databases or non-deterministic components.

## Tasks (Easy → Medium → Hard)

1. **`logp_targeting`** (easy)  
   Reach the optimal logP window `[2.0, 3.0]` starting from benzene (max 6 steps).

2. **`qed_maximization`** (medium)  
   Maximize QED (drug-likeness) starting from aspirin (max 10 steps).

3. **`multi_objective`** (hard)  
   Simultaneously improve QED, reduce SA Score & rotatable bonds, and keep zero Lipinski violations (max 12 steps).

## Action & Observation Spaces

**Primary Action**  
`modify_molecule(new_smiles: str)` — submit a new candidate SMILES.

**Helper Tools**  
- `get_properties()`  
- `get_task_info()`

**Observation** (Pydantic `MolOptObservation`) contains:
- Current SMILES + all RDKit-computed properties
- Step / steps remaining
- Dense reward breakdown (`RewardModel`)
- `last_action_error` and `final_score`

All grading and reward logic is **100% deterministic** (see `rubrics.py`).

## Reward Design

Dense and informative:
- Positive reward for progress toward objective
- Bonus for improvement
- `-0.5` for invalid SMILES
- `-0.1` for no-change or repetition
- Penalty for regression

Final episode score is always normalized in `[0.0, 1.0]`.

## Baseline Scores (measured with Qwen2.5-72B-Instruct)

| Task               | Success | Final Score | Notes                     |
|--------------------|---------|-------------|---------------------------|
| logp_targeting     | true    | **0.978**   | Excellent                 |
| qed_maximization   | true    | **0.726**   | Strong                    |
| multi_objective    | false   | **0.450**   | Still improving (repetition loop) |

Average baseline: **0.718**

## Project Structure

```text
.
├── openenv.yaml
├── pyproject.toml
├── Dockerfile
├── inference.py                 # root-level, exact required format
├── models.py
├── rubrics.py
├── env.py
├── client.py
├── server/
│   ├── app.py
│   ├── molopt_environment.py
│   └── sascorer.py              # official RDKit SA_Score
└── README.md