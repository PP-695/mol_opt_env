from __future__ import annotations

from typing import Optional

from models import MoleculeProperties, RewardModel, TaskSpec


TASKS: dict[str, TaskSpec] = {
    "logp_targeting": TaskSpec(
        name="logp_targeting",
        description=(
            "Move the molecule into the medicinal-chemistry-friendly logP window [2.0, 3.0]. "
            "Start from benzene and make small, realistic structural edits."
        ),
        start_smiles="c1ccccc1",
        max_steps=8,
        difficulty="easy",
        success_threshold=0.8,
    ),
    "qed_maximization": TaskSpec(
        name="qed_maximization",
        description=(
            "Increase the QED drug-likeness score starting from aspirin without making invalid or repetitive edits."
        ),
        start_smiles="CC(=O)Oc1ccccc1C(=O)O",
        max_steps=12,
        difficulty="medium",
        success_threshold=0.65,
    ),
    "multi_objective": TaskSpec(
        name="multi_objective",
        description=(
            "Balance three real medicinal chemistry objectives at once: maximize QED, lower SA score, "
            "and keep Lipinski violations at zero."
        ),
        start_smiles="ClC1=CC=C(N2CCCCC2)C(=C1)C(=O)N(C)C",
        max_steps=15,
        difficulty="hard",
        success_threshold=0.55,
    ),
}

DEFAULT_TASK = "logp_targeting"


def _objective_score(task_name: str, props: MoleculeProperties) -> float:
    if task_name == "logp_targeting":
        target_min, target_max = 2.0, 3.0
        midpoint = (target_min + target_max) / 2.0
        half_width = (target_max - target_min) / 2.0
        if target_min <= props.logp <= target_max:
            return max(0.0, min(1.0, 0.7 + 0.3 * (1.0 - abs(props.logp - midpoint) / half_width)))
        distance = min(abs(props.logp - target_min), abs(props.logp - target_max))
        return max(0.0, min(1.0, 0.7 - 0.15 * distance))

    if task_name == "qed_maximization":
        return props.qed

    sa_norm = max(0.0, (10.0 - props.sa_score) / 9.0)
    lip_norm = max(0.0, 1.0 - 0.25 * props.lipinski_violations)
    return max(0.0, min(1.0, 0.45 * props.qed + 0.30 * sa_norm + 0.25 * lip_norm))


def compute_reward(
    task_name: str,
    props: MoleculeProperties,
    prev_props: Optional[MoleculeProperties],
    *,
    invalid: bool = False,
    repeated: bool = False,
    unchanged: bool = False,
) -> RewardModel:
    if invalid:
        return RewardModel(
            value=-0.5,
            objective_score=0.0,
            progress_delta=0.0,
            penalty=-0.5,
            reason="invalid_smiles",
        )
    if unchanged:
        score = _objective_score(task_name, props)
        return RewardModel(
            value=-0.1,
            objective_score=score,
            progress_delta=0.0,
            penalty=-0.1,
            reason="no_change",
        )
    if repeated:
        score = _objective_score(task_name, props)
        return RewardModel(
            value=-0.1,
            objective_score=score,
            progress_delta=0.0,
            penalty=-0.1,
            reason="repeated_state",
        )

    score = _objective_score(task_name, props)
    prev_score = _objective_score(task_name, prev_props) if prev_props is not None else 0.0
    delta = score - prev_score
    bonus = max(0.0, delta) * 0.3
    regression_penalty = min(0.0, delta) * 0.2
    value = max(0.0, min(1.0, score + bonus + regression_penalty))
    reason = "improved" if delta > 0 else "regressed" if delta < 0 else "steady"
    return RewardModel(
        value=value,
        objective_score=score,
        progress_delta=delta,
        penalty=regression_penalty,
        reason=reason,
    )


def grade_episode(task_name: str, props: MoleculeProperties) -> float:
    return round(_objective_score(task_name, props), 4)
