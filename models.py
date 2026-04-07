from __future__ import annotations

from typing import List, Literal, Optional

from pydantic import BaseModel, Field, field_validator


class MoleculeAction(BaseModel):
    action_type: Literal["modify_molecule"] = "modify_molecule"
    new_smiles: str = Field(..., min_length=1, description="Candidate molecule as a SMILES string.")

    @field_validator("new_smiles")
    @classmethod
    def strip_smiles(cls, value: str) -> str:
        cleaned = value.strip()
        if not cleaned:
            raise ValueError("SMILES must not be empty.")
        return cleaned


class MoleculeProperties(BaseModel):
    smiles: str
    qed: float = Field(..., ge=0.0, le=1.0)
    logp: float
    molecular_weight: float = Field(..., ge=0.0)
    hbd: int = Field(..., ge=0)
    hba: int = Field(..., ge=0)
    tpsa: float = Field(..., ge=0.0)
    rotatable_bonds: int = Field(..., ge=0)
    sa_score: float = Field(..., ge=1.0, le=10.0)
    lipinski_violations: int = Field(..., ge=0)


class RewardModel(BaseModel):
    value: float = Field(..., ge=-1.0, le=1.0)
    objective_score: float = Field(..., ge=0.0, le=1.0)
    progress_delta: float = Field(default=0.0, ge=-1.0, le=1.0)
    penalty: float = Field(default=0.0, ge=-1.0, le=0.0)
    reason: str


class TaskSpec(BaseModel):
    name: str
    description: str
    start_smiles: str
    max_steps: int = Field(..., ge=1)
    difficulty: Literal["easy", "medium", "hard"]
    success_threshold: float = Field(..., ge=0.0, le=1.0)


class EpisodeState(BaseModel):
    task_name: str
    current_smiles: str
    step_count: int = Field(..., ge=0)
    max_steps: int = Field(..., ge=1)
    done: bool = False
    last_action_error: Optional[str] = None
    visited_smiles: List[str] = Field(default_factory=list)


class MolOptObservation(BaseModel):
    task_name: str
    difficulty: Literal["easy", "medium", "hard"]
    step: int = Field(..., ge=0)
    steps_remaining: int = Field(..., ge=0)
    done: bool
    properties: MoleculeProperties
    reward: RewardModel
    message: str
    last_action_error: Optional[str] = None
    final_score: Optional[float] = Field(default=None, ge=0.0, le=1.0)
