"""
MolOpt Environment - Molecule Optimization for Drug Design.

A real-world OpenEnv environment for iterative lead optimization.
An AI agent modifies molecules (SMILES strings) to optimize
drug-likeness properties such as QED, logP, and Lipinski compliance.
"""

from openenv.core.env_server.mcp_types import CallToolAction, ListToolsAction

from .client import MolOptEnv
from .env import MolOptEnvironment, compute_properties
from .models import MolOptObservation, MoleculeAction, MoleculeProperties, RewardModel, TaskSpec
from .rubrics import DEFAULT_TASK, TASKS, compute_reward, grade_episode

__all__ = [
    "MolOptEnv",
    "MolOptEnvironment",
    "MoleculeAction",
    "MoleculeProperties",
    "RewardModel",
    "MolOptObservation",
    "TaskSpec",
    "TASKS",
    "DEFAULT_TASK",
    "compute_properties",
    "compute_reward",
    "grade_episode",
    "CallToolAction",
    "ListToolsAction",
]
