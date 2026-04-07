from __future__ import annotations

import json
import importlib
from typing import Any, Optional
from uuid import uuid4

import rdkit.Chem as Chem
import rdkit.Chem.QED as QED
from fastmcp import FastMCP
from rdkit.Chem import Descriptors, Lipinski

from models import EpisodeState, MolOptObservation, MoleculeAction, MoleculeProperties, RewardModel
from rubrics import DEFAULT_TASK, TASKS, compute_reward, grade_episode

try:
    from openenv.core.env_server.mcp_environment import MCPEnvironment
    from openenv.core.env_server.types import Action, Observation, State
except ImportError:
    from openenv.core.env_server.mcp_environment import MCPEnvironment
    from openenv.core.env_server.types import Action, Observation, State

try:
    _sascorer = importlib.import_module("server.sascorer")
    _sascorer.readFragmentScores()
    _HAS_SA_SCORER = True
except Exception:
    _sascorer = None
    _HAS_SA_SCORER = False


def sa_score_from_mol(mol: Chem.Mol) -> float:
    if _HAS_SA_SCORER and _sascorer is not None:
        try:
            return float(_sascorer.calculateScore(mol))
        except Exception:
            pass
    bertz = Descriptors.BertzCT(mol)
    return float(max(1.0, min(10.0, 1.0 + bertz / 250.0)))


def compute_properties(smiles: str) -> Optional[MoleculeProperties]:
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None

    canonical = Chem.MolToSmiles(mol)
    mw = round(Descriptors.MolWt(mol), 2)
    logp = round(Descriptors.MolLogP(mol), 4)
    hbd = int(Lipinski.NumHDonors(mol))
    hba = int(Lipinski.NumHAcceptors(mol))
    violations = int(mw > 500) + int(logp > 5) + int(hbd > 5) + int(hba > 10)

    return MoleculeProperties(
        smiles=canonical,
        qed=round(QED.default(mol), 4),
        logp=logp,
        molecular_weight=mw,
        hbd=hbd,
        hba=hba,
        tpsa=round(Descriptors.TPSA(mol), 2),
        rotatable_bonds=int(Lipinski.NumRotatableBonds(mol)),
        sa_score=round(sa_score_from_mol(mol), 3),
        lipinski_violations=violations,
    )


class MolOptEnvironment(MCPEnvironment):
    def __init__(self) -> None:
        mcp = FastMCP("molopt_env")

        @mcp.tool
        def get_properties() -> str:
            return self._current_properties_json()

        @mcp.tool
        def get_task_info() -> str:
            task = TASKS[self._task_name]
            payload = {
                "task_name": task.name,
                "difficulty": task.difficulty,
                "description": task.description,
                "start_smiles": task.start_smiles,
                "max_steps": task.max_steps,
                "step": self._step_count,
                "steps_remaining": max(task.max_steps - self._step_count, 0),
            }
            return json.dumps(payload, indent=2)

        @mcp.tool
        def modify_molecule(new_smiles: str) -> str:
            parsed = MoleculeAction(new_smiles=new_smiles)
            return json.dumps(self._apply_modification(parsed.new_smiles), indent=2)

        super().__init__(mcp)
        self._task_name = DEFAULT_TASK
        self._step_count = 0
        self._done = False
        self._last_action_error: Optional[str] = None
        self._last_reward = RewardModel(value=0.0, objective_score=0.0, progress_delta=0.0, penalty=0.0, reason="reset")
        self._episode = EpisodeState(
            task_name=DEFAULT_TASK,
            current_smiles=TASKS[DEFAULT_TASK].start_smiles,
            step_count=0,
            max_steps=TASKS[DEFAULT_TASK].max_steps,
            visited_smiles=[],
        )
        self._state = State(episode_id=str(uuid4()), step_count=0)
        self._current_props = compute_properties(TASKS[DEFAULT_TASK].start_smiles)
        self._previous_props = self._current_props

    def _current_properties_json(self) -> str:
        props = self._current_props or compute_properties(self._episode.current_smiles)
        payload = (props.model_dump() if props is not None else {"error": "invalid_molecule"})
        payload["step"] = self._step_count
        payload["steps_remaining"] = max(self._episode.max_steps - self._step_count, 0)
        return json.dumps(payload, indent=2)

    def _build_observation_model(self, final_score: Optional[float] = None) -> MolOptObservation:
        task = TASKS[self._task_name]
        props = self._current_props
        if props is None:
            raise RuntimeError("Current properties are unavailable.")
        return MolOptObservation(
            task_name=self._task_name,
            difficulty=task.difficulty,
            step=self._step_count,
            steps_remaining=max(task.max_steps - self._step_count, 0),
            done=self._done,
            properties=props,
            reward=self._last_reward,
            message=task.description,
            last_action_error=self._last_action_error,
            final_score=final_score,
        )

    def _apply_modification(self, new_smiles: str) -> dict[str, Any]:
        current_props = self._current_props
        proposed_props = compute_properties(new_smiles)
        self._last_action_error = None

        if proposed_props is None:
            self._last_action_error = f"Invalid SMILES: {new_smiles}"
            fallback_props = current_props
            if fallback_props is None:
                raise RuntimeError("Environment has no current molecule state.")
            self._last_reward = compute_reward(self._task_name, fallback_props, self._previous_props, invalid=True)
            return {"success": False, "error": self._last_action_error, "reward": self._last_reward.value}

        if current_props is not None and proposed_props.smiles == current_props.smiles:
            self._last_action_error = "No change: submitted the same molecule"
            self._last_reward = compute_reward(self._task_name, current_props, self._previous_props, unchanged=True)
            return {"success": False, "error": self._last_action_error, "reward": self._last_reward.value}

        if proposed_props.smiles in self._episode.visited_smiles:
            self._last_action_error = "Repetition: molecule already visited"
            self._last_reward = compute_reward(self._task_name, proposed_props, self._previous_props, repeated=True)
            return {"success": False, "error": self._last_action_error, "reward": self._last_reward.value}

        self._last_reward = compute_reward(self._task_name, proposed_props, self._previous_props)
        self._previous_props = current_props
        self._current_props = proposed_props
        self._episode.current_smiles = proposed_props.smiles
        self._episode.visited_smiles.append(proposed_props.smiles)
        return {
            "success": True,
            "reward": self._last_reward.value,
            "objective_score": self._last_reward.objective_score,
            "properties": proposed_props.model_dump(),
        }

    def reset(
        self,
        seed: Optional[int] = None,
        episode_id: Optional[str] = None,
        task: str = DEFAULT_TASK,
        **kwargs: Any,
    ) -> Observation:
        if task not in TASKS:
            task = DEFAULT_TASK
        spec = TASKS[task]
        start_props = compute_properties(spec.start_smiles)
        if start_props is None:
            raise ValueError(f"Invalid starting SMILES for task {task}: {spec.start_smiles}")

        self._task_name = task
        self._step_count = 0
        self._done = False
        self._last_action_error = None
        self._current_props = start_props
        self._previous_props = start_props
        self._last_reward = RewardModel(
            value=0.0,
            objective_score=grade_episode(task, start_props),
            progress_delta=0.0,
            penalty=0.0,
            reason="reset",
        )
        self._episode = EpisodeState(
            task_name=task,
            current_smiles=start_props.smiles,
            step_count=0,
            max_steps=spec.max_steps,
            done=False,
            visited_smiles=[start_props.smiles],
        )
        self._state = State(episode_id=episode_id or str(uuid4()), step_count=0)
        obs_model = self._build_observation_model()
        return Observation(done=False, reward=0.0, metadata=obs_model.model_dump())

    def _step_impl(self, action: Action, timeout_s: Optional[float] = None, **kwargs: Any) -> Observation:
        return Observation(
            done=self._done,
            reward=self._last_reward.value,
            metadata={"error": f"Unsupported action type {type(action).__name__}. Use MCP tool actions."},
        )

    def _finalize_observation(self, obs: Observation) -> Observation:
        self._done = self._step_count >= self._episode.max_steps
        self._episode.step_count = self._step_count
        self._episode.done = self._done
        final_score = grade_episode(self._task_name, self._current_props) if (self._done and self._current_props is not None) else None
        obs.reward = self._last_reward.value
        obs.done = self._done
        if obs.metadata is None:
            obs.metadata = {}
        obs.metadata.update(self._build_observation_model(final_score=final_score).model_dump())
        return obs

    def step(self, action: Action, timeout_s: Optional[float] = None, **kwargs: Any) -> Observation:
        self._step_count += 1
        self._state.step_count = self._step_count
        obs = super().step(action, timeout_s=timeout_s, **kwargs)
        return self._finalize_observation(obs)

    async def step_async(self, action: Action, timeout_s: Optional[float] = None, **kwargs: Any) -> Observation:
        self._step_count += 1
        self._state.step_count = self._step_count
        obs = await super().step_async(action, timeout_s=timeout_s, **kwargs)
        return self._finalize_observation(obs)

    @property
    def state(self) -> State:
        return self._state
