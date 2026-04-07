import os
import sys
import textwrap
import json
from typing import List, Optional, Tuple
import asyncio

from dotenv import load_dotenv
from openai import OpenAI

from client import MolOptEnv
from env import MolOptEnvironment, compute_properties
from openenv.core.containers.runtime.providers import LocalDockerProvider
from openenv.core.client_types import StepResult
from openenv.core.env_server.mcp_types import CallToolAction, CallToolObservation, Observation
from models import MoleculeProperties
from rubrics import TASKS, grade_episode

load_dotenv()

API_BASE_URL = os.getenv("API_BASE_URL", "https://router.huggingface.co/v1")
MODEL_NAME = os.getenv("MODEL_NAME", "Qwen/Qwen2.5-72B-Instruct:novita")
HF_TOKEN = os.getenv("HF_TOKEN")
LOCAL_IMAGE_NAME = os.getenv("LOCAL_IMAGE_NAME") or os.getenv("IMAGE_NAME")
DOCKER_READY_TIMEOUT_S = float(os.getenv("DOCKER_READY_TIMEOUT_S", "90"))

if HF_TOKEN is None:
    raise ValueError("HF_TOKEN environment variable is required")

client = OpenAI(base_url=API_BASE_URL, api_key=HF_TOKEN)

TEMPERATURE = 0.0
MAX_TOKENS = 96 #64
BENCHMARK = "molopt_env"
MODEL_REQUESTS_DISABLED = False

SYSTEM_PROMPT = textwrap.dedent(
    """
    You are an expert medicinal chemist doing lead optimization.
    Your ONLY output must be exactly one valid SMILES string.
    Make one small, chemically plausible structural change that improves the stated goal.
    NEVER repeat any SMILES you have already proposed in this episode.
    Do not add any explanation, markdown, quotes, prefixes, or extra text.
    Return nothing but the SMILES string.

    Example 1 (logP targeting):
    Input: Task: logp_targeting | Goal: logP in [2,3] | Current SMILES: c1ccccc1
    Output: Cc1ccccc1

    Example 2 (QED maximization):
    Input: Task: qed_maximization | Goal: maximize QED | Current SMILES: CC(=O)Oc1ccccc1C(=O)O
    Output: CC(=O)Nc1ccccc1C(=O)O

    Example 3 (multi-objective):
    Input: Task: multi_objective | Goal: raise QED, lower SA & rotatable bonds, Lipinski=0 | Current SMILES: CCN(CC)CCNC(=O)c1cc(Cl)ccc1N1CCN(CCOCC)CC1
    Output: CCN(CC)CCNC(=O)c1cc(Cl)ccc1N1CCN(CCO)CC1
    """
).strip()


def log_start(task: str, env: str, model: str) -> None:
    print(f"[START] task={task} env={env} model={model}", flush=True)


def log_step(step: int, action: str, reward: float, done: bool, error: Optional[str]) -> None:
    action_clean = action.replace("\n", " ").replace("\r", " ").strip()
    error_val = error if error else "null"
    print(
        f"[STEP] step={step} action={action_clean} reward={reward:.2f} "
        f"done={str(done).lower()} error={error_val}",
        flush=True,
    )


def log_end(success: bool, steps: int, score: float, rewards: List[float]) -> None:
    rewards_str = ",".join(f"{reward:.2f}" for reward in rewards)
    print(
        f"[END] success={str(success).lower()} steps={steps} score={score:.3f} rewards={rewards_str}",
        flush=True,
    )


def build_user_prompt(task_name: str, metadata: dict, history: List[str]) -> str:
    props = metadata.get("properties", {})
    smiles = props.get("smiles", TASKS[task_name].start_smiles)
    steps_left = metadata.get("steps_remaining", TASKS[task_name].max_steps)

    short_goals = {
        "logp_targeting": "logP in [2,3]",
        "qed_maximization": "maximize QED",
        "multi_objective": "raise QED, lower SA & rotatable bonds, Lipinski violations=0",
    }

    # Show last 4 moves clearly so model avoids repetition
    history_block = "\n".join(history[-4:]) if history else "none"

    return (
        f"Task: {task_name}\n"
        f"Goal: {short_goals[task_name]}\n"
        f"Current SMILES: {smiles}\n"
        f"QED:{props.get('qed')} logP:{props.get('logp')} SA:{props.get('sa_score')} "
        f"Lip:{props.get('lipinski_violations')} RB:{props.get('rotatable_bonds')}\n"
        f"Steps left: {steps_left}\n"
        f"Recent proposals (DO NOT repeat any of these):\n{history_block}\n"
        f"Next SMILES:"
    )


def get_model_smiles(task_name: str, metadata: dict, history: List[str]) -> str:
    global MODEL_REQUESTS_DISABLED
    fallback = metadata.get("properties", {}).get("smiles", TASKS[task_name].start_smiles)
    if MODEL_REQUESTS_DISABLED:
        return fallback
    try:
        completion = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": build_user_prompt(task_name, metadata, history)},
            ],
            temperature=TEMPERATURE,
            max_tokens=MAX_TOKENS,
            stream=False,
        )
        text = (completion.choices[0].message.content or "").strip()
        text = text.strip("`\"' ")
        return text.splitlines()[0].strip() if text else fallback
    except Exception as exc:
        error_text = str(exc)
        if "401" in error_text or "402" in error_text:
            MODEL_REQUESTS_DISABLED = True
        print(f"[DEBUG] Model request failed: {exc}", file=sys.stderr, flush=True)
        return fallback


def unwrap_tool_result(result: object) -> object:
    payload = result
    if hasattr(payload, "data"):
        payload = getattr(payload, "data")
    if isinstance(payload, dict) and "data" in payload:
        payload = payload["data"]
    if isinstance(payload, str):
        text = payload.strip()
        if text:
            try:
                return json.loads(text)
            except json.JSONDecodeError:
                return text
    return payload


def build_local_metadata(
    task_name: str,
    props: MoleculeProperties,
    *,
    step: int,
    done: bool,
    last_action_error: Optional[str],
) -> dict:
    return {
        "task_name": task_name,
        "difficulty": TASKS[task_name].difficulty,
        "step": step,
        "steps_remaining": max(TASKS[task_name].max_steps - step, 0),
        "done": done,
        "properties": props.model_dump(),
        "last_action_error": last_action_error,
        "final_score": grade_episode(task_name, props) if done else None,
    }


async def create_env() -> Tuple[object, bool]:
    if LOCAL_IMAGE_NAME:
        provider = None
        try:
            provider = LocalDockerProvider()
            base_url = provider.start_container(LOCAL_IMAGE_NAME)
            provider.wait_for_ready(base_url, timeout_s=DOCKER_READY_TIMEOUT_S)
            async_client = MolOptEnv(base_url=base_url, provider=provider)
            await async_client.connect()
            return async_client, True
        except Exception as exc:
            if provider is not None:
                try:
                    provider.stop_container()
                except Exception:
                    pass
            print(
                f"[DEBUG] Docker-backed environment startup failed for image '{LOCAL_IMAGE_NAME}' "
                f"within {DOCKER_READY_TIMEOUT_S:.1f}s: {exc}. "
                "Falling back to in-process environment.",
                file=sys.stderr,
                flush=True,
            )
    return MolOptEnvironment(), False


async def reset_env(env_obj: object, task_name: str, uses_client: bool) -> StepResult[Observation]:
    if uses_client:
        return await env_obj.reset(task=task_name)  # type: ignore[return-value]
    observation = env_obj.reset(task=task_name)  # type: ignore[call-arg]
    return StepResult(observation=observation, reward=0.0, done=bool(observation.done))


async def step_env(env_obj: object, candidate_smiles: str, uses_client: bool) -> StepResult[Observation]:
    action = CallToolAction(tool_name="modify_molecule", arguments={"new_smiles": candidate_smiles})
    if uses_client:
        return await env_obj.step(action)  # type: ignore[return-value]
    observation = env_obj.step(action)  # type: ignore[call-arg]
    return StepResult(observation=observation, reward=observation.reward, done=bool(observation.done))


async def run_task(task_name: str, env_obj: object, uses_client: bool) -> None:
    rewards: List[float] = []
    history: List[str] = []
    steps_taken = 0
    score = 0.0
    success = False
    start_props = compute_properties(TASKS[task_name].start_smiles)
    if start_props is None:
        raise RuntimeError(f"Invalid starting SMILES for task {task_name}")
    current_props = start_props
    metadata: dict = build_local_metadata(
        task_name,
        current_props,
        step=0,
        done=False,
        last_action_error=None,
    )

    log_start(task=task_name, env=BENCHMARK, model=MODEL_NAME)

    try:
        result = await reset_env(env_obj, task_name, uses_client)
        if not uses_client:
            metadata = result.observation.metadata or metadata
            props_payload = metadata.get("properties")
            if isinstance(props_payload, dict):
                current_props = MoleculeProperties.model_validate(props_payload)
        max_steps = TASKS[task_name].max_steps

        for step in range(1, max_steps + 1):
            if result.done:
                break

            candidate_smiles = get_model_smiles(task_name, metadata, history)
            result = await step_env(env_obj, candidate_smiles, uses_client)

            observation = result.observation
            reward = float(result.reward or 0.0)
            done = bool(result.done)

            if uses_client:
                tool_payload = {}
                if isinstance(observation, CallToolObservation):
                    raw_payload = unwrap_tool_result(observation.result)
                    if isinstance(raw_payload, dict):
                        tool_payload = raw_payload
                error = tool_payload.get("error")
                props_payload = tool_payload.get("properties")
                if tool_payload.get("success") and isinstance(props_payload, dict):
                    current_props = MoleculeProperties.model_validate(props_payload)
                metadata = build_local_metadata(
                    task_name,
                    current_props,
                    step=step,
                    done=done,
                    last_action_error=error,
                )
            else:
                metadata = observation.metadata or metadata
                error = metadata.get("last_action_error")
                props_payload = metadata.get("properties")
                if isinstance(props_payload, dict):
                    current_props = MoleculeProperties.model_validate(props_payload)

            rewards.append(reward)
            history.append(f"step={step} smiles={candidate_smiles} reward={reward:.2f}")
            steps_taken = step

            log_step(step=step, action=candidate_smiles, reward=reward, done=done, error=error)

            if done:
                final_score = metadata.get("final_score")
                if final_score is not None:
                    score = float(final_score)
                else:
                    score = grade_episode(task_name, current_props)
                success = score >= TASKS[task_name].success_threshold
                break

        if not rewards:
            rewards = []
        if score == 0.0 and steps_taken > 0:
            final_score = metadata.get("final_score")
            score = float(final_score) if final_score is not None else grade_episode(task_name, current_props)
            success = score >= TASKS[task_name].success_threshold
    except Exception as exc:
        print(f"[DEBUG] Task '{task_name}' failed: {exc}", file=sys.stderr, flush=True)
    finally:
        log_end(success=success, steps=steps_taken, score=score, rewards=rewards)


async def close_env(env_obj: object, uses_client: bool) -> None:
    close_fn = getattr(env_obj, "close", None)
    if not callable(close_fn):
        return

    if uses_client:
        await close_fn()
    else:
        close_fn()


async def main() -> None:
    env_obj, uses_client = await create_env()
    try:
        for task_name in TASKS:
            await run_task(task_name, env_obj, uses_client)
    finally:
        try:
            await close_env(env_obj, uses_client)
        except Exception as exc:
            print(f"[DEBUG] env.close() failed: {exc}", file=sys.stderr, flush=True)


if __name__ == "__main__":
    asyncio.run(main())
