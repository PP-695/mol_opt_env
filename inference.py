import os
import sys
import textwrap
from typing import List, Optional, Tuple
import asyncio

from dotenv import load_dotenv
from openai import OpenAI

from client import MolOptEnv
from env import MolOptEnvironment
from openenv.core.client_types import StepResult
from openenv.core.env_server.mcp_types import CallToolAction, Observation
from rubrics import TASKS

load_dotenv()

API_BASE_URL = os.getenv("API_BASE_URL", "https://router.huggingface.co/v1")
MODEL_NAME = os.getenv("MODEL_NAME", "Qwen/Qwen2.5-72B-Instruct:novita")
HF_TOKEN = os.getenv("HF_TOKEN")
LOCAL_IMAGE_NAME = os.getenv("LOCAL_IMAGE_NAME") or os.getenv("IMAGE_NAME")

if HF_TOKEN is None:
    raise ValueError("HF_TOKEN environment variable is required")

client = OpenAI(base_url=API_BASE_URL, api_key=HF_TOKEN)

TEMPERATURE = 0.0
MAX_TOKENS = 64
BENCHMARK = "molopt_env"

SYSTEM_PROMPT = textwrap.dedent(
    """
    You are an expert medicinal chemist optimizing molecules.
    Output exactly one valid SMILES string and nothing else.
    Make a small, chemically plausible edit that improves the task objective.
    Never include markdown, explanations, or quotes.
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
    spec = TASKS[task_name]
    props = metadata.get("properties", {})
    history_block = "\n".join(history[-4:]) if history else "None"
    return textwrap.dedent(
        f"""
        Task: {task_name}
        Objective: {spec.description}
        Current SMILES: {props.get('smiles', spec.start_smiles)}
        Current properties:
        - QED: {props.get('qed')}
        - logP: {props.get('logp')}
        - Molecular weight: {props.get('molecular_weight')}
        - SA score: {props.get('sa_score')}
        - Lipinski violations: {props.get('lipinski_violations')}
        Steps remaining: {metadata.get('steps_remaining', spec.max_steps)}
        Recent proposals:
        {history_block}

        Output the next improved SMILES now.
        """
    ).strip()


def get_model_smiles(task_name: str, metadata: dict, history: List[str]) -> str:
    fallback = metadata.get("properties", {}).get("smiles", TASKS[task_name].start_smiles)
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
        print(f"[DEBUG] Model request failed: {exc}", file=sys.stderr, flush=True)
        return fallback


def create_env() -> Tuple[object, bool]:
    if LOCAL_IMAGE_NAME:
        try:
            async_client = asyncio.run(MolOptEnv.from_docker_image(LOCAL_IMAGE_NAME))
            return async_client.sync(), True
        except Exception as exc:
            print(
                f"[DEBUG] Docker-backed environment startup failed for image '{LOCAL_IMAGE_NAME}': {exc}. "
                "Falling back to in-process environment.",
                file=sys.stderr,
                flush=True,
            )
    return MolOptEnvironment(), False


def reset_env(env_obj: object, task_name: str, uses_client: bool) -> StepResult[Observation]:
    if uses_client:
        return env_obj.reset(task=task_name)  # type: ignore[return-value]
    observation = env_obj.reset(task=task_name)  # type: ignore[call-arg]
    return StepResult(observation=observation, reward=0.0, done=bool(observation.done))


def step_env(env_obj: object, candidate_smiles: str, uses_client: bool) -> StepResult[Observation]:
    action = CallToolAction(tool_name="modify_molecule", arguments={"new_smiles": candidate_smiles})
    if uses_client:
        return env_obj.step(action)  # type: ignore[return-value]
    observation = env_obj.step(action)  # type: ignore[call-arg]
    return StepResult(observation=observation, reward=observation.reward, done=bool(observation.done))


def run_task(task_name: str, env_obj: object, uses_client: bool) -> None:
    rewards: List[float] = []
    history: List[str] = []
    steps_taken = 0
    score = 0.0
    success = False
    metadata: dict = {}

    log_start(task=task_name, env=BENCHMARK, model=MODEL_NAME)

    try:
        result = reset_env(env_obj, task_name, uses_client)
        metadata = result.observation.metadata or {}
        max_steps = TASKS[task_name].max_steps

        for step in range(1, max_steps + 1):
            if result.done:
                break

            candidate_smiles = get_model_smiles(task_name, metadata, history)
            result = step_env(env_obj, candidate_smiles, uses_client)

            observation = result.observation
            metadata = observation.metadata or {}
            reward = float(result.reward or 0.0)
            done = bool(result.done)
            error = metadata.get("last_action_error")

            rewards.append(reward)
            history.append(f"step={step} smiles={candidate_smiles} reward={reward:.2f}")
            steps_taken = step

            log_step(step=step, action=candidate_smiles, reward=reward, done=done, error=error)

            if done:
                final_score = metadata.get("final_score")
                score = float(final_score) if final_score is not None else max(0.0, min(1.0, reward))
                success = score >= TASKS[task_name].success_threshold
                break

        if not rewards:
            rewards = []
        if score == 0.0 and metadata.get("final_score") is not None:
            score = float(metadata["final_score"])
            success = score >= TASKS[task_name].success_threshold
    except Exception as exc:
        print(f"[DEBUG] Task '{task_name}' failed: {exc}", file=sys.stderr, flush=True)
    finally:
        log_end(success=success, steps=steps_taken, score=score, rewards=rewards)


def main() -> None:
    env_obj, uses_client = create_env()
    try:
        for task_name in TASKS:
            run_task(task_name, env_obj, uses_client)
    finally:
        close_fn = getattr(env_obj, "close", None)
        if callable(close_fn):
            try:
                close_fn()
            except Exception as exc:
                print(f"[DEBUG] env.close() failed: {exc}", file=sys.stderr, flush=True)


if __name__ == "__main__":
    main()
