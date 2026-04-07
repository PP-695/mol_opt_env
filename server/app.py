"""
FastAPI application for the MolOpt environment.
"""

from typing import Any

from fastapi.responses import HTMLResponse, JSONResponse

try:
    from openenv.core.env_server.http_server import create_app
    from openenv.core.env_server.mcp_types import CallToolAction, CallToolObservation

    from .molopt_environment import DEFAULT_TASK, TASKS, MolOptEnvironment
except ImportError:
    from openenv.core.env_server.http_server import create_app
    from openenv.core.env_server.mcp_types import CallToolAction, CallToolObservation
    from server.molopt_environment import DEFAULT_TASK, TASKS, MolOptEnvironment


def build_custom_task_ui(
    web_manager: Any,
    action_fields: Any,
    metadata: Any,
    is_chat_env: bool,
    title: str,
    quick_start_md: str,
) -> Any:
    import gradio as gr

    task_names = list(TASKS.keys())

    def task_summary(task_name: str) -> str:
        task = TASKS[task_name]
        return (
            f"### {task.name}\n"
            f"- Difficulty: {task.difficulty}\n"
            f"- Max steps: {task.max_steps}\n"
            f"- Success threshold: {task.success_threshold}\n"
            f"- Start SMILES: {task.start_smiles}\n\n"
            f"{task.description}"
        )

    async def reset_selected_task(task_name: str) -> tuple[dict[str, Any], str]:
        try:
            response = await web_manager.reset_environment({"task": task_name})
            return response, f"Reset complete for task '{task_name}'."
        except Exception as exc:
            return {"error": str(exc)}, f"Reset failed for task '{task_name}'."

    async def call_get_task_info() -> tuple[dict[str, Any], str]:
        try:
            response = await web_manager.step_environment(
                {"type": "call_tool", "tool_name": "get_task_info", "arguments": {}}
            )
            return response, "Called get_task_info."
        except Exception as exc:
            return {"error": str(exc)}, "get_task_info failed."

    async def call_get_properties() -> tuple[dict[str, Any], str]:
        try:
            response = await web_manager.step_environment(
                {"type": "call_tool", "tool_name": "get_properties", "arguments": {}}
            )
            return response, "Called get_properties."
        except Exception as exc:
            return {"error": str(exc)}, "get_properties failed."

    async def call_modify_molecule(new_smiles: str) -> tuple[dict[str, Any], str]:
        smiles = new_smiles.strip()
        if not smiles:
            return {"error": "SMILES is required."}, "No action sent."
        try:
            response = await web_manager.step_environment(
                {
                    "type": "call_tool",
                    "tool_name": "modify_molecule",
                    "arguments": {"new_smiles": smiles},
                }
            )
            return response, f"Called modify_molecule for '{smiles}'."
        except Exception as exc:
            return {"error": str(exc)}, "modify_molecule failed."

    def get_state() -> tuple[dict[str, Any], str]:
        try:
            return web_manager.get_state(), "Fetched environment state."
        except Exception as exc:
            return {"error": str(exc)}, "get_state failed."

    with gr.Blocks() as demo:
        gr.Markdown(
            "## Task Runner\n"
            "Use this tab to switch tasks and call tools without manually crafting action JSON."
        )

        task_dropdown = gr.Dropdown(
            choices=task_names,
            value=DEFAULT_TASK,
            label="Select Task",
        )
        task_description = gr.Markdown(task_summary(DEFAULT_TASK))
        task_dropdown.change(fn=task_summary, inputs=task_dropdown, outputs=task_description)

        with gr.Row():
            reset_button = gr.Button("Reset With Selected Task", variant="primary")
            state_button = gr.Button("Get State")

        smiles_input = gr.Textbox(
            label="SMILES for modify_molecule",
            placeholder="e.g. Cc1ccccc1",
        )
        with gr.Row():
            task_info_button = gr.Button("Tool: get_task_info")
            properties_button = gr.Button("Tool: get_properties")
            modify_button = gr.Button("Tool: modify_molecule")

        status_text = gr.Textbox(label="Status", interactive=False)
        response_json = gr.JSON(label="Response")

        reset_button.click(fn=reset_selected_task, inputs=task_dropdown, outputs=[response_json, status_text])
        state_button.click(fn=get_state, outputs=[response_json, status_text])
        task_info_button.click(fn=call_get_task_info, outputs=[response_json, status_text])
        properties_button.click(fn=call_get_properties, outputs=[response_json, status_text])
        modify_button.click(fn=call_modify_molecule, inputs=smiles_input, outputs=[response_json, status_text])

    return demo

app = create_app(
    MolOptEnvironment,
    CallToolAction,
    CallToolObservation,
    env_name="molopt_env",
    gradio_builder=build_custom_task_ui,
)


@app.get("/", response_class=HTMLResponse)
def root() -> str:
    return """
    <html>
      <head><title>MolOpt Environment</title></head>
      <body style="font-family: sans-serif; max-width: 760px; margin: 40px auto; line-height: 1.5;">
        <h1>MolOpt Environment</h1>
        <p>This container serves the OpenEnv molecule optimization benchmark.</p>
        <ul>
          <li><a href="/health">/health</a> health check</li>
          <li>POST <code>/reset</code> to start an episode</li>
          <li>POST <code>/step</code> to execute an action</li>
          <li>GET <code>/state</code> to inspect environment state</li>
        </ul>
      </body>
    </html>
    """.strip()


@app.get("/health", response_class=JSONResponse)
def health() -> dict:
    return {"status": "ok", "env": "molopt_env"}


def main() -> None:
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)


if __name__ == "__main__":
    main()
