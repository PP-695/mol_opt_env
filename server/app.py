"""
FastAPI application for the MolOpt environment.
"""

from fastapi.responses import HTMLResponse, JSONResponse

try:
    from openenv.core.env_server.http_server import create_app
    from openenv.core.env_server.mcp_types import CallToolAction, CallToolObservation

    from .molopt_environment import MolOptEnvironment
except ImportError:
    from openenv.core.env_server.http_server import create_app
    from openenv.core.env_server.mcp_types import CallToolAction, CallToolObservation
    from server.molopt_environment import MolOptEnvironment

app = create_app(
    MolOptEnvironment,
    CallToolAction,
    CallToolObservation,
    env_name="molopt_env",
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
