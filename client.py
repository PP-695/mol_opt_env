# Copyright (c) 2026 MolOpt Environment Authors.
# Licensed under the BSD-3-Clause License.

"""
MolOpt Environment Client.

This module provides the client for connecting to a MolOpt Environment server.
MolOptEnv extends MCPToolClient to provide tool-calling style interactions.

Example:
    >>> with MolOptEnv(base_url="http://localhost:8000") as env:
    ...     env.reset()
    ...
    ...     # Discover tools
    ...     tools = env.list_tools()
    ...     print([t.name for t in tools])
    ...
    ...     # Get current molecule properties
    ...     props = env.call_tool("get_properties")
    ...     print(props)
    ...
    ...     # Modify the molecule
    ...     result = env.call_tool("modify_molecule", new_smiles="c1ccc(O)cc1")
    ...     print(result)

Example with Docker:
    >>> env = MolOptEnv.from_docker_image("molopt-env:latest")
    >>> try:
    ...     env.reset()
    ...     tools = env.list_tools()
    ...     result = env.call_tool("get_properties")
    ... finally:
    ...     env.close()

Example with HuggingFace Space:
    >>> env = MolOptEnv.from_env("openenv/molopt-env")
    >>> try:
    ...     env.reset()
    ...     result = env.call_tool("get_properties")
    ... finally:
    ...     env.close()
"""

from openenv.core.mcp_client import MCPToolClient


class MolOptEnv(MCPToolClient):
    """
    Client for the Molecule Optimization Environment.

    Inherits all functionality from MCPToolClient:
    - `list_tools()`: Discover available tools
    - `call_tool(name, **kwargs)`: Call a tool by name
    - `reset(**kwargs)`: Reset the environment
    - `step(action)`: Execute an action (for advanced use)
    """

    pass  # MCPToolClient provides all needed functionality
