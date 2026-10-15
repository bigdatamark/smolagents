#!/usr/bin/env python
# coding=utf-8

# Copyright 2025 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# [MODULE DOC]
# Description: Provides MCPClient, a wrapper that connects to Model Context Protocol servers and returns their
#   tools as smolagents Tools.
# Inputs: depends on warnings, smolagents.tools.Tool and, lazily, the optional mcpadapt package (and mcp for
#   StdioServerParameters type hints).
# Outputs: exports MCPClient (__all__).
# Exceptions: none raised directly.
# Side effects: none at import.

from __future__ import annotations

import warnings
from types import TracebackType
from typing import TYPE_CHECKING, Any

from smolagents.tools import Tool


__all__ = ["MCPClient"]

if TYPE_CHECKING:
    from mcpadapt.core import StdioServerParameters


class MCPClient:
    # [CLASS DOC]
    # Description: Connects to one or more MCP servers through mcpadapt's MCPAdapt with the SmolAgentsAdapter
    #   and exposes the servers' tools as smolagents Tool objects. Connects immediately in the constructor;
    #   usable as a context manager that yields the tool list and disconnects on exit.
    # Inputs: server_parameters (StdioServerParameters, a dict with url and optional transport, or a list of
    #   these); adapter_kwargs (dict or None, extra MCPAdapt kwargs); structured_output (bool or None; None
    #   warns and means False). Key attributes: _adapter (MCPAdapt), _tools (list of Tool or None).
    # Outputs: the instance; get_tools/__enter__ return the tool list.
    # Exceptions: none raised directly.
    # Side effects: spawns subprocesses or opens network connections to MCP servers.
    """Manages the connection to an MCP server and make its tools available to SmolAgents.

    Note: tools can only be accessed after the connection has been started with the
        `connect()` method, done during the init. If you don't use the context manager
        we strongly encourage to use "try ... finally" to ensure the connection is cleaned up.

    Args:
        server_parameters (StdioServerParameters | dict[str, Any] | list[StdioServerParameters | dict[str, Any]]):
            Configuration parameters to connect to the MCP server. Can be a list if you want to connect multiple MCPs at once.

            - An instance of `mcp.StdioServerParameters` for connecting a Stdio MCP server via standard input/output using a subprocess.

            - A `dict` with at least:
              - "url": URL of the server.
              - "transport": Transport protocol to use, one of:
                - "streamable-http": Streamable HTTP transport (default).
                - "sse": Legacy HTTP+SSE transport (deprecated).
        adapter_kwargs (dict[str, Any], optional):
            Additional keyword arguments to be passed directly to `MCPAdapt`.
        structured_output (bool, optional, defaults to False):
            Whether to enable structured output features for MCP tools. If True, enables:
            - Support for outputSchema in MCP tools
            - Structured content handling (structuredContent from MCP responses)
            - JSON parsing fallback for structured data
            If False, uses the original simple text-only behavior for backwards compatibility.

    Example:
        ```python
        # fully managed context manager + stdio
        with MCPClient(...) as tools:
            # tools are now available

        # context manager + Streamable HTTP transport:
        with MCPClient({"url": "http://localhost:8000/mcp", "transport": "streamable-http"}) as tools:
            # tools are now available

        # Enable structured output for advanced MCP tools:
        with MCPClient(server_parameters, structured_output=True) as tools:
            # tools with structured output support are now available

        # manually manage the connection via the mcp_client object:
        try:
            mcp_client = MCPClient(...)
            tools = mcp_client.get_tools()

            # use your tools here.
        finally:
            mcp_client.disconnect()
        ```
    """

    def __init__(
        self,
        server_parameters: "StdioServerParameters" | dict[str, Any] | list["StdioServerParameters" | dict[str, Any]],
        adapter_kwargs: dict[str, Any] | None = None,
        structured_output: bool | None = None,
    ):
        # [FUNCTION DOC]
        # Description: Emits a FutureWarning and uses False when structured_output is None; imports mcpadapt
        #   lazily; for a single dict, defaults its transport to streamable-http and validates it is sse or
        #   streamable-http; builds the MCPAdapt adapter and connects.
        # Inputs: server_parameters, adapter_kwargs (default None meaning empty), structured_output (default
        #   None).
        # Outputs: none.
        # Exceptions: ModuleNotFoundError if mcpadapt is not installed; ValueError for an unsupported
        #   transport; connection errors from connect propagate.
        # Side effects: warning emission; mutates a dict server_parameters by inserting the transport key;
        #   connects to servers.
        # KNOWN ISSUE: The warning says the default changes to True in version 1.25, but the package is 1.26
        #   and the default is still False.
        # KNOWN ISSUE: Transport defaulting/validation is skipped for dicts inside a list.
        # KNOWN ISSUE: The caller's dict is mutated in place.
        # Handle future warning for structured_output default value change
        if structured_output is None:
            warnings.warn(
                "Parameter 'structured_output' was not specified. "
                "Currently it defaults to False, but in version 1.25, the default will change to True. "
                "To suppress this warning, explicitly set structured_output=True (new behavior) or structured_output=False (legacy behavior). "
                "See documentation at https://huggingface.co/docs/smolagents/tutorials/tools#structured-output-and-output-schema-support for more details.",
                FutureWarning,
                stacklevel=2,
            )
            structured_output = False

        try:
            from mcpadapt.core import MCPAdapt
            from mcpadapt.smolagents_adapter import SmolAgentsAdapter
        except ModuleNotFoundError:
            raise ModuleNotFoundError("Please install 'mcp' extra to use MCPClient: `pip install 'smolagents[mcp]'`")
        if isinstance(server_parameters, dict):
            transport = server_parameters.get("transport")
            if transport is None:
                transport = "streamable-http"
                server_parameters["transport"] = transport
            if transport not in {"sse", "streamable-http"}:
                raise ValueError(
                    f"Unsupported transport: {transport}. Supported transports are 'streamable-http' and 'sse'."
                )
        adapter_kwargs = adapter_kwargs or {}
        self._adapter = MCPAdapt(
            server_parameters, SmolAgentsAdapter(structured_output=structured_output), **adapter_kwargs
        )
        self._tools: list[Tool] | None = None
        self.connect()

    def connect(self):
        # [FUNCTION DOC]
        # Description: Enters the MCPAdapt context, which starts the server connections and returns the
        #   adapted tools, and stores them.
        # Inputs: none.
        # Outputs: None.
        # Exceptions: connection errors propagate.
        # Side effects: opens connections/subprocesses; sets _tools.
        """Connect to the MCP server and initialize the tools."""
        self._tools: list[Tool] = self._adapter.__enter__()

    def disconnect(
        self,
        exc_type: type[BaseException] | None = None,
        exc_value: BaseException | None = None,
        exc_traceback: TracebackType | None = None,
    ):
        # [FUNCTION DOC]
        # Description: Exits the MCPAdapt context, closing server connections, forwarding any exception info.
        # Inputs: exc_type, exc_value, exc_traceback (optional, default None).
        # Outputs: None.
        # Exceptions: errors from the adapter's exit propagate.
        # Side effects: closes connections/subprocesses.
        # KNOWN ISSUE: _tools is not reset, so get_tools keeps returning tools that are no longer connected.
        """Disconnect from the MCP server"""
        self._adapter.__exit__(exc_type, exc_value, exc_traceback)

    def get_tools(self) -> list[Tool]:
        # [FUNCTION DOC]
        # Description: Returns the tool list obtained at connect time.
        # Inputs: none.
        # Outputs: list of Tool.
        # Exceptions: ValueError if no tools have been loaded (_tools is None).
        # Side effects: none.
        """The SmolAgents tools available from the MCP server.

        Note: for now, this always returns the tools available at the creation of the session,
        but it will in a future release return also new tools available from the MCP server if
        any at call time.

        Raises:
            ValueError: If the MCP server tools is None (usually assuming the server is not started).

        Returns:
            list[Tool]: The SmolAgents tools available from the MCP server.
        """
        if self._tools is None:
            raise ValueError(
                "Couldn't retrieve tools from MCP server, run `mcp_client.connect()` first before accessing `tools`"
            )
        return self._tools

    def __enter__(self) -> list[Tool]:
        # [FUNCTION DOC]
        # Description: Context-manager entry; returns the already-loaded tool list (connection was made in the
        #   constructor).
        # Inputs: none.
        # Outputs: list of Tool (or None if never connected).
        # Exceptions: none raised directly.
        # Side effects: none.
        # KNOWN ISSUE: The docstring says it connects, but it only returns the stored tools.
        """Connect to the MCP server and return the tools directly.

        Note that because of the `.connect` in the init, the mcp_client
        is already connected at this point.
        """
        return self._tools

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_value: BaseException | None,
        exc_traceback: TracebackType | None,
    ):
        # [FUNCTION DOC]
        # Description: Context-manager exit; disconnects, passing the exception info through.
        # Inputs: exc_type, exc_value, exc_traceback.
        # Outputs: None (exceptions are not suppressed).
        # Exceptions: errors from disconnect propagate.
        # Side effects: closes connections.
        """Disconnect from the MCP server."""
        self.disconnect(exc_type, exc_value, exc_traceback)
