# [MODULE DOC]
# Description: Integration tests for MCPClient against small FastMCP servers launched as python -c
#   subprocesses over stdio.
# Inputs: depends on pytest, mcp (StdioServerParameters, FastMCP in the child) and MCPClient.
# Outputs: fixtures and test functions.
# Exceptions: none raised directly.
# Side effects: tests spawn subprocesses.
import json
from textwrap import dedent

import pytest
from mcp import StdioServerParameters

from smolagents.mcp_client import MCPClient


@pytest.fixture
def echo_server_script():
    # [FUNCTION DOC]
    # Description: HELPER: Fixture returning Python source for a FastMCP server with one echo_tool that
    #   returns 'Echo: ' plus the text.
    # Inputs: none.
    # Outputs: str of source code.
    # Exceptions: none raised directly.
    # Side effects: none.
    return dedent(
        '''
        from mcp.server.fastmcp import FastMCP

        mcp = FastMCP("Echo Server")

        @mcp.tool()
        def echo_tool(text: str) -> str:
            """Echo the input text"""
            return f"Echo: {text}"

        mcp.run()
        '''
    )


@pytest.fixture
def structured_output_server_script():
    # [FUNCTION DOC]
    # Description: HELPER: Fixture returning Python source for a FastMCP server with user_info_tool that
    #   returns a dict with name, age 25, lower-cased email and active True.
    # Inputs: none.
    # Outputs: str of source code.
    # Exceptions: none raised directly.
    # Side effects: none.
    return dedent(
        '''
        from mcp.server.fastmcp import FastMCP
        from typing import Any

        mcp = FastMCP("Structured Output Server")

        @mcp.tool()
        def user_info_tool(name: str) -> dict[str, Any]:
            """Get user information as structured data"""
            user_data = {
                "name": name,
                "age": 25,
                "email": f"{name.lower()}@example.com",
                "active": True
            }
            return user_data

        mcp.run()
        '''
    )


# Ignore FutureWarning about structured_output default value change: this test intentionally uses default behavior
@pytest.mark.filterwarnings("ignore:.*structured_output:FutureWarning")
def test_mcp_client_with_syntax(echo_server_script: str):
    # [FUNCTION DOC]
    # Description: Uses MCPClient as a context manager (structured_output unspecified, warning ignored) and
    #   checks one echo_tool is exposed and echoes.
    # Inputs: echo_server_script.
    # Outputs: none (passes or fails via assertions).
    # Exceptions: AssertionError on failure.
    # Side effects: spawns a server subprocess.
    """Test the MCPClient with the context manager syntax."""
    server_parameters = StdioServerParameters(command="python", args=["-c", echo_server_script])
    with MCPClient(server_parameters) as tools:
        assert len(tools) == 1
        assert tools[0].name == "echo_tool"
        assert tools[0].forward(**{"text": "Hello, world!"}) == "Echo: Hello, world!"


def test_mcp_client_with_structured_output(structured_output_server_script: str):
    # [FUNCTION DOC]
    # Description: With structured_output True, checks the tool has object output type, an object output
    #   schema, and returns a parsed dict with the expected fields.
    # Inputs: structured_output_server_script.
    # Outputs: none (passes or fails via assertions).
    # Exceptions: AssertionError on failure.
    # Side effects: spawns a server subprocess.
    """Test the MCPClient with structured_output=True parameter."""
    server_parameters = StdioServerParameters(command="python", args=["-c", structured_output_server_script])
    with MCPClient(server_parameters, structured_output=True) as tools:
        assert len(tools) == 1
        assert tools[0].name == "user_info_tool"
        assert tools[0].output_type == "object"  # Should be object due to outputSchema

        # Check the output schema {'additionalProperties': True, 'title': 'user_info_toolDictOutput', 'type': 'object'}
        assert tools[0].output_schema is not None
        schema = tools[0].output_schema
        assert isinstance(schema, dict)
        assert schema.get("type") == "object"

        # Test that structured output is properly parsed
        result = tools[0].forward(**{"name": "Alice"})
        assert isinstance(result, dict)
        assert result["name"] == "Alice"
        assert result["age"] == 25
        assert result["email"] == "alice@example.com"
        assert result["active"] is True


def test_mcp_client_without_structured_output(structured_output_server_script: str):
    # [FUNCTION DOC]
    # Description: With structured_output False, checks the tool still reports object output type but returns
    #   a JSON string that decodes to the expected data.
    # Inputs: structured_output_server_script.
    # Outputs: none (passes or fails via assertions).
    # Exceptions: AssertionError on failure.
    # Side effects: spawns a server subprocess.
    # KNOWN ISSUE: Docstring calls False the default; the actual default is None, which warns and then behaves
    #   as False.
    """Test the MCPClient with structured_output=False (default) for comparison."""
    server_parameters = StdioServerParameters(command="python", args=["-c", structured_output_server_script])
    with MCPClient(server_parameters, structured_output=False) as tools:
        assert len(tools) == 1
        assert tools[0].name == "user_info_tool"
        assert tools[0].output_type == "object"

        # Test that output is returned as raw text
        result = tools[0].forward(**{"name": "Alice"})
        assert isinstance(result, str)
        # Should be JSON string, not parsed object
        parsed_result = json.loads(result)
        assert parsed_result["name"] == "Alice"


# Ignore FutureWarning about structured_output default value change: this test intentionally uses default behavior
@pytest.mark.filterwarnings("ignore:.*structured_output:FutureWarning")
def test_mcp_client_try_finally_syntax(echo_server_script: str):
    # [FUNCTION DOC]
    # Description: Creates MCPClient directly, uses get_tools and always disconnects in a finally block;
    #   checks the echo tool works.
    # Inputs: echo_server_script.
    # Outputs: none (passes or fails via assertions).
    # Exceptions: AssertionError on failure.
    # Side effects: spawns a server subprocess.
    """Test the MCPClient with the try ... finally syntax."""
    server_parameters = StdioServerParameters(command="python", args=["-c", echo_server_script])
    mcp_client = MCPClient(server_parameters)
    try:
        tools = mcp_client.get_tools()
        assert len(tools) == 1
        assert tools[0].name == "echo_tool"
        assert tools[0].forward(**{"text": "Hello, world!"}) == "Echo: Hello, world!"
    finally:
        mcp_client.disconnect()


# Ignore FutureWarning about structured_output default value change: this test intentionally uses default behavior
@pytest.mark.filterwarnings("ignore:.*structured_output:FutureWarning")
def test_multiple_servers(echo_server_script: str):
    # [FUNCTION DOC]
    # Description: Passes a list of two stdio server parameters and checks two echo tools are returned and
    #   both work.
    # Inputs: echo_server_script.
    # Outputs: none (passes or fails via assertions).
    # Exceptions: AssertionError on failure.
    # Side effects: spawns two server subprocesses.
    """Test the MCPClient with multiple servers."""
    server_parameters = [
        StdioServerParameters(command="python", args=["-c", echo_server_script]),
        StdioServerParameters(command="python", args=["-c", echo_server_script]),
    ]
    with MCPClient(server_parameters) as tools:
        assert len(tools) == 2
        assert tools[0].name == "echo_tool"
        assert tools[1].name == "echo_tool"
        assert tools[0].forward(**{"text": "Hello, world!"}) == "Echo: Hello, world!"
        assert tools[1].forward(**{"text": "Hello, world!"}) == "Echo: Hello, world!"
