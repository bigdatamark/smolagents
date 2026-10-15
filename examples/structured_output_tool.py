# [MODULE DOC]
# Example script (runnable with uv thanks to its inline script metadata): starts an inline FastMCP weather
#   server over stdio, connects to it with MCPClient using structured_output so tool results keep their
#   pydantic schema, and asks a CodeAgent (LiteLLM mistral-small-latest) for Tokyo's temperature in
#   Fahrenheit.
# Inputs: Mistral API key for LiteLLM (environment); depends on mcp, pydantic and smolagents with mcp and
#   litellm extras.
# Outputs: functions weather_server_script and main.
# Exceptions: none at import.
# Side effects: none at import; main spawns a python subprocess and performs network calls.
# How to run with uv:
#   uv run structured_output_tool.py
#
# Modify the smolagents dependency to point to the local smolagents repo or
# remove `@ file:///<path-to-smolagents>`
#
# /// script
# requires-python = ">=3.10"
# dependencies = [
#   "smolagents[mcp,litellm] @ file:///<path-to-smolagents>",
#   "pydantic",
# ]
# ///

from textwrap import dedent

from mcp import StdioServerParameters

from smolagents import CodeAgent, InferenceClientModel, LiteLLMModel, MCPClient  # noqa: F401


def weather_server_script() -> str:
    # [FUNCTION DOC]
    # HELPER: Builds the source code of a tiny MCP server.
    # Inputs: none.
    # Outputs: dedented Python source string defining a FastMCP "Weather Service" with a get_weather_info tool
    #   that returns a fixed WeatherInfo (location equals the city, 22.5 C, partly cloudy, 65 percent
    #   humidity) and runs the server.
    # Exceptions: none raised directly.
    # Side effects: none.
    """Return an inline MCP server script that exposes a weather tool."""
    return dedent(
        '''
        from pydantic import BaseModel, Field
        from mcp.server.fastmcp import FastMCP

        mcp = FastMCP("Weather Service")

        class WeatherInfo(BaseModel):
            location: str = Field(description="The location name")
            temperature: float = Field(description="Temperature in Celsius")
            conditions: str = Field(description="Weather conditions")
            humidity: int = Field(description="Humidity percentage", ge=0, le=100)

        @mcp.tool(
            name="get_weather_info",
            description="Get weather information for a location as structured data.",
        )
        def get_weather_info(city: str) -> WeatherInfo:
            """Get weather information for a city."""
            return WeatherInfo(
                location=city,
                temperature=22.5,
                conditions="partly cloudy",
                humidity=65
            )

        mcp.run()
        '''
    )


def main() -> None:
    # [FUNCTION DOC]
    # Runs the demo: creates the LiteLLM model, launches the inline server as a "python -c" stdio subprocess,
    #   opens an MCPClient with structured output, builds a CodeAgent with the MCP tools and runs the Tokyo
    #   Fahrenheit query.
    # Inputs: none.
    # Outputs: None (the agent answer is not printed or returned).
    # Exceptions: MCP connection, model and agent errors propagate.
    # Side effects: spawns a subprocess (closed when the MCPClient context exits), network calls, agent
    #   logging.
    # Configure your inference model
    # model = InferenceClientModel()
    model = LiteLLMModel(
        model_id="mistral/mistral-small-latest",
        # model_id="openai/gpt-4o-mini",
    )

    # Start the Weather MCP server from an inline script in this same file
    serverparams = StdioServerParameters(command="python", args=["-c", weather_server_script()])

    # Bridge MCP tools into SmolAgents with structured outputs enabled
    with MCPClient(
        serverparams,
        structured_output=True,
    ) as tools:
        agent = CodeAgent(tools=tools, model=model)
        # Example query that encourages tool use and unit conversion
        agent.run("What is the temperature in Tokyo in Fahrenheit?")


if __name__ == "__main__":
    main()
