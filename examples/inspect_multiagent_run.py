# [MODULE DOC]
# Example script: enables OpenTelemetry tracing to Phoenix (register plus SmolagentsInstrumentor) and runs a
#   multi-agent setup: a manager CodeAgent delegating to a ToolCallingAgent "search_agent" (web search and
#   page visit tools), both on an InferenceClientModel via nebius, with full results returned. Prints the
#   manager's token usage and timing.
# Inputs: none; depends on openinference-instrumentation-smolagents, arize-phoenix (phoenix.otel), smolagents
#   and HF credentials; a Phoenix collector should be reachable.
# Outputs: none (script); module globals model, search_agent, manager_agent, run_result.
# Exceptions: tracing/model/network errors propagate.
# Side effects: at import, installs global tracing instrumentation, exports spans, performs network calls and
#   prints.
from openinference.instrumentation.smolagents import SmolagentsInstrumentor
from phoenix.otel import register


register()
SmolagentsInstrumentor().instrument(skip_dep_check=True)


from smolagents import (
    CodeAgent,
    InferenceClientModel,
    ToolCallingAgent,
    VisitWebpageTool,
    WebSearchTool,
)


# Then we run the agentic part!
model = InferenceClientModel(provider="nebius")

search_agent = ToolCallingAgent(
    tools=[WebSearchTool(), VisitWebpageTool()],
    model=model,
    name="search_agent",
    description="This is an agent that can do web search.",
    return_full_result=True,
)

manager_agent = CodeAgent(
    tools=[],
    model=model,
    managed_agents=[search_agent],
    return_full_result=True,
)
run_result = manager_agent.run(
    "If the US keeps it 2024 growth rate, how many years would it take for the GDP to double?"
)
print("Here is the token usage for the manager agent", run_result.token_usage)
print("Here are the timing details for the manager agent:", run_result.timing)
