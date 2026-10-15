# [MODULE DOC]
# Example Starlette app exposing a CodeAgent over HTTP asynchronously. Exports app (Starlette, debug on) with
#   one route POST /run-agent; the synchronous agent.run is offloaded to a worker thread via anyio so the
#   event loop is not blocked.
# Inputs: depends on anyio, starlette and smolagents (CodeAgent, InferenceClientModel with
#   Qwen3-Next-80B-A3B-Thinking); needs Hugging Face inference credentials at request time.
# Outputs: module globals routes and app.
# Exceptions: none raised at import.
# Side effects: none at import beyond constructing the app; requests trigger model network calls.
"""
Async CodeAgent Example with Starlette

This example demonstrates how to use a CodeAgent in an async Starlette app,
running the agent in a background thread using anyio.to_thread.run_sync.
"""

import anyio.to_thread
from starlette.applications import Starlette
from starlette.requests import Request
from starlette.responses import JSONResponse
from starlette.routing import Route

from smolagents import CodeAgent, InferenceClientModel


# Create a simple agent instance (customize as needed)
def get_agent():
    # [FUNCTION DOC]
    # HELPER: Builds a fresh CodeAgent with no tools and an InferenceClientModel for
    #   Qwen/Qwen3-Next-80B-A3B-Thinking.
    # Inputs: none.
    # Outputs: a new CodeAgent instance (a new one per call, so no memory is shared between requests).
    # Exceptions: propagates model/agent construction errors.
    # Side effects: none beyond object construction.
    # You can set custom model, or tools as needed
    return CodeAgent(
        model=InferenceClientModel(model_id="Qwen/Qwen3-Next-80B-A3B-Thinking"),
        tools=[],
    )


async def run_agent_in_thread(task: str):
    # [FUNCTION DOC]
    # HELPER: Creates a new agent and runs it on the task inside an anyio worker thread, awaiting the result.
    # Inputs: task (str), the prompt for the agent.
    # Outputs: whatever agent.run returns (the final answer).
    # Exceptions: propagates any exception raised by the agent run.
    # Side effects: LLM network calls; uses a thread from anyio's pool.
    agent = get_agent()
    # The agent's run method is synchronous
    result = await anyio.to_thread.run_sync(agent.run, task)
    return result


async def run_agent_endpoint(request: Request):
    # [FUNCTION DOC]
    # Starlette handler for POST /run-agent. Reads the JSON body, takes its "task" field, runs the agent and
    #   returns JSON.
    # Inputs: request (starlette Request) whose JSON body should contain a non-empty "task".
    # Outputs: JSONResponse with the result under "result" (status 200); with an error message and status 400
    #   if task is missing/empty; with the stringified exception and status 500 if the agent run fails.
    # Exceptions: invalid JSON body is not caught and propagates (Starlette turns it into a 500).
    # Side effects: runs the agent (network calls).
    # KNOWN ISSUE: the agent result is put directly into JSONResponse; a non-JSON-serializable final answer
    #   raises outside the try block and yields an unhandled server error.
    data = await request.json()
    task = data.get("task")
    if not task:
        return JSONResponse({"error": 'Missing "task" in request body.'}, status_code=400)
    try:
        result = await run_agent_in_thread(task)
        return JSONResponse({"result": result})
    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=500)


routes = [
    Route("/run-agent", run_agent_endpoint, methods=["POST"]),
]

app = Starlette(debug=True, routes=routes)
