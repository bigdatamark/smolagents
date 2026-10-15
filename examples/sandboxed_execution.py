# [MODULE DOC]
# Example script: runs the same CodeAgent task (WebSearchTool, default InferenceClientModel) four times, each
#   with a different remote/sandboxed code executor (blaxel, docker, e2b, modal), using the agent as a context
#   manager so the executor is cleaned up, and prints each result.
# Inputs: none; requires credentials/runtime for every executor (Blaxel, a Docker daemon, E2B API key, Modal)
#   and HF inference credentials.
# Outputs: none (script); prints four results.
# Exceptions: the first executor that fails to start aborts the whole script (no per-executor handling).
# Side effects: at import, starts sandboxes/containers, network calls, console output.
from smolagents import CodeAgent, InferenceClientModel, WebSearchTool


model = InferenceClientModel()

# Blaxel executor example
with CodeAgent(tools=[WebSearchTool()], model=model, executor_type="blaxel") as agent:
    output = agent.run("How many seconds would it take for a leopard at full speed to run through Pont des Arts?")
print("Blaxel executor result:", output)

# Docker executor example
with CodeAgent(tools=[WebSearchTool()], model=model, executor_type="docker") as agent:
    output = agent.run("How many seconds would it take for a leopard at full speed to run through Pont des Arts?")
print("Docker executor result:", output)

# E2B executor example
with CodeAgent(tools=[WebSearchTool()], model=model, executor_type="e2b") as agent:
    output = agent.run("How many seconds would it take for a leopard at full speed to run through Pont des Arts?")
print("E2B executor result:", output)

# Modal executor example
with CodeAgent(tools=[WebSearchTool()], model=model, executor_type="modal") as agent:
    output = agent.run("How many seconds would it take for a leopard at full speed to run through Pont des Arts?")
print("Modal executor result:", output)
