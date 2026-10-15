# [MODULE DOC]
# Example script: builds a streaming CodeAgent (WebSearchTool, Llama-3.3-70B via fireworks-ai through
#   InferenceClientModel, planning every 3 steps, named example_agent) and launches a Gradio chat UI for it
#   with file uploads saved under ./data.
# Inputs: none; depends on smolagents with the gradio extra and HF inference credentials.
# Outputs: none (script); module globals agent.
# Exceptions: launch/model errors propagate.
# Side effects: at import, starts a blocking local Gradio web server; files uploaded through the UI are
#   written to ./data.
from smolagents import CodeAgent, GradioUI, InferenceClientModel, WebSearchTool


agent = CodeAgent(
    tools=[WebSearchTool()],
    model=InferenceClientModel(model_id="meta-llama/Llama-3.3-70B-Instruct", provider="fireworks-ai"),
    verbosity_level=1,
    planning_interval=3,
    name="example_agent",
    description="This is an example agent.",
    step_callbacks=[],
    stream_outputs=True,
    # use_structured_outputs_internally=True,
)

GradioUI(agent, file_upload_folder="./data").launch()
