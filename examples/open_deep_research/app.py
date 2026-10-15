# [MODULE DOC]
# Example entry point: builds the Open Deep Research manager agent via run.create_agent (default model "o1")
#   and wraps it in a GradioUI; when executed as a script, launches the Gradio server.
# Inputs: depends on the sibling run module (and thus its env vars HF_TOKEN, SERPAPI_API_KEY etc.) and
#   smolagents.gradio_ui.
# Outputs: module globals agent and demo.
# Exceptions: errors from agent creation (e.g. Hugging Face login) propagate at import.
# Side effects: importing run performs dotenv loading, HF login and creates ./downloads_folder; launching
#   starts a web server.
from run import create_agent

from smolagents.gradio_ui import GradioUI


agent = create_agent()

demo = GradioUI(agent)

if __name__ == "__main__":
    demo.launch()
