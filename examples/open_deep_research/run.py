# [MODULE DOC]
# Open Deep Research CLI: answers one question with a two-level agent team. A manager CodeAgent (visualizer
#   and text inspector tools, any imports authorized, max 12 steps, planning every 4) delegates web browsing
#   to a managed ToolCallingAgent "search_agent" (Google search via serper, text-browser navigation tools,
#   archive search, text inspector; max 20 steps, planning every 4). Both use a LiteLLMModel.
# Inputs: CLI args (question, --model-id default "o1"); environment variables HF_TOKEN, SERPAPI_API_KEY,
#   SERPER_API_KEY and the LLM provider key; depends on the local scripts package and smolagents.
# Outputs: functions parse_args, create_agent, main; globals custom_role_conversions, user_agent,
#   BROWSER_CONFIG, append_answer_lock.
# Exceptions: login errors propagate at import.
# Side effects: at import, loads .env (overriding env vars), logs in to the Hugging Face Hub, creates
#   ./downloads_folder.
# KNOWN ISSUE: append_answer_lock is created but never used in this module.
import argparse
import os
import threading

from dotenv import load_dotenv
from huggingface_hub import login
from scripts.text_inspector_tool import TextInspectorTool
from scripts.text_web_browser import (
    ArchiveSearchTool,
    FinderTool,
    FindNextTool,
    PageDownTool,
    PageUpTool,
    SimpleTextBrowser,
    VisitTool,
)
from scripts.visual_qa import visualizer

from smolagents import (
    CodeAgent,
    GoogleSearchTool,
    # InferenceClientModel,
    LiteLLMModel,
    ToolCallingAgent,
)


load_dotenv(override=True)
login(os.getenv("HF_TOKEN"))

append_answer_lock = threading.Lock()


def parse_args():
    # [FUNCTION DOC]
    # HELPER: Parses command-line arguments.
    # Inputs: sys.argv; positional question (str) and optional --model-id (str, default "o1").
    # Outputs: argparse Namespace with question and model_id.
    # Exceptions: SystemExit on invalid arguments or --help.
    # Side effects: may print usage to stderr.
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "question", type=str, help="for example: 'How many studio albums did Mercedes Sosa release before 2007?'"
    )
    parser.add_argument("--model-id", type=str, default="o1")
    return parser.parse_args()


custom_role_conversions = {"tool-call": "assistant", "tool-response": "user"}

user_agent = "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/119.0.0.0 Safari/537.36 Edg/119.0.0.0"

BROWSER_CONFIG = {
    "viewport_size": 1024 * 5,
    "downloads_folder": "downloads_folder",
    "request_kwargs": {
        "headers": {"User-Agent": user_agent},
        "timeout": 300,
    },
    "serpapi_key": os.getenv("SERPAPI_API_KEY"),
}

os.makedirs(f"./{BROWSER_CONFIG['downloads_folder']}", exist_ok=True)


def create_agent(model_id="o1"):
    # [FUNCTION DOC]
    # Builds the manager agent and its managed web-search agent sharing a single LiteLLMModel.
    # Inputs: model_id (str, default "o1"); for "o1", reasoning_effort is set to "high". max_completion_tokens
    #   is 8192 and tool-call/tool-response roles are mapped to assistant/user.
    # Outputs: the manager CodeAgent (with the ToolCallingAgent in managed_agents). The search agent's
    #   managed-agent task prompt is extended with hints about .txt files, the inspect_file_as_text tool and
    #   asking for clarification via final_answer.
    # Exceptions: propagates model/tool construction errors.
    # Side effects: creates a SimpleTextBrowser (which reads module COOKIES); no network until run.
    model_params = {
        "model_id": model_id,
        "custom_role_conversions": custom_role_conversions,
        "max_completion_tokens": 8192,
    }
    if model_id == "o1":
        model_params["reasoning_effort"] = "high"
    model = LiteLLMModel(**model_params)

    text_limit = 100000
    browser = SimpleTextBrowser(**BROWSER_CONFIG)
    WEB_TOOLS = [
        GoogleSearchTool(provider="serper"),
        VisitTool(browser),
        PageUpTool(browser),
        PageDownTool(browser),
        FinderTool(browser),
        FindNextTool(browser),
        ArchiveSearchTool(browser),
        TextInspectorTool(model, text_limit),
    ]
    text_webbrowser_agent = ToolCallingAgent(
        model=model,
        tools=WEB_TOOLS,
        max_steps=20,
        verbosity_level=2,
        planning_interval=4,
        name="search_agent",
        description="""A team member that will search the internet to answer your question.
    Ask him for all your questions that require browsing the web.
    Provide him as much context as possible, in particular if you need to search on a specific timeframe!
    And don't hesitate to provide him with a complex search task, like finding a difference between two webpages.
    Your request must be a real sentence, not a google search! Like "Find me this information (...)" rather than a few keywords.
    """,
        provide_run_summary=True,
    )
    text_webbrowser_agent.prompt_templates["managed_agent"]["task"] += """You can navigate to .txt online files.
    If a non-html page is in another format, especially .pdf or a Youtube video, use tool 'inspect_file_as_text' to inspect it.
    Additionally, if after some searching you find out that you need more information to answer the question, you can use `final_answer` with your request for clarification as argument to request for more information."""

    manager_agent = CodeAgent(
        model=model,
        tools=[visualizer, TextInspectorTool(model, text_limit)],
        max_steps=12,
        verbosity_level=2,
        additional_authorized_imports=["*"],
        planning_interval=4,
        managed_agents=[text_webbrowser_agent],
    )

    return manager_agent


def main():
    # [FUNCTION DOC]
    # CLI entry: parses arguments, creates the agent for the requested model, runs it on the question and
    #   prints the answer.
    # Inputs: command-line arguments.
    # Outputs: None.
    # Exceptions: agent errors propagate.
    # Side effects: network calls, console output, possible file downloads into the downloads folder.
    args = parse_args()

    agent = create_agent(model_id=args.model_id)

    answer = agent.run(args.question)

    print(f"Got this answer: {answer}")


if __name__ == "__main__":
    main()
