# [MODULE DOC]
# GAIA benchmark runner for Open Deep Research: downloads the GAIA dataset, skips questions already answered
#   in the output JSONL, and answers the rest concurrently with a fresh manager/search agent team per
#   question, appending annotated results to output/<set>/<run-name>.jsonl.
# Inputs: CLI args (--concurrency default 8, --model-id default "o1", --run-name required, --set-to-run
#   default "validation", --use-open-models, --use-raw-dataset); env vars HF_TOKEN, SERPAPI_API_KEY,
#   SERPER_API_KEY and LLM keys; depends on the local scripts package, datasets, pandas, huggingface_hub,
#   tqdm, smolagents.
# Outputs: functions parse_args, create_agent_team, load_gaia_dataset, append_answer, answer_single_question,
#   get_examples_to_answer, main; globals append_answer_lock, custom_role_conversions, BROWSER_CONFIG.
# Exceptions: login errors propagate at import.
# Side effects: at import, loads .env (override), logs in to the Hub, prints a VPN warning, creates
#   ./downloads_folder.
# KNOWN ISSUE: --use-open-models is parsed (with type bool, so any non-empty string is True) but never used.
# EXAMPLE COMMAND: from folder examples/open_deep_research, run: python run_gaia.py --concurrency 32 --run-name generate-traces-03-apr-noplanning --model-id gpt-4o
import argparse
import json
import os
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path
from typing import Any

import datasets
import pandas as pd
from dotenv import load_dotenv
from huggingface_hub import login, snapshot_download
from scripts.reformulator import prepare_response
from scripts.run_agents import (
    get_single_file_description,
    get_zip_description,
)
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
from tqdm import tqdm

from smolagents import (
    CodeAgent,
    GoogleSearchTool,
    LiteLLMModel,
    Model,
    TokenUsage,
    ToolCallingAgent,
)


load_dotenv(override=True)
login(os.getenv("HF_TOKEN"))

append_answer_lock = threading.Lock()


def parse_args():
    # [FUNCTION DOC]
    # HELPER: Parses the benchmark command-line arguments.
    # Inputs: sys.argv.
    # Outputs: argparse Namespace with concurrency, model_id, run_name, set_to_run, use_open_models,
    #   use_raw_dataset.
    # Exceptions: SystemExit on invalid/missing arguments (run-name is required).
    # Side effects: may print usage.
    parser = argparse.ArgumentParser()
    parser.add_argument("--concurrency", type=int, default=8)
    parser.add_argument("--model-id", type=str, default="o1")
    parser.add_argument("--run-name", type=str, required=True)
    parser.add_argument("--set-to-run", type=str, default="validation")
    parser.add_argument("--use-open-models", type=bool, default=False)
    parser.add_argument("--use-raw-dataset", action="store_true")
    return parser.parse_args()


### IMPORTANT: EVALUATION SWITCHES

print("Make sure you deactivated any VPN like Tailscale, else some URLs will be blocked!")

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


def create_agent_team(model: Model, token_counts: TokenUsage):
    # [FUNCTION DOC]
    # Builds the manager CodeAgent (visualizer plus a text inspector, max 12 steps, planning every 4, all
    #   imports allowed) and its managed ToolCallingAgent "search_agent" (serper Google search, browser
    #   navigation, archive search, text inspector; max 20 steps, planning every 4) on the given model.
    # Inputs: model (smolagents Model); token_counts (TokenUsage-like accumulator that the search agent's
    #   token totals are added to when it gives a final answer).
    # Outputs: the manager CodeAgent.
    # Exceptions: propagates construction errors.
    # Side effects: creates a SimpleTextBrowser; registers a final-answer check that mutates token_counts.
    text_limit = 100000
    ti_tool = TextInspectorTool(model, text_limit)

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

    def increment_web_agent_token_counts(final_answer, memory_step, agent):
        # [FUNCTION DOC]
        # HELPER: Final-answer check used purely as a hook to add the web agent's total input/output token
        #   counts into the enclosing token_counts.
        # Inputs: final_answer (ignored), memory_step (ignored), agent (the search agent).
        # Outputs: always True (never rejects the answer).
        # Exceptions: propagates attribute/key errors (see KNOWN ISSUE).
        # Side effects: mutates the closed-over token_counts.
        # KNOWN ISSUE: subscripts the monitor result with "input"/"output" keys while writing
        #   input_tokens/output_tokens attributes; the TokenUsage returned by the monitor is not subscriptable
        #   and the dict passed by answer_single_question has no such attributes, so this raises.
        token_counts_web = agent.monitor.get_total_token_counts()
        token_counts.input_tokens += token_counts_web["input"]
        token_counts.output_tokens += token_counts_web["output"]
        return True

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
        final_answer_checks=[increment_web_agent_token_counts],
    )
    text_webbrowser_agent.prompt_templates["managed_agent"]["task"] += """You can navigate to .txt online files.
    If a non-html page is in another format, especially .pdf or a Youtube video, use tool 'inspect_file_as_text' to inspect it.
    Additionally, if after some searching you find out that you need more information to answer the question, you can use `final_answer` with your request for clarification as argument to request for more information."""

    manager_agent = CodeAgent(
        model=model,
        tools=[visualizer, ti_tool],
        max_steps=12,
        verbosity_level=2,
        additional_authorized_imports=["*"],
        planning_interval=4,
        managed_agents=[text_webbrowser_agent],
    )
    return manager_agent


def load_gaia_dataset(use_raw_dataset: bool, set_to_run: str) -> datasets.Dataset:
    # [FUNCTION DOC]
    # Ensures the GAIA dataset is present in data/gaia (downloading it from the Hub if the folder is missing)
    #   and loads the requested split via the dataset's GAIA.py loading script with config "2023_all".
    # Inputs: use_raw_dataset (bool; True downloads gaia-benchmark/GAIA, False the gated
    #   smolagents/GAIA-annotated); set_to_run (str split name, e.g. "validation" or "test").
    # Outputs: datasets.Dataset with columns renamed to question, true_answer, task, and non-empty file_name
    #   values prefixed with data/gaia/<split>/.
    # Exceptions: Hub/network/auth errors (gated dataset) and dataset loading errors propagate.
    # Side effects: may download files to data/gaia.
    if not os.path.exists("data/gaia"):
        if use_raw_dataset:
            snapshot_download(
                repo_id="gaia-benchmark/GAIA",
                repo_type="dataset",
                local_dir="data/gaia",
                ignore_patterns=[".gitattributes", "README.md"],
            )
        else:
            # WARNING: this dataset is gated: make sure you visit the repo to require access.
            snapshot_download(
                repo_id="smolagents/GAIA-annotated",
                repo_type="dataset",
                local_dir="data/gaia",
                ignore_patterns=[".gitattributes", "README.md"],
            )

    def preprocess_file_paths(row):
        # [FUNCTION DOC]
        # HELPER: Map function that prefixes a non-empty file_name with the local split folder path.
        # Inputs: row (dict-like dataset row).
        # Outputs: the same row (mutated in place).
        # Exceptions: none raised directly.
        # Side effects: mutates row.
        if len(row["file_name"]) > 0:
            row["file_name"] = f"data/gaia/{set_to_run}/" + row["file_name"]
        return row

    eval_ds = datasets.load_dataset(
        "data/gaia/GAIA.py",
        name="2023_all",
        split=set_to_run,
        # data_files={"validation": "validation/metadata.jsonl", "test": "test/metadata.jsonl"},
    )

    eval_ds = eval_ds.rename_columns({"Question": "question", "Final answer": "true_answer", "Level": "task"})
    eval_ds = eval_ds.map(preprocess_file_paths)
    return eval_ds


def append_answer(entry: dict, jsonl_file: str) -> None:
    # [FUNCTION DOC]
    # HELPER: Appends one JSON record as a line to a JSONL file, thread-safely.
    # Inputs: entry (dict, must be JSON serializable), jsonl_file (str path).
    # Outputs: None.
    # Exceptions: TypeError if entry is not serializable; AssertionError if the file does not exist after
    #   writing.
    # Side effects: creates parent directories, writes to the file under the module lock, prints the absolute
    #   path.
    jsonl_path = Path(jsonl_file)
    jsonl_path.parent.mkdir(parents=True, exist_ok=True)
    with append_answer_lock, open(jsonl_file, "a", encoding="utf-8") as fp:
        fp.write(json.dumps(entry) + "\n")
    assert jsonl_path.exists(), "File not found!"
    print("Answer exported to file:", jsonl_path.resolve())


def answer_single_question(
    example: dict, model_id: str, answers_file: str, visual_inspection_tool: TextInspectorTool
) -> None:
    # [FUNCTION DOC]
    # Answers one GAIA example end to end and records the result. Builds a LiteLLMModel (o1 gets high
    #   reasoning effort and 8192 completion tokens, others 4096 max tokens), a text inspector, and an agent
    #   team; augments the question with an insistent preamble plus descriptions of attached files (zip
    #   archives are extracted and every file described); runs the agent, reformulates the final answer from
    #   the memory with prepare_response, detects parsing errors and iteration-limit stops, then appends an
    #   annotated record to answers_file.
    # Inputs: example (dict with question, file_name, task, task_id, true_answer), model_id (str),
    #   answers_file (str path), visual_inspection_tool (tool used to caption images; actually the visualizer
    #   tool despite the type hint).
    # Outputs: None.
    # Exceptions: errors during the agent run are caught and recorded as agent_error; errors in file
    #   description, token accounting or writing propagate.
    # Side effects: network calls, possible file extraction/downloads, appends to answers_file, prints; clears
    #   model_input_messages on every memory step.
    # KNOWN ISSUE: total_token_counts is a plain dict but input_tokens/output_tokens attributes are
    #   incremented on it, and the monitor result is subscripted with string keys; this raises
    #   AttributeError/TypeError after every run.
    # KNOWN ISSUE: the parsing-error check looks for a substring in message objects, which is likely never
    #   true.
    model_params: dict[str, Any] = {
        "model_id": model_id,
        "custom_role_conversions": custom_role_conversions,
    }
    if model_id == "o1":
        model_params["reasoning_effort"] = "high"
        model_params["max_completion_tokens"] = 8192
    else:
        model_params["max_tokens"] = 4096
    model = LiteLLMModel(**model_params)
    # model = InferenceClientModel(model_id="Qwen/Qwen3-32B", provider="novita", max_tokens=4096)
    document_inspection_tool = TextInspectorTool(model, 100000)

    total_token_counts: TokenUsage = {
        "input": 0,
        "output": 0,
    }
    agent = create_agent_team(model, total_token_counts)

    augmented_question = """You have one question to answer. It is paramount that you provide a correct answer.
Give it all you can: I know for a fact that you have access to all the relevant tools to solve it and find the correct answer (the answer does exist).
Failure or 'I cannot answer' or 'None found' will not be tolerated, success will be rewarded.
Run verification steps if that's needed, you must make sure you find the correct answer! Here is the task:

""" + example["question"]

    if example["file_name"]:
        if ".zip" in example["file_name"]:
            prompt_use_files = "\n\nTo solve the task above, you will have to use these attached files:\n"
            prompt_use_files += get_zip_description(
                example["file_name"], example["question"], visual_inspection_tool, document_inspection_tool
            )
        else:
            prompt_use_files = "\n\nTo solve the task above, you will have to use this attached file:\n"
            prompt_use_files += get_single_file_description(
                example["file_name"], example["question"], visual_inspection_tool, document_inspection_tool
            )
        augmented_question += prompt_use_files

    start_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    try:
        # Run agent 🚀
        final_result = agent.run(augmented_question)

        agent_memory = agent.write_memory_to_messages()

        final_result = prepare_response(augmented_question, agent_memory, reformulation_model=model)

        output = str(final_result)
        for memory_step in agent.memory.steps:
            memory_step.model_input_messages = None
        intermediate_steps = agent_memory

        # Check for parsing errors which indicate the LLM failed to follow the required format
        parsing_error = True if any(["AgentParsingError" in step for step in intermediate_steps]) else False

        # check if iteration limit exceeded
        iteration_limit_exceeded = True if "Agent stopped due to iteration limit or time limit." in output else False
        raised_exception = False

    except Exception as e:
        print("Error on ", augmented_question, e)
        output = None
        intermediate_steps = []
        parsing_error = False
        iteration_limit_exceeded = False
        exception = e
        raised_exception = True
    end_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    token_counts_manager = agent.monitor.get_total_token_counts()
    total_token_counts.input_tokens += token_counts_manager["input"]
    total_token_counts.output_tokens += token_counts_manager["output"]
    annotated_example = {
        "agent_name": model.model_id,
        "question": example["question"],
        "augmented_question": augmented_question,
        "prediction": output,
        "intermediate_steps": intermediate_steps,
        "parsing_error": parsing_error,
        "iteration_limit_exceeded": iteration_limit_exceeded,
        "agent_error": str(exception) if raised_exception else None,
        "task": example["task"],
        "task_id": example["task_id"],
        "true_answer": example["true_answer"],
        "start_time": start_time,
        "end_time": end_time,
        "token_counts": total_token_counts,
    }
    append_answer(annotated_example, answers_file)


def get_examples_to_answer(answers_file: str, eval_ds: datasets.Dataset) -> list[dict]:
    # [FUNCTION DOC]
    # Determines which dataset rows still need answering by reading questions already present in answers_file.
    # Inputs: answers_file (str JSONL path), eval_ds (datasets.Dataset).
    # Outputs: list of row dicts whose question is not yet answered and which have a non-empty file_name.
    # Exceptions: none raised; any error reading the file is printed and treated as no prior answers.
    # Side effects: reads the file, prints progress.
    # KNOWN ISSUE: questions without an attached file are silently excluded, so most of GAIA is never run.
    print(f"Loading answers from {answers_file}...")
    try:
        done_questions = pd.read_json(answers_file, lines=True)["question"].tolist()
        print(f"Found {len(done_questions)} previous results!")
    except Exception as e:
        print("Error when loading records: ", e)
        print("No usable records! ▶️ Starting new.")
        done_questions = []
    return [line for line in eval_ds.to_list() if line["question"] not in done_questions and line["file_name"]]


def main():
    # [FUNCTION DOC]
    # CLI entry: parses args, loads the dataset, prints task counts, computes output/<set>/<run-name>.jsonl,
    #   and answers remaining questions in a thread pool of the requested concurrency using the visualizer
    #   tool for images.
    # Inputs: command-line arguments.
    # Outputs: None.
    # Exceptions: the first worker exception propagates when its future result is collected.
    # Side effects: network calls, file writes, progress bar and console output.
    args = parse_args()
    print(f"Starting run with arguments: {args}")

    eval_ds = load_gaia_dataset(args.use_raw_dataset, args.set_to_run)
    print("Loaded evaluation dataset:")
    print(pd.DataFrame(eval_ds)["task"].value_counts())

    answers_file = f"output/{args.set_to_run}/{args.run_name}.jsonl"
    tasks_to_run = get_examples_to_answer(answers_file, eval_ds)

    with ThreadPoolExecutor(max_workers=args.concurrency) as exe:
        futures = [
            exe.submit(answer_single_question, example, args.model_id, answers_file, visualizer)
            for example in tasks_to_run
        ]
        for f in tqdm(as_completed(futures), total=len(tasks_to_run), desc="Processing tasks"):
            f.result()

    # for example in tasks_to_run:
    #     answer_single_question(example, args.model_id, answers_file, visualizer)
    print("All tasks processed.")


if __name__ == "__main__":
    main()
