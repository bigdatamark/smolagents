#!/usr/bin/env python
# coding=utf-8

# Copyright 2024 The HuggingFace Inc. team. All rights reserved.
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
# Core agent implementations of smolagents: the abstract ReAct-style MultiStepAgent base class and its two
#   concrete subclasses ToolCallingAgent (JSON/native tool calls) and CodeAgent (model writes Python code that
#   is executed by a local or remote Python executor).
# Exports: populate_template, ActionOutput, ToolOutput, the prompt TypedDicts (PlanningPromptTemplate,
#   ManagedAgentPromptTemplate, FinalAnswerPromptTemplate, PromptTemplates), EMPTY_PROMPT_TEMPLATES, RunResult,
#   the StreamEvent type alias, MultiStepAgent, ToolCallingAgent, CodeAgent, and AGENT_REGISTRY (class-name to
#   class map used for safe deserialization in from_dict/from_folder).
# Depends on: jinja2 (prompt templating), yaml (prompt files in smolagents.prompts), rich (console rendering),
#   huggingface_hub (push/pull agents to Spaces), and smolagents modules memory, models, monitoring, tools,
#   default_tools, local_python_executor, remote_executors, agent_types and utils.
# Exceptions: none raised at import time beyond missing dependencies (ImportError).
# Side effects: creates a module-level logger named after the module; no other import-time side effects.
import importlib
import json
import os
import tempfile
import textwrap
import time
import warnings
from abc import ABC, abstractmethod
from collections.abc import Callable, Generator
from concurrent.futures import ThreadPoolExecutor, as_completed
from contextvars import copy_context
from dataclasses import dataclass
from logging import getLogger
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal, Type, TypeAlias, TypedDict, Union

import yaml
from huggingface_hub import create_repo, metadata_update, snapshot_download, upload_folder
from jinja2 import StrictUndefined, Template
from rich.console import Group
from rich.live import Live
from rich.markdown import Markdown
from rich.panel import Panel
from rich.rule import Rule
from rich.text import Text


if TYPE_CHECKING:
    import PIL.Image

from .agent_types import AgentAudio, AgentImage, handle_agent_output_types
from .default_tools import TOOL_MAPPING, FinalAnswerTool
from .local_python_executor import BASE_BUILTIN_MODULES, LocalPythonExecutor, PythonExecutor, fix_final_answer_code
from .memory import (
    ActionStep,
    AgentMemory,
    CallbackRegistry,
    FinalAnswerStep,
    MemoryStep,
    PlanningStep,
    SystemPromptStep,
    TaskStep,
    Timing,
    ToolCall,
)
from .models import (
    CODEAGENT_RESPONSE_FORMAT,
    MODEL_REGISTRY,
    ChatMessage,
    ChatMessageStreamDelta,
    ChatMessageToolCall,
    MessageRole,
    Model,
    agglomerate_stream_deltas,
    parse_json_if_needed,
)
from .monitoring import (
    YELLOW_HEX,
    AgentLogger,
    LogLevel,
    Monitor,
    TokenUsage,
)
from .remote_executors import BlaxelExecutor, DockerExecutor, E2BExecutor, ModalExecutor
from .tools import BaseTool, Tool, validate_tool_arguments
from .utils import (
    AgentError,
    AgentExecutionError,
    AgentGenerationError,
    AgentMaxStepsError,
    AgentParsingError,
    AgentToolCallError,
    AgentToolExecutionError,
    create_agent_gradio_app_template,
    extract_code_from_text,
    is_valid_name,
    make_init_file,
    parse_code_blobs,
    truncate_content,
)


logger = getLogger(__name__)


def populate_template(template: str, variables: dict[str, Any]) -> str:
    # [FUNCTION DOC]
    # Renders a Jinja2 template string with the given variables, using StrictUndefined so that any referenced
    #   but missing variable is an error.
    # Inputs: template (str, Jinja2 source), variables (dict of name to value made available in the template).
    # Outputs: the rendered string.
    # Exceptions: raises a plain Exception whose message is "Error during jinja template rendering: <type>:
    #   <msg>" for any rendering error (including undefined variables). Template syntax errors raised while
    #   compiling (before the try block) propagate unwrapped as jinja2 TemplateSyntaxError.
    # Side effects: none.
    # KNOWN ISSUE: the wrapping Exception is raised without chaining to the original error and uses the
    #   generic Exception type, which makes it hard to catch specifically.
    compiled_template = Template(template, undefined=StrictUndefined)
    try:
        return compiled_template.render(**variables)
    except Exception as e:
        raise Exception(f"Error during jinja template rendering: {type(e).__name__}: {e}")


@dataclass
class ActionOutput:
    # [CLASS DOC]
    # Dataclass emitted at the end of every action step (by _step_stream) describing the step result.
    # Inputs/attributes: output (Any, the value produced by the step, or None), is_final_answer (bool, True
    #   when the output is the agent's final answer).
    # Outputs: n/a (plain data container with generated init/repr/eq).
    # Exceptions: none raised directly.
    # Side effects: none.
    output: Any
    is_final_answer: bool


@dataclass
class ToolOutput:
    # [CLASS DOC]
    # Dataclass describing the result of executing one tool call in ToolCallingAgent.
    # Inputs/attributes: id (str, tool call id), output (Any, raw tool return value), is_final_answer (bool,
    #   True if the tool was final_answer), observation (str, text shown to the model), tool_call (ToolCall, the
    #   originating call).
    # Outputs: n/a (plain data container).
    # Exceptions: none raised directly.
    # Side effects: none.
    id: str
    output: Any
    is_final_answer: bool
    observation: str
    tool_call: ToolCall


class PlanningPromptTemplate(TypedDict):
    # [CLASS DOC]
    # TypedDict with the Jinja2 prompt strings used for planning steps.
    # Inputs/attributes: initial_plan (str, prompt for the first plan; receives task, tools, managed_agents),
    #   update_plan_pre_messages (str, system prompt preceding memory when re-planning; receives task),
    #   update_plan_post_messages (str, user prompt following memory when re-planning; receives task, tools,
    #   managed_agents, remaining_steps).
    # Outputs: n/a (type definition only; instances are ordinary dicts).
    # Exceptions: none raised directly.
    # Side effects: none.
    # KNOWN ISSUE: the docstring names the first key "plan" but the actual key is initial_plan.
    """
    Prompt templates for the planning step.

    Args:
        plan (`str`): Initial plan prompt.
        update_plan_pre_messages (`str`): Update plan pre-messages prompt.
        update_plan_post_messages (`str`): Update plan post-messages prompt.
    """

    initial_plan: str
    update_plan_pre_messages: str
    update_plan_post_messages: str


class ManagedAgentPromptTemplate(TypedDict):
    # [CLASS DOC]
    # TypedDict with the Jinja2 prompts used when this agent is called as a managed agent by another agent.
    # Inputs/attributes: task (str, wraps the incoming task; receives name and task), report (str, wraps the
    #   final answer returned to the caller; receives name and final_answer).
    # Outputs: n/a (type definition only).
    # Exceptions: none raised directly.
    # Side effects: none.
    """
    Prompt templates for the managed agent.

    Args:
        task (`str`): Task prompt.
        report (`str`): Report prompt.
    """

    task: str
    report: str


class FinalAnswerPromptTemplate(TypedDict):
    # [CLASS DOC]
    # TypedDict with the prompts used to force a final answer when max steps are reached.
    # Inputs/attributes: pre_messages (str, used verbatim as a system message, not templated), post_messages
    #   (str, Jinja2 user prompt appended after memory; receives task).
    # Outputs: n/a (type definition only).
    # Exceptions: none raised directly.
    # Side effects: none.
    """
    Prompt templates for the final answer.

    Args:
        pre_messages (`str`): Pre-messages prompt.
        post_messages (`str`): Post-messages prompt.
    """

    pre_messages: str
    post_messages: str


class PromptTemplates(TypedDict):
    # [CLASS DOC]
    # TypedDict grouping all prompt templates an agent needs.
    # Inputs/attributes: system_prompt (str, Jinja2 system prompt), planning (PlanningPromptTemplate),
    #   managed_agent (ManagedAgentPromptTemplate), final_answer (FinalAnswerPromptTemplate). Custom templates
    #   passed to an agent must contain every key and sub-key present in EMPTY_PROMPT_TEMPLATES.
    # Outputs: n/a (type definition only).
    # Exceptions: none raised directly.
    # Side effects: none.
    """
    Prompt templates for the agent.

    Args:
        system_prompt (`str`): System prompt.
        planning ([`~agents.PlanningPromptTemplate`]): Planning prompt templates.
        managed_agent ([`~agents.ManagedAgentPromptTemplate`]): Managed agent prompt templates.
        final_answer ([`~agents.FinalAnswerPromptTemplate`]): Final answer prompt templates.
    """

    system_prompt: str
    planning: PlanningPromptTemplate
    managed_agent: ManagedAgentPromptTemplate
    final_answer: FinalAnswerPromptTemplate


EMPTY_PROMPT_TEMPLATES = PromptTemplates(
    system_prompt="",
    planning=PlanningPromptTemplate(
        initial_plan="",
        update_plan_pre_messages="",
        update_plan_post_messages="",
    ),
    managed_agent=ManagedAgentPromptTemplate(task="", report=""),
    final_answer=FinalAnswerPromptTemplate(pre_messages="", post_messages=""),
)


@dataclass
class RunResult:
    # [CLASS DOC]
    # Dataclass holding extended information about one agent run, returned by MultiStepAgent.run when full
    #   results are requested.
    # Inputs/attributes: output (Any or None, final answer), state ("success" or "max_steps_error"), steps
    #   (list of step dicts from memory), token_usage (TokenUsage or None if any step lacked usage), timing
    #   (Timing of the run). A deprecated messages argument/property aliases steps.
    # Outputs: n/a (data container; see dict for serialization).
    # Exceptions: ValueError from the constructor when both messages and steps are given.
    # Side effects: emits FutureWarning when the deprecated messages argument or property is used.
    # KNOWN ISSUE: the custom init replaces the dataclass-generated one, so every field defaults to None even
    #   though the type hints (e.g. state, timing) do not allow None; dict then fails if timing is None.
    """Holds extended information about an agent run.

    Attributes:
        output (Any | None): The final output of the agent run, if available.
        state (Literal["success", "max_steps_error"]): The final state of the agent after the run.
        steps (list[dict]): The agent's memory, as a list of steps.
        token_usage (TokenUsage | None): Count of tokens used during the run.
        timing (Timing): Timing details of the agent run: start time, end time, duration.
        messages (list[dict]): The agent's memory, as a list of messages.
            <Deprecated version="1.22.0">
            Parameter 'messages' is deprecated and will be removed in version 1.25. Please use 'steps' instead.
            </Deprecated>
    """

    output: Any | None
    state: Literal["success", "max_steps_error"]
    steps: list[dict]
    token_usage: TokenUsage | None
    timing: Timing

    def __init__(self, output=None, state=None, steps=None, token_usage=None, timing=None, messages=None):
        # [FUNCTION DOC]
        # Initializes the run result, supporting the deprecated messages alias for steps.
        # Inputs: output (Any, default None), state (str, default None), steps (list of dicts, default None),
        #   token_usage (TokenUsage, default None), timing (Timing, default None), messages (deprecated alias
        #   for steps, default None).
        # Outputs: none (initializes attributes output, state, steps, token_usage, timing).
        # Exceptions: ValueError if both messages and steps are provided.
        # Side effects: emits a FutureWarning when messages is provided.
        # Handle deprecated 'messages' parameter
        if messages is not None:
            if steps is not None:
                raise ValueError("Cannot specify both 'messages' and 'steps' parameters. Use 'steps' instead.")
            warnings.warn(
                "Parameter 'messages' is deprecated and will be removed in version 1.25. Please use 'steps' instead.",
                FutureWarning,
                stacklevel=2,
            )
            steps = messages

        # Initialize with dataclass fields
        self.output = output
        self.state = state
        self.steps = steps
        self.token_usage = token_usage
        self.timing = timing

    @property
    def messages(self):
        # [FUNCTION DOC]
        # Deprecated read-only property kept for backward compatibility.
        # Inputs: none.
        # Outputs: the steps list.
        # Exceptions: none raised directly.
        # Side effects: emits a FutureWarning on every access.
        """Backward compatibility property that returns steps."""
        warnings.warn(
            "Parameter 'messages' is deprecated and will be removed in version 1.25. Please use 'steps' instead.",
            FutureWarning,
            stacklevel=2,
        )
        return self.steps

    def dict(self):
        # [FUNCTION DOC]
        # Serializes the run result to a plain dictionary.
        # Inputs: none.
        # Outputs: dict with keys output, state, steps, token_usage (its dict form or None) and timing (its
        #   dict form).
        # Exceptions: AttributeError if timing is None (timing is serialized unconditionally).
        # Side effects: none.
        return {
            "output": self.output,
            "state": self.state,
            "steps": self.steps,
            "token_usage": self.token_usage.dict() if self.token_usage is not None else None,
            "timing": self.timing.dict(),
        }


StreamEvent: TypeAlias = Union[
    ChatMessageStreamDelta,
    ChatMessageToolCall,
    ActionOutput,
    ToolCall,
    ToolOutput,
    PlanningStep,
    ActionStep,
    FinalAnswerStep,
]


class MultiStepAgent(ABC):
    # [CLASS DOC]
    # Abstract base class for agents solving a task step by step with the ReAct loop: repeatedly generate an
    #   action with the model, execute it, observe the result, optionally re-plan every planning_interval steps,
    #   until a final answer is produced or max_steps is reached (then a final answer is forced from memory).
    # Constructor args: tools (list of Tool), model (Model), prompt_templates (PromptTemplates or None,
    #   defaults to empty templates; if given, must contain every key/sub-key), instructions (str, inserted in
    #   system prompt), max_steps (int, default 20), add_base_tools (bool, default False), verbosity_level
    #   (LogLevel, default INFO), managed_agents (list of agents with name and description), step_callbacks
    #   (list of callables registered for ActionStep, or dict from step class to callable or list of callables),
    #   planning_interval (int or None), name (str, must be a valid identifier), description (str),
    #   provide_run_summary (bool), final_answer_checks (list of callables taking final answer, memory and agent
    #   keyword), return_full_result (bool), logger (AgentLogger, created from verbosity_level if None).
    # Key attributes: tools (dict name to tool, always containing final_answer), managed_agents (dict name to
    #   agent), memory (AgentMemory), monitor (Monitor), step_callbacks (CallbackRegistry), state (dict of
    #   variables shared with tools/executor), step_number, task, stream_outputs, interrupt_switch.
    # Subclasses must implement initialize_system_prompt and _step_stream.
    # Exceptions: AssertionError for missing prompt templates or invalid tools/managed agents, ValueError for
    #   bad names, duplicate names or bad step_callbacks type.
    # Side effects: mutates managed agents (sets their inputs and output_type); may instantiate base tools.
    """
    Agent class that solves the given task step by step, using the ReAct framework:
    While the objective is not reached, the agent will perform a cycle of action (given by the LLM) and observation (obtained from the environment).

    Args:
        tools (`list[Tool]`): [`Tool`]s that the agent can use.
        model (`Callable[[list[dict[str, str]]], ChatMessage]`): Model that will generate the agent's actions.
        prompt_templates ([`~agents.PromptTemplates`], *optional*): Prompt templates.
        instructions (`str`, *optional*): Custom instructions for the agent, will be inserted in the system prompt.
        max_steps (`int`, default `20`): Maximum number of steps the agent can take to solve the task.
        add_base_tools (`bool`, default `False`): Whether to add the base tools to the agent's tools.
        verbosity_level (`LogLevel`, default `LogLevel.INFO`): Level of verbosity of the agent's logs.
        managed_agents (`list`, *optional*): Managed agents that the agent can call.
        step_callbacks (`list[Callable]` | `dict[Type[MemoryStep], Callable | list[Callable]]`, *optional*): Callbacks that will be called at each step.
        planning_interval (`int`, *optional*): Interval at which the agent will run a planning step.
        name (`str`, *optional*): Necessary for a managed agent only - the name by which this agent can be called.
        description (`str`, *optional*): Necessary for a managed agent only - the description of this agent.
        provide_run_summary (`bool`, *optional*): Whether to provide a run summary when called as a managed agent.
        final_answer_checks (`list[Callable]`, *optional*): List of validation functions to run before accepting a final answer.
            Each function should:
            - Take the final answer, the agent's memory, and the agent itself as arguments.
            - Return a boolean indicating whether the final answer is valid.
        return_full_result (`bool`, default `False`): Whether to return the full [`RunResult`] object or just the final answer output from the agent run.
    """

    def __init__(
        self,
        tools: list[Tool],
        model: Model,
        prompt_templates: PromptTemplates | None = None,
        instructions: str | None = None,
        max_steps: int = 20,
        add_base_tools: bool = False,
        verbosity_level: LogLevel = LogLevel.INFO,
        managed_agents: list | None = None,
        step_callbacks: list[Callable] | dict[Type[MemoryStep], Callable | list[Callable]] | None = None,
        planning_interval: int | None = None,
        name: str | None = None,
        description: str | None = None,
        provide_run_summary: bool = False,
        final_answer_checks: list[Callable] | None = None,
        return_full_result: bool = False,
        logger: AgentLogger | None = None,
    ):
        # [FUNCTION DOC]
        # Builds the agent: validates prompt templates, stores settings, sets up managed agents, tools and
        #   validates unique names, creates AgentMemory from the system prompt, the logger, the Monitor and step
        #   callbacks.
        # Inputs: see class doc (tools, model, prompt_templates, instructions, max_steps=20,
        #   add_base_tools=False, verbosity_level=LogLevel.INFO, managed_agents=None, step_callbacks=None,
        #   planning_interval=None, name=None, description=None, provide_run_summary=False,
        #   final_answer_checks=None, return_full_result=False, logger=None).
        # Outputs: none.
        # Exceptions: AssertionError if prompt_templates lacks keys/sub-keys or tools/managed agents are
        #   invalid; ValueError for invalid agent name, duplicate tool/agent names, or invalid step_callbacks
        #   type; any error from initialize_system_prompt (template rendering) propagates.
        # Side effects: mutates passed managed agents' inputs/output_type; sets stream_outputs to False.
        # KNOWN ISSUE: validation relies on assert statements, which are skipped when Python runs with -O.
        self.agent_name = self.__class__.__name__
        self.model = model
        self.prompt_templates = prompt_templates or EMPTY_PROMPT_TEMPLATES
        if prompt_templates is not None:
            missing_keys = set(EMPTY_PROMPT_TEMPLATES.keys()) - set(prompt_templates.keys())
            assert not missing_keys, (
                f"Some prompt templates are missing from your custom `prompt_templates`: {missing_keys}"
            )
            for key, value in EMPTY_PROMPT_TEMPLATES.items():
                if isinstance(value, dict):
                    for subkey in value.keys():
                        assert key in prompt_templates.keys() and (subkey in prompt_templates[key].keys()), (
                            f"Some prompt templates are missing from your custom `prompt_templates`: {subkey} under {key}"
                        )

        self.max_steps = max_steps
        self.step_number = 0
        self.planning_interval = planning_interval
        self.state: dict[str, Any] = {}
        self.name = self._validate_name(name)
        self.description = description
        self.provide_run_summary = provide_run_summary
        self.final_answer_checks = final_answer_checks if final_answer_checks is not None else []
        self.return_full_result = return_full_result
        self.instructions = instructions
        self._setup_managed_agents(managed_agents)
        self._setup_tools(tools, add_base_tools)
        self._validate_tools_and_managed_agents(tools, managed_agents)

        self.task: str | None = None
        self.memory = AgentMemory(self.system_prompt)

        if logger is None:
            self.logger = AgentLogger(level=verbosity_level)
        else:
            self.logger = logger

        self.monitor = Monitor(self.model, self.logger)
        self._setup_step_callbacks(step_callbacks)
        self.stream_outputs = False

    @property
    def system_prompt(self) -> str:
        # [FUNCTION DOC]
        # Property getter returning the current system prompt, rendered fresh on each access.
        # Inputs: none.
        # Outputs: the string returned by initialize_system_prompt.
        # Exceptions: propagates template rendering errors.
        # Side effects: none beyond those of initialize_system_prompt.
        return self.initialize_system_prompt()

    @system_prompt.setter
    def system_prompt(self, value: str):
        # [FUNCTION DOC]
        # Property setter that forbids assignment of system_prompt.
        # Inputs: value (str, ignored).
        # Outputs: none.
        # Exceptions: always raises AttributeError telling the user to edit prompt_templates["system_prompt"]
        #   instead.
        # Side effects: none.
        raise AttributeError(
            """The 'system_prompt' property is read-only. Use 'self.prompt_templates["system_prompt"]' instead."""
        )

    def _validate_name(self, name: str | None) -> str | None:
        # [FUNCTION DOC]
        # HELPER: Validates the optional agent name.
        # Inputs: name (str or None).
        # Outputs: the same name (or None).
        # Exceptions: ValueError if name is given and is not a valid Python identifier or is a reserved
        #   keyword.
        # Side effects: none.
        if name is not None and not is_valid_name(name):
            raise ValueError(f"Agent name '{name}' must be a valid Python identifier and not a reserved keyword.")
        return name

    def _setup_managed_agents(self, managed_agents: list | None = None) -> None:
        # [FUNCTION DOC]
        # HELPER: Stores managed agents keyed by name and makes them callable like tools.
        # Inputs: managed_agents (list of agents or None, default None).
        # Outputs: none (sets self.managed_agents to a dict, empty if none given).
        # Exceptions: AssertionError if any managed agent lacks a name or description.
        # Side effects: sets on each managed agent an inputs schema (task: string; additional_args: nullable
        #   object) and output_type "string".
        """Setup managed agents with proper logging."""
        self.managed_agents = {}
        if managed_agents:
            assert all(agent.name and agent.description for agent in managed_agents), (
                "All managed agents need both a name and a description!"
            )
            self.managed_agents = {agent.name: agent for agent in managed_agents}
            # Ensure managed agents can be called as tools by the model: set their inputs and output_type
            for agent in self.managed_agents.values():
                agent.inputs = {
                    "task": {"type": "string", "description": "Long detailed description of the task."},
                    "additional_args": {
                        "type": "object",
                        "description": "Dictionary of extra inputs to pass to the managed agent, e.g. images, dataframes, or any other contextual data it may need.",
                        "nullable": True,
                    },
                }
                agent.output_type = "string"

    def _setup_tools(self, tools, add_base_tools):
        # [FUNCTION DOC]
        # HELPER: Builds the tools dict keyed by tool name, optionally adding base tools, and always ensures a
        #   final_answer tool exists.
        # Inputs: tools (list of BaseTool), add_base_tools (bool): when true, instantiates every tool in
        #   TOOL_MAPPING except python_interpreter, which is only added when the class name is exactly
        #   ToolCallingAgent.
        # Outputs: none (sets self.tools).
        # Exceptions: AssertionError if any element is not a BaseTool.
        # Side effects: instantiates tool classes. User-provided tools with the same name as base tools are
        #   overwritten.
        assert all(isinstance(tool, BaseTool) for tool in tools), (
            "All elements must be instance of BaseTool (or a subclass)"
        )
        self.tools = {tool.name: tool for tool in tools}
        if add_base_tools:
            self.tools.update(
                {
                    name: cls()
                    for name, cls in TOOL_MAPPING.items()
                    if name != "python_interpreter" or self.__class__.__name__ == "ToolCallingAgent"
                }
            )
        self.tools.setdefault("final_answer", FinalAnswerTool())

    def _validate_tools_and_managed_agents(self, tools, managed_agents):
        # [FUNCTION DOC]
        # HELPER: Ensures names of provided tools, managed agents and the agent itself are all unique.
        # Inputs: tools (list of tools), managed_agents (list or None).
        # Outputs: none.
        # Exceptions: ValueError listing duplicated names.
        # Side effects: none.
        tool_and_managed_agent_names = [tool.name for tool in tools]
        if managed_agents is not None:
            tool_and_managed_agent_names += [agent.name for agent in managed_agents]
        if self.name:
            tool_and_managed_agent_names.append(self.name)
        if len(tool_and_managed_agent_names) != len(set(tool_and_managed_agent_names)):
            raise ValueError(
                "Each tool or managed_agent should have a unique name! You passed these duplicate names: "
                f"{[name for name in tool_and_managed_agent_names if tool_and_managed_agent_names.count(name) > 1]}"
            )

    def _setup_step_callbacks(self, step_callbacks):
        # [FUNCTION DOC]
        # HELPER: Creates the CallbackRegistry and registers user callbacks plus the monitor's update_metrics.
        # Inputs: step_callbacks (list: each registered for ActionStep; dict: step class to callable or list
        #   of callables; or None/empty).
        # Outputs: none (sets self.step_callbacks).
        # Exceptions: ValueError if a non-empty value is neither list nor dict.
        # Side effects: always registers self.monitor.update_metrics for ActionStep (so monitor must already
        #   exist).
        # Initialize step callbacks registry
        self.step_callbacks = CallbackRegistry()
        if step_callbacks:
            # Register callbacks list only for ActionStep for backward compatibility
            if isinstance(step_callbacks, list):
                for callback in step_callbacks:
                    self.step_callbacks.register(ActionStep, callback)
            # Register callbacks dict for specific step classes
            elif isinstance(step_callbacks, dict):
                for step_cls, callbacks in step_callbacks.items():
                    if not isinstance(callbacks, list):
                        callbacks = [callbacks]
                    for callback in callbacks:
                        self.step_callbacks.register(step_cls, callback)
            else:
                raise ValueError("step_callbacks must be a list or a dict")
        # Register monitor update_metrics only for ActionStep for backward compatibility
        self.step_callbacks.register(ActionStep, self.monitor.update_metrics)

    def run(
        self,
        task: str,
        stream: bool = False,
        reset: bool = True,
        images: list["PIL.Image.Image"] | None = None,
        additional_args: dict | None = None,
        max_steps: int | None = None,
        return_full_result: bool | None = None,
    ) -> Any | RunResult:
        # [FUNCTION DOC]
        # Runs the agent on a task, either returning the final result or a generator of streamed events.
        # Inputs: task (str), stream (bool, default False), reset (bool, default True: clear memory and
        #   monitor), images (list of PIL images or None), additional_args (dict or None: merged into self.state
        #   and described at the end of the task text), max_steps (int or None: falls back to self.max_steps),
        #   return_full_result (bool or None: falls back to self.return_full_result).
        # Outputs: if stream, the _run_stream generator (yielding stream deltas, tool calls, tool outputs,
        #   action outputs, planning/action steps and finally a FinalAnswerStep). Otherwise the final answer
        #   output, or a RunResult (output, aggregated token usage or None if any step lacked usage, step dicts,
        #   timing, and state "max_steps_error" if the last step's error is AgentMaxStepsError else "success").
        # Exceptions: AgentGenerationError and other errors from _run_stream propagate (non-generation
        #   AgentErrors are recorded in steps instead); AgentError if interrupted; AssertionError if the last
        #   step is not a FinalAnswerStep.
        # Side effects: sets self.task and interrupt_switch, updates state, refreshes memory.system_prompt,
        #   appends a TaskStep, logs the task, sends state and tools to the python executor if present.
        # KNOWN ISSUE: max_steps uses a truthiness fallback so 0 is treated as unset; return_full_result is
        #   ignored when stream is true.
        """
        Run the agent for the given task.

        Args:
            task (`str`): Task to perform.
            stream (`bool`): Whether to run in streaming mode.
                If `True`, returns a generator that yields each step as it is executed. You must iterate over this generator to process the individual steps (e.g., using a for loop or `next()`).
                If `False`, executes all steps internally and returns only the final answer after completion.
            reset (`bool`): Whether to reset the conversation or keep it going from previous run.
            images (`list[PIL.Image.Image]`, *optional*): Image(s) objects.
            additional_args (`dict`, *optional*): Any other variables that you want to pass to the agent run, for instance images or dataframes. Give them clear names!
            max_steps (`int`, *optional*): Maximum number of steps the agent can take to solve the task. if not provided, will use the agent's default value.
            return_full_result (`bool`, *optional*): Whether to return the full [`RunResult`] object or just the final answer output.
                If `None` (default), the agent's `self.return_full_result` setting is used.

        Example:
        ```py
        from smolagents import CodeAgent
        agent = CodeAgent(tools=[])
        agent.run("What is the result of 2 power 3.7384?")
        ```
        """
        max_steps = max_steps or self.max_steps
        self.task = task
        self.interrupt_switch = False
        if additional_args:
            self.state.update(additional_args)
            self.task += f"""
You have been provided with these additional arguments, that you can access directly using the keys as variables:
{str(additional_args)}."""

        self.memory.system_prompt = SystemPromptStep(system_prompt=self.system_prompt)
        if reset:
            self.memory.reset()
            self.monitor.reset()

        self.logger.log_task(
            content=self.task.strip(),
            subtitle=f"{type(self.model).__name__} - {(self.model.model_id if hasattr(self.model, 'model_id') else '')}",
            level=LogLevel.INFO,
            title=self.name if hasattr(self, "name") else None,
        )
        self.memory.steps.append(TaskStep(task=self.task, task_images=images))

        if getattr(self, "python_executor", None):
            self.python_executor.send_variables(variables=self.state)
            self.python_executor.send_tools({**self.tools, **self.managed_agents})

        if stream:
            # The steps are returned as they are executed through a generator to iterate on.
            return self._run_stream(task=self.task, max_steps=max_steps, images=images)

        run_start_time = time.time()
        steps = list(self._run_stream(task=self.task, max_steps=max_steps, images=images))

        # Outputs are returned only at the end. We only look at the last step.
        assert isinstance(steps[-1], FinalAnswerStep)
        output = steps[-1].output

        return_full_result = return_full_result if return_full_result is not None else self.return_full_result
        if return_full_result:
            total_input_tokens = 0
            total_output_tokens = 0
            correct_token_usage = True
            for step in self.memory.steps:
                if isinstance(step, (ActionStep, PlanningStep)):
                    if step.token_usage is None:
                        correct_token_usage = False
                        break
                    else:
                        total_input_tokens += step.token_usage.input_tokens
                        total_output_tokens += step.token_usage.output_tokens
            if correct_token_usage:
                token_usage = TokenUsage(input_tokens=total_input_tokens, output_tokens=total_output_tokens)
            else:
                token_usage = None

            if self.memory.steps and isinstance(getattr(self.memory.steps[-1], "error", None), AgentMaxStepsError):
                state = "max_steps_error"
            else:
                state = "success"

            step_dicts = self.memory.get_full_steps()

            return RunResult(
                output=output,
                token_usage=token_usage,
                steps=step_dicts,
                timing=Timing(start_time=run_start_time, end_time=time.time()),
                state=state,
            )

        return output

    def _run_stream(
        self, task: str, max_steps: int, images: list["PIL.Image.Image"] | None = None
    ) -> Generator[ActionStep | PlanningStep | FinalAnswerStep | ChatMessageStreamDelta]:
        # [FUNCTION DOC]
        # HELPER: Generator implementing the main ReAct loop.
        # Inputs: task (str), max_steps (int), images (list of PIL images or None, attached to each
        #   ActionStep).
        # Outputs (yields): for scheduled planning steps (step 1 and every planning_interval steps), the
        #   planning stream events and the PlanningStep; for each action step, every event from _step_stream
        #   then the ActionStep; if max steps are exhausted, the last action step again; finally a
        #   FinalAnswerStep wrapping the output-type-converted final answer.
        # Exceptions: AgentError if interrupt_switch is set; AgentGenerationError re-raised; AgentError from
        #   final answer checks is caught as a step error; other AgentErrors are stored on the step and the loop
        #   continues; non-agent exceptions propagate.
        # Side effects: resets step_number to 1 and increments it; appends planning/action steps to memory;
        #   runs step callbacks; logs final answer.
        # KNOWN ISSUE: when max steps is reached it yields the previous action_step a second time (not the
        #   max-steps step); and if max_steps is below 1 the loop never runs, so action_step and final_answer
        #   are undefined (UnboundLocalError).
        self.step_number = 1
        returned_final_answer = False
        while not returned_final_answer and self.step_number <= max_steps:
            if self.interrupt_switch:
                raise AgentError("Agent interrupted.", self.logger)

            # Run a planning step if scheduled
            if self.planning_interval is not None and (
                self.step_number == 1 or (self.step_number - 1) % self.planning_interval == 0
            ):
                planning_start_time = time.time()
                planning_step = None
                for element in self._generate_planning_step(
                    task, is_first_step=len(self.memory.steps) == 1, step=self.step_number
                ):  # Don't use the attribute step_number here, because there can be steps from previous runs
                    yield element
                    planning_step = element
                assert isinstance(planning_step, PlanningStep)  # Last yielded element should be a PlanningStep
                planning_end_time = time.time()
                planning_step.timing = Timing(
                    start_time=planning_start_time,
                    end_time=planning_end_time,
                )
                self._finalize_step(planning_step)
                self.memory.steps.append(planning_step)

            # Start action step!
            action_step_start_time = time.time()
            action_step = ActionStep(
                step_number=self.step_number,
                timing=Timing(start_time=action_step_start_time),
                observations_images=images,
            )
            self.logger.log_rule(f"Step {self.step_number}", level=LogLevel.INFO)
            try:
                for output in self._step_stream(action_step):
                    # Yield all
                    yield output

                    if isinstance(output, ActionOutput) and output.is_final_answer:
                        final_answer = output.output
                        self.logger.log(
                            Text(f"Final answer: {final_answer}", style=f"bold {YELLOW_HEX}"),
                            level=LogLevel.INFO,
                        )

                        if self.final_answer_checks:
                            self._validate_final_answer(final_answer)
                        returned_final_answer = True
                        action_step.is_final_answer = True

            except AgentGenerationError as e:
                # Agent generation errors are not caused by a Model error but an implementation error: so we should raise them and exit.
                raise e
            except AgentError as e:
                # Other AgentError types are caused by the Model, so we should log them and iterate.
                action_step.error = e
            finally:
                self._finalize_step(action_step)
                self.memory.steps.append(action_step)
                yield action_step
                self.step_number += 1

        if not returned_final_answer and self.step_number == max_steps + 1:
            final_answer = self._handle_max_steps_reached(task)
            yield action_step
        final_answer_step = FinalAnswerStep(handle_agent_output_types(final_answer))
        self._finalize_step(final_answer_step)
        yield final_answer_step

    def _validate_final_answer(self, final_answer: Any):
        # [FUNCTION DOC]
        # HELPER: Runs every final_answer_checks function on a candidate final answer.
        # Inputs: final_answer (Any).
        # Outputs: none.
        # Exceptions: AgentError naming the failed check if a check returns a falsy value or raises.
        # Side effects: whatever the check functions do.
        for check_function in self.final_answer_checks:
            try:
                assert check_function(final_answer, self.memory, agent=self)
            except Exception as e:
                raise AgentError(f"Check {check_function.__name__} failed with error: {e}", self.logger)

    def _finalize_step(self, memory_step: ActionStep | PlanningStep | FinalAnswerStep):
        # [FUNCTION DOC]
        # HELPER: Closes a memory step and runs registered callbacks for its type.
        # Inputs: memory_step (ActionStep, PlanningStep or FinalAnswerStep).
        # Outputs: none.
        # Exceptions: propagates callback errors.
        # Side effects: sets timing.end_time to now for non-final steps; invokes callbacks with agent=self.
        if not isinstance(memory_step, FinalAnswerStep):
            memory_step.timing.end_time = time.time()
        self.step_callbacks.callback(memory_step, agent=self)

    def _handle_max_steps_reached(self, task: str) -> Any:
        # [FUNCTION DOC]
        # HELPER: Produces a forced final answer after max steps and records it as an ActionStep with an
        #   AgentMaxStepsError.
        # Inputs: task (str).
        # Outputs: the content of the model's final answer message.
        # Exceptions: none raised directly (provide_final_answer swallows model errors).
        # Side effects: calls the model, appends the step to memory, runs callbacks.
        action_step_start_time = time.time()
        final_answer = self.provide_final_answer(task)
        final_memory_step = ActionStep(
            step_number=self.step_number,
            error=AgentMaxStepsError("Reached max steps.", self.logger),
            timing=Timing(start_time=action_step_start_time, end_time=time.time()),
            token_usage=final_answer.token_usage,
        )
        final_memory_step.action_output = final_answer.content
        self._finalize_step(final_memory_step)
        self.memory.steps.append(final_memory_step)
        return final_answer.content

    def _generate_planning_step(
        self, task, is_first_step: bool, step: int
    ) -> Generator[ChatMessageStreamDelta | PlanningStep]:
        # [FUNCTION DOC]
        # HELPER: Generator producing an initial or updated plan with the model.
        # Inputs: task (str), is_first_step (bool: use initial_plan template on a single user message;
        #   otherwise use summary-mode memory between update_plan_pre_messages system and
        #   update_plan_post_messages user messages), step (int, current step number used to compute
        #   remaining_steps).
        # Outputs (yields): stream deltas when streaming is enabled and the model supports it, then one
        #   PlanningStep with input messages, wrapped plan text, model output message, token usage and timing.
        #   Generation stops at "<end_plan>".
        # Exceptions: propagates model and template errors.
        # Side effects: calls the model, renders live markdown when streaming, logs the plan.
        # KNOWN ISSUE: remaining_steps uses self.max_steps, ignoring a max_steps override passed to run; when
        #   streaming, input tokens are overwritten per event while output tokens are summed.
        start_time = time.time()
        if is_first_step:
            input_messages = [
                ChatMessage(
                    role=MessageRole.USER,
                    content=[
                        {
                            "type": "text",
                            "text": populate_template(
                                self.prompt_templates["planning"]["initial_plan"],
                                variables={"task": task, "tools": self.tools, "managed_agents": self.managed_agents},
                            ),
                        }
                    ],
                )
            ]
            if self.stream_outputs and hasattr(self.model, "generate_stream"):
                plan_message_content = ""
                output_stream = self.model.generate_stream(input_messages, stop_sequences=["<end_plan>"])  # type: ignore
                input_tokens, output_tokens = 0, 0
                with Live("", console=self.logger.console, vertical_overflow="visible") as live:
                    for event in output_stream:
                        if event.content is not None:
                            plan_message_content += event.content
                            live.update(Markdown(plan_message_content))
                            if event.token_usage:
                                input_tokens = event.token_usage.input_tokens
                                output_tokens += event.token_usage.output_tokens
                        yield event
            else:
                plan_message = self.model.generate(input_messages, stop_sequences=["<end_plan>"])
                plan_message_content = plan_message.content
                input_tokens, output_tokens = 0, 0
                if plan_message.token_usage:
                    input_tokens = plan_message.token_usage.input_tokens
                    output_tokens = plan_message.token_usage.output_tokens
            plan = textwrap.dedent(
                f"""Here are the facts I know and the plan of action that I will follow to solve the task:\n```\n{plan_message_content}\n```"""
            )
        else:
            # Summary mode removes the system prompt and previous planning messages output by the model.
            # Removing previous planning messages avoids influencing too much the new plan.
            memory_messages = self.write_memory_to_messages(summary_mode=True)
            plan_update_pre = ChatMessage(
                role=MessageRole.SYSTEM,
                content=[
                    {
                        "type": "text",
                        "text": populate_template(
                            self.prompt_templates["planning"]["update_plan_pre_messages"], variables={"task": task}
                        ),
                    }
                ],
            )
            plan_update_post = ChatMessage(
                role=MessageRole.USER,
                content=[
                    {
                        "type": "text",
                        "text": populate_template(
                            self.prompt_templates["planning"]["update_plan_post_messages"],
                            variables={
                                "task": task,
                                "tools": self.tools,
                                "managed_agents": self.managed_agents,
                                "remaining_steps": (self.max_steps - step),
                            },
                        ),
                    }
                ],
            )
            input_messages = [plan_update_pre] + memory_messages + [plan_update_post]
            if self.stream_outputs and hasattr(self.model, "generate_stream"):
                plan_message_content = ""
                input_tokens, output_tokens = 0, 0
                with Live("", console=self.logger.console, vertical_overflow="visible") as live:
                    for event in self.model.generate_stream(
                        input_messages,
                        stop_sequences=["<end_plan>"],
                    ):  # type: ignore
                        if event.content is not None:
                            plan_message_content += event.content
                            live.update(Markdown(plan_message_content))
                            if event.token_usage:
                                input_tokens = event.token_usage.input_tokens
                                output_tokens += event.token_usage.output_tokens
                        yield event
            else:
                plan_message = self.model.generate(input_messages, stop_sequences=["<end_plan>"])
                plan_message_content = plan_message.content
                input_tokens, output_tokens = 0, 0
                if plan_message.token_usage:
                    input_tokens = plan_message.token_usage.input_tokens
                    output_tokens = plan_message.token_usage.output_tokens
            plan = textwrap.dedent(
                f"""I still need to solve the task I was given:\n```\n{self.task}\n```\n\nHere are the facts I know and my new/updated plan of action to solve the task:\n```\n{plan_message_content}\n```"""
            )
        log_headline = "Initial plan" if is_first_step else "Updated plan"
        self.logger.log(Rule(f"[bold]{log_headline}", style="orange"), Text(plan), level=LogLevel.INFO)
        yield PlanningStep(
            model_input_messages=input_messages,
            plan=plan,
            model_output_message=ChatMessage(role=MessageRole.ASSISTANT, content=plan_message_content),
            token_usage=TokenUsage(input_tokens=input_tokens, output_tokens=output_tokens),
            timing=Timing(start_time=start_time, end_time=time.time()),
        )

    @abstractmethod
    def initialize_system_prompt(self) -> str:
        # [FUNCTION DOC]
        # Abstract method: subclasses return the rendered system prompt.
        # Inputs: none.
        # Outputs: system prompt string (base implementation returns None).
        # Exceptions: none raised directly.
        # Side effects: none.
        """To be implemented in child classes"""
        ...

    def interrupt(self):
        # [FUNCTION DOC]
        # Requests interruption of a running agent; the loop raises AgentError at the start of the next step.
        # Inputs: none.
        # Outputs: none.
        # Exceptions: none raised directly.
        # Side effects: sets interrupt_switch to True.
        """Interrupts the agent execution."""
        self.interrupt_switch = True

    def write_memory_to_messages(
        self,
        summary_mode: bool = False,
    ) -> list[ChatMessage]:
        # [FUNCTION DOC]
        # Converts the system prompt step and all memory steps into a flat list of chat messages for the
        #   model.
        # Inputs: summary_mode (bool, default False: passed to each step's to_messages, dropping e.g. system
        #   prompt and plans).
        # Outputs: list of ChatMessage.
        # Exceptions: none raised directly.
        # Side effects: none.
        """
        Reads past llm_outputs, actions, and observations or errors from the memory into a series of messages
        that can be used as input to the LLM. Adds a number of keywords (such as PLAN, error, etc) to help
        the LLM.
        """
        messages = self.memory.system_prompt.to_messages(summary_mode=summary_mode)
        for memory_step in self.memory.steps:
            messages.extend(memory_step.to_messages(summary_mode=summary_mode))
        return messages

    def _step_stream(
        self, memory_step: ActionStep
    ) -> Generator[ChatMessageStreamDelta | ToolCall | ToolOutput | ActionOutput]:
        # [FUNCTION DOC]
        # Abstract-style hook performing one ReAct step; subclasses must override.
        # Inputs: memory_step (ActionStep to fill).
        # Outputs: generator expected to yield stream deltas, ToolCall, ToolOutput and finally an
        #   ActionOutput.
        # Exceptions: always raises NotImplementedError in the base class.
        # Side effects: none.
        # KNOWN ISSUE: docstring says it finally yields None or the final answer, but implementations yield an
        #   ActionOutput.
        """
        Perform one step in the ReAct framework: the agent thinks, acts, and observes the result.
        Yields ChatMessageStreamDelta during the run if streaming is enabled.
        At the end, yields either None if the step is not final, or the final answer.
        """
        raise NotImplementedError("This method should be implemented in child classes")

    def step(self, memory_step: ActionStep) -> Any:
        # [FUNCTION DOC]
        # Runs one step non-streaming by exhausting _step_stream.
        # Inputs: memory_step (ActionStep).
        # Outputs: the last yielded element (in practice an ActionOutput).
        # Exceptions: propagates errors from _step_stream; IndexError if it yields nothing.
        # Side effects: those of _step_stream.
        # KNOWN ISSUE: docstring says it returns None or the final answer, but it returns the ActionOutput
        #   object.
        """
        Perform one step in the ReAct framework: the agent thinks, acts, and observes the result.
        Returns either None if the step is not final, or the final answer.
        """
        return list(self._step_stream(memory_step))[-1]

    def extract_action(self, model_output: str, split_token: str) -> tuple[str, str]:
        # [FUNCTION DOC]
        # Splits model output into rationale and action using the last two segments around split_token.
        # Inputs: model_output (str), split_token (str).
        # Outputs: tuple (rationale stripped, action stripped).
        # Exceptions: AgentParsingError if the token is absent (indexing fails).
        # Side effects: none.
        """
        Parse action from the LLM output

        Args:
            model_output (`str`): Output of the LLM
            split_token (`str`): Separator for the action. Should match the example in the system prompt.
        """
        try:
            split = model_output.split(split_token)
            rationale, action = (
                split[-2],
                split[-1],
            )  # NOTE: using indexes starting from the end solves for when you have more than one split_token in the output
        except Exception:
            raise AgentParsingError(
                f"No '{split_token}' token provided in your output.\nYour output:\n{model_output}\n. Be sure to include an action, prefaced with '{split_token}'!",
                self.logger,
            )
        return rationale.strip(), action.strip()

    def provide_final_answer(self, task: str) -> ChatMessage:
        # [FUNCTION DOC]
        # Asks the model for a final answer from the whole memory (used when max steps are reached).
        # Inputs: task (str).
        # Outputs: the model's ChatMessage; on any model error, an assistant ChatMessage containing the error
        #   text.
        # Exceptions: none from the model call (swallowed); template rendering errors propagate.
        # Side effects: calls the model.
        # KNOWN ISSUE: docstring documents an images argument that does not exist and a str return, but a
        #   ChatMessage is returned.
        """
        Provide the final answer to the task, based on the logs of the agent's interactions.

        Args:
            task (`str`): Task to perform.
            images (`list[PIL.Image.Image]`, *optional*): Image(s) objects.

        Returns:
            `str`: Final answer to the task.
        """
        messages = [
            ChatMessage(
                role=MessageRole.SYSTEM,
                content=[
                    {
                        "type": "text",
                        "text": self.prompt_templates["final_answer"]["pre_messages"],
                    }
                ],
            )
        ]
        messages += self.write_memory_to_messages()[1:]
        messages.append(
            ChatMessage(
                role=MessageRole.USER,
                content=[
                    {
                        "type": "text",
                        "text": populate_template(
                            self.prompt_templates["final_answer"]["post_messages"], variables={"task": task}
                        ),
                    }
                ],
            )
        )
        try:
            chat_message: ChatMessage = self.model.generate(messages)
            return chat_message
        except Exception as e:
            return ChatMessage(
                role=MessageRole.ASSISTANT,
                content=[{"type": "text", "text": f"Error in generating final LLM output: {e}"}],
            )

    def visualize(self):
        # [FUNCTION DOC]
        # Prints a rich tree of the agent, its tools and managed agents.
        # Inputs: none. Outputs: none.
        # Exceptions: none raised directly.
        # Side effects: console output through the logger.
        """Creates a rich tree visualization of the agent's structure."""
        self.logger.visualize_agent_tree(self)

    def replay(self, detailed: bool = False):
        # [FUNCTION DOC]
        # Prints a replay of the agent's memory steps.
        # Inputs: detailed (bool, default False: also show memory at each step).
        # Outputs: none.
        # Exceptions: none raised directly.
        # Side effects: console output through the logger.
        """Prints a pretty replay of the agent's steps.

        Args:
            detailed (bool, optional): If True, also displays the memory at each step. Defaults to False.
                Careful: will increase log length exponentially. Use only for debugging.
        """
        self.memory.replay(self.logger, detailed=detailed)

    def __call__(self, task: str, **kwargs):
        # [FUNCTION DOC]
        # Entry point used when this agent is invoked as a managed agent: wraps the task with the
        #   managed_agent task template, runs it, and wraps the result with the report template.
        # Inputs: task (str), kwargs (forwarded to run, e.g. additional_args).
        # Outputs: report string; if provide_run_summary, followed by a summary_of_work section with truncated
        #   summary-mode memory messages.
        # Exceptions: propagates errors from run and template rendering.
        # Side effects: runs the agent (memory reset, model calls, logging).
        """Adds additional prompting for the managed agent, runs it, and wraps the output.
        This method is called only by a managed agent.
        """
        full_task = populate_template(
            self.prompt_templates["managed_agent"]["task"],
            variables=dict(name=self.name, task=task),
        )
        result = self.run(full_task, **kwargs)
        if isinstance(result, RunResult):
            report = result.output
        else:
            report = result
        answer = populate_template(
            self.prompt_templates["managed_agent"]["report"], variables=dict(name=self.name, final_answer=report)
        )
        if self.provide_run_summary:
            answer += "\n\nFor more detail, find below a summary of this agent's work:\n<summary_of_work>\n"
            for message in self.write_memory_to_messages(summary_mode=True):
                content = message.content
                answer += "\n" + truncate_content(str(content)) + "\n---"
            answer += "\n</summary_of_work>"
        return answer

    def save(self, output_dir: str | Path, relative_path: str | None = None):
        # [FUNCTION DOC]
        # Writes the agent to a folder: __init__.py, tools/<name>.py for each tool, managed_agents/<name>/
        #   recursively, prompts.yaml, agent.json (to_dict with tools as names and managed_agents as name to
        #   class map), requirements.txt and a Gradio app.py.
        # Inputs: output_dir (str or Path), relative_path (str or None: dotted import prefix used for nested
        #   managed agents).
        # Outputs: none.
        # Exceptions: propagates I/O and serialization errors.
        # Side effects: creates directories and files on disk; may log about ignored callbacks.
        # KNOWN ISSUE: docstring mentions prompt.yaml but the file written is prompts.yaml; relative_path
        #   undocumented.
        """
        Saves the relevant code files for your agent. This will copy the code of your agent in `output_dir` as well as autogenerate:

        - a `tools` folder containing the logic for each of the tools under `tools/{tool_name}.py`.
        - a `managed_agents` folder containing the logic for each of the managed agents.
        - an `agent.json` file containing a dictionary representing your agent.
        - a `prompt.yaml` file containing the prompt templates used by your agent.
        - an `app.py` file providing a UI for your agent when it is exported to a Space with `agent.push_to_hub()`
        - a `requirements.txt` containing the names of the modules used by your tool (as detected when inspecting its
          code)

        Args:
            output_dir (`str` or `Path`): The folder in which you want to save your agent.
        """
        make_init_file(output_dir)

        # Recursively save managed agents
        if self.managed_agents:
            make_init_file(os.path.join(output_dir, "managed_agents"))
            for agent_name, agent in self.managed_agents.items():
                agent_suffix = f"managed_agents.{agent_name}"
                if relative_path:
                    agent_suffix = relative_path + "." + agent_suffix
                agent.save(os.path.join(output_dir, "managed_agents", agent_name), relative_path=agent_suffix)

        class_name = self.__class__.__name__

        # Save tools to different .py files
        for tool in self.tools.values():
            make_init_file(os.path.join(output_dir, "tools"))
            tool.save(os.path.join(output_dir, "tools"), tool_file_name=tool.name, make_gradio_app=False)

        # Save prompts to yaml
        yaml_prompts = yaml.safe_dump(
            self.prompt_templates,
            default_style="|",  # This forces block literals for all strings
            default_flow_style=False,
            width=float("inf"),
            sort_keys=False,
            allow_unicode=True,
            indent=2,
        )

        with open(os.path.join(output_dir, "prompts.yaml"), "w", encoding="utf-8") as f:
            f.write(yaml_prompts)

        # Save agent dictionary to json
        agent_dict = self.to_dict()
        agent_dict["tools"] = [tool.name for tool in self.tools.values()]
        agent_dict["managed_agents"] = {agent.name: agent.__class__.__name__ for agent in self.managed_agents.values()}
        with open(os.path.join(output_dir, "agent.json"), "w", encoding="utf-8") as f:
            json.dump(agent_dict, f, indent=4)

        # Save requirements
        with open(os.path.join(output_dir, "requirements.txt"), "w", encoding="utf-8") as f:
            f.writelines(f"{r}\n" for r in agent_dict["requirements"])

        # Make agent.py file with Gradio UI
        agent_name = f"agent_{self.name}" if getattr(self, "name", None) else "agent"
        managed_agent_relative_path = relative_path + "." if relative_path is not None else ""
        app_template = create_agent_gradio_app_template()

        # Render the app.py file from Jinja2 template
        app_text = app_template.render(
            {
                "agent_name": agent_name,
                "class_name": class_name,
                "agent_dict": agent_dict,
                "tools": self.tools,
                "managed_agents": self.managed_agents,
                "managed_agent_relative_path": managed_agent_relative_path,
            }
        )

        with open(os.path.join(output_dir, "app.py"), "w", encoding="utf-8") as f:
            f.write(app_text + "\n")  # Append newline at the end

    def to_dict(self) -> dict[str, Any]:
        # [FUNCTION DOC]
        # Serializes the agent configuration to a dict.
        # Inputs: none.
        # Outputs: dict with class, tools (tool dicts), model (class and data), managed_agents (their dicts),
        #   prompt_templates, max_steps, verbosity_level (int), planning_interval, name, description and sorted
        #   requirements (tool and managed agent requirements plus top-level packages of non-builtin authorized
        #   imports).
        # Exceptions: propagates errors from tool/model to_dict.
        # Side effects: logs that step_callbacks/final_answer_checks are ignored when set.
        # KNOWN ISSUE: step_callbacks is always a non-empty registry, so that message is always logged;
        #   instructions, return_full_result and provide_run_summary are not serialized.
        """Convert the agent to a dictionary representation.

        Returns:
            `dict`: Dictionary representation of the agent.
        """
        # TODO: handle serializing step_callbacks and final_answer_checks
        for attr in ["final_answer_checks", "step_callbacks"]:
            if getattr(self, attr, None):
                self.logger.log(f"This agent has {attr}: they will be ignored by this method.", LogLevel.INFO)

        tool_dicts = [tool.to_dict() for tool in self.tools.values()]
        tool_requirements = {req for tool in self.tools.values() for req in tool.to_dict()["requirements"]}
        managed_agents_requirements = {
            req for managed_agent in self.managed_agents.values() for req in managed_agent.to_dict()["requirements"]
        }
        requirements = tool_requirements | managed_agents_requirements
        if hasattr(self, "authorized_imports"):
            requirements.update(
                {package.split(".")[0] for package in self.authorized_imports if package not in BASE_BUILTIN_MODULES}
            )

        agent_dict = {
            "class": self.__class__.__name__,
            "tools": tool_dicts,
            "model": {
                "class": self.model.__class__.__name__,
                "data": self.model.to_dict(),
            },
            "managed_agents": [managed_agent.to_dict() for managed_agent in self.managed_agents.values()],
            "prompt_templates": self.prompt_templates,
            "max_steps": self.max_steps,
            "verbosity_level": int(self.logger.level),
            "planning_interval": self.planning_interval,
            "name": self.name,
            "description": self.description,
            "requirements": sorted(requirements),
        }
        return agent_dict

    @classmethod
    def from_dict(cls, agent_dict: dict[str, Any], **kwargs) -> "MultiStepAgent":
        # [FUNCTION DOC]
        # Classmethod rebuilding an agent from to_dict output.
        # Inputs: agent_dict (dict), kwargs (override constructor args; also forwarded to managed agents'
        #   from_dict).
        # Outputs: a new instance of cls.
        # Exceptions: ValueError for model or managed agent classes not in MODEL_REGISTRY/AGENT_REGISTRY;
        #   KeyError if required keys are missing.
        # Side effects: executes tool source code via Tool.from_code.
        """Create agent from a dictionary representation.

        Args:
            agent_dict (`dict[str, Any]`): Dictionary representation of the agent.
            **kwargs: Additional keyword arguments that will override agent_dict values.

        Returns:
            `MultiStepAgent`: Instance of the agent class.
        """
        # Load model
        model_info = agent_dict["model"]
        model_class = MODEL_REGISTRY.get(model_info["class"])
        if model_class is None:
            raise ValueError(
                f"Unknown model class '{model_info['class']}'. "
                f"Supported models: {', '.join(sorted(MODEL_REGISTRY.keys()))}"
            )
        model = model_class.from_dict(model_info["data"])
        # Load tools
        tools = []
        for tool_info in agent_dict["tools"]:
            tools.append(Tool.from_code(tool_info["code"]))
        # Load managed agents
        managed_agents = []
        for managed_agent_dict in agent_dict["managed_agents"]:
            agent_class = AGENT_REGISTRY.get(managed_agent_dict["class"])
            if agent_class is None:
                raise ValueError(
                    f"Unknown agent class '{managed_agent_dict['class']}'. "
                    f"Supported agents: {', '.join(sorted(AGENT_REGISTRY.keys()))}"
                )
            managed_agent = agent_class.from_dict(managed_agent_dict, **kwargs)
            managed_agents.append(managed_agent)
        # Extract base agent parameters
        agent_args = {
            "model": model,
            "tools": tools,
            "managed_agents": managed_agents,
            "prompt_templates": agent_dict.get("prompt_templates"),
            "max_steps": agent_dict.get("max_steps"),
            "verbosity_level": agent_dict.get("verbosity_level"),
            "planning_interval": agent_dict.get("planning_interval"),
            "name": agent_dict.get("name"),
            "description": agent_dict.get("description"),
        }
        # Filter out None values to use defaults from __init__
        agent_args = {k: v for k, v in agent_args.items() if v is not None}
        # Update with any additional kwargs
        agent_args.update(kwargs)
        # Create agent instance
        return cls(**agent_args)

    @classmethod
    def from_hub(
        cls,
        repo_id: str,
        token: str | None = None,
        trust_remote_code: bool = False,
        **kwargs,
    ):
        # [FUNCTION DOC]
        # Classmethod downloading an agent Space from the Hugging Face Hub and loading it with from_folder.
        # Inputs: repo_id (str), token (str or None), trust_remote_code (bool, default False), kwargs
        #   (cache_dir, force_download, proxies, revision, local_files_only go to snapshot_download; the rest to
        #   from_folder).
        # Outputs: the loaded agent.
        # Exceptions: ValueError if trust_remote_code is false; network/hub errors propagate.
        # Side effects: network download to the local cache; executes downloaded tool code.
        """
        Loads an agent defined on the Hub.

        <Tip warning={true}>

        Loading a tool from the Hub means that you'll download the tool and execute it locally.
        ALWAYS inspect the tool you're downloading before loading it within your runtime, as you would do when
        installing a package using pip/npm/apt.

        </Tip>

        Args:
            repo_id (`str`):
                The name of the repo on the Hub where your tool is defined.
            token (`str`, *optional*):
                The token to identify you on hf.co. If unset, will use the token generated when running
                `huggingface-cli login` (stored in `~/.huggingface`).
            trust_remote_code(`bool`, *optional*, defaults to False):
                This flags marks that you understand the risk of running remote code and that you trust this tool.
                If not setting this to True, loading the tool from Hub will fail.
            kwargs (additional keyword arguments, *optional*):
                Additional keyword arguments that will be split in two: all arguments relevant to the Hub (such as
                `cache_dir`, `revision`, `subfolder`) will be used when downloading the files for your agent, and the
                others will be passed along to its init.
        """
        if not trust_remote_code:
            raise ValueError(
                "Loading an agent from Hub requires to acknowledge you trust its code: to do so, pass `trust_remote_code=True`."
            )

        # Get the agent's Hub folder.
        download_kwargs = {"token": token, "repo_type": "space"} | {
            key: kwargs.pop(key)
            for key in [
                "cache_dir",
                "force_download",
                "proxies",
                "revision",
                "local_files_only",
            ]
            if key in kwargs
        }

        download_folder = Path(snapshot_download(repo_id=repo_id, **download_kwargs))
        return cls.from_folder(download_folder, **kwargs)

    @classmethod
    def from_folder(cls, folder: str | Path, **kwargs):
        # [FUNCTION DOC]
        # Classmethod loading an agent saved by save: reads agent.json, renames legacy HfApiModel to
        #   InferenceClientModel, recursively loads managed agents, reads tool sources, then calls from_dict.
        # Inputs: folder (str or Path), kwargs (passed to from_dict, not to managed agents).
        # Outputs: the loaded agent.
        # Exceptions: ValueError for unknown managed agent class; file and JSON errors propagate.
        # Side effects: reads files; logs a warning for HfApiModel; executes tool code.
        """Loads an agent from a local folder.

        Args:
            folder (`str` or `Path`): The folder where the agent is saved.
            **kwargs: Additional keyword arguments that will be passed to the agent's init.
        """
        # Load agent.json
        folder = Path(folder)
        agent_dict = json.loads((folder / "agent.json").read_text())
        # Handle HfApiModel -> InferenceClientModel rename for old agents
        if agent_dict.get("model", {}).get("class") == "HfApiModel":
            agent_dict["model"]["class"] = "InferenceClientModel"
            logger.warning(
                "The agent you're loading uses the deprecated 'HfApiModel' class: it was automatically updated to 'InferenceClientModel'."
            )
        # Load managed agents from their respective folders, recursively
        managed_agents = []
        for managed_agent_name, managed_agent_class_name in agent_dict["managed_agents"].items():
            agent_cls = AGENT_REGISTRY.get(managed_agent_class_name)
            if agent_cls is None:
                raise ValueError(
                    f"Unknown agent class '{managed_agent_class_name}'. "
                    f"Supported agents: {', '.join(sorted(AGENT_REGISTRY.keys()))}"
                )
            managed_agents.append(agent_cls.from_folder(folder / "managed_agents" / managed_agent_name))
        agent_dict["managed_agents"] = {}

        # Load tools
        tools = []
        for tool_name in agent_dict["tools"]:
            tool_code = (folder / "tools" / f"{tool_name}.py").read_text()
            tools.append({"name": tool_name, "code": tool_code})
        agent_dict["tools"] = tools

        # Add managed agents to kwargs to override the empty list in from_dict
        if managed_agents:
            kwargs["managed_agents"] = managed_agents

        return cls.from_dict(agent_dict, **kwargs)

    def push_to_hub(
        self,
        repo_id: str,
        commit_message: str = "Upload agent",
        private: bool | None = None,
        token: bool | str | None = None,
        create_pr: bool = False,
    ) -> str:
        # [FUNCTION DOC]
        # Uploads the agent as a Gradio Space on the Hub.
        # Inputs: repo_id (str), commit_message (str, default "Upload agent"), private (bool or None), token
        #   (bool, str or None), create_pr (bool, default False).
        # Outputs: the result of upload_folder (commit info / URL).
        # Exceptions: hub/network errors propagate.
        # Side effects: creates the Space if needed, sets tags smolagents and agent, saves to a temp dir and
        #   uploads it.
        """
        Upload the agent to the Hub.

        Parameters:
            repo_id (`str`):
                The name of the repository you want to push to. It should contain your organization name when
                pushing to a given organization.
            commit_message (`str`, *optional*, defaults to `"Upload agent"`):
                Message to commit while pushing.
            private (`bool`, *optional*, defaults to `None`):
                Whether to make the repo private. If `None`, the repo will be public unless the organization's default is private. This value is ignored if the repo already exists.
            token (`bool` or `str`, *optional*):
                The token to use as HTTP bearer authorization for remote files. If unset, will use the token generated
                when running `huggingface-cli login` (stored in `~/.huggingface`).
            create_pr (`bool`, *optional*, defaults to `False`):
                Whether to create a PR with the uploaded files or directly commit.
        """
        repo_url = create_repo(
            repo_id=repo_id,
            token=token,
            private=private,
            exist_ok=True,
            repo_type="space",
            space_sdk="gradio",
        )
        repo_id = repo_url.repo_id
        metadata_update(
            repo_id,
            {"tags": ["smolagents", "agent"]},
            repo_type="space",
            token=token,
            overwrite=True,
        )

        with tempfile.TemporaryDirectory() as work_dir:
            self.save(work_dir)
            logger.info(f"Uploading the following files to {repo_id}: {','.join(os.listdir(work_dir))}")
            return upload_folder(
                repo_id=repo_id,
                commit_message=commit_message,
                folder_path=work_dir,
                token=token,
                create_pr=create_pr,
                repo_type="space",
            )


class ToolCallingAgent(MultiStepAgent):
    # [CLASS DOC]
    # Agent whose actions are native/JSON tool calls produced by the model; multiple calls in one step run in
    #   parallel threads.
    # Constructor args: tools, model, prompt_templates (defaults to prompts/toolcalling_agent.yaml),
    #   planning_interval, stream_outputs (bool, default False), max_tool_threads (int or None), kwargs passed
    #   to MultiStepAgent.
    # Key attributes: stream_outputs, max_tool_threads, tools_and_managed_agents.
    # Exceptions: ValueError if stream_outputs is set but the model lacks generate_stream.
    # Side effects: reads the packaged YAML prompt file.
    """
    This agent uses JSON-like tool calls, using method `model.get_tool_call` to leverage the LLM engine's tool calling capabilities.

    Args:
        tools (`list[Tool]`): [`Tool`]s that the agent can use.
        model (`Model`): Model that will generate the agent's actions.
        prompt_templates ([`~agents.PromptTemplates`], *optional*): Prompt templates.
        planning_interval (`int`, *optional*): Interval at which the agent will run a planning step.
        stream_outputs (`bool`, *optional*, default `False`): Whether to stream outputs during execution.
        max_tool_threads (`int`, *optional*): Maximum number of threads for parallel tool calls.
            Higher values increase concurrency but resource usage as well.
            Defaults to `ThreadPoolExecutor`'s default.
        **kwargs: Additional keyword arguments.
    """

    def __init__(
        self,
        tools: list[Tool],
        model: Model,
        prompt_templates: PromptTemplates | None = None,
        planning_interval: int | None = None,
        stream_outputs: bool = False,
        max_tool_threads: int | None = None,
        **kwargs,
    ):
        # [FUNCTION DOC]
        # Loads default prompts if none given, initializes the base agent, and configures streaming and
        #   threading.
        # Inputs: tools (list of Tool), model (Model), prompt_templates (default None), planning_interval
        #   (default None), stream_outputs (bool, default False), max_tool_threads (int or None), kwargs (base
        #   class args).
        # Outputs: none.
        # Exceptions: ValueError if streaming requested without generate_stream; base class exceptions.
        # Side effects: reads prompts YAML from package resources.
        prompt_templates = prompt_templates or yaml.safe_load(
            importlib.resources.files("smolagents.prompts").joinpath("toolcalling_agent.yaml").read_text()
        )
        super().__init__(
            tools=tools,
            model=model,
            prompt_templates=prompt_templates,
            planning_interval=planning_interval,
            **kwargs,
        )
        # Streaming setup
        self.stream_outputs = stream_outputs
        if self.stream_outputs and not hasattr(self.model, "generate_stream"):
            raise ValueError(
                "`stream_outputs` is set to True, but the model class implements no `generate_stream` method."
            )
        # Tool calling setup
        self.max_tool_threads = max_tool_threads

    @property
    def tools_and_managed_agents(self):
        # [FUNCTION DOC]
        # Property returning tools followed by managed agents as a list (passed to the model as callable
        #   tools).
        # Inputs: none. Outputs: list.
        # Exceptions: none raised directly.
        # Side effects: none.
        """Returns a combined list of tools and managed agents."""
        return list(self.tools.values()) + list(self.managed_agents.values())

    def initialize_system_prompt(self) -> str:
        # [FUNCTION DOC]
        # Renders the system prompt template with tools, managed_agents and custom_instructions.
        # Inputs: none. Outputs: rendered string.
        # Exceptions: template rendering Exception propagates.
        # Side effects: none.
        system_prompt = populate_template(
            self.prompt_templates["system_prompt"],
            variables={
                "tools": self.tools,
                "managed_agents": self.managed_agents,
                "custom_instructions": self.instructions,
            },
        )
        return system_prompt

    def _step_stream(
        self, memory_step: ActionStep
    ) -> Generator[ChatMessageStreamDelta | ToolCall | ToolOutput | ActionOutput]:
        # [FUNCTION DOC]
        # HELPER: One ReAct step: send memory to the model with tools (stop sequences "Observation:", "Calling
        #   tools:"), stream or generate the reply, parse tool calls (from text if none native, else JSON-decode
        #   arguments), run them via process_tool_calls, and yield the results.
        # Inputs: memory_step (ActionStep, filled with input messages, model output, token usage, tool calls,
        #   observations).
        # Outputs (yields): stream deltas (if streaming), ToolCall and ToolOutput events, then an ActionOutput
        #   whose output is the final answer (resolved from state if it names a state key) or None.
        # Exceptions: AgentGenerationError for model errors; AgentParsingError for tool call parsing errors;
        #   AgentExecutionError if final_answer is combined with other calls; AgentToolExecutionError for
        #   multiple final answers; tool errors from execute_tool_call.
        # Side effects: model call, live console rendering, logging, memory step mutation.
        # KNOWN ISSUE: docstring says it yields None or the final answer, but it yields an ActionOutput.
        """
        Perform one step in the ReAct framework: the agent thinks, acts, and observes the result.
        Yields ChatMessageStreamDelta during the run if streaming is enabled.
        At the end, yields either None if the step is not final, or the final answer.
        """
        memory_messages = self.write_memory_to_messages()

        input_messages = memory_messages.copy()

        # Add new step in logs
        memory_step.model_input_messages = input_messages

        try:
            if self.stream_outputs and hasattr(self.model, "generate_stream"):
                output_stream = self.model.generate_stream(
                    input_messages,
                    stop_sequences=["Observation:", "Calling tools:"],
                    tools_to_call_from=self.tools_and_managed_agents,
                )

                chat_message_stream_deltas: list[ChatMessageStreamDelta] = []
                with Live("", console=self.logger.console, vertical_overflow="visible") as live:
                    for event in output_stream:
                        chat_message_stream_deltas.append(event)
                        live.update(
                            Markdown(agglomerate_stream_deltas(chat_message_stream_deltas).render_as_markdown())
                        )
                        yield event
                chat_message = agglomerate_stream_deltas(chat_message_stream_deltas)
            else:
                chat_message: ChatMessage = self.model.generate(
                    input_messages,
                    stop_sequences=["Observation:", "Calling tools:"],
                    tools_to_call_from=self.tools_and_managed_agents,
                )
                self.logger.log_markdown(
                    content=str(chat_message.content or chat_message.raw or ""),
                    title="Output message of the LLM:",
                    level=LogLevel.DEBUG,
                )

            # Record model output
            memory_step.model_output_message = chat_message
            memory_step.model_output = chat_message.content
            memory_step.token_usage = chat_message.token_usage
        except Exception as e:
            raise AgentGenerationError(f"Error while generating output:\n{e}", self.logger) from e

        if chat_message.tool_calls is None or len(chat_message.tool_calls) == 0:
            try:
                chat_message = self.model.parse_tool_calls(chat_message)
            except Exception as e:
                raise AgentParsingError(f"Error while parsing tool call from model output: {e}", self.logger)
        else:
            for tool_call in chat_message.tool_calls:
                tool_call.function.arguments = parse_json_if_needed(tool_call.function.arguments)
        final_answer, got_final_answer = None, False
        for output in self.process_tool_calls(chat_message, memory_step):
            yield output
            if isinstance(output, ToolOutput):
                if output.is_final_answer:
                    if len(chat_message.tool_calls) > 1:
                        raise AgentExecutionError(
                            "If you want to return an answer, please do not perform any other tool calls than the final answer tool call!",
                            self.logger,
                        )
                    if got_final_answer:
                        raise AgentToolExecutionError(
                            "You returned multiple final answers. Please return only one single final answer!",
                            self.logger,
                        )
                    final_answer = output.output
                    got_final_answer = True

                    # Manage state variables
                    if isinstance(final_answer, str) and final_answer in self.state.keys():
                        final_answer = self.state[final_answer]
        yield ActionOutput(
            output=final_answer,
            is_final_answer=got_final_answer,
        )

    def process_tool_calls(
        self, chat_message: ChatMessage, memory_step: ActionStep
    ) -> Generator[ToolCall | ToolOutput]:
        # [FUNCTION DOC]
        # Converts model tool calls to ToolCall objects, executes them (directly for one call, in a
        #   ThreadPoolExecutor with copied context for several), and records results on the memory step.
        # Inputs: chat_message (ChatMessage with tool_calls), memory_step (ActionStep).
        # Outputs (yields): each ToolCall first, then each ToolOutput as it completes.
        # Exceptions: AssertionError if tool_calls is None; tool execution errors propagate.
        # Side effects: logging; image/audio results stored in self.state; sets memory_step.tool_calls and
        #   appends observations joined by newlines.
        # KNOWN ISSUE: tool calls and observations are ordered by sorting call ids as strings, not by model
        #   order.
        """Process tool calls from the model output and update agent memory.

        Args:
            chat_message (`ChatMessage`): Chat message containing tool calls from the model.
            memory_step (`ActionStep)`: Memory ActionStep to update with results.

        Yields:
            `ToolCall | ToolOutput`: The tool call or tool output.
        """
        parallel_calls: dict[str, ToolCall] = {}
        assert chat_message.tool_calls is not None
        for chat_tool_call in chat_message.tool_calls:
            tool_call = ToolCall(
                name=chat_tool_call.function.name, arguments=chat_tool_call.function.arguments, id=chat_tool_call.id
            )
            yield tool_call
            parallel_calls[tool_call.id] = tool_call

        # Helper function to process a single tool call
        def process_single_tool_call(tool_call: ToolCall) -> ToolOutput:
            # [FUNCTION DOC]
            # HELPER: Executes one ToolCall and builds its ToolOutput.
            # Inputs: tool_call (ToolCall; missing arguments treated as an empty dict).
            # Outputs: ToolOutput; AgentImage/AgentAudio results are stored in state as image.png/audio.mp3
            #   and the observation says so, otherwise the observation is the stripped string of the result.
            # Exceptions: propagates errors from execute_tool_call.
            # Side effects: logs the call and observations; mutates self.state for image/audio results (same
            #   key reused, so earlier results are overwritten).
            tool_name = tool_call.name
            tool_arguments = tool_call.arguments or {}
            self.logger.log(
                Panel(Text(f"Calling tool: '{tool_name}' with arguments: {tool_arguments}")),
                level=LogLevel.INFO,
            )
            tool_call_result = self.execute_tool_call(tool_name, tool_arguments)
            tool_call_result_type = type(tool_call_result)
            if tool_call_result_type in [AgentImage, AgentAudio]:
                if tool_call_result_type == AgentImage:
                    observation_name = "image.png"
                elif tool_call_result_type == AgentAudio:
                    observation_name = "audio.mp3"
                # TODO: tool_call_result naming could allow for different names of same type
                self.state[observation_name] = tool_call_result
                observation = f"Stored '{observation_name}' in memory."
            else:
                observation = str(tool_call_result).strip()
            self.logger.log(
                f"Observations: {observation.replace('[', '|')}",  # escape potential rich-tag-like components
                level=LogLevel.INFO,
            )
            is_final_answer = tool_name == "final_answer"

            return ToolOutput(
                id=tool_call.id,
                output=tool_call_result,
                is_final_answer=is_final_answer,
                observation=observation,
                tool_call=tool_call,
            )

        # Process tool calls in parallel
        outputs = {}
        if len(parallel_calls) == 1:
            # If there's only one call, process it directly
            tool_call = list(parallel_calls.values())[0]
            tool_output = process_single_tool_call(tool_call)
            outputs[tool_output.id] = tool_output
            yield tool_output
        else:
            # If multiple tool calls, process them in parallel
            with ThreadPoolExecutor(self.max_tool_threads) as executor:
                futures = []
                for tool_call in parallel_calls.values():
                    ctx = copy_context()
                    futures.append(executor.submit(ctx.run, process_single_tool_call, tool_call))
                for future in as_completed(futures):
                    tool_output = future.result()
                    outputs[tool_output.id] = tool_output
                    yield tool_output

        memory_step.tool_calls = [parallel_calls[k] for k in sorted(parallel_calls.keys())]
        memory_step.observations = memory_step.observations or ""
        for tool_output in [outputs[k] for k in sorted(outputs.keys())]:
            memory_step.observations += tool_output.observation + "\n"
        memory_step.observations = (
            memory_step.observations.rstrip("\n") if memory_step.observations else memory_step.observations
        )

    def _substitute_state_variables(self, arguments: dict[str, str] | str) -> dict[str, Any] | str:
        # [FUNCTION DOC]
        # HELPER: Replaces string argument values that match a state key with the stored state value.
        # Inputs: arguments (dict or str).
        # Outputs: new dict with substitutions, or the string unchanged.
        # Exceptions: none raised directly.
        # Side effects: none.
        """Replace string values in arguments with their corresponding state values if they exist."""
        if isinstance(arguments, dict):
            return {
                key: self.state.get(value, value) if isinstance(value, str) else value
                for key, value in arguments.items()
            }
        return arguments

    def execute_tool_call(self, tool_name: str, arguments: dict[str, str] | str) -> Any:
        # [FUNCTION DOC]
        # Looks up a tool or managed agent by name, substitutes state variables, validates arguments and calls
        #   it (tools with sanitize_inputs_outputs enabled; dict arguments are passed as keywords, otherwise
        #   positionally).
        # Inputs: tool_name (str), arguments (dict or str).
        # Outputs: the tool's return value.
        # Exceptions: AgentToolExecutionError for unknown tool, unexpected validation errors or execution
        #   failures; AgentToolCallError for ValueError/TypeError during argument validation.
        # Side effects: those of the tool or managed agent run.
        """
        Execute a tool or managed agent with the provided arguments.

        The arguments are replaced with the actual values from the state if they refer to state variables.

        Args:
            tool_name (`str`): Name of the tool or managed agent to execute.
            arguments (dict[str, str] | str): Arguments passed to the tool call.
        """
        # Check if the tool exists
        available_tools = {**self.tools, **self.managed_agents}
        if tool_name not in available_tools:
            raise AgentToolExecutionError(
                f"Unknown tool {tool_name}, should be one of: {', '.join(available_tools)}.", self.logger
            )

        # Get the tool and substitute state variables in arguments
        tool = available_tools[tool_name]
        arguments = self._substitute_state_variables(arguments)
        is_managed_agent = tool_name in self.managed_agents

        try:
            validate_tool_arguments(tool, arguments)
        except (ValueError, TypeError) as e:
            raise AgentToolCallError(str(e), self.logger) from e
        except Exception as e:
            error_msg = f"Error executing tool '{tool_name}' with arguments {str(arguments)}: {type(e).__name__}: {e}"
            raise AgentToolExecutionError(error_msg, self.logger) from e

        try:
            # Call tool with appropriate arguments
            if isinstance(arguments, dict):
                return tool(**arguments) if is_managed_agent else tool(**arguments, sanitize_inputs_outputs=True)
            else:
                return tool(arguments) if is_managed_agent else tool(arguments, sanitize_inputs_outputs=True)

        except Exception as e:
            # Handle execution errors
            if is_managed_agent:
                error_msg = (
                    f"Error executing request to team member '{tool_name}' with arguments {str(arguments)}: {e}\n"
                    "Please try again or request to another team member"
                )
            else:
                error_msg = (
                    f"Error executing tool '{tool_name}' with arguments {str(arguments)}: {type(e).__name__}: {e}\n"
                    "Please try again or use another tool"
                )
            raise AgentToolExecutionError(error_msg, self.logger) from e


class CodeAgent(MultiStepAgent):
    # [CLASS DOC]
    # Agent whose actions are Python code blobs written by the model, extracted and run by a Python executor
    #   (local sandboxed interpreter or a remote blaxel/e2b/modal/docker executor).
    # Constructor args: tools, model, prompt_templates (defaults to code_agent.yaml, or
    #   structured_code_agent.yaml when use_structured_outputs_internally), additional_authorized_imports (list
    #   of str, "*" allows all), planning_interval, executor (PythonExecutor or None), executor_type (default
    #   "local"), executor_kwargs, max_print_outputs_length, stream_outputs (bool),
    #   use_structured_outputs_internally (bool), code_block_tags (tuple of opening/closing tags, "markdown", or
    #   None for <code>/</code>), kwargs for MultiStepAgent.
    # Key attributes: authorized_imports (sorted base builtins plus additional), code_block_tags,
    #   python_executor, executor_type, executor_kwargs. Usable as a context manager that cleans up the
    #   executor.
    # Exceptions: ValueError for unsupported code_block_tags string or streaming without generate_stream.
    # Side effects: creates an executor (possibly remote resources); logs a caution when all imports allowed.
    # KNOWN ISSUE: the docstring says "markdown" maps to a regex opening tag for python or py with a newline-
    #   prefixed closing tag, but code uses the literal pair of a python-tagged fence and a plain fence.
    """
    In this agent, the tool calls will be formulated by the LLM in code format, then parsed and executed.

    Args:
        tools (`list[Tool]`): [`Tool`]s that the agent can use.
        model (`Model`): Model that will generate the agent's actions.
        prompt_templates ([`~agents.PromptTemplates`], *optional*): Prompt templates.
        additional_authorized_imports (`list[str]`, *optional*): Additional authorized imports for the agent.
        planning_interval (`int`, *optional*): Interval at which the agent will run a planning step.
        executor ([`PythonExecutor`], *optional*): Custom Python code executor. If not provided, a default executor will be created based on `executor_type`.
        executor_type (`Literal["local", "blaxel", "e2b", "modal", "docker"]`, default `"local"`): Type of code executor.
        executor_kwargs (`dict`, *optional*): Additional arguments to pass to initialize the executor.
        max_print_outputs_length (`int`, *optional*): Maximum length of the print outputs.
        stream_outputs (`bool`, *optional*, default `False`): Whether to stream outputs during execution.
        use_structured_outputs_internally (`bool`, default `False`): Whether to use structured generation at each action step: improves performance for many models.

            <Added version="1.17.0"/>
        code_block_tags (`tuple[str, str]` | `Literal["markdown"]`, *optional*): Opening and closing tags for code blocks (regex strings). Pass a custom tuple, or pass 'markdown' to use ("```(?:python|py)", "\\n```"), leave empty to use ("<code>", "</code>").
        **kwargs: Additional keyword arguments.
    """

    def __init__(
        self,
        tools: list[Tool],
        model: Model,
        prompt_templates: PromptTemplates | None = None,
        additional_authorized_imports: list[str] | None = None,
        planning_interval: int | None = None,
        executor: PythonExecutor = None,
        executor_type: Literal["local", "blaxel", "e2b", "modal", "docker"] = "local",
        executor_kwargs: dict[str, Any] | None = None,
        max_print_outputs_length: int | None = None,
        stream_outputs: bool = False,
        use_structured_outputs_internally: bool = False,
        code_block_tags: str | tuple[str, str] | None = None,
        **kwargs,
    ):
        # [FUNCTION DOC]
        # Configures imports, prompts, code block tags, base agent, streaming and the Python executor.
        # Inputs: see class doc (tools, model, prompt_templates=None, additional_authorized_imports=None,
        #   planning_interval=None, executor=None, executor_type="local", executor_kwargs=None,
        #   max_print_outputs_length=None, stream_outputs=False, use_structured_outputs_internally=False,
        #   code_block_tags=None, kwargs).
        # Outputs: none.
        # Exceptions: ValueError for bad code_block_tags or streaming unsupported; errors from
        #   create_python_executor and the base constructor.
        # Side effects: reads prompt YAML, logs, creates executor.
        self.additional_authorized_imports = additional_authorized_imports if additional_authorized_imports else []
        self.authorized_imports = sorted(set(BASE_BUILTIN_MODULES) | set(self.additional_authorized_imports))
        self.max_print_outputs_length = max_print_outputs_length
        self._use_structured_outputs_internally = use_structured_outputs_internally
        if self._use_structured_outputs_internally:
            prompt_templates = prompt_templates or yaml.safe_load(
                importlib.resources.files("smolagents.prompts").joinpath("structured_code_agent.yaml").read_text()
            )
        else:
            prompt_templates = prompt_templates or yaml.safe_load(
                importlib.resources.files("smolagents.prompts").joinpath("code_agent.yaml").read_text()
            )

        if isinstance(code_block_tags, str) and not code_block_tags == "markdown":
            raise ValueError("Only 'markdown' is supported for a string argument to `code_block_tags`.")
        self.code_block_tags = (
            code_block_tags
            if isinstance(code_block_tags, tuple)
            else ("```python", "```")
            if code_block_tags == "markdown"
            else ("<code>", "</code>")
        )

        super().__init__(
            tools=tools,
            model=model,
            prompt_templates=prompt_templates,
            planning_interval=planning_interval,
            **kwargs,
        )
        self.stream_outputs = stream_outputs
        if self.stream_outputs and not hasattr(self.model, "generate_stream"):
            raise ValueError(
                "`stream_outputs` is set to True, but the model class implements no `generate_stream` method."
            )
        if "*" in self.additional_authorized_imports:
            self.logger.log(
                "Caution: you set an authorization for all imports, meaning your agent can decide to import any package it deems necessary. This might raise issues if the package is not installed in your environment.",
                level=LogLevel.INFO,
            )
        self.executor_type = executor_type
        self.executor_kwargs: dict[str, Any] = executor_kwargs or {}
        self.python_executor = executor or self.create_python_executor()

    def __enter__(self):
        # [FUNCTION DOC]
        # Context manager entry.
        # Inputs: none. Outputs: self.
        # Exceptions: none raised directly. Side effects: none.
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        # [FUNCTION DOC]
        # Context manager exit; calls cleanup and does not suppress exceptions.
        # Inputs: exc_type, exc_value, traceback (ignored). Outputs: None.
        # Exceptions: propagates cleanup errors. Side effects: releases executor resources.
        self.cleanup()

    def cleanup(self):
        # [FUNCTION DOC]
        # Calls the executor's cleanup method if it has one.
        # Inputs: none. Outputs: none.
        # Exceptions: propagates executor errors.
        # Side effects: may shut down remote sandboxes/containers.
        """Clean up resources used by the agent, such as the remote Python executor."""
        if hasattr(self.python_executor, "cleanup"):
            self.python_executor.cleanup()

    def create_python_executor(self) -> PythonExecutor:
        # [FUNCTION DOC]
        # Creates the executor for executor_type: LocalPythonExecutor (with additional imports and
        #   max_print_outputs_length merged with executor_kwargs) or the matching remote executor class (with
        #   additional imports, logger and executor_kwargs).
        # Inputs: none (uses instance attributes).
        # Outputs: a PythonExecutor.
        # Exceptions: ValueError for unsupported type; generic Exception if managed agents are combined with a
        #   remote executor; executor construction errors.
        # Side effects: may start remote sandboxes or containers.
        if self.executor_type not in {"local", "blaxel", "e2b", "modal", "docker"}:
            raise ValueError(f"Unsupported executor type: {self.executor_type}")

        if self.executor_type == "local":
            return LocalPythonExecutor(
                self.additional_authorized_imports,
                **{"max_print_outputs_length": self.max_print_outputs_length} | self.executor_kwargs,
            )
        else:
            if self.managed_agents:
                raise Exception("Managed agents are not yet supported with remote code execution.")
            remote_executors = {
                "blaxel": BlaxelExecutor,
                "e2b": E2BExecutor,
                "docker": DockerExecutor,
                "modal": ModalExecutor,
            }
            return remote_executors[self.executor_type](
                self.additional_authorized_imports, self.logger, **self.executor_kwargs
            )

    def initialize_system_prompt(self) -> str:
        # [FUNCTION DOC]
        # Renders the system prompt with tools, managed_agents, authorized_imports (a sentence if "*" is
        #   allowed, else the list as a string), custom_instructions and the code block opening/closing tags.
        # Inputs: none. Outputs: rendered string.
        # Exceptions: template rendering Exception propagates.
        # Side effects: none.
        system_prompt = populate_template(
            self.prompt_templates["system_prompt"],
            variables={
                "tools": self.tools,
                "managed_agents": self.managed_agents,
                "authorized_imports": (
                    "You can import from any package you want."
                    if "*" in self.authorized_imports
                    else str(self.authorized_imports)
                ),
                "custom_instructions": self.instructions,
                "code_block_opening_tag": self.code_block_tags[0],
                "code_block_closing_tag": self.code_block_tags[1],
            },
        )
        return system_prompt

    def _step_stream(
        self, memory_step: ActionStep
    ) -> Generator[ChatMessageStreamDelta | ToolCall | ToolOutput | ActionOutput]:
        # [FUNCTION DOC]
        # HELPER: One ReAct step: query the model (stop sequences "Observation:", "Calling tools:" and the
        #   closing code tag when not contained in the opening tag; JSON response format if structured outputs),
        #   ensure the output ends with the closing tag, extract the code (JSON "code" field or code blob
        #   parsing), fix final_answer usage, execute it, and record logs/output as observations.
        # Inputs: memory_step (ActionStep; filled with input messages, model output, token usage, code_action,
        #   tool_calls, observations, action_output).
        # Outputs (yields): stream deltas (if streaming), a python_interpreter ToolCall, then an ActionOutput
        #   with the execution output and is_final_answer flag.
        # Exceptions: AgentGenerationError for model errors; AgentParsingError for code extraction failures;
        #   AgentExecutionError for execution failures (after logging any captured print output and an
        #   unauthorized import hint).
        # Side effects: model call, code execution (may have arbitrary effects), logging and live rendering.
        # KNOWN ISSUE: docstring says it yields None or the final answer, but it yields an ActionOutput.
        """
        Perform one step in the ReAct framework: the agent thinks, acts, and observes the result.
        Yields ChatMessageStreamDelta during the run if streaming is enabled.
        At the end, yields either None if the step is not final, or the final answer.
        """
        memory_messages = self.write_memory_to_messages()

        input_messages = memory_messages.copy()
        ### Generate model output ###
        memory_step.model_input_messages = input_messages
        stop_sequences = ["Observation:", "Calling tools:"]
        if self.code_block_tags[1] not in self.code_block_tags[0]:
            # If the closing tag is contained in the opening tag, adding it as a stop sequence would cut short any code generation
            stop_sequences.append(self.code_block_tags[1])
        try:
            additional_args: dict[str, Any] = {}
            if self._use_structured_outputs_internally:
                additional_args["response_format"] = CODEAGENT_RESPONSE_FORMAT
            if self.stream_outputs:
                output_stream = self.model.generate_stream(
                    input_messages,
                    stop_sequences=stop_sequences,
                    **additional_args,
                )
                chat_message_stream_deltas: list[ChatMessageStreamDelta] = []
                with Live("", console=self.logger.console, vertical_overflow="visible") as live:
                    for event in output_stream:
                        chat_message_stream_deltas.append(event)
                        live.update(
                            Markdown(agglomerate_stream_deltas(chat_message_stream_deltas).render_as_markdown())
                        )
                        yield event
                chat_message = agglomerate_stream_deltas(chat_message_stream_deltas)
                memory_step.model_output_message = chat_message
                output_text = chat_message.content
            else:
                chat_message: ChatMessage = self.model.generate(
                    input_messages,
                    stop_sequences=stop_sequences,
                    **additional_args,
                )
                memory_step.model_output_message = chat_message
                output_text = chat_message.content
                self.logger.log_markdown(
                    content=output_text or "",
                    title="Output message of the LLM:",
                    level=LogLevel.DEBUG,
                )

            if not self._use_structured_outputs_internally:
                # This adds the end code sequence (i.e. the closing code block tag) to the history.
                # This will nudge subsequent LLM calls to finish with this end code sequence, thus efficiently stopping generation.
                if output_text and not output_text.strip().endswith(self.code_block_tags[1]):
                    output_text += self.code_block_tags[1]
                    memory_step.model_output_message.content = output_text

            memory_step.token_usage = chat_message.token_usage
            memory_step.model_output = output_text
        except Exception as e:
            raise AgentGenerationError(f"Error in generating model output:\n{e}", self.logger) from e

        ### Parse output ###
        try:
            if self._use_structured_outputs_internally:
                code_action = json.loads(output_text)["code"]
                code_action = extract_code_from_text(code_action, self.code_block_tags) or code_action
            else:
                code_action = parse_code_blobs(output_text, self.code_block_tags)
            code_action = fix_final_answer_code(code_action)
            memory_step.code_action = code_action
        except Exception as e:
            error_msg = f"Error in code parsing:\n{e}\nMake sure to provide correct code blobs."
            raise AgentParsingError(error_msg, self.logger)

        tool_call = ToolCall(
            name="python_interpreter",
            arguments=code_action,
            id=f"call_{len(self.memory.steps)}",
        )
        yield tool_call
        memory_step.tool_calls = [tool_call]

        ### Execute action ###
        self.logger.log_code(title="Executing parsed code:", content=code_action, level=LogLevel.INFO)
        try:
            code_output = self.python_executor(code_action)
            execution_outputs_console = []
            if len(code_output.logs) > 0:
                execution_outputs_console += [
                    Text("Execution logs:", style="bold"),
                    Text(code_output.logs),
                ]
            observation = "Execution logs:\n" + code_output.logs
        except Exception as e:
            if hasattr(self.python_executor, "state") and "_print_outputs" in self.python_executor.state:
                execution_logs = str(self.python_executor.state["_print_outputs"])
                if len(execution_logs) > 0:
                    execution_outputs_console = [
                        Text("Execution logs:", style="bold"),
                        Text(execution_logs),
                    ]
                    memory_step.observations = "Execution logs:\n" + execution_logs
                    self.logger.log(Group(*execution_outputs_console), level=LogLevel.INFO)
            error_msg = str(e)
            if "Import of " in error_msg and " is not allowed" in error_msg:
                self.logger.log(
                    "[bold red]Warning to user: Code execution failed due to an unauthorized import - Consider passing said import under `additional_authorized_imports` when initializing your CodeAgent.",
                    level=LogLevel.INFO,
                )
            raise AgentExecutionError(error_msg, self.logger)

        truncated_output = truncate_content(str(code_output.output))
        observation += "Last output from code snippet:\n" + truncated_output
        memory_step.observations = observation

        if not code_output.is_final_answer:
            execution_outputs_console += [
                Text(
                    f"Out: {truncated_output}",
                ),
            ]
        self.logger.log(Group(*execution_outputs_console), level=LogLevel.INFO)
        memory_step.action_output = code_output.output
        yield ActionOutput(output=code_output.output, is_final_answer=code_output.is_final_answer)

    def to_dict(self) -> dict[str, Any]:
        # [FUNCTION DOC]
        # Extends the base dict with authorized_imports, executor_type, executor_kwargs and
        #   max_print_outputs_length.
        # Inputs: none. Outputs: dict.
        # Exceptions: as base to_dict. Side effects: as base to_dict.
        # KNOWN ISSUE: code_block_tags and use_structured_outputs_internally are not serialized, although
        #   from_dict reads code_block_tags.
        """Convert the agent to a dictionary representation.

        Returns:
            `dict`: Dictionary representation of the agent.
        """
        agent_dict = super().to_dict()
        agent_dict["authorized_imports"] = self.authorized_imports
        agent_dict["executor_type"] = self.executor_type
        agent_dict["executor_kwargs"] = self.executor_kwargs
        agent_dict["max_print_outputs_length"] = self.max_print_outputs_length
        return agent_dict

    @classmethod
    def from_dict(cls, agent_dict: dict[str, Any], **kwargs) -> "CodeAgent":
        # [FUNCTION DOC]
        # Classmethod rebuilding a CodeAgent: maps authorized_imports to additional_authorized_imports and
        #   forwards executor_type, executor_kwargs, max_print_outputs_length and code_block_tags (non-None
        #   ones), then delegates to MultiStepAgent.from_dict with user kwargs taking precedence.
        # Inputs: agent_dict (dict), kwargs (overrides).
        # Outputs: a CodeAgent instance.
        # Exceptions: as MultiStepAgent.from_dict.
        # Side effects: as MultiStepAgent.from_dict.
        # KNOWN ISSUE: the CodeAgent-specific kwargs are also forwarded to managed agents' from_dict, which
        #   may be ToolCallingAgents that do not accept them.
        """Create CodeAgent from a dictionary representation.

        Args:
            agent_dict (`dict[str, Any]`): Dictionary representation of the agent.
            **kwargs: Additional keyword arguments that will override agent_dict values.

        Returns:
            `CodeAgent`: Instance of the CodeAgent class.
        """
        # Add CodeAgent-specific parameters to kwargs
        code_agent_kwargs = {
            "additional_authorized_imports": agent_dict.get("authorized_imports"),
            "executor_type": agent_dict.get("executor_type"),
            "executor_kwargs": agent_dict.get("executor_kwargs"),
            "max_print_outputs_length": agent_dict.get("max_print_outputs_length"),
            "code_block_tags": agent_dict.get("code_block_tags"),
        }
        # Filter out None values
        code_agent_kwargs = {k: v for k, v in code_agent_kwargs.items() if v is not None}
        # Update with any additional kwargs
        code_agent_kwargs.update(kwargs)
        # Call the parent class's from_dict method
        return super().from_dict(agent_dict, **code_agent_kwargs)


# Agent Registry for secure deserialization
# This registry maps agent class names to their actual classes.
# Only classes listed here can be instantiated during deserialization (from_dict/from_folder).
# This prevents arbitrary code execution via importlib-based dynamic loading.
AGENT_REGISTRY = {
    "ToolCallingAgent": ToolCallingAgent,
    "CodeAgent": CodeAgent,
}
