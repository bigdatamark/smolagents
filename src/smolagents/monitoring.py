#!/usr/bin/env python
# coding=utf-8
# [MODULE DOC]
# Purpose: Monitoring and console logging utilities for agents: token usage and timing dataclasses, a Monitor
#   accumulating per-step metrics, the LogLevel enum and the rich-based AgentLogger.
# Exports: AgentLogger, LogLevel, Monitor, TokenUsage, Timing; also the YELLOW_HEX colour constant.
# Dependencies: rich (console rendering), smolagents.utils.sanitize_for_rich, json.
# Side effects: none at import.

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
import json
from dataclasses import dataclass, field
from enum import IntEnum

from rich import box
from rich.console import Console, Group
from rich.panel import Panel
from rich.rule import Rule
from rich.syntax import Syntax
from rich.table import Table
from rich.text import Text
from rich.tree import Tree

from smolagents.utils import sanitize_for_rich


__all__ = ["AgentLogger", "LogLevel", "Monitor", "TokenUsage", "Timing"]


@dataclass
class TokenUsage:
    # [CLASS DOC]
    # Dataclass with input and output token counts; total_tokens is computed automatically as their sum after
    #   init.
    # Inputs: input_tokens (int), output_tokens (int).
    # Attributes: total_tokens (int, not an init argument).
    # Exceptions: TypeError if counts cannot be added.
    # Side effects: none.
    """
    Contains the token usage information for a given step or run.
    """

    input_tokens: int
    output_tokens: int
    total_tokens: int = field(init=False)

    def __post_init__(self):
        # [FUNCTION DOC]
        # HELPER: Computes total_tokens as input_tokens plus output_tokens.
        # Inputs: none.
        # Outputs: none.
        # Exceptions: TypeError for non-numeric counts.
        # Side effects: sets total_tokens.
        self.total_tokens = self.input_tokens + self.output_tokens

    def dict(self):
        # [FUNCTION DOC]
        # Returns a dict with input_tokens, output_tokens and total_tokens.
        # Inputs: none.
        # Outputs: dict.
        # Exceptions: none raised directly.
        # Side effects: none.
        return {
            "input_tokens": self.input_tokens,
            "output_tokens": self.output_tokens,
            "total_tokens": self.total_tokens,
        }


@dataclass
class Timing:
    # [CLASS DOC]
    # Dataclass with a start time and optional end time (floats, seconds); exposes a derived duration.
    # Inputs: start_time (float), end_time (float or None, default None).
    # Exceptions: none raised directly.
    # Side effects: none.
    """
    Contains the timing information for a given step or run.
    """

    start_time: float
    end_time: float | None = None

    @property
    def duration(self):
        # [FUNCTION DOC]
        # Property returning end_time minus start_time, or None if the end time is not set.
        # Inputs: none.
        # Outputs: float or None.
        # Exceptions: none raised directly.
        # Side effects: none.
        return None if self.end_time is None else self.end_time - self.start_time

    def dict(self):
        # [FUNCTION DOC]
        # Returns a dict with start_time, end_time and duration.
        # Inputs: none.
        # Outputs: dict.
        # Exceptions: none raised directly.
        # Side effects: none.
        return {
            "start_time": self.start_time,
            "end_time": self.end_time,
            "duration": self.duration,
        }

    def __repr__(self) -> str:
        # [FUNCTION DOC]
        # Returns a readable representation including start_time, end_time and duration.
        # Inputs: none.
        # Outputs: str.
        # Exceptions: none raised directly.
        # Side effects: none.
        return f"Timing(start_time={self.start_time}, end_time={self.end_time}, duration={self.duration})"


class Monitor:
    # [CLASS DOC]
    # Accumulates metrics across an agent run: list of step durations and cumulative input/output token
    #   counts, logging a one-line summary after each step.
    # Inputs: tracked_model (the model being monitored; stored but unused), logger (AgentLogger used for
    #   output).
    # Attributes: step_durations, tracked_model, logger, total_input_token_count, total_output_token_count.
    # Exceptions: none raised directly.
    # Side effects: logs through the given logger.
    def __init__(self, tracked_model, logger):
        # [FUNCTION DOC]
        # Stores the model and logger and initialises empty durations and zero token counters.
        # Inputs: tracked_model (any), logger (AgentLogger).
        # Outputs: none.
        # Exceptions: none raised directly.
        # Side effects: sets instance attributes.
        self.step_durations = []
        self.tracked_model = tracked_model
        self.logger = logger
        self.total_input_token_count = 0
        self.total_output_token_count = 0

    def get_total_token_counts(self) -> TokenUsage:
        # [FUNCTION DOC]
        # Returns the cumulative token counts as a TokenUsage.
        # Inputs: none.
        # Outputs: TokenUsage.
        # Exceptions: none raised directly.
        # Side effects: none.
        return TokenUsage(
            input_tokens=self.total_input_token_count,
            output_tokens=self.total_output_token_count,
        )

    def reset(self):
        # [FUNCTION DOC]
        # Clears step durations and resets both token counters to zero.
        # Inputs: none.
        # Outputs: none.
        # Exceptions: none raised directly.
        # Side effects: mutates monitor state.
        self.step_durations = []
        self.total_input_token_count = 0
        self.total_output_token_count = 0

    def update_metrics(self, step_log):
        # [FUNCTION DOC]
        # Records the step's duration, adds its token usage (if any) to the running totals and logs a dim line
        #   with step index, duration and cumulative token counts at INFO level.
        # Inputs: step_log (a memory step exposing timing and token_usage).
        # Outputs: none.
        # Exceptions: TypeError if the step's duration is None (formatting with two decimals fails);
        #   AttributeError if the step lacks timing/token_usage.
        # Side effects: mutates counters and durations; console output via logger.
        # KNOWN ISSUE: crashes when timing has no end time; the token section lacks a leading space before the
        #   first separator.
        """Update the metrics of the monitor.

        Args:
            step_log ([`MemoryStep`]): Step log to update the monitor with.
        """
        step_duration = step_log.timing.duration
        self.step_durations.append(step_duration)
        console_outputs = f"[Step {len(self.step_durations)}: Duration {step_duration:.2f} seconds"

        if step_log.token_usage is not None:
            self.total_input_token_count += step_log.token_usage.input_tokens
            self.total_output_token_count += step_log.token_usage.output_tokens
            console_outputs += (
                f"| Input tokens: {self.total_input_token_count:,} | Output tokens: {self.total_output_token_count:,}"
            )
        console_outputs += "]"
        self.logger.log(Text(console_outputs, style="dim"), level=1)


class LogLevel(IntEnum):
    # [CLASS DOC]
    # Integer enum of verbosity levels: OFF (-1), ERROR (0), INFO (1, default), DEBUG (2). A message is
    #   printed when its level is less than or equal to the logger's level.
    # Inputs: none.
    # Exceptions: none raised directly.
    # Side effects: none.
    OFF = -1  # No output
    ERROR = 0  # Only errors
    INFO = 1  # Normal output (default)
    DEBUG = 2  # Detailed output


YELLOW_HEX = "#d4b702"


class AgentLogger:
    # [CLASS DOC]
    # Rich-console logger filtering messages by LogLevel and offering helpers for markdown, code, rules, task
    #   panels, message dumps and an agent tree visualisation.
    # Inputs: level (LogLevel, default INFO), console (rich Console or None; a new non-highlighting Console is
    #   created if None).
    # Attributes: level, console.
    # Exceptions: none raised directly.
    # Side effects: writes to the console.
    def __init__(self, level: LogLevel = LogLevel.INFO, console: Console | None = None):
        # [FUNCTION DOC]
        # Stores the level and the given console or creates a new one with highlighting disabled.
        # Inputs: level (LogLevel, default INFO), console (Console or None).
        # Outputs: none.
        # Exceptions: none raised directly.
        # Side effects: may create a rich Console.
        self.level = level
        if console is None:
            self.console = Console(highlight=False)
        else:
            self.console = console

    def log(self, *args, level: int | str | LogLevel = LogLevel.INFO, **kwargs) -> None:
        # [FUNCTION DOC]
        # Prints the given renderables to the console if the message level is at or below the logger level;
        #   string levels are converted by uppercase name lookup.
        # Inputs: *args (renderables), level (int, str or LogLevel, default INFO), **kwargs (passed to console
        #   print).
        # Outputs: none.
        # Exceptions: KeyError for an unknown level name.
        # Side effects: console output.
        """Logs a message to the console.

        Args:
            level (LogLevel, optional): Defaults to LogLevel.INFO.
        """
        if isinstance(level, str):
            level = LogLevel[level.upper()]
        if level <= self.level:
            self.console.print(*args, **kwargs)

    def log_error(self, error_message: str) -> None:
        # [FUNCTION DOC]
        # Logs a sanitised error message in bold red at ERROR level.
        # Inputs: error_message (str).
        # Outputs: none.
        # Exceptions: none raised directly.
        # Side effects: console output.
        self.log(Text(sanitize_for_rich(error_message), style="bold red"), level=LogLevel.ERROR)

    def log_markdown(self, content: str, title: str | None = None, level=LogLevel.INFO, style=YELLOW_HEX) -> None:
        # [FUNCTION DOC]
        # Logs content syntax-highlighted as markdown; when a title is given, it is preceded by a left-aligned
        #   styled rule with the title.
        # Inputs: content (str), title (str or None), level (default INFO), style (rule colour, default
        #   YELLOW_HEX).
        # Outputs: none.
        # Exceptions: none raised directly.
        # Side effects: console output.
        # KNOWN ISSUE: the title is inserted into rich markup unsanitised, so stray markup in titles could
        #   break rendering.
        markdown_content = Syntax(
            content,
            lexer="markdown",
            theme="github-dark",
            word_wrap=True,
        )
        if title:
            self.log(
                Group(
                    Rule(
                        "[bold italic]" + title,
                        align="left",
                        style=style,
                    ),
                    markdown_content,
                ),
                level=level,
            )
        else:
            self.log(markdown_content, level=level)

    def log_code(self, title: str, content: str, level: int = LogLevel.INFO) -> None:
        # [FUNCTION DOC]
        # Logs content as a Python-highlighted code panel with the given title and horizontal-line box.
        # Inputs: title (str), content (str), level (default INFO).
        # Outputs: none.
        # Exceptions: none raised directly.
        # Side effects: console output.
        self.log(
            Panel(
                Syntax(
                    content,
                    lexer="python",
                    theme="monokai",
                    word_wrap=True,
                ),
                title="[bold]" + title,
                title_align="left",
                box=box.HORIZONTALS,
            ),
            level=level,
        )

    def log_rule(self, title: str, level: int = LogLevel.INFO) -> None:
        # [FUNCTION DOC]
        # Logs a horizontal rule with a bold white title in the yellow style.
        # Inputs: title (str), level (default INFO).
        # Outputs: none.
        # Exceptions: none raised directly.
        # Side effects: console output.
        # KNOWN ISSUE: the level argument is ignored; the rule is always logged at INFO level.
        self.log(
            Rule(
                "[bold white]" + title,
                characters="━",
                style=YELLOW_HEX,
            ),
            level=LogLevel.INFO,
        )

    def log_task(self, content: str, subtitle: str, title: str | None = None, level: LogLevel = LogLevel.INFO) -> None:
        # [FUNCTION DOC]
        # Logs a "New run" panel (optionally suffixed by the title) containing the sanitised task content in
        #   bold and the sanitised subtitle, rendered as Text so stray markup cannot break rich.
        # Inputs: content (any; sanitised), subtitle (any; sanitised), title (str or None), level (default
        #   INFO).
        # Outputs: none.
        # Exceptions: none raised directly.
        # Side effects: console output.
        # Important: `content` can contain arbitrary tool logs / payloads. If we embed it
        # inside Rich markup (e.g. f"[bold]{content}"), any stray "[/...]" sequences or
        # binary-ish characters can crash Rich's markup parser. Render the content as
        # `Text` instead, and apply styling via Text/style, not markup.
        safe_content = sanitize_for_rich(content)
        safe_subtitle = sanitize_for_rich(subtitle)
        content_text = Text("\n") + Text(safe_content, style="bold") + Text("\n")
        subtitle_text = Text(safe_subtitle)
        self.log(
            Panel(
                content_text,
                title="[bold]New run" + (f" - {title}" if title else ""),
                subtitle=subtitle_text,
                border_style=YELLOW_HEX,
                subtitle_align="left",
            ),
            level=level,
        )

    def log_messages(self, messages: list[dict], level: LogLevel = LogLevel.DEBUG) -> None:
        # [FUNCTION DOC]
        # Logs messages as indented JSON dumps (one per message, via each message's dict method) highlighted
        #   as markdown.
        # Inputs: messages (list of objects with a dict method, e.g. ChatMessage), level (default DEBUG).
        # Outputs: none.
        # Exceptions: AttributeError if a message has no dict method; TypeError if not JSON-serialisable.
        # Side effects: console output.
        # KNOWN ISSUE: the type hint says list of dicts, but plain dicts would fail since a dict method is
        #   called on each.
        messages_as_string = "\n".join([json.dumps(message.dict(), indent=4) for message in messages])
        self.log(
            Syntax(
                messages_as_string,
                lexer="markdown",
                theme="github-dark",
                word_wrap=True,
            ),
            level=level,
        )

    def visualize_agent_tree(self, agent):
        # [FUNCTION DOC]
        # Prints a rich tree describing the agent: headline with class name and model id, authorized imports
        #   for CodeAgents, a tools table, and recursively each managed agent with its description.
        # Inputs: agent (MultiStepAgent with model, tools, managed_agents attributes).
        # Outputs: none.
        # Exceptions: AttributeError if expected agent attributes are missing.
        # Side effects: console output (printed directly, ignoring the log level).
        def create_tools_section(tools_dict):
            # [FUNCTION DOC]
            # HELPER: Builds a table with name, description and formatted arguments (type, optional flag,
            #   description) for each tool, grouped under a "Tools:" header.
            # Inputs: tools_dict (dict of name to tool).
            # Outputs: rich Group.
            # Exceptions: none raised directly.
            # Side effects: none.
            table = Table(show_header=True, header_style="bold")
            table.add_column("Name", style="#1E90FF")
            table.add_column("Description")
            table.add_column("Arguments")

            for name, tool in tools_dict.items():
                args = [
                    f"{arg_name} (`{info.get('type', 'Any')}`{', optional' if info.get('optional') else ''}): {info.get('description', '')}"
                    for arg_name, info in getattr(tool, "inputs", {}).items()
                ]
                table.add_row(name, getattr(tool, "description", str(tool)), "\n".join(args))

            return Group("🛠️ [italic #1E90FF]Tools:[/italic #1E90FF]", table)

        def get_agent_headline(agent, name: str | None = None):
            # [FUNCTION DOC]
            # HELPER: Formats the bold yellow headline "name | ClassName | model_id" (name part omitted when
            #   not given).
            # Inputs: agent (with model.model_id), name (str or None).
            # Outputs: str (rich markup).
            # Exceptions: AttributeError if agent has no model.
            # Side effects: none.
            name_headline = f"{name} | " if name else ""
            return f"[bold {YELLOW_HEX}]{name_headline}{agent.__class__.__name__} | {agent.model.model_id}"

        def build_agent_tree(parent_tree, agent_obj):
            # [FUNCTION DOC]
            # HELPER: Adds the tools section to the tree node, then a branch for managed agents with headline,
            #   imports (for CodeAgents), description and their own subtree, recursively.
            # Inputs: parent_tree (rich Tree node), agent_obj (agent).
            # Outputs: none.
            # Exceptions: AttributeError for missing agent attributes.
            # Side effects: mutates the given tree.
            """Recursively builds the agent tree."""
            parent_tree.add(create_tools_section(agent_obj.tools))

            if agent_obj.managed_agents:
                agents_branch = parent_tree.add("🤖 [italic #1E90FF]Managed agents:")
                for name, managed_agent in agent_obj.managed_agents.items():
                    agent_tree = agents_branch.add(get_agent_headline(managed_agent, name))
                    if managed_agent.__class__.__name__ == "CodeAgent":
                        agent_tree.add(
                            f"✅ [italic #1E90FF]Authorized imports:[/italic #1E90FF] {managed_agent.additional_authorized_imports}"
                        )
                    agent_tree.add(f"📝 [italic #1E90FF]Description:[/italic #1E90FF] {managed_agent.description}")
                    build_agent_tree(agent_tree, managed_agent)

        main_tree = Tree(get_agent_headline(agent))
        if agent.__class__.__name__ == "CodeAgent":
            main_tree.add(
                f"✅ [italic #1E90FF]Authorized imports:[/italic #1E90FF] {agent.additional_authorized_imports}"
            )
        build_agent_tree(main_tree, agent)
        self.console.print(main_tree)
