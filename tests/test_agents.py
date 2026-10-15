# coding=utf-8
# Copyright 2024 HuggingFace Inc.
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
# Purpose: pytest suite for smolagents.agents (MultiStepAgent, CodeAgent, ToolCallingAgent, RunResult),
#   covering runs with scripted fake models, memory/step bookkeeping, planning, final-answer checks,
#   callbacks, tool/managed-agent validation, serialization (to_dict/from_dict/from_folder/save), streaming
#   tool calls and multi-agent orchestration.
# Inputs: depends on smolagents (agents, models, memory, monitoring, tools, default_tools, agent_types,
#   utils), huggingface_hub chat-completion types, rich Console, pytest and unittest.mock; uses fixtures
#   shared_datadir, test_tool and get_agent_dict from conftest plus local fixtures agent_logger and
#   prompt_templates.
# Outputs: defines fake streaming dataclasses (ChoiceDelta*), scripted fake Model subclasses, mock tool/agent
#   helpers, test classes TestAgent, TestRunResult, TestMultiStepAgent, TestToolCallingAgent, TestCodeAgent,
#   TestMultiAgents and module-level tests.
# Exceptions: none raised at import time beyond import failures of dependencies.
# Side effects: tests create temp dirs/files, patch registries and clients, capture console output; one test
#   (test_transformers_toolcalling_agent) downloads and runs a real HF model and test_multiagents_save builds
#   InferenceClientModel objects (no network calls are made by save).
import io
import json
import os
import re
import tempfile
import uuid
from collections.abc import Generator
from contextlib import nullcontext as does_not_raise
from dataclasses import dataclass
from pathlib import Path
from textwrap import dedent
from typing import Optional
from unittest.mock import MagicMock, patch

import pytest
from huggingface_hub import (
    ChatCompletionOutputFunctionDefinition,
    ChatCompletionOutputMessage,
    ChatCompletionOutputToolCall,
)
from rich.console import Console

from smolagents import EMPTY_PROMPT_TEMPLATES
from smolagents.agent_types import AgentImage, AgentText
from smolagents.agents import (
    AgentError,
    AgentMaxStepsError,
    AgentToolCallError,
    CodeAgent,
    MultiStepAgent,
    RunResult,
    ToolCall,
    ToolCallingAgent,
    ToolOutput,
    populate_template,
)
from smolagents.default_tools import DuckDuckGoSearchTool, FinalAnswerTool, PythonInterpreterTool, VisitWebpageTool
from smolagents.memory import (
    ActionStep,
    CallbackRegistry,
    FinalAnswerStep,
    MemoryStep,
    PlanningStep,
    SystemPromptStep,
    TaskStep,
)
from smolagents.models import (
    ChatMessage,
    ChatMessageToolCall,
    ChatMessageToolCallFunction,
    InferenceClientModel,
    MessageRole,
    Model,
    TransformersModel,
)
from smolagents.monitoring import AgentLogger, LogLevel, Timing, TokenUsage
from smolagents.tools import Tool, tool
from smolagents.utils import (
    BASE_BUILTIN_MODULES,
    AgentExecutionError,
    AgentGenerationError,
    AgentToolExecutionError,
)


@dataclass
class ChoiceDeltaToolCallFunction:
    # [CLASS DOC]
    # HELPER: Minimal stand-in for an OpenAI streaming delta tool-call function fragment.
    # Inputs: arguments (str or None, default None; partial JSON argument text), name (str or None, default
    #   None; tool name).
    # Outputs: dataclass instance with those two attributes.
    # Exceptions: none raised directly.
    # Side effects: none.
    arguments: Optional[str] = None
    name: Optional[str] = None


@dataclass
class ChoiceDeltaToolCall:
    # [CLASS DOC]
    # HELPER: Stand-in for an OpenAI streaming delta tool call.
    # Inputs: index (int or None; position of the tool call being streamed), id (str or None), function
    #   (ChoiceDeltaToolCallFunction or None), type (str or None, e.g. "function"); all default None.
    # Outputs: dataclass instance with those attributes.
    # Exceptions: none raised directly.
    # Side effects: none.
    index: Optional[int] = None
    id: Optional[str] = None
    function: Optional[ChoiceDeltaToolCallFunction] = None
    type: Optional[str] = None


@dataclass
class ChoiceDelta:
    # [CLASS DOC]
    # HELPER: Stand-in for an OpenAI streaming chunk delta.
    # Inputs: content, function_call, refusal, role (all str or None) and tool_calls (list of
    #   ChoiceDeltaToolCall or None); all default None.
    # Outputs: dataclass instance with those attributes.
    # Exceptions: none raised directly.
    # Side effects: none.
    content: Optional[str] = None
    function_call: Optional[str] = None
    refusal: Optional[str] = None
    role: Optional[str] = None
    tool_calls: Optional[list] = None


def get_new_path(suffix="") -> str:
    # [FUNCTION DOC]
    # HELPER: Builds a unique path inside a freshly created temporary directory.
    # Inputs: suffix (str, default empty; appended to the file name).
    # Outputs: string path made of a new temp dir plus a random UUID4 name and the suffix; the file itself is
    #   not created.
    # Exceptions: OSError propagates if the temp directory cannot be created.
    # Side effects: creates a new temporary directory on disk that is never cleaned up.
    # KNOWN ISSUE: unused anywhere in this module, and the temp directory it creates leaks.
    directory = tempfile.mkdtemp()
    return os.path.join(directory, str(uuid.uuid4()) + suffix)


@pytest.fixture
def agent_logger():
    # [FUNCTION DOC]
    # HELPER: pytest fixture providing an AgentLogger at DEBUG level whose rich Console records output.
    # Inputs: none.
    # Outputs: AgentLogger writing to an in-memory StringIO console (record enabled, no color, not a
    #   terminal), so tests can call export_text on its console.
    # Exceptions: none raised directly.
    # Side effects: none outside the in-memory buffer.
    return AgentLogger(
        LogLevel.DEBUG, console=Console(record=True, no_color=True, force_terminal=False, file=io.StringIO())
    )


class FakeToolCallModel(Model):
    # [CLASS DOC]
    # HELPER: Scripted Model emulating a tool-calling LLM that first calls python_interpreter then
    #   final_answer.
    # Inputs: no constructor args (uses Model defaults).
    # Outputs: see generate.
    # Exceptions: none raised directly.
    # Side effects: none.
    def generate(self, messages, tools_to_call_from=None, stop_sequences=None):
        # [FUNCTION DOC]
        # HELPER: Returns a canned assistant ChatMessage depending on conversation length.
        # Inputs: messages (list of messages), tools_to_call_from (ignored, default None), stop_sequences
        #   (ignored, default None).
        # Outputs: when fewer than 3 messages, a message with content "I will call the python interpreter."
        #   and one tool call (id call_0) to python_interpreter with code argument "2*3.6452"; otherwise a
        #   message "I will return the final answer." with a final_answer tool call (id call_1) and answer
        #   "7.2904".
        # Exceptions: none raised directly.
        # Side effects: none.
        if len(messages) < 3:
            return ChatMessage(
                role=MessageRole.ASSISTANT,
                content="I will call the python interpreter.",
                tool_calls=[
                    ChatMessageToolCall(
                        id="call_0",
                        type="function",
                        function=ChatMessageToolCallFunction(
                            name="python_interpreter", arguments={"code": "2*3.6452"}
                        ),
                    )
                ],
            )
        else:
            return ChatMessage(
                role=MessageRole.ASSISTANT,
                content="I will return the final answer.",
                tool_calls=[
                    ChatMessageToolCall(
                        id="call_1",
                        type="function",
                        function=ChatMessageToolCallFunction(name="final_answer", arguments={"answer": "7.2904"}),
                    )
                ],
            )


class FakeToolCallModelImage(Model):
    # [CLASS DOC]
    # HELPER: Scripted tool-calling Model that first calls an image generation tool, then returns the stored
    #   image by state key.
    # Inputs: no constructor args.
    # Outputs: see generate.
    # Exceptions: none raised directly.
    # Side effects: none.
    def generate(self, messages, tools_to_call_from=None, stop_sequences=None):
        # [FUNCTION DOC]
        # HELPER: Returns canned tool calls for an image-generation flow.
        # Inputs: messages (list), tools_to_call_from and stop_sequences (ignored, default None).
        # Outputs: with fewer than 3 messages, an empty-content message calling fake_image_generation_tool
        #   with prompt "An image of a cat"; otherwise a final_answer call whose arguments are the plain
        #   string "image.png" (a state variable name).
        # Exceptions: none raised directly.
        # Side effects: none.
        if len(messages) < 3:
            return ChatMessage(
                role=MessageRole.ASSISTANT,
                content="",
                tool_calls=[
                    ChatMessageToolCall(
                        id="call_0",
                        type="function",
                        function=ChatMessageToolCallFunction(
                            name="fake_image_generation_tool",
                            arguments={"prompt": "An image of a cat"},
                        ),
                    )
                ],
            )
        else:
            return ChatMessage(
                role=MessageRole.ASSISTANT,
                content="",
                tool_calls=[
                    ChatMessageToolCall(
                        id="call_1",
                        type="function",
                        function=ChatMessageToolCallFunction(name="final_answer", arguments="image.png"),
                    )
                ],
            )


class FakeToolCallModelVL(Model):
    # [CLASS DOC]
    # HELPER: Scripted tool-calling Model emulating a vision-language flow.
    # Inputs: no constructor args.
    # Outputs: see generate.
    # Exceptions: none raised directly.
    # Side effects: none.
    def generate(self, messages, tools_to_call_from=None, stop_sequences=None):
        # [FUNCTION DOC]
        # HELPER: Returns canned tool calls for image understanding.
        # Inputs: messages (list), tools_to_call_from and stop_sequences (ignored).
        # Outputs: with fewer than 3 messages, a call to fake_image_understanding_tool with prompt "What is in
        #   this image?" and image "image.png"; otherwise a final_answer call with string argument "The image
        #   is a cat.".
        # Exceptions: none raised directly.
        # Side effects: none.
        if len(messages) < 3:
            return ChatMessage(
                role=MessageRole.ASSISTANT,
                content="",
                tool_calls=[
                    ChatMessageToolCall(
                        id="call_0",
                        type="function",
                        function=ChatMessageToolCallFunction(
                            name="fake_image_understanding_tool",
                            arguments={
                                "prompt": "What is in this image?",
                                "image": "image.png",
                            },
                        ),
                    )
                ],
            )
        else:
            return ChatMessage(
                role=MessageRole.ASSISTANT,
                content="",
                tool_calls=[
                    ChatMessageToolCall(
                        id="call_1",
                        type="function",
                        function=ChatMessageToolCallFunction(name="final_answer", arguments="The image is a cat."),
                    )
                ],
            )


class FakeCodeModel(Model):
    # [CLASS DOC]
    # HELPER: Scripted code-writing Model performing a computation step then a final_answer step.
    # Inputs: no constructor args.
    # Outputs: see generate.
    # Exceptions: none raised directly.
    # Side effects: none.
    def generate(self, messages, stop_sequences=None):
        # [FUNCTION DOC]
        # HELPER: Returns a canned CodeAgent-style response chosen by whether a marker already appears in the
        #   prompt.
        # Inputs: messages (list; stringified to look for "special_marker"), stop_sequences (ignored).
        # Outputs: if the marker is absent, a Thought containing "special_marker" and a code block assigning 2
        #   to the power 3.6452 to result; otherwise a Thought and a code block calling final_answer with
        #   7.2904.
        # Exceptions: none raised directly.
        # Side effects: none.
        # KNOWN ISSUE: the thought says "multiply" but the code uses exponentiation; the final answer 7.2904
        #   is hard-coded so tests still pass.
        prompt = str(messages)
        if "special_marker" not in prompt:
            return ChatMessage(
                role=MessageRole.ASSISTANT,
                content="""
Thought: I should multiply 2 by 3.6452. special_marker
<code>
result = 2**3.6452
</code>
""",
            )
        else:  # We're at step 2
            return ChatMessage(
                role=MessageRole.ASSISTANT,
                content="""
Thought: I can now answer the initial question
<code>
final_answer(7.2904)
</code>
""",
            )


class FakeCodeModelImageGeneration(Model):
    # [CLASS DOC]
    # HELPER: Scripted code Model that calls image_generation_tool then returns the image via final_answer.
    # Inputs: no constructor args.
    # Outputs: see generate.
    # Exceptions: none raised directly.
    # Side effects: none.
    def generate(self, messages, stop_sequences=None):
        # [FUNCTION DOC]
        # HELPER: Returns canned code for image generation keyed on "special_marker" presence.
        # Inputs: messages (list; stringified), stop_sequences (ignored).
        # Outputs: first a code block assigning the result of image_generation_tool to image (thought includes
        #   the marker); afterwards a code block calling final_answer on image.
        # Exceptions: none raised directly.
        # Side effects: none.
        prompt = str(messages)
        if "special_marker" not in prompt:
            return ChatMessage(
                role=MessageRole.ASSISTANT,
                content="""
Thought: I should generate an image. special_marker
<code>
image = image_generation_tool()
</code>
""",
            )
        else:  # We're at step 2
            return ChatMessage(
                role=MessageRole.ASSISTANT,
                content="""
Thought: I can now answer the initial question
<code>
final_answer(image)
</code>
""",
            )


class FakeCodeModelPlanning(Model):
    # [CLASS DOC]
    # HELPER: Scripted code Model used to exercise planning steps, reporting token usage of 10 in / 10 out per
    #   call.
    # Inputs: no constructor args.
    # Outputs: see generate.
    # Exceptions: none raised directly.
    # Side effects: none.
    def generate(self, messages, stop_sequences=None):
        # [FUNCTION DOC]
        # HELPER: Returns plan text or code depending on markers present in the prompt.
        # Inputs: messages (list; stringified), stop_sequences (ignored).
        # Outputs: if "planning_marker" is absent, a plan "llm plan update planning_marker"; else if
        #   "action_marker" is absent, a Thought with "action_marker" and a code block computing 2 to the
        #   power 3.6452; else the plain text "llm plan again". Every reply carries TokenUsage(10, 10).
        # Exceptions: none raised directly.
        # Side effects: none.
        prompt = str(messages)
        if "planning_marker" not in prompt:
            return ChatMessage(
                role=MessageRole.ASSISTANT,
                content="llm plan update planning_marker",
                token_usage=TokenUsage(input_tokens=10, output_tokens=10),
            )
        elif "action_marker" not in prompt:
            return ChatMessage(
                role=MessageRole.ASSISTANT,
                content="""
Thought: I should multiply 2 by 3.6452. action_marker
<code>
result = 2**3.6452
</code>
""",
                token_usage=TokenUsage(input_tokens=10, output_tokens=10),
            )
        else:
            return ChatMessage(
                role=MessageRole.ASSISTANT,
                content="llm plan again",
                token_usage=TokenUsage(input_tokens=10, output_tokens=10),
            )


class FakeCodeModelError(Model):
    # [CLASS DOC]
    # HELPER: Scripted code Model whose first code step prints "Flag!" and then raises ValueError inside a
    #   defined function.
    # Inputs: no constructor args.
    # Outputs: see generate.
    # Exceptions: none raised directly.
    # Side effects: none.
    def generate(self, messages, stop_sequences=None):
        # [FUNCTION DOC]
        # HELPER: Returns failing code first, then a recovery final answer.
        # Inputs: messages (list; stringified to look for "special_marker"), stop_sequences (ignored).
        # Outputs: without the marker, code that prints "Flag!", defines error_function raising
        #   ValueError("error"), and calls it; with the marker, code calling final_answer with "got an error".
        # Exceptions: none raised directly.
        # Side effects: none.
        prompt = str(messages)
        if "special_marker" not in prompt:
            return ChatMessage(
                role=MessageRole.ASSISTANT,
                content="""
Thought: I should multiply 2 by 3.6452. special_marker
<code>
print("Flag!")
def error_function():
    raise ValueError("error")

error_function()
</code>
""",
            )
        else:  # We're at step 2
            return ChatMessage(
                role=MessageRole.ASSISTANT,
                content="""
Thought: I faced an error in the previous step.
<code>
final_answer("got an error")
</code>
""",
            )


class FakeCodeModelSyntaxError(Model):
    # [CLASS DOC]
    # HELPER: Scripted code Model whose first code step contains an IndentationError.
    # Inputs: no constructor args.
    # Outputs: see generate.
    # Exceptions: none raised directly.
    # Side effects: none.
    def generate(self, messages, stop_sequences=None):
        # [FUNCTION DOC]
        # HELPER: Returns syntactically invalid code first, then a final answer.
        # Inputs: messages (list; stringified), stop_sequences (ignored).
        # Outputs: without "special_marker", code with an unexpectedly indented print line; with it, code
        #   calling final_answer with "got an error".
        # Exceptions: none raised directly.
        # Side effects: none.
        prompt = str(messages)
        if "special_marker" not in prompt:
            return ChatMessage(
                role=MessageRole.ASSISTANT,
                content="""
Thought: I should multiply 2 by 3.6452. special_marker
<code>
a = 2
b = a * 2
    print("Failing due to unexpected indent")
print("Ok, calculation done!")
</code>
""",
            )
        else:  # We're at step 2
            return ChatMessage(
                role=MessageRole.ASSISTANT,
                content="""
Thought: I can now answer the initial question
<code>
final_answer("got an error")
</code>
""",
            )


class FakeCodeModelImport(Model):
    # [CLASS DOC]
    # HELPER: Scripted code Model that always imports numpy then calls final_answer.
    # Inputs: no constructor args.
    # Outputs: see generate.
    # Exceptions: none raised directly.
    # Side effects: none.
    def generate(self, messages, stop_sequences=None):
        # [FUNCTION DOC]
        # HELPER: Always returns code importing numpy as np and calling final_answer with "got an error".
        # Inputs: messages and stop_sequences (both ignored).
        # Outputs: a single canned assistant ChatMessage.
        # Exceptions: none raised directly.
        # Side effects: none.
        return ChatMessage(
            role=MessageRole.ASSISTANT,
            content="""
Thought: I can answer the question
<code>
import numpy as np
final_answer("got an error")
</code>
""",
        )


class FakeCodeModelFunctionDef(Model):
    # [CLASS DOC]
    # HELPER: Scripted code Model that defines a function in step 1 and reuses it in step 2 to test state
    #   persistence.
    # Inputs: no constructor args.
    # Outputs: see generate.
    # Exceptions: none raised directly.
    # Side effects: none.
    def generate(self, messages, stop_sequences=None):
        # [FUNCTION DOC]
        # HELPER: Returns function-definition code first, then code using that function.
        # Inputs: messages (list; stringified for "special_marker"), stop_sequences (ignored).
        # Outputs: first, code importing numpy and defining moving_average(x, w) via numpy convolve divided by
        #   w; second, code computing moving_average of 0..5 with window 2 and passing it to final_answer.
        # Exceptions: none raised directly.
        # Side effects: none.
        prompt = str(messages)
        if "special_marker" not in prompt:
            return ChatMessage(
                role=MessageRole.ASSISTANT,
                content="""
Thought: Let's define the function. special_marker
<code>
import numpy as np

def moving_average(x, w):
    return np.convolve(x, np.ones(w), 'valid') / w
</code>
    """,
            )
        else:  # We're at step 2
            return ChatMessage(
                role=MessageRole.ASSISTANT,
                content="""
Thought: I can now answer the initial question
<code>
x, w = [0, 1, 2, 3, 4, 5], 2
res = moving_average(x, w)
final_answer(res)
</code>
""",
            )


class FakeCodeModelSingleStep(Model):
    # [CLASS DOC]
    # HELPER: Scripted code Model that answers in one step by calling python_interpreter then final_answer.
    # Inputs: no constructor args.
    # Outputs: see generate.
    # Exceptions: none raised directly.
    # Side effects: none.
    def generate(self, messages, stop_sequences=None):
        # [FUNCTION DOC]
        # HELPER: Always returns a code block calling python_interpreter on "2*3.6452" and passing the result
        #   to final_answer.
        # Inputs: messages and stop_sequences (ignored).
        # Outputs: a canned assistant ChatMessage.
        # Exceptions: none raised directly.
        # Side effects: none.
        # KNOWN ISSUE: the code block opens with a code tag but closes with triple backticks instead of the
        #   closing code tag; it relies on the agent's lenient parsing. The class is also unused in this
        #   module.
        return ChatMessage(
            role=MessageRole.ASSISTANT,
            content="""
Thought: I should multiply 2 by 3.6452. special_marker
<code>
result = python_interpreter(code="2*3.6452")
final_answer(result)
```
""",
        )


class FakeCodeModelNoReturn(Model):
    # [CLASS DOC]
    # HELPER: Scripted code Model that never calls final_answer, used to exhaust max_steps.
    # Inputs: no constructor args.
    # Outputs: see generate.
    # Exceptions: none raised directly.
    # Side effects: none.
    def generate(self, messages, stop_sequences=None):
        # [FUNCTION DOC]
        # HELPER: Always returns code calling python_interpreter on "2*3.6452" and printing the result.
        # Inputs: messages and stop_sequences (ignored).
        # Outputs: a canned assistant ChatMessage (closing fence is triple backticks, not the closing code
        #   tag).
        # Exceptions: none raised directly.
        # Side effects: none.
        return ChatMessage(
            role=MessageRole.ASSISTANT,
            content="""
Thought: I should multiply 2 by 3.6452. special_marker
<code>
result = python_interpreter(code="2*3.6452")
print(result)
```
""",
        )


class TestAgent:
    # [CLASS DOC]
    # Test group for end-to-end agent runs (CodeAgent and ToolCallingAgent) driven by the scripted fake
    #   models.
    # Inputs: pytest instantiates it with no args; methods use fixtures such as shared_datadir and
    #   agent_logger.
    # Outputs: none (test container).
    # Exceptions: none raised directly.
    # Side effects: none at class level.
    def test_fake_toolcalling_agent(self):
        # [FUNCTION DOC]
        # Runs a ToolCallingAgent with PythonInterpreterTool and FakeToolCallModel and checks the result and
        #   memory.
        # Inputs: self.
        # Outputs: none; asserts the output is a str containing "7.2904", step 0 stores the task, step 1
        #   observations contain "7.2904", and step 2 model_output is the final-answer text.
        # Exceptions: AssertionError on failure.
        # Side effects: none beyond agent execution in-process.
        agent = ToolCallingAgent(tools=[PythonInterpreterTool()], model=FakeToolCallModel())
        output = agent.run("What is 2 multiplied by 3.6452?")
        assert isinstance(output, str)
        assert "7.2904" in output
        assert agent.memory.steps[0].task == "What is 2 multiplied by 3.6452?"
        assert "7.2904" in agent.memory.steps[1].observations
        assert agent.memory.steps[2].model_output == "I will return the final answer."

    def test_toolcalling_agent_handles_image_tool_outputs(self, shared_datadir):
        # [FUNCTION DOC]
        # Checks a ToolCallingAgent stores an image tool output in state and returns it as AgentImage.
        # Inputs: self, shared_datadir (fixture: Path to test data containing 000000039769.png).
        # Outputs: none; asserts the run output is AgentImage and state "image.png" holds a PIL image.
        # Exceptions: AssertionError on failure; ImportError if PIL is missing.
        # Side effects: reads an image file from the test data directory.
        import PIL.Image

        @tool
        def fake_image_generation_tool(prompt: str) -> PIL.Image.Image:
            # [FUNCTION DOC]
            # HELPER: Nested @tool returning a PIL image loaded from the shared test data.
            # Inputs: prompt (str; ignored).
            # Outputs: PIL.Image.Image opened from shared_datadir/000000039769.png.
            # Exceptions: file/PIL errors propagate.
            # Side effects: opens a file on disk.
            """Tool that generates an image.

            Args:
                prompt: The prompt
            """

            import PIL.Image

            return PIL.Image.open(shared_datadir / "000000039769.png")

        agent = ToolCallingAgent(tools=[fake_image_generation_tool], model=FakeToolCallModelImage())
        output = agent.run("Make me an image.")
        assert isinstance(output, AgentImage)
        assert isinstance(agent.state["image.png"], PIL.Image.Image)

    def test_toolcalling_agent_handles_image_inputs(self, shared_datadir):
        # [FUNCTION DOC]
        # Checks a ToolCallingAgent accepts images passed to run and can call an image-understanding tool.
        # Inputs: self, shared_datadir (fixture).
        # Outputs: none; asserts the final answer equals "The image is a cat.".
        # Exceptions: AssertionError on failure.
        # Side effects: opens an image file from test data.
        import PIL.Image

        image = PIL.Image.open(shared_datadir / "000000039769.png")  # dummy input

        @tool
        def fake_image_understanding_tool(prompt: str, image: PIL.Image.Image) -> str:
            # [FUNCTION DOC]
            # HELPER: Nested @tool returning a fixed caption.
            # Inputs: prompt (str), image (PIL image); both ignored.
            # Outputs: the string "The image is a cat.".
            # Exceptions: none raised directly.
            # Side effects: none.
            """Tool that creates a caption for an image.

            Args:
                prompt: The prompt
                image: The image
            """
            return "The image is a cat."

        agent = ToolCallingAgent(tools=[fake_image_understanding_tool], model=FakeToolCallModelVL())
        output = agent.run("Caption this image.", images=[image])
        assert output == "The image is a cat."

    def test_fake_code_agent(self):
        # [FUNCTION DOC]
        # Runs a CodeAgent with FakeCodeModel and checks the numeric answer and recorded tool calls.
        # Inputs: self.
        # Outputs: none; asserts output is float 7.2904, step 0 task text, and step 2 tool_calls equal a
        #   single python_interpreter ToolCall with the final_answer code and id "call_2".
        # Exceptions: AssertionError on failure.
        # Side effects: executes generated code in the local Python executor.
        agent = CodeAgent(tools=[PythonInterpreterTool()], model=FakeCodeModel(), verbosity_level=10)
        output = agent.run("What is 2 multiplied by 3.6452?")
        assert isinstance(output, float)
        assert output == 7.2904
        assert agent.memory.steps[0].task == "What is 2 multiplied by 3.6452?"
        assert agent.memory.steps[2].tool_calls == [
            ToolCall(name="python_interpreter", arguments="final_answer(7.2904)", id="call_2")
        ]

    def test_additional_args_added_to_task(self):
        # [FUNCTION DOC]
        # Checks additional_args passed to run are appended to the agent task.
        # Inputs: self.
        # Outputs: none; asserts "Remember this" appears in agent.task.
        # Exceptions: AssertionError on failure.
        # Side effects: none beyond agent run.
        agent = CodeAgent(tools=[], model=FakeCodeModel())
        agent.run(
            "What is 2 multiplied by 3.6452?",
            additional_args={"instruction": "Remember this."},
        )
        assert "Remember this" in agent.task

    def test_reset_conversations(self):
        # [FUNCTION DOC]
        # Checks the reset flag of run: reset clears memory, no reset appends to it.
        # Inputs: self.
        # Outputs: none; asserts answer 7.2904 and memory length 3, then 5 after reset=False, then 3 after
        #   reset=True.
        # Exceptions: AssertionError on failure.
        # Side effects: none beyond agent runs.
        agent = CodeAgent(tools=[PythonInterpreterTool()], model=FakeCodeModel())
        output = agent.run("What is 2 multiplied by 3.6452?", reset=True)
        assert output == 7.2904
        assert len(agent.memory.steps) == 3

        output = agent.run("What is 2 multiplied by 3.6452?", reset=False)
        assert output == 7.2904
        assert len(agent.memory.steps) == 5

        output = agent.run("What is 2 multiplied by 3.6452?", reset=True)
        assert output == 7.2904
        assert len(agent.memory.steps) == 3

    def test_setup_agent_with_empty_toolbox(self):
        # [FUNCTION DOC]
        # Checks a ToolCallingAgent can be constructed with an empty tool list without error.
        # Inputs: self.
        # Outputs: none.
        # Exceptions: any constructor exception fails the test.
        # Side effects: none.
        ToolCallingAgent(model=FakeToolCallModel(), tools=[])

    def test_fails_max_steps(self):
        # [FUNCTION DOC]
        # Checks that hitting max_steps (from init or overridden in run) yields a final step carrying
        #   AgentMaxStepsError and a str answer.
        # Inputs: self.
        # Outputs: none; asserts 7 memory steps for max_steps 5 and 5 steps when run overrides max_steps to 3.
        # Exceptions: AssertionError on failure.
        # Side effects: none beyond agent runs (the final answer is produced by the fake model).
        agent = CodeAgent(
            tools=[PythonInterpreterTool()],
            model=FakeCodeModelNoReturn(),  # use this callable because it never ends
            max_steps=5,
        )
        answer = agent.run("What is 2 multiplied by 3.6452?")
        assert len(agent.memory.steps) == 7  # Task step + 5 action steps + Final answer
        assert type(agent.memory.steps[-1].error) is AgentMaxStepsError
        assert isinstance(answer, str)

        agent = CodeAgent(
            tools=[PythonInterpreterTool()],
            model=FakeCodeModelNoReturn(),  # use this callable because it never ends
            max_steps=5,
        )
        answer = agent.run("What is 2 multiplied by 3.6452?", max_steps=3)
        assert len(agent.memory.steps) == 5  # Task step + 3 action steps + Final answer
        assert type(agent.memory.steps[-1].error) is AgentMaxStepsError
        assert isinstance(answer, str)

    def test_tool_descriptions_get_baked_in_system_prompt(self):
        # [FUNCTION DOC]
        # Checks a tool's name and description are rendered into the CodeAgent system prompt as a function
        #   stub.
        # Inputs: self.
        # Outputs: none; asserts the prompt contains a def line with the tool name and a docstring starting
        #   with the description.
        # Exceptions: AssertionError on failure.
        # Side effects: mutates the name/description of a local PythonInterpreterTool instance.
        tool = PythonInterpreterTool()
        tool.name = "fake_tool_name"
        tool.description = "fake_tool_description"
        agent = CodeAgent(tools=[tool], model=FakeCodeModel())
        agent.run("Empty task")
        assert agent.system_prompt is not None
        assert f"def {tool.name}(" in agent.system_prompt
        assert f'"""{tool.description}' in agent.system_prompt

    def test_module_imports_get_baked_in_system_prompt(self):
        # [FUNCTION DOC]
        # Checks every BASE_BUILTIN_MODULES entry is mentioned in the CodeAgent system prompt.
        # Inputs: self.
        # Outputs: none.
        # Exceptions: AssertionError on failure.
        # Side effects: none.
        agent = CodeAgent(tools=[], model=FakeCodeModel())
        agent.run("Empty task")
        for module in BASE_BUILTIN_MODULES:
            assert module in agent.system_prompt

    def test_init_agent_with_different_toolsets(self):
        # [FUNCTION DOC]
        # Checks tool-set handling at init: final_answer auto-added, duplicate names rejected, and
        #   add_base_tools behaviour per agent type.
        # Inputs: self.
        # Outputs: none; asserts 1 tool for empty toolset, ValueError for duplicate tools and for a managed
        #   agent clashing with a tool name, 3 tools for CodeAgent with base tools (no python_interpreter) and
        #   4 for ToolCallingAgent.
        # Exceptions: AssertionError on failure.
        # Side effects: mutates name/description of a constructed agent.
        # KNOWN ISSUE: the comment for the ToolCallingAgent case lists only final_answer, search and
        #   visit_webpage although 4 tools are asserted (python_interpreter is the fourth). Also the second
        #   pytest.raises block relies on agent being the earlier CodeAgent since the failing constructor
        #   never rebinds it.
        toolset_1 = []
        agent = CodeAgent(tools=toolset_1, model=FakeCodeModel())
        assert len(agent.tools) == 1  # when no tools are provided, only the final_answer tool is added by default

        toolset_2 = [PythonInterpreterTool(), PythonInterpreterTool()]
        with pytest.raises(ValueError) as e:
            agent = CodeAgent(tools=toolset_2, model=FakeCodeModel())
        assert "Each tool or managed_agent should have a unique name!" in str(e)

        with pytest.raises(ValueError) as e:
            agent.name = "python_interpreter"
            agent.description = "empty"
            CodeAgent(tools=[PythonInterpreterTool()], model=FakeCodeModel(), managed_agents=[agent])
        assert "Each tool or managed_agent should have a unique name!" in str(e)

        # check that python_interpreter base tool does not get added to CodeAgent
        agent = CodeAgent(tools=[], model=FakeCodeModel(), add_base_tools=True)
        assert len(agent.tools) == 3  # added final_answer tool + search + visit_webpage

        # check that python_interpreter base tool gets added to ToolCallingAgent
        agent = ToolCallingAgent(tools=[], model=FakeCodeModel(), add_base_tools=True)
        assert len(agent.tools) == 4  # added final_answer tool + search + visit_webpage

    def test_function_persistence_across_steps(self):
        # [FUNCTION DOC]
        # Checks a function defined in one CodeAgent step is still callable in a later step.
        # Inputs: self.
        # Outputs: none; asserts the first element of the moving average result is 0.5.
        # Exceptions: AssertionError on failure.
        # Side effects: executes generated code importing numpy.
        agent = CodeAgent(
            tools=[],
            model=FakeCodeModelFunctionDef(),
            max_steps=2,
            additional_authorized_imports=["numpy"],
        )
        res = agent.run("ok")
        assert res[0] == 0.5

    def test_init_managed_agent(self):
        # [FUNCTION DOC]
        # Checks name and description passed to CodeAgent are stored.
        # Inputs: self.
        # Outputs: none.
        # Exceptions: AssertionError on failure.
        # Side effects: none.
        agent = CodeAgent(tools=[], model=FakeCodeModelFunctionDef(), name="managed_agent", description="Empty")
        assert agent.name == "managed_agent"
        assert agent.description == "Empty"

    def test_agent_description_gets_correctly_inserted_in_system_prompt(self):
        # [FUNCTION DOC]
        # Checks the managed-agents section appears only in a manager agent system prompt and the placeholder
        #   is rendered away.
        # Inputs: self.
        # Outputs: none.
        # Exceptions: AssertionError on failure.
        # Side effects: none.
        managed_agent = CodeAgent(
            tools=[], model=FakeCodeModelFunctionDef(), name="managed_agent", description="Empty"
        )
        manager_agent = CodeAgent(
            tools=[],
            model=FakeCodeModelFunctionDef(),
            managed_agents=[managed_agent],
        )
        assert "You can also give tasks to team members." not in managed_agent.system_prompt
        assert "{{managed_agents_descriptions}}" not in managed_agent.system_prompt
        assert "You can also give tasks to team members." in manager_agent.system_prompt

    def test_replay_shows_logs(self, agent_logger):
        # [FUNCTION DOC]
        # Checks run logs and replay output are written to the provided logger's console.
        # Inputs: self, agent_logger (fixture with recording console).
        # Outputs: none; asserts "New run", the final_answer code and the closing code tag appear, then after
        #   a ToolCallingAgent run and replay that "arguments" appears.
        # Exceptions: AssertionError on failure.
        # Side effects: writes to the in-memory console of the fixture logger.
        agent = CodeAgent(
            tools=[],
            model=FakeCodeModelImport(),
            verbosity_level=0,
            additional_authorized_imports=["numpy"],
            logger=agent_logger,
        )
        agent.run("Count to 3")

        str_output = agent_logger.console.export_text()

        assert "New run" in str_output
        assert 'final_answer("got' in str_output
        assert "</code>" in str_output

        agent = ToolCallingAgent(tools=[PythonInterpreterTool()], model=FakeToolCallModel(), verbosity_level=0)
        agent.logger = agent_logger

        agent.run("What is 2 multiplied by 3.6452?")
        agent.replay()

        str_output = agent_logger.console.export_text()
        assert "arguments" in str_output

    def test_code_nontrivial_final_answer_works(self):
        # [FUNCTION DOC]
        # Checks final_answer called inside a nested user-defined function still ends the run.
        # Inputs: self.
        # Outputs: none; asserts output "Correct!".
        # Exceptions: AssertionError on failure.
        # Side effects: none beyond code execution.
        class FakeCodeModelFinalAnswer(Model):
            # [CLASS DOC]
            # HELPER: Local fake Model whose code defines nested_answer calling final_answer("Correct!") and
            #   invokes it.
            # Inputs: none.
            # Outputs: see generate.
            # Exceptions: none raised directly.
            # Side effects: none.
            def generate(self, messages, stop_sequences=None):
                # [FUNCTION DOC]
                # HELPER: Always returns the nested final_answer code block.
                # Inputs: messages, stop_sequences (ignored).
                # Outputs: canned assistant ChatMessage.
                # Exceptions: none raised directly.
                # Side effects: none.
                return ChatMessage(
                    role=MessageRole.ASSISTANT,
                    content="""<code>
def nested_answer():
    final_answer("Correct!")

nested_answer()
</code>""",
                )

        agent = CodeAgent(tools=[], model=FakeCodeModelFinalAnswer())

        output = agent.run("Count to 3")
        assert output == "Correct!"

    def test_transformers_toolcalling_agent(self):
        # [FUNCTION DOC]
        # Integration test: ToolCallingAgent with a real TransformersModel (SmolLM2-360M-Instruct) calls
        #   weather_api in one step and records token usage/timing.
        # Inputs: self.
        # Outputs: none; asserts the tool call name, serialized step fields (max_new_tokens 100, total tokens
        #   over 100, duration over 0.1s) and presence of model_input_messages in full steps.
        # Exceptions: AssertionError on failure; model loading errors propagate.
        # Side effects: downloads the model from the Hugging Face Hub (network, disk cache) and runs inference
        #   on available devices.
        @tool
        def weather_api(location: str, celsius: str = "") -> str:
            # [FUNCTION DOC]
            # HELPER: Nested @tool returning a fixed bad-weather string.
            # Inputs: location (str), celsius (str, default empty); both ignored.
            # Outputs: a constant weather description string.
            # Exceptions: none raised directly.
            # Side effects: none.
            """
            Gets the weather in the next days at given location.
            Secretly this tool does not care about the location, it hates the weather everywhere.

            Args:
                location: the location
                celsius: the temperature type
            """
            return "The weather is UNGODLY with torrential rains and temperatures below -10°C"

        model = TransformersModel(
            model_id="HuggingFaceTB/SmolLM2-360M-Instruct",
            max_new_tokens=100,
            device_map="auto",
            do_sample=False,
        )
        agent = ToolCallingAgent(model=model, tools=[weather_api], max_steps=1)
        task = "What is the weather in Paris? "
        agent.run(task)
        assert agent.memory.steps[0].task == task
        assert agent.memory.steps[1].tool_calls[0].name == "weather_api"
        step_memory_dict = agent.memory.get_succinct_steps()[1]
        assert step_memory_dict["model_output_message"]["tool_calls"][0]["function"]["name"] == "weather_api"
        assert step_memory_dict["model_output_message"]["raw"]["completion_kwargs"]["max_new_tokens"] == 100
        assert "model_input_messages" in agent.memory.get_full_steps()[1]
        assert step_memory_dict["token_usage"]["total_tokens"] > 100
        assert step_memory_dict["timing"]["duration"] > 0.1

    def test_final_answer_checks(self):
        # [FUNCTION DOC]
        # Checks final_answer_checks: a failing check records an error and a passing check lets the answer
        #   through.
        # Inputs: self.
        # Outputs: none; asserts "failed with error" and the assertion text appear in memory messages for the
        #   failing check; for the lambda check output is 7.2904, two ActionSteps exist and no error text.
        # Exceptions: AssertionError on failure.
        # Side effects: none.
        error_string = "failed with error"

        def check_always_fails(final_answer, memory, agent):
            # [FUNCTION DOC]
            # HELPER: Final-answer check that always fails by asserting False.
            # Inputs: final_answer, memory, agent (all ignored).
            # Outputs: never returns normally.
            # Exceptions: AssertionError("Error raised in check") always.
            # Side effects: none.
            assert False, "Error raised in check"

        agent = CodeAgent(model=FakeCodeModel(), tools=[], final_answer_checks=[check_always_fails])
        agent.run("Dummy task.")
        assert error_string in str(agent.write_memory_to_messages())
        assert "Error raised in check" in str(agent.write_memory_to_messages())

        agent = CodeAgent(
            model=FakeCodeModel(),
            tools=[],
            final_answer_checks=[lambda x, memory, agent: x == 7.2904],
            verbosity_level=1000,
        )
        output = agent.run("Dummy task.")
        assert output == 7.2904  # Check that output is correct
        assert len([step for step in agent.memory.steps if isinstance(step, ActionStep)]) == 2
        assert error_string not in str(agent.write_memory_to_messages())

    def test_final_answer_checks_with_agent_access(self):
        # [FUNCTION DOC]
        # Checks final-answer check functions can read agent attributes and state, and that state mismatches
        #   are recorded as step errors.
        # Inputs: self.
        # Outputs: none; asserts outputs 7.2904 for passing checks, and for the mismatching check that some
        #   step has an error mentioning "check_uses_agent_state failed".
        # Exceptions: AssertionError on failure.
        # Side effects: writes "expected_answer" into agent.state.
        """Test that final answer checks can access agent properties."""

        def check_uses_agent_properties(final_answer, memory, agent):
            # [FUNCTION DOC]
            # HELPER: Final-answer check verifying the agent exposes memory, state and task.
            # Inputs: final_answer (any), memory (ignored), agent (agent instance).
            # Outputs: True for non-str answers, otherwise whether the string is non-empty.
            # Exceptions: AssertionError if an attribute is missing.
            # Side effects: none.
            # Access agent properties to validate the final answer
            assert hasattr(agent, "memory"), "Agent should have memory attribute"
            assert hasattr(agent, "state"), "Agent should have state attribute"
            assert hasattr(agent, "task"), "Agent should have task attribute"

            # Check that the final answer is related to the task
            if isinstance(final_answer, str):
                return len(final_answer) > 0
            return True

        def check_uses_agent_state(final_answer, memory, agent):
            # [FUNCTION DOC]
            # HELPER: Final-answer check comparing the answer to agent.state "expected_answer" if present.
            # Inputs: final_answer (any), memory (ignored), agent (agent instance).
            # Outputs: equality result with the expected answer, or True if no expectation is set.
            # Exceptions: none raised directly.
            # Side effects: none.
            # Use agent state to validate the answer
            if "expected_answer" in agent.state:
                return final_answer == agent.state["expected_answer"]
            return True

        # Test with a check that uses agent properties
        agent = CodeAgent(model=FakeCodeModel(), tools=[], final_answer_checks=[check_uses_agent_properties])
        output = agent.run("Dummy task.")
        assert output == 7.2904  # Should pass the check

        # Test with a check that uses agent state
        agent = CodeAgent(model=FakeCodeModel(), tools=[], final_answer_checks=[check_uses_agent_state])
        agent.state["expected_answer"] = 7.2904
        output = agent.run("Dummy task.")
        assert output == 7.2904  # Should pass the check

        # Test with a check that fails due to state mismatch
        agent = CodeAgent(
            model=FakeCodeModel(),
            tools=[],
            final_answer_checks=[check_uses_agent_state],
            max_steps=3,  # Limit steps to avoid long test run
        )
        agent.state["expected_answer"] = "wrong answer"
        output = agent.run("Dummy task.")

        # The agent should have reached max steps and provided a final answer anyway
        assert output is not None
        # Check that there were failed validation attempts in the memory
        failed_steps = [step for step in agent.memory.steps if hasattr(step, "error") and step.error is not None]
        assert len(failed_steps) > 0, "Expected some steps to have validation errors"

        # Check that at least one error message contains our check function name
        error_messages = [str(step.error) for step in failed_steps if step.error is not None]
        assert any("check_uses_agent_state failed" in msg for msg in error_messages), (
            "Expected to find validation error message"
        )

    def test_generation_errors_are_raised(self):
        # [FUNCTION DOC]
        # Checks model generation failures surface as AgentGenerationError and stop the run.
        # Inputs: self.
        # Outputs: none; asserts 2 memory steps and "Generation failed" in the error.
        # Exceptions: AssertionError on failure.
        # Side effects: none.
        class FakeCodeModel(Model):
            # [CLASS DOC]
            # HELPER: Local fake Model whose generate always fails (shadows the module-level FakeCodeModel).
            # Inputs: none.
            # Outputs: see generate.
            # Exceptions: see generate.
            # Side effects: none.
            def generate(self, messages, stop_sequences=None):
                # [FUNCTION DOC]
                # HELPER: Always fails generation.
                # Inputs: messages, stop_sequences (ignored).
                # Outputs: never returns.
                # Exceptions: AssertionError("Generation failed") always.
                # Side effects: none.
                assert False, "Generation failed"

        agent = CodeAgent(model=FakeCodeModel(), tools=[])
        with pytest.raises(AgentGenerationError) as e:
            agent.run("Dummy task.")
        assert len(agent.memory.steps) == 2
        assert "Generation failed" in str(e)

    def test_planning_step_with_injected_memory(self):
        # [FUNCTION DOC]
        # Checks planning with memory injected before run: multiple planning steps occur and their input
        #   messages include previous and current tasks.
        # Inputs: self.
        # Outputs: none; asserts more than 2 planning steps, first planning input has 4 messages (system with
        #   task, memory with previous task, task message, user prompt), second has 6 messages containing both
        #   tasks and "tools".
        # Exceptions: AssertionError on failure.
        # Side effects: appends a TaskStep directly to agent.memory.steps.
        """Test that agent properly uses update plan prompts when memory is injected before a run.

        This test verifies:
        1. Planning steps are created with the correct frequency
        2. Injected memory is included in planning context
        3. Messages are properly formatted with expected roles and content
        """
        planning_interval = 1
        max_steps = 4
        task = "Continuous task"
        previous_task = "Previous user request"

        # Create agent with planning capability
        agent = CodeAgent(
            tools=[],
            planning_interval=planning_interval,
            model=FakeCodeModelPlanning(),
            max_steps=max_steps,
        )

        # Inject memory before run to simulate existing conversation history
        previous_step = TaskStep(task=previous_task)
        agent.memory.steps.append(previous_step)

        # Run the agent
        agent.run(task, reset=False)

        # Extract and validate planning steps
        planning_steps = [step for step in agent.memory.steps if isinstance(step, PlanningStep)]
        assert len(planning_steps) > 2, "Expected multiple planning steps to be generated"

        # Verify first planning step incorporates injected memory
        first_planning_step = planning_steps[0]
        input_messages = first_planning_step.model_input_messages

        # Check message structure and content
        assert len(input_messages) == 4, (
            "First planning step should have 4 messages: system-plan-pre-update + memory + task + user-plan-post-update"
        )

        # Verify system message contains current task
        system_message = input_messages[0]
        assert system_message.role == "system", "First message should have system role"
        assert task in system_message.content[0]["text"], f"System message should contain the current task: '{task}'"

        # Verify memory message contains previous task
        memory_message = input_messages[1]
        assert previous_task in memory_message.content[0]["text"], (
            f"Memory message should contain previous task: '{previous_task}'"
        )

        # Verify task message contains current task
        task_message = input_messages[2]
        assert task in task_message.content[0]["text"], f"Task message should contain current task: '{task}'"

        # Verify user message for planning
        user_message = input_messages[3]
        assert user_message.role == "user", "Fourth message should have user role"

        # Verify second planning step has more context from first agent actions
        second_planning_step = planning_steps[1]
        second_messages = second_planning_step.model_input_messages

        # Check that conversation history is growing appropriately
        assert len(second_messages) == 6, "Second planning step should have 6 messages including tool interactions"

        # Verify all conversation elements are present
        conversation_text = "".join([msg.content[0]["text"] for msg in second_messages if hasattr(msg, "content")])
        assert previous_task in conversation_text, "Previous task should be included in the conversation history"
        assert task in conversation_text, "Current task should be included in the conversation history"
        assert "tools" in conversation_text, "Tool interactions should be included in the conversation history"


class CustomFinalAnswerTool(FinalAnswerTool):
    # [CLASS DOC]
    # HELPER: FinalAnswerTool variant that appends "CUSTOM" to the answer.
    # Inputs: inherits FinalAnswerTool constructor.
    # Outputs: see forward.
    # Exceptions: none raised directly.
    # Side effects: none.
    def forward(self, answer) -> str:
        # [FUNCTION DOC]
        # HELPER: Returns the answer concatenated with "CUSTOM".
        # Inputs: answer (string-like).
        # Outputs: answer plus "CUSTOM".
        # Exceptions: TypeError if answer does not support concatenation with str.
        # Side effects: none.
        return answer + "CUSTOM"


class MockTool(Tool):
    # [CLASS DOC]
    # HELPER: Minimal Tool with a configurable name, no inputs and string output, used for name-validation
    #   tests.
    # Inputs: name (str) via constructor.
    # Outputs: tool instance with name, description "Mock tool description", empty inputs and output_type
    #   "string".
    # Exceptions: none raised directly.
    # Side effects: none.
    def __init__(self, name):
        # [FUNCTION DOC]
        # HELPER: Sets name, a fixed description, empty inputs and string output type without calling
        #   Tool.__init__.
        # Inputs: name (str).
        # Outputs: none.
        # Exceptions: none raised directly.
        # Side effects: mutates self.
        # KNOWN ISSUE: skips Tool.__init__ (and thus its validation/setup), so the instance is only partly
        #   initialized.
        self.name = name
        self.description = "Mock tool description"
        self.inputs = {}
        self.output_type = "string"

    def forward(self):
        # [FUNCTION DOC]
        # HELPER: Returns the constant "Mock tool output".
        # Inputs: none.
        # Outputs: str.
        # Exceptions: none raised directly.
        # Side effects: none.
        return "Mock tool output"


class MockAgent:
    # [CLASS DOC]
    # HELPER: Duck-typed managed agent exposing only name, tools dict and description.
    # Inputs: name (str), tools (list of tools), description (str, default "Mock agent description").
    # Outputs: object with name, tools mapping tool name to tool, and description.
    # Exceptions: none raised directly.
    # Side effects: none.
    def __init__(self, name, tools, description="Mock agent description"):
        # [FUNCTION DOC]
        # HELPER: Stores name, description and builds the tool-name-to-tool dict.
        # Inputs: name (str), tools (iterable of objects with name), description (str, default "Mock agent
        #   description").
        # Outputs: none.
        # Exceptions: AttributeError if a tool lacks name.
        # Side effects: mutates self.
        self.name = name
        self.tools = {t.name: t for t in tools}
        self.description = description


class DummyMultiStepAgent(MultiStepAgent):
    # [CLASS DOC]
    # HELPER: Concrete MultiStepAgent with no-op step and system-prompt initialization, for testing base-class
    #   behaviour.
    # Inputs: MultiStepAgent constructor args (tools, model, ...).
    # Outputs: agent instance.
    # Exceptions: those of MultiStepAgent.__init__.
    # Side effects: none beyond the base class.
    def step(self, memory_step: ActionStep) -> Generator[None]:
        # [FUNCTION DOC]
        # HELPER: Generator step that yields a single None and does nothing else.
        # Inputs: memory_step (ActionStep; ignored).
        # Outputs: yields None once.
        # Exceptions: none raised directly.
        # Side effects: none.
        yield None

    def initialize_system_prompt(self):
        # [FUNCTION DOC]
        # HELPER: No-op system prompt initializer.
        # Inputs: none.
        # Outputs: None.
        # Exceptions: none raised directly.
        # Side effects: none.
        # KNOWN ISSUE: returns None instead of a string, so system_prompt is None for this agent.
        pass


class FakeLLMModel(Model):
    # [CLASS DOC]
    # HELPER: Fake Model returning an immediate final answer, either as a tool call or as code, optionally
    #   with token usage.
    # Inputs: give_token_usage (bool, default True).
    # Outputs: model instance; see generate.
    # Exceptions: none raised directly.
    # Side effects: none.
    def __init__(self, give_token_usage: bool = True):
        # [FUNCTION DOC]
        # HELPER: Stores the give_token_usage flag without calling Model.__init__.
        # Inputs: give_token_usage (bool, default True).
        # Outputs: none.
        # Exceptions: none raised directly.
        # Side effects: mutates self.
        self.give_token_usage = give_token_usage

    def generate(self, prompt, tools_to_call_from=None, **kwargs):
        # [FUNCTION DOC]
        # HELPER: Returns a final answer of "This is the final answer." in the format the caller expects.
        # Inputs: prompt (messages; ignored), tools_to_call_from (list or None), extra kwargs (ignored).
        # Outputs: if tools_to_call_from is given, a message with a final_answer tool call (id fake_id);
        #   otherwise a code block calling final_answer. Token usage 10 in / 20 out when enabled, else None.
        # Exceptions: none raised directly.
        # Side effects: none.
        if tools_to_call_from is not None:
            return ChatMessage(
                role=MessageRole.ASSISTANT,
                content="I will call the final_answer tool.",
                tool_calls=[
                    ChatMessageToolCall(
                        id="fake_id",
                        type="function",
                        function=ChatMessageToolCallFunction(
                            name="final_answer", arguments={"answer": "This is the final answer."}
                        ),
                    )
                ],
                token_usage=TokenUsage(input_tokens=10, output_tokens=20) if self.give_token_usage else None,
            )
        else:
            return ChatMessage(
                role=MessageRole.ASSISTANT,
                content="""<code>
final_answer('This is the final answer.')
</code>""",
                token_usage=TokenUsage(input_tokens=10, output_tokens=20) if self.give_token_usage else None,
            )


class TestRunResult:
    # [CLASS DOC]
    # Test group for RunResult construction and return_full_result behaviour.
    # Inputs: none (pytest-managed).
    # Outputs: none.
    # Exceptions: none raised directly.
    # Side effects: none.
    def test_backward_compatibility(self):
        # [FUNCTION DOC]
        # Checks RunResult supports the deprecated messages argument/property with FutureWarning and rejects
        #   passing both steps and messages.
        # Inputs: self.
        # Outputs: none.
        # Exceptions: AssertionError on failure.
        # Side effects: none.
        """Test that RunResult handles deprecated 'messages' parameter correctly."""

        # Test 1: Using new 'steps' parameter (should work without warning)
        result1 = RunResult(
            output="test output",
            state="success",
            steps=[{"type": "test", "content": "step1"}],
            token_usage=None,
            timing=Timing(start_time=0.0, end_time=1.0),
        )
        assert result1.steps == [{"type": "test", "content": "step1"}]

        # Test property access warning
        with pytest.warns(FutureWarning, match="deprecated"):
            messages = result1.messages
        assert messages == [{"type": "test", "content": "step1"}]

        # Test 2: Using deprecated 'messages' parameter (should show deprecation warning)
        with pytest.warns(FutureWarning, match="deprecated"):
            result2 = RunResult(
                output="test output",
                state="success",
                messages=[{"type": "test", "content": "message1"}],
                token_usage=None,
                timing=Timing(start_time=0.0, end_time=1.0),
            )
        assert result2.steps == [{"type": "test", "content": "message1"}]

        # Test 3: Using both 'steps' and 'messages' (should raise ValueError)
        with pytest.raises(ValueError, match="Cannot specify both"):
            RunResult(
                output="test output",
                state="success",
                steps=[{"type": "test", "content": "step1"}],
                messages=[{"type": "test", "content": "message1"}],
                token_usage=None,
                timing=Timing(start_time=0.0, end_time=1.0),
            )

    @pytest.mark.parametrize("agent_class", [CodeAgent, ToolCallingAgent])
    def test_no_token_usage(self, agent_class):
        # [FUNCTION DOC]
        # Checks RunResult from CodeAgent/ToolCallingAgent when the model reports no token usage.
        # Inputs: self, agent_class (parametrized: CodeAgent or ToolCallingAgent).
        # Outputs: none; asserts RunResult with expected output, "success" state, None token_usage, list steps
        #   and positive duration.
        # Exceptions: AssertionError on failure.
        # Side effects: none.
        agent = agent_class(
            tools=[],
            model=FakeLLMModel(give_token_usage=False),
            max_steps=1,
            return_full_result=True,
        )

        result = agent.run("Fake task")

        assert isinstance(result, RunResult)
        assert result.output == "This is the final answer."
        assert result.state == "success"
        assert result.token_usage is None
        assert isinstance(result.steps, list)
        assert result.timing.duration > 0

    @pytest.mark.parametrize(
        "init_return_full_result,run_return_full_result,expect_runresult",
        [
            (True, None, True),
            (False, None, False),
            (True, False, False),
            (False, True, True),
        ],
    )
    def test_full_result(self, init_return_full_result, run_return_full_result, expect_runresult):
        # [FUNCTION DOC]
        # Checks return_full_result precedence: run argument overrides the init setting when not None.
        # Inputs: self, init_return_full_result (bool), run_return_full_result (bool or None),
        #   expect_runresult (bool), all parametrized.
        # Outputs: none; asserts a RunResult with token usage 10/20 when expected, else a str.
        # Exceptions: AssertionError on failure.
        # Side effects: none.
        agent = ToolCallingAgent(
            tools=[],
            model=FakeLLMModel(),
            max_steps=1,
            return_full_result=init_return_full_result,
        )
        result = agent.run("Fake task", return_full_result=run_return_full_result)

        if expect_runresult:
            assert isinstance(result, RunResult)
            assert result.output == "This is the final answer."
            assert result.state == "success"
            assert result.token_usage == TokenUsage(input_tokens=10, output_tokens=20)
            assert isinstance(result.steps, list)
            assert result.timing.duration > 0
        else:
            assert isinstance(result, str)


class TestMultiStepAgent:
    # [CLASS DOC]
    # Test group for MultiStepAgent base behaviour: init, prompt templates, callbacks, planning, final answer,
    #   interrupts, validation and serialization.
    # Inputs: none (pytest-managed).
    # Outputs: none.
    # Exceptions: none raised directly.
    # Side effects: none.
    def test_instantiation_disables_logging_to_terminal(self):
        # [FUNCTION DOC]
        # Checks the logger level is -1 during tests (terminal logging disabled by a conftest fixture).
        # Inputs: self.
        # Outputs: none.
        # Exceptions: AssertionError on failure.
        # Side effects: none.
        fake_model = MagicMock()
        agent = DummyMultiStepAgent(tools=[], model=fake_model)
        assert agent.logger.level == -1, "logging to terminal should be disabled for testing using a fixture"

    def test_instantiation_with_prompt_templates(self, prompt_templates):
        # [FUNCTION DOC]
        # Checks custom prompt templates are stored as given.
        # Inputs: self, prompt_templates (fixture dict).
        # Outputs: none.
        # Exceptions: AssertionError on failure.
        # Side effects: none.
        agent = DummyMultiStepAgent(tools=[], model=MagicMock(), prompt_templates=prompt_templates)
        assert agent.prompt_templates == prompt_templates
        assert agent.prompt_templates["system_prompt"] == "This is a test system prompt."
        assert "managed_agent" in agent.prompt_templates
        assert agent.prompt_templates["managed_agent"]["task"] == "Task for {{name}}: {{task}}"
        assert agent.prompt_templates["managed_agent"]["report"] == "Report for {{name}}: {{final_answer}}"

    @pytest.mark.parametrize(
        "tools, expected_final_answer_tool",
        [([], FinalAnswerTool), ([CustomFinalAnswerTool()], CustomFinalAnswerTool)],
    )
    def test_instantiation_with_final_answer_tool(self, tools, expected_final_answer_tool):
        # [FUNCTION DOC]
        # Checks a final_answer tool is always present, using a custom one if provided.
        # Inputs: self, tools (list), expected_final_answer_tool (class), parametrized.
        # Outputs: none.
        # Exceptions: AssertionError on failure.
        # Side effects: none.
        agent = DummyMultiStepAgent(tools=tools, model=MagicMock())
        assert "final_answer" in agent.tools
        assert isinstance(agent.tools["final_answer"], expected_final_answer_tool)

    def test_system_prompt_property(self):
        # [FUNCTION DOC]
        # Checks system_prompt is read from initialize_system_prompt and that assigning it raises
        #   AttributeError with guidance.
        # Inputs: self.
        # Outputs: none.
        # Exceptions: AssertionError on failure.
        # Side effects: none.
        """Test that system_prompt property is read-only and calls initialize_system_prompt."""

        class SimpleAgent(MultiStepAgent):
            # [CLASS DOC]
            # HELPER: Local MultiStepAgent subclass with a fixed system prompt and a no-op step.
            # Inputs: MultiStepAgent constructor args.
            # Outputs: agent instance.
            # Exceptions: none raised directly.
            # Side effects: none.
            def initialize_system_prompt(self) -> str:
                # [FUNCTION DOC]
                # HELPER: Returns the fixed string "Test system prompt".
                # Inputs: none.
                # Outputs: str.
                # Exceptions: none raised directly.
                # Side effects: none.
                return "Test system prompt"

            def step(self, memory_step: ActionStep) -> Generator[None]:
                # [FUNCTION DOC]
                # HELPER: Generator yielding a single None.
                # Inputs: memory_step (ignored).
                # Outputs: yields None once.
                # Exceptions: none raised directly.
                # Side effects: none.
                yield None

        # Create a simple agent with mocked model
        model = MagicMock()
        agent = SimpleAgent(tools=[], model=model)

        # Test reading the property works and calls initialize_system_prompt
        assert agent.system_prompt == "Test system prompt"

        # Test setting the property raises AttributeError with correct message
        with pytest.raises(
            AttributeError,
            match=re.escape(
                """The 'system_prompt' property is read-only. Use 'self.prompt_templates["system_prompt"]' instead."""
            ),
        ):
            agent.system_prompt = "New system prompt"

        # assert "read-only" in str(exc_info.value)
        # assert "Use 'self.prompt_templates[\"system_prompt\"]' instead" in str(exc_info.value)

    @pytest.mark.parametrize(
        "step_callbacks, expected_registry_state",
        [
            # Case 0: None as input (initializes empty registry)
            (
                None,
                {
                    "MemoryStep": 0,
                    "ActionStep": 1,
                    "PlanningStep": 0,
                    "TaskStep": 0,
                    "SystemPromptStep": 0,
                    "FinalAnswerStep": 0,
                },  # Only monitor.update_metrics is registered for ActionStep
            ),
            # Case 1: List of callbacks (registers only for ActionStep: backward compatibility)
            (
                [MagicMock(), MagicMock()],
                {
                    "MemoryStep": 0,
                    "ActionStep": 3,
                    "PlanningStep": 0,
                    "TaskStep": 0,
                    "SystemPromptStep": 0,
                    "FinalAnswerStep": 0,
                },
            ),
            # Case 2: Dict mapping specific step types to callbacks
            (
                {ActionStep: MagicMock(), PlanningStep: MagicMock()},
                {
                    "MemoryStep": 0,
                    "ActionStep": 2,
                    "PlanningStep": 1,
                    "TaskStep": 0,
                    "SystemPromptStep": 0,
                    "FinalAnswerStep": 0,
                },
            ),
            # Case 3: Dict with list of callbacks for a step type
            (
                {ActionStep: [MagicMock(), MagicMock()]},
                {
                    "MemoryStep": 0,
                    "ActionStep": 3,
                    "PlanningStep": 0,
                    "TaskStep": 0,
                    "SystemPromptStep": 0,
                    "FinalAnswerStep": 0,
                },
            ),
            # Case 4: Dict with mixed single and list callbacks
            (
                {ActionStep: MagicMock(), MemoryStep: [MagicMock(), MagicMock()]},
                {
                    "MemoryStep": 2,
                    "ActionStep": 2,
                    "PlanningStep": 0,
                    "TaskStep": 0,
                    "SystemPromptStep": 0,
                    "FinalAnswerStep": 0,
                },
            ),
        ],
    )
    def test_setup_step_callbacks(self, step_callbacks, expected_registry_state):
        # [FUNCTION DOC]
        # Checks _setup_step_callbacks builds a CallbackRegistry with the expected number of callbacks per
        #   step type (monitor.update_metrics always added for ActionStep).
        # Inputs: self, step_callbacks (None, list or dict), expected_registry_state (dict of step type name
        #   to count), parametrized.
        # Outputs: none.
        # Exceptions: AssertionError on failure.
        # Side effects: replaces agent.monitor with a MagicMock; reads the private _callbacks of the registry.
        """Test that _setup_step_callbacks correctly sets up the callback registry."""
        # Create a dummy agent
        agent = DummyMultiStepAgent(tools=[], model=MagicMock())
        # Mock the monitor
        agent.monitor = MagicMock()

        # Call the method
        agent._setup_step_callbacks(step_callbacks)

        # Check that step_callbacks is a CallbackRegistry
        assert isinstance(agent.step_callbacks, CallbackRegistry)

        # Count callbacks for each step type
        actual_registry_state = {}
        for step_type in [MemoryStep, ActionStep, PlanningStep, TaskStep, SystemPromptStep, FinalAnswerStep]:
            callbacks = agent.step_callbacks._callbacks.get(step_type, [])
            actual_registry_state[step_type.__name__] = len(callbacks)

        # Verify registry state matches expected
        assert actual_registry_state == expected_registry_state

    def test_finalize_step_callbacks_with_list(self):
        # [FUNCTION DOC]
        # Checks list-form step callbacks run only for ActionStep via _finalize_step.
        # Inputs: self.
        # Outputs: none; asserts both callbacks called once with the action step and agent, and not called for
        #   a PlanningStep.
        # Exceptions: AssertionError on failure.
        # Side effects: none.
        # KNOWN ISSUE: the in-code comment says callbacks are "called again with the planning step" but the
        #   assertions check they are not called.
        # Create mock callbacks
        callback1 = MagicMock()
        callback2 = MagicMock()

        # Create a test agent with a list of callbacks
        agent = DummyMultiStepAgent(tools=[], model=MagicMock(), step_callbacks=[callback1, callback2])

        # Create steps of different types
        action_step = ActionStep(step_number=1, timing=Timing(start_time=0.0))
        planning_step = PlanningStep(
            timing=Timing(start_time=1.0),
            model_input_messages=[],
            model_output_message=ChatMessage(role="assistant", content="Test plan"),
            plan="Test planning step",
        )

        # Test with ActionStep
        agent._finalize_step(action_step)

        # Verify all callbacks were called
        callback1.assert_called_once_with(action_step, agent=agent)
        callback2.assert_called_once_with(action_step, agent=agent)

        # Reset mocks
        callback1.reset_mock()
        callback2.reset_mock()

        # Test with PlanningStep
        agent._finalize_step(planning_step)

        # Verify all callbacks were called again with the planning step
        callback1.assert_not_called()
        callback2.assert_not_called()

    def test_finalize_step_callbacks_by_type(self):
        # [FUNCTION DOC]
        # Checks dict-form step callbacks dispatch by step type, with MemoryStep callbacks firing for every
        #   step.
        # Inputs: self.
        # Outputs: none; asserts the right callbacks fire for ActionStep, PlanningStep and FinalAnswerStep.
        # Exceptions: AssertionError on failure.
        # Side effects: none.
        # KNOWN ISSUE: the comment before the final-answer section says "Test with PlanningStep" although a
        #   FinalAnswerStep is used.
        # Create mock callbacks for different step types
        action_step_callback = MagicMock()
        action_step_callback_2 = MagicMock()
        planning_step_callback = MagicMock()
        step_callback = MagicMock()
        final_answer_step_callback = MagicMock()

        # Register callbacks for different step types
        step_callbacks = {
            ActionStep: [action_step_callback, action_step_callback_2],
            PlanningStep: planning_step_callback,
            MemoryStep: step_callback,
            FinalAnswerStep: final_answer_step_callback,
        }
        agent = DummyMultiStepAgent(tools=[], model=MagicMock(), step_callbacks=step_callbacks)

        # Create steps of different types
        action_step = ActionStep(step_number=1, timing=Timing(start_time=0.0))
        planning_step = PlanningStep(
            timing=Timing(start_time=1.0),
            model_input_messages=[],
            model_output_message=ChatMessage(role="assistant", content="Test plan"),
            plan="Test planning step",
        )
        final_answer_step = FinalAnswerStep(output="Sample output")

        # Test with ActionStep
        agent._finalize_step(action_step)

        # Verify correct callbacks were called
        action_step_callback.assert_called_once_with(action_step, agent=agent)
        action_step_callback_2.assert_called_once_with(action_step, agent=agent)
        step_callback.assert_called_once_with(action_step, agent=agent)
        planning_step_callback.assert_not_called()
        final_answer_step_callback.assert_not_called()

        # Reset mocks
        action_step_callback.reset_mock()
        action_step_callback_2.reset_mock()
        planning_step_callback.reset_mock()
        step_callback.reset_mock()
        final_answer_step_callback.reset_mock()

        # Test with PlanningStep
        agent._finalize_step(planning_step)

        # Verify correct callbacks were called
        planning_step_callback.assert_called_once_with(planning_step, agent=agent)
        step_callback.assert_called_once_with(planning_step, agent=agent)
        action_step_callback.assert_not_called()
        action_step_callback_2.assert_not_called()
        final_answer_step_callback.assert_not_called()

        # Reset mocks
        action_step_callback.reset_mock()
        action_step_callback_2.reset_mock()
        planning_step_callback.reset_mock()
        step_callback.reset_mock()
        final_answer_step_callback.reset_mock()

        # Test with PlanningStep
        agent._finalize_step(final_answer_step)

        # Verify correct callbacks were called
        planning_step_callback.assert_not_called()
        step_callback.assert_called_once_with(final_answer_step, agent=agent)
        action_step_callback.assert_not_called()
        action_step_callback_2.assert_not_called()
        final_answer_step_callback.assert_called_once_with(final_answer_step, agent=agent)

    def test_logs_display_thoughts_even_if_error(self):
        # [FUNCTION DOC]
        # Checks model output text is logged even when no tool call or code is produced, for both agent types.
        # Inputs: self.
        # Outputs: none; asserts "don't" and "want" appear in captured console output.
        # Exceptions: AssertionError on failure.
        # Side effects: captures console output.
        class FakeJsonModelNoCall(Model):
            # [CLASS DOC]
            # HELPER: Local fake Model that returns text without tool calls.
            # Inputs: none.
            # Outputs: see generate.
            # Exceptions: none raised directly.
            # Side effects: none.
            def generate(self, messages, stop_sequences=None, tools_to_call_from=None):
                # [FUNCTION DOC]
                # HELPER: Returns "I don't want to call tools today" with no tool calls.
                # Inputs: messages, stop_sequences, tools_to_call_from (ignored).
                # Outputs: assistant ChatMessage with that content and raw text.
                # Exceptions: none raised directly.
                # Side effects: none.
                return ChatMessage(
                    role=MessageRole.ASSISTANT,
                    content="""I don't want to call tools today""",
                    tool_calls=None,
                    raw="""I don't want to call tools today""",
                )

        agent_toolcalling = ToolCallingAgent(model=FakeJsonModelNoCall(), tools=[], max_steps=1, verbosity_level=10)
        with agent_toolcalling.logger.console.capture() as capture:
            agent_toolcalling.run("Dummy task")
        assert "don't" in capture.get() and "want" in capture.get()

        class FakeCodeModelNoCall(Model):
            # [CLASS DOC]
            # HELPER: Local fake Model that returns text without a code block.
            # Inputs: none.
            # Outputs: see generate.
            # Exceptions: none raised directly.
            # Side effects: none.
            def generate(self, messages, stop_sequences=None):
                # [FUNCTION DOC]
                # HELPER: Returns "I don't want to write an action today".
                # Inputs: messages, stop_sequences (ignored).
                # Outputs: assistant ChatMessage.
                # Exceptions: none raised directly.
                # Side effects: none.
                return ChatMessage(
                    role=MessageRole.ASSISTANT,
                    content="""I don't want to write an action today""",
                )

        agent_code = CodeAgent(model=FakeCodeModelNoCall(), tools=[], max_steps=1, verbosity_level=10)
        with agent_code.logger.console.capture() as capture:
            agent_code.run("Dummy task")
        assert "don't" in capture.get() and "want" in capture.get()

    def test_step_number(self):
        # [FUNCTION DOC]
        # Checks step_number starts at 0 and equals max_steps + 1 after a run that never answers.
        # Inputs: self.
        # Outputs: none.
        # Exceptions: AssertionError on failure.
        # Side effects: none.
        fake_model = MagicMock()
        fake_model.generate.return_value = ChatMessage(
            role=MessageRole.ASSISTANT,
            content="Model output.",
            tool_calls=None,
            raw="Model output.",
            token_usage=None,
        )
        max_steps = 2
        agent = CodeAgent(tools=[], model=fake_model, max_steps=max_steps)
        assert hasattr(agent, "step_number"), "step_number attribute should be defined"
        assert agent.step_number == 0, "step_number should be initialized to 0"
        agent.run("Test task")
        assert hasattr(agent, "step_number"), "step_number attribute should be defined"
        assert agent.step_number == max_steps + 1, "step_number should be max_steps + 1 after run method is called"

    @pytest.mark.parametrize(
        "step, expected_messages_list",
        [
            (
                1,
                [
                    [
                        ChatMessage(
                            role=MessageRole.USER, content=[{"type": "text", "text": "INITIAL_PLAN_USER_PROMPT"}]
                        ),
                    ],
                ],
            ),
            (
                2,
                [
                    [
                        ChatMessage(
                            role=MessageRole.SYSTEM,
                            content=[{"type": "text", "text": "UPDATE_PLAN_SYSTEM_PROMPT"}],
                        ),
                        ChatMessage(
                            role=MessageRole.USER,
                            content=[{"type": "text", "text": "UPDATE_PLAN_USER_PROMPT"}],
                        ),
                    ],
                ],
            ),
        ],
    )
    def test_planning_step(self, step, expected_messages_list):
        # [FUNCTION DOC]
        # Checks _generate_planning_step builds the right model input messages for the initial plan (step 1)
        #   and plan updates (step 2), and calls the model once.
        # Inputs: self, step (int), expected_messages_list (list of lists of ChatMessage with placeholder
        #   texts), parametrized.
        # Outputs: none.
        # Exceptions: AssertionError on failure.
        # Side effects: mutates the parametrized expected ChatMessage contents in place, replacing
        #   placeholders with rendered templates.
        fake_model = MagicMock()
        agent = CodeAgent(
            tools=[],
            model=fake_model,
        )
        task = "Test task"

        planning_step = list(agent._generate_planning_step(task, is_first_step=(step == 1), step=step))[-1]
        expected_message_texts = {
            "INITIAL_PLAN_USER_PROMPT": populate_template(
                agent.prompt_templates["planning"]["initial_plan"],
                variables=dict(
                    task=task,
                    tools=agent.tools,
                    managed_agents=agent.managed_agents,
                    answer_facts=planning_step.model_output_message.content,
                ),
            ),
            "UPDATE_PLAN_SYSTEM_PROMPT": populate_template(
                agent.prompt_templates["planning"]["update_plan_pre_messages"], variables=dict(task=task)
            ),
            "UPDATE_PLAN_USER_PROMPT": populate_template(
                agent.prompt_templates["planning"]["update_plan_post_messages"],
                variables=dict(
                    task=task,
                    tools=agent.tools,
                    managed_agents=agent.managed_agents,
                    facts_update=planning_step.model_output_message.content,
                    remaining_steps=agent.max_steps - step,
                ),
            ),
        }
        for expected_messages in expected_messages_list:
            for expected_message in expected_messages:
                expected_message.content[0]["text"] = expected_message_texts[expected_message.content[0]["text"]]
        assert isinstance(planning_step, PlanningStep)
        expected_model_input_messages = expected_messages_list[0]
        model_input_messages = planning_step.model_input_messages
        assert isinstance(model_input_messages, list)
        assert len(model_input_messages) == len(expected_model_input_messages)  # 2
        for message, expected_message in zip(model_input_messages, expected_model_input_messages):
            assert isinstance(message, ChatMessage)
            assert message.role in MessageRole.__members__.values()
            assert message.role == expected_message.role
            assert isinstance(message.content, list)
            for content, expected_content in zip(message.content, expected_message.content):
                assert content == expected_content
        # Test calls to model
        assert len(fake_model.generate.call_args_list) == 1
        for call_args, expected_messages in zip(fake_model.generate.call_args_list, expected_messages_list):
            assert len(call_args.args) == 1
            messages = call_args.args[0]
            assert isinstance(messages, list)
            assert len(messages) == len(expected_messages)
            for message, expected_message in zip(messages, expected_messages):
                assert isinstance(message, ChatMessage)
                assert message.role in MessageRole.__members__.values()
                assert message.role == expected_message.role
                assert isinstance(message.content, list)
                for content, expected_content in zip(message.content, expected_message.content):
                    assert content == expected_content

    @pytest.mark.parametrize(
        "expected_messages_list",
        [
            [
                [
                    ChatMessage(
                        role=MessageRole.SYSTEM,
                        content=[{"type": "text", "text": "FINAL_ANSWER_SYSTEM_PROMPT"}],
                    ),
                    ChatMessage(
                        role=MessageRole.USER,
                        content=[{"type": "text", "text": "FINAL_ANSWER_USER_PROMPT"}],
                    ),
                ]
            ],
            [
                [
                    ChatMessage(
                        role=MessageRole.SYSTEM,
                        content=[
                            {"type": "text", "text": "FINAL_ANSWER_SYSTEM_PROMPT"},
                            {"type": "image", "image": "image1.png"},
                        ],
                    ),
                    ChatMessage(
                        role=MessageRole.USER,
                        content=[{"type": "text", "text": "FINAL_ANSWER_USER_PROMPT"}],
                    ),
                ]
            ],
        ],
    )
    def test_provide_final_answer(self, expected_messages_list):
        # [FUNCTION DOC]
        # Checks provide_final_answer sends the final-answer pre/post messages to the model and returns its
        #   content.
        # Inputs: self, expected_messages_list (parametrized; second case adds an image item to the system
        #   message).
        # Outputs: none.
        # Exceptions: AssertionError on failure.
        # Side effects: mutates the parametrized expected message texts in place.
        # KNOWN ISSUE: the image case cannot really be verified since no images are given to the agent; the
        #   comparison uses zip over contents, so the extra expected image item is silently ignored.
        fake_model = MagicMock()
        fake_model.generate.return_value = ChatMessage(
            role=MessageRole.ASSISTANT,
            content="Final answer.",
            tool_calls=None,
            raw="Final answer.",
            token_usage=None,
        )
        agent = CodeAgent(
            tools=[],
            model=fake_model,
        )
        task = "Test task"
        final_answer = agent.provide_final_answer(task).content
        expected_message_texts = {
            "FINAL_ANSWER_SYSTEM_PROMPT": agent.prompt_templates["final_answer"]["pre_messages"],
            "FINAL_ANSWER_USER_PROMPT": populate_template(
                agent.prompt_templates["final_answer"]["post_messages"], variables=dict(task=task)
            ),
        }
        for expected_messages in expected_messages_list:
            for expected_message in expected_messages:
                for expected_content in expected_message.content:
                    if "text" in expected_content:
                        expected_content["text"] = expected_message_texts[expected_content["text"]]
        assert final_answer == "Final answer."
        # Test calls to model
        assert len(fake_model.generate.call_args_list) == 1
        for call_args, expected_messages in zip(fake_model.generate.call_args_list, expected_messages_list):
            assert len(call_args.args) == 1
            messages = call_args.args[0]
            assert isinstance(messages, list)
            assert len(messages) == len(expected_messages)
            for message, expected_message in zip(messages, expected_messages):
                assert isinstance(message, ChatMessage)
                assert message.role in MessageRole.__members__.values()
                assert message.role == expected_message.role
                assert isinstance(message.content, list)
                for content, expected_content in zip(message.content, expected_message.content):
                    assert content == expected_content

    def test_interrupt(self):
        # [FUNCTION DOC]
        # Checks calling agent.interrupt from a step callback aborts run with AgentError "Agent interrupted".
        # Inputs: self.
        # Outputs: none.
        # Exceptions: AssertionError on failure.
        # Side effects: none.
        fake_model = MagicMock()
        fake_model.generate.return_value = ChatMessage(
            role=MessageRole.ASSISTANT,
            content="Model output.",
            tool_calls=None,
            raw="Model output.",
            token_usage=None,
        )

        def interrupt_callback(memory_step, agent):
            # [FUNCTION DOC]
            # HELPER: Step callback that interrupts the agent.
            # Inputs: memory_step (ignored), agent (agent instance).
            # Outputs: None.
            # Exceptions: none raised directly.
            # Side effects: sets the agent's interrupt flag.
            agent.interrupt()

        agent = CodeAgent(
            tools=[],
            model=fake_model,
            step_callbacks=[interrupt_callback],
        )
        with pytest.raises(AgentError) as e:
            agent.run("Test task")
        assert "Agent interrupted" in str(e)

    @pytest.mark.parametrize(
        "tools, managed_agents, name, expectation",
        [
            # Valid case: no duplicates
            (
                [MockTool("tool1"), MockTool("tool2")],
                [MockAgent("agent1", [MockTool("tool3")])],
                "test_agent",
                does_not_raise(),
            ),
            # Invalid case: duplicate tool names
            ([MockTool("tool1"), MockTool("tool1")], [], "test_agent", pytest.raises(ValueError)),
            # Invalid case: tool name same as managed agent name
            (
                [MockTool("tool1")],
                [MockAgent("tool1", [MockTool("final_answer")])],
                "test_agent",
                pytest.raises(ValueError),
            ),
            # Valid case: tool name same as managed agent's tool name
            ([MockTool("tool1")], [MockAgent("agent1", [MockTool("tool1")])], "test_agent", does_not_raise()),
            # Invalid case: duplicate managed agent name and managed agent tool name
            ([MockTool("tool1")], [], "tool1", pytest.raises(ValueError)),
            # Valid case: duplicate tool names across managed agents
            (
                [MockTool("tool1")],
                [
                    MockAgent("agent1", [MockTool("tool2"), MockTool("final_answer")]),
                    MockAgent("agent2", [MockTool("tool2"), MockTool("final_answer")]),
                ],
                "test_agent",
                does_not_raise(),
            ),
        ],
    )
    def test_validate_tools_and_managed_agents(self, tools, managed_agents, name, expectation):
        # [FUNCTION DOC]
        # Checks name uniqueness validation across tools, managed agents and the agent name.
        # Inputs: self, tools (list of MockTool), managed_agents (list of MockAgent), name (str), expectation
        #   (context manager: no raise or raises ValueError), parametrized.
        # Outputs: none.
        # Exceptions: AssertionError/ValueError mismatch fails the test.
        # Side effects: none.
        # KNOWN ISSUE: the case comment "duplicate managed agent name and managed agent tool name" actually
        #   tests the agent name clashing with a tool name.
        fake_model = MagicMock()
        with expectation:
            DummyMultiStepAgent(
                tools=tools,
                model=fake_model,
                name=name,
                managed_agents=managed_agents,
            )

    def test_from_dict(self):
        # [FUNCTION DOC]
        # Checks MultiStepAgent.from_dict rebuilds model (via patched MODEL_REGISTRY), tools from code,
        #   settings, and that kwargs override dict values.
        # Inputs: self.
        # Outputs: none.
        # Exceptions: AssertionError on failure.
        # Side effects: temporarily patches smolagents.models.MODEL_REGISTRY; executes tool source code from
        #   the dict.
        # Create a test agent dictionary
        agent_dict = {
            "model": {"class": "TransformersModel", "data": {"model_id": "test/model"}},
            "tools": [
                {
                    "name": "valid_tool_function",
                    "code": 'from smolagents import Tool\nfrom typing import Any, Optional\n\nclass SimpleTool(Tool):\n    name = "valid_tool_function"\n    description = "A valid tool function."\n    inputs = {"input":{"type":"string","description":"Input string."}}\n    output_type = "string"\n\n    def forward(self, input: str) -> str:\n        """A valid tool function.\n\n        Args:\n            input (str): Input string.\n        """\n        return input.upper()',
                    "requirements": {"smolagents"},
                }
            ],
            "managed_agents": {},
            "prompt_templates": EMPTY_PROMPT_TEMPLATES,
            "max_steps": 15,
            "verbosity_level": 2,
            "planning_interval": 3,
            "name": "test_agent",
            "description": "Test agent description",
        }

        # Call from_dict: mock the MODEL_REGISTRY to return a mock model class
        mock_model_class = MagicMock()
        mock_model_instance = MagicMock()
        mock_model_class.from_dict.return_value = mock_model_instance

        with patch.dict("smolagents.models.MODEL_REGISTRY", {"TransformersModel": mock_model_class}):
            agent = DummyMultiStepAgent.from_dict(agent_dict)

        # Verify the agent was created correctly
        assert agent.model == mock_model_instance
        assert mock_model_class.from_dict.call_args.args[0] == {"model_id": "test/model"}
        assert agent.max_steps == 15
        assert agent.logger.level == 2
        assert agent.planning_interval == 3
        assert agent.name == "test_agent"
        assert agent.description == "Test agent description"
        # Verify the tool was created correctly
        assert sorted(agent.tools.keys()) == ["final_answer", "valid_tool_function"]
        assert agent.tools["valid_tool_function"].name == "valid_tool_function"
        assert agent.tools["valid_tool_function"].description == "A valid tool function."
        assert agent.tools["valid_tool_function"].inputs == {
            "input": {"type": "string", "description": "Input string."}
        }
        assert agent.tools["valid_tool_function"]("test") == "TEST"

        # Test overriding with kwargs
        with patch.dict("smolagents.models.MODEL_REGISTRY", {"TransformersModel": mock_model_class}):
            agent = DummyMultiStepAgent.from_dict(agent_dict, max_steps=30)
        assert agent.max_steps == 30

    def test_multiagent_to_dict_from_dict_roundtrip(self):
        # [FUNCTION DOC]
        # Checks to_dict serializes managed agents and from_dict recreates them with names, descriptions and
        #   max_steps.
        # Inputs: self.
        # Outputs: none.
        # Exceptions: AssertionError on failure.
        # Side effects: temporarily patches MODEL_REGISTRY with a "MagicMock" entry.
        """Test that to_dict() and from_dict() work correctly for agents with managed agents."""
        # Create a managed agent
        managed_agent = CodeAgent(
            tools=[], model=MagicMock(), name="managed_agent", description="A managed agent for testing", max_steps=5
        )

        # Create a main agent with the managed agent
        main_agent = ToolCallingAgent(
            tools=[],
            managed_agents=[managed_agent],
            model=MagicMock(),
            name="main_agent",
            description="Main agent with managed agents",
            max_steps=10,
        )

        # Convert to dict
        agent_dict = main_agent.to_dict()

        # Verify managed_agents structure in dict
        assert "managed_agents" in agent_dict
        assert isinstance(agent_dict["managed_agents"], list)
        assert len(agent_dict["managed_agents"]) == 1

        managed_agent_dict = agent_dict["managed_agents"][0]
        assert managed_agent_dict["name"] == "managed_agent"
        assert managed_agent_dict["class"] == "CodeAgent"
        assert managed_agent_dict["description"] == "A managed agent for testing"
        assert managed_agent_dict["max_steps"] == 5

        # Test round-trip: from_dict should recreate the agent
        # Mock the model class for the test
        mock_model_class = MagicMock()
        mock_model_instance = MagicMock()
        mock_model_class.from_dict.return_value = mock_model_instance

        with patch.dict("smolagents.models.MODEL_REGISTRY", {"MagicMock": mock_model_class}):
            recreated_agent = ToolCallingAgent.from_dict(agent_dict)

        # Verify the recreated agent has the same structure
        assert recreated_agent.name == "main_agent"
        assert recreated_agent.description == "Main agent with managed agents"
        assert recreated_agent.max_steps == 10
        assert len(recreated_agent.managed_agents) == 1

        recreated_managed_agent = list(recreated_agent.managed_agents.values())[0]
        assert recreated_managed_agent.name == "managed_agent"
        assert recreated_managed_agent.description == "A managed agent for testing"
        assert recreated_managed_agent.max_steps == 5

    def test_from_dict_invalid_model_class(self):
        # [FUNCTION DOC]
        # Checks from_dict raises ValueError naming an unknown model class and listing supported models.
        # Inputs: self.
        # Outputs: none.
        # Exceptions: AssertionError on failure.
        # Side effects: none.
        """Test that from_dict raises ValueError with helpful message for invalid model class."""
        agent_dict = {
            "class": "CodeAgent",
            "model": {"class": "InvalidModelClass", "data": {}},
            "tools": [],
            "managed_agents": [],
        }

        with pytest.raises(ValueError) as exc_info:
            CodeAgent.from_dict(agent_dict)

        error_message = str(exc_info.value)
        assert "InvalidModelClass" in error_message
        assert "Unknown model class" in error_message
        assert "Supported models:" in error_message

    def test_from_dict_invalid_agent_class(self):
        # [FUNCTION DOC]
        # Checks from_dict raises ValueError naming an unknown managed agent class and listing supported
        #   agents.
        # Inputs: self.
        # Outputs: none.
        # Exceptions: AssertionError on failure.
        # Side effects: temporarily patches MODEL_REGISTRY.
        """Test that from_dict raises ValueError with helpful message for invalid agent class."""
        # Create a valid agent first
        agent = CodeAgent(tools=[], model=MagicMock(), name="test_agent")
        agent_dict = agent.to_dict()

        # Add a managed agent with invalid class
        agent_dict["managed_agents"] = [
            {
                "class": "InvalidAgentClass",
                "model": {"class": "MagicMock", "data": {}},
                "tools": [],
                "managed_agents": [],
            }
        ]

        # Mock the model registry to allow the main agent's model and managed agent's model
        mock_model_class = MagicMock()
        mock_model_instance = MagicMock()
        mock_model_class.from_dict.return_value = mock_model_instance

        with patch.dict("smolagents.models.MODEL_REGISTRY", {"MagicMock": mock_model_class}):
            with pytest.raises(ValueError) as exc_info:
                CodeAgent.from_dict(agent_dict)

            error_message = str(exc_info.value)
            assert "InvalidAgentClass" in error_message
            assert "Unknown agent class" in error_message
            assert "Supported agents:" in error_message


class TestToolCallingAgent:
    # [CLASS DOC]
    # Test group for ToolCallingAgent-specific behaviour.
    # Inputs: none (pytest-managed); methods use the test_tool fixture.
    # Outputs: none.
    # Exceptions: none raised directly.
    # Side effects: none.
    def test_toolcalling_agent_instructions(self):
        # [FUNCTION DOC]
        # Checks instructions are stored and rendered into the ToolCallingAgent system prompt.
        # Inputs: self.
        # Outputs: none.
        # Exceptions: AssertionError on failure.
        # Side effects: none.
        agent = ToolCallingAgent(tools=[], model=MagicMock(), instructions="Test instructions")
        assert agent.instructions == "Test instructions"
        assert "Test instructions" in agent.system_prompt

    def test_toolcalling_agent_passes_both_tools_and_managed_agents(self, test_tool):
        # [FUNCTION DOC]
        # Checks the model receives both regular tools and managed agents (plus final_answer) in
        #   tools_to_call_from.
        # Inputs: self, test_tool (fixture tool named test_tool).
        # Outputs: none.
        # Exceptions: AssertionError on failure.
        # Side effects: advances a streaming run by one event only (generator left unfinished).
        """Test that both tools and managed agents are passed to the model."""
        managed_agent = MagicMock()
        managed_agent.name = "managed_agent"
        model = MagicMock()
        model.generate.return_value = ChatMessage(
            role=MessageRole.ASSISTANT,
            content="",
            tool_calls=[
                ChatMessageToolCall(
                    id="call_0",
                    type="function",
                    function=ChatMessageToolCallFunction(name="test_tool", arguments={"input": "test_value"}),
                )
            ],
        )
        agent = ToolCallingAgent(tools=[test_tool], managed_agents=[managed_agent], model=model)
        # Run the agent one step to trigger the model call
        next(agent.run("Test task", stream=True))
        # Check that the model was called with both tools and managed agents:
        # - Get all tool_to_call_from names passed to the model
        tools_to_call_from_names = [tool.name for tool in model.generate.call_args.kwargs["tools_to_call_from"]]
        # - Verify both regular tools and managed agents are included
        assert "test_tool" in tools_to_call_from_names  # The regular tool
        assert "managed_agent" in tools_to_call_from_names  # The managed agent
        assert "final_answer" in tools_to_call_from_names  # The final_answer tool (added by default)

    @patch("huggingface_hub.InferenceClient")
    def test_toolcalling_agent_api(self, mock_inference_client):
        # [FUNCTION DOC]
        # Checks ToolCallingAgent with InferenceClientModel parses tool calls from JSON content and from
        #   native tool_calls.
        # Inputs: self, mock_inference_client (patched huggingface_hub.InferenceClient).
        # Outputs: none; asserts tool name, parsed arguments and observations in both modes.
        # Exceptions: AssertionError on failure.
        # Side effects: none (client is mocked).
        mock_client = mock_inference_client.return_value
        mock_response = mock_client.chat_completion.return_value
        mock_response.choices[0].message = ChatCompletionOutputMessage(
            role=MessageRole.ASSISTANT,
            content='{"name": "weather_api", "arguments": {"location": "Paris", "date": "today"}}',
        )
        mock_response.usage.prompt_tokens = 10
        mock_response.usage.completion_tokens = 20

        model = InferenceClientModel(model_id="test-model")

        from smolagents import tool

        @tool
        def weather_api(location: str, date: str) -> str:
            # [FUNCTION DOC]
            # HELPER: Nested @tool returning a sunny weather string.
            # Inputs: location (str), date (str).
            # Outputs: formatted string naming location and date.
            # Exceptions: none raised directly.
            # Side effects: none.
            """
            Gets the weather in the next days at given location.
            Args:
                location: the location
                date: the date
            """
            return f"The weather in {location} on date:{date} is sunny."

        agent = ToolCallingAgent(model=model, tools=[weather_api], max_steps=1)
        agent.run("What's the weather in Paris?")
        assert agent.memory.steps[0].task == "What's the weather in Paris?"
        assert agent.memory.steps[1].tool_calls[0].name == "weather_api"
        assert agent.memory.steps[1].tool_calls[0].arguments == {"location": "Paris", "date": "today"}
        assert agent.memory.steps[1].observations == "The weather in Paris on date:today is sunny."

        mock_response.choices[0].message = ChatCompletionOutputMessage(
            role=MessageRole.ASSISTANT,
            content=None,
            tool_calls=[
                ChatCompletionOutputToolCall(
                    function=ChatCompletionOutputFunctionDefinition(
                        name="weather_api", arguments='{"location": "Paris", "date": "today"}'
                    ),
                    id="call_0",
                    type="function",
                )
            ],
        )

        agent.run("What's the weather in Paris?")
        assert agent.memory.steps[0].task == "What's the weather in Paris?"
        assert agent.memory.steps[1].tool_calls[0].name == "weather_api"
        assert agent.memory.steps[1].tool_calls[0].arguments == {"location": "Paris", "date": "today"}
        assert agent.memory.steps[1].observations == "The weather in Paris on date:today is sunny."

    @patch("openai.OpenAI")
    def test_toolcalling_agent_stream_logs_multiple_tool_calls_observations(self, mock_openai_client, test_tool):
        # [FUNCTION DOC]
        # Checks streamed tool-call deltas from OpenAIModel are assembled into two tool calls whose
        #   observations are both logged.
        # Inputs: self, mock_openai_client (patched openai.OpenAI), test_tool (fixture).
        # Outputs: none; asserts two test_tool calls and observations "Processed: output1" and "Processed:
        #   output2" joined by newline.
        # Exceptions: AssertionError on failure.
        # Side effects: none (client mocked).
        # KNOWN ISSUE: mock_usage is built but never used.
        """Test that ToolCallingAgent with stream_outputs=True logs the observations of all tool calls when multiple are called."""
        mock_client = mock_openai_client.return_value
        from smolagents import OpenAIModel

        # Mock streaming response with multiple tool calls
        mock_deltas = [
            ChoiceDelta(role=MessageRole.ASSISTANT),
            ChoiceDelta(
                tool_calls=[
                    ChoiceDeltaToolCall(
                        index=0,
                        id="call_1",
                        function=ChoiceDeltaToolCallFunction(name="test_tool"),
                        type="function",
                    )
                ]
            ),
            ChoiceDelta(
                tool_calls=[ChoiceDeltaToolCall(index=0, function=ChoiceDeltaToolCallFunction(arguments='{"in'))]
            ),
            ChoiceDelta(
                tool_calls=[ChoiceDeltaToolCall(index=0, function=ChoiceDeltaToolCallFunction(arguments='put"'))]
            ),
            ChoiceDelta(
                tool_calls=[ChoiceDeltaToolCall(index=0, function=ChoiceDeltaToolCallFunction(arguments=': "out'))]
            ),
            ChoiceDelta(
                tool_calls=[ChoiceDeltaToolCall(index=0, function=ChoiceDeltaToolCallFunction(arguments="put1"))]
            ),
            ChoiceDelta(
                tool_calls=[ChoiceDeltaToolCall(index=0, function=ChoiceDeltaToolCallFunction(arguments='"}'))]
            ),
            ChoiceDelta(
                tool_calls=[
                    ChoiceDeltaToolCall(
                        index=1,
                        id="call_2",
                        function=ChoiceDeltaToolCallFunction(name="test_tool"),
                        type="function",
                    )
                ]
            ),
            ChoiceDelta(
                tool_calls=[ChoiceDeltaToolCall(index=1, function=ChoiceDeltaToolCallFunction(arguments='{"in'))]
            ),
            ChoiceDelta(
                tool_calls=[ChoiceDeltaToolCall(index=1, function=ChoiceDeltaToolCallFunction(arguments='put"'))]
            ),
            ChoiceDelta(
                tool_calls=[ChoiceDeltaToolCall(index=1, function=ChoiceDeltaToolCallFunction(arguments=': "out'))]
            ),
            ChoiceDelta(
                tool_calls=[ChoiceDeltaToolCall(index=1, function=ChoiceDeltaToolCallFunction(arguments="put2"))]
            ),
            ChoiceDelta(
                tool_calls=[ChoiceDeltaToolCall(index=1, function=ChoiceDeltaToolCallFunction(arguments='"}'))]
            ),
        ]

        class MockChoice:
            # [CLASS DOC]
            # HELPER: Mock streaming choice wrapping a delta.
            # Inputs: delta (ChoiceDelta).
            # Outputs: object with delta attribute.
            # Exceptions: none raised directly.
            # Side effects: none.
            def __init__(self, delta):
                # [FUNCTION DOC]
                # HELPER: Stores the delta.
                # Inputs: delta.
                # Outputs: none.
                # Exceptions: none raised directly.
                # Side effects: mutates self.
                self.delta = delta

        class MockChunk:
            # [CLASS DOC]
            # HELPER: Mock streaming chunk holding one choice and no usage.
            # Inputs: delta (ChoiceDelta).
            # Outputs: object with choices list and usage None.
            # Exceptions: none raised directly.
            # Side effects: none.
            def __init__(self, delta):
                # [FUNCTION DOC]
                # HELPER: Wraps delta in a MockChoice list and sets usage to None.
                # Inputs: delta.
                # Outputs: none.
                # Exceptions: none raised directly.
                # Side effects: mutates self.
                self.choices = [MockChoice(delta)]
                self.usage = None

        mock_client.chat.completions.create.return_value = (MockChunk(delta) for delta in mock_deltas)

        # Mock usage for non-streaming fallback
        mock_usage = MagicMock()
        mock_usage.prompt_tokens = 10
        mock_usage.completion_tokens = 20

        model = OpenAIModel(model_id="fakemodel")

        agent = ToolCallingAgent(model=model, tools=[test_tool], max_steps=1, stream_outputs=True)
        agent.run("Dummy task")
        assert agent.memory.steps[1].model_output_message.tool_calls[0].function.name == "test_tool"
        assert agent.memory.steps[1].model_output_message.tool_calls[1].function.name == "test_tool"
        assert agent.memory.steps[1].observations == "Processed: output1\nProcessed: output2"

    @patch("openai.OpenAI")
    def test_toolcalling_agent_final_answer_cannot_be_called_with_parallel_tool_calls(
        self, mock_openai_client, test_tool
    ):
        # [FUNCTION DOC]
        # Checks that a final_answer tool call issued in parallel with another tool call is rejected with an
        #   error on the step.
        # Inputs: self, mock_openai_client (patched openai.OpenAI), test_tool (fixture).
        # Outputs: none; asserts step 1 error message forbids other tool calls alongside final answer.
        # Exceptions: AssertionError on failure.
        # Side effects: prints the tool calls to stdout via the mock model_dump.
        # KNOWN ISSUE: the docstring talks about stream_outputs and returning all tool calls, but the test
        #   uses non-streaming mode and checks for an error. ExtendedChatMessage is defined but unused.
        """Test that ToolCallingAgent with stream_outputs=True returns the all tool calls when multiple are called."""
        mock_client = mock_openai_client.return_value

        from smolagents import OpenAIModel

        class ExtendedChatMessage(ChatMessage):
            # [CLASS DOC]
            # HELPER: Unused ChatMessage subclass accepting a usage keyword.
            # Inputs: ChatMessage args plus required keyword usage (discarded).
            # Outputs: ChatMessage instance.
            # Exceptions: those of ChatMessage.
            # Side effects: none.
            def __init__(self, *args, usage, **kwargs):
                # [FUNCTION DOC]
                # HELPER: Calls the ChatMessage constructor, dropping usage.
                # Inputs: positional/keyword ChatMessage args, usage (ignored).
                # Outputs: none.
                # Exceptions: those of ChatMessage.__init__.
                # Side effects: mutates self.
                super().__init__(*args, **kwargs)

            def model_dump(self, include=None):
                # [FUNCTION DOC]
                # HELPER: Returns the JSON string dump of the message.
                # Inputs: include (ignored).
                # Outputs: JSON string.
                # Exceptions: those of model_dump_json.
                # Side effects: none.
                # KNOWN ISSUE: calls super().model_dump_json, which ChatMessage (a dataclass) may not provide;
                #   unused so never triggered.
                return super().model_dump_json()

        class MockChoice:
            # [CLASS DOC]
            # HELPER: Mock non-streaming choice wrapping a message.
            # Inputs: chat_message.
            # Outputs: object with message attribute.
            # Exceptions: none raised directly.
            # Side effects: none.
            def __init__(self, chat_message):
                # [FUNCTION DOC]
                # HELPER: Stores the message.
                # Inputs: chat_message.
                # Outputs: none.
                # Exceptions: none raised directly.
                # Side effects: mutates self.
                self.message = chat_message

        class MockChatCompletion:
            # [CLASS DOC]
            # HELPER: Mock OpenAI chat completion with one choice and fixed usage 10/20.
            # Inputs: chat_message.
            # Outputs: object with choices and usage.
            # Exceptions: none raised directly.
            # Side effects: none.
            def __init__(self, chat_message):
                # [FUNCTION DOC]
                # HELPER: Builds the choices list and a MockTokenUsage(10, 20).
                # Inputs: chat_message.
                # Outputs: none.
                # Exceptions: none raised directly.
                # Side effects: mutates self.
                self.choices = [MockChoice(chat_message)]
                self.usage = MockTokenUsage(prompt_tokens=10, completion_tokens=20)

        class MockTokenUsage:
            # [CLASS DOC]
            # HELPER: Mock usage object with prompt and completion token counts.
            # Inputs: prompt_tokens (int), completion_tokens (int).
            # Outputs: object holding the counts.
            # Exceptions: none raised directly.
            # Side effects: none.
            def __init__(self, prompt_tokens, completion_tokens):
                # [FUNCTION DOC]
                # HELPER: Stores the two token counts.
                # Inputs: prompt_tokens, completion_tokens.
                # Outputs: none.
                # Exceptions: none raised directly.
                # Side effects: mutates self.
                self.prompt_tokens = prompt_tokens
                self.completion_tokens = completion_tokens

        from dataclasses import asdict

        class ExtendedChatCompletionOutputMessage(ChatCompletionOutputMessage):
            # [CLASS DOC]
            # HELPER: huggingface_hub ChatCompletionOutputMessage subclass that also carries usage and dumps
            #   to a plain dict.
            # Inputs: ChatCompletionOutputMessage args plus required keyword usage.
            # Outputs: message instance.
            # Exceptions: those of the base dataclass.
            # Side effects: none.
            def __init__(self, *args, usage, **kwargs):
                # [FUNCTION DOC]
                # HELPER: Initializes the base message and stores usage.
                # Inputs: base args, usage.
                # Outputs: none.
                # Exceptions: those of the base init.
                # Side effects: mutates self.
                super().__init__(*args, **kwargs)
                self.usage = usage

            def model_dump(self, include=None):
                # [FUNCTION DOC]
                # HELPER: Returns a dict with role, content and tool_calls converted via dataclass asdict.
                # Inputs: include (ignored).
                # Outputs: dict.
                # Exceptions: TypeError if tool_calls is None or contains non-dataclasses.
                # Side effects: prints "TOOL CALLS" and the tool calls to stdout.
                print("TOOL CALLS", self.tool_calls)
                return {
                    "role": self.role,
                    "content": self.content,
                    "tool_calls": [asdict(tc) for tc in self.tool_calls],
                }

        mock_client.chat.completions.create.return_value = MockChatCompletion(
            ExtendedChatCompletionOutputMessage(
                role=MessageRole.ASSISTANT,
                content=None,
                tool_calls=[
                    ChatMessageToolCall(
                        id="call_0",
                        type="function",
                        function=ChatMessageToolCallFunction(name="test_tool", arguments={"input": "out1"}),
                    ),
                    ChatMessageToolCall(
                        id="1",
                        type="function",
                        function=ChatMessageToolCallFunction(name="final_answer", arguments={"answer": "out1"}),
                    ),
                ],
                usage=MockTokenUsage(prompt_tokens=10, completion_tokens=20),
            )
        )

        model = OpenAIModel(model_id="fakemodel")

        agent = ToolCallingAgent(model=model, tools=[test_tool], max_steps=1)
        agent.run("Dummy task")
        assert agent.memory.steps[1].error is not None
        assert (
            "do not perform any other tool calls than the final answer tool call!"
            in agent.memory.steps[1].error.message
        )

    @patch("huggingface_hub.InferenceClient")
    def test_toolcalling_agent_api_misformatted_output(self, mock_inference_client):
        # [FUNCTION DOC]
        # Checks malformed JSON tool output records a parse error on the step and the run continues to a final
        #   answer.
        # Inputs: self, mock_inference_client (patched huggingface_hub.InferenceClient).
        # Outputs: none; asserts tool_calls None, invalid-JSON error message, "Error while parsing" logged and
        #   4 memory steps.
        # Exceptions: AssertionError on failure.
        # Side effects: captures console output.
        """Test that even misformatted json blobs don't interrupt the run for a ToolCallingAgent."""
        mock_client = mock_inference_client.return_value
        mock_response = mock_client.chat_completion.return_value
        mock_response.choices[0].message = ChatCompletionOutputMessage(
            role=MessageRole.ASSISTANT,
            content='{"name": weather_api", "arguments": {"location": "Paris", "date": "today"}}',
        )

        mock_response.usage.prompt_tokens = 10
        mock_response.usage.completion_tokens = 20

        model = InferenceClientModel(model_id="test-model")

        logger = AgentLogger(console=Console(markup=False, no_color=True))

        agent = ToolCallingAgent(model=model, tools=[], max_steps=2, verbosity_level=1, logger=logger)
        with agent.logger.console.capture() as capture:
            agent.run("What's the weather in Paris?")
        assert agent.memory.steps[0].task == "What's the weather in Paris?"
        assert agent.memory.steps[1].tool_calls is None
        assert "The JSON blob you used is invalid" in agent.memory.steps[1].error.message
        assert "Error while parsing" in capture.get()
        assert len(agent.memory.steps) == 4

    @pytest.mark.skip(
        reason="Test is not properly implemented (GH-1255) because fake_tools should have the same name. "
        "Additionally, it uses CodeAgent instead of ToolCallingAgent (GH-1409)"
    )
    def test_change_tools_after_init(self):
        # [FUNCTION DOC]
        # Skipped test intended to check that replacing tools after init takes effect.
        # Inputs: self.
        # Outputs: none; would assert the answer "2CUSTOM".
        # Exceptions: AssertionError on failure (if run).
        # Side effects: mutates agent.tools.
        # KNOWN ISSUE: skipped as broken (GH-1255, GH-1409): uses CodeAgent in the ToolCallingAgent group and
        #   the replacement tool has a different name.
        from smolagents import tool

        @tool
        def fake_tool_1() -> str:
            # [FUNCTION DOC]
            # HELPER: Nested @tool returning "1".
            # Inputs: none.
            # Outputs: "1".
            # Exceptions: none raised directly.
            # Side effects: none.
            """Fake tool"""
            return "1"

        @tool
        def fake_tool_2() -> str:
            # [FUNCTION DOC]
            # HELPER: Nested @tool returning "2".
            # Inputs: none.
            # Outputs: "2".
            # Exceptions: none raised directly.
            # Side effects: none.
            """Fake tool"""
            return "2"

        class FakeCodeModel(Model):
            # [CLASS DOC]
            # HELPER: Local fake Model returning code that passes fake_tool_1() to final_answer.
            # Inputs: none.
            # Outputs: see generate.
            # Exceptions: none raised directly.
            # Side effects: none.
            def generate(self, messages, stop_sequences=None):
                # [FUNCTION DOC]
                # HELPER: Returns the fixed code block.
                # Inputs: messages, stop_sequences (ignored).
                # Outputs: assistant ChatMessage.
                # Exceptions: none raised directly.
                # Side effects: none.
                return ChatMessage(role=MessageRole.ASSISTANT, content="<code>\nfinal_answer(fake_tool_1())\n</code>")

        agent = CodeAgent(tools=[fake_tool_1], model=FakeCodeModel())

        agent.tools["final_answer"] = CustomFinalAnswerTool()
        agent.tools["fake_tool_1"] = fake_tool_2

        answer = agent.run("Fake task.")
        assert answer == "2CUSTOM"

    def test_custom_final_answer_with_custom_inputs(self, test_tool):
        # [FUNCTION DOC]
        # Checks a custom final-answer tool with two inputs is called with both arguments by ToolCallingAgent.
        # Inputs: self, test_tool (fixture).
        # Outputs: none; asserts answer "1 and 2" and last step tool call name final_answer.
        # Exceptions: AssertionError on failure.
        # Side effects: none.
        class CustomFinalAnswerToolWithCustomInputs(FinalAnswerTool):
            # [CLASS DOC]
            # HELPER: FinalAnswerTool with inputs answer1 and answer2 joining them with " and ".
            # Inputs: inherits FinalAnswerTool constructor.
            # Outputs: tool instance.
            # Exceptions: none raised directly.
            # Side effects: none.
            inputs = {
                "answer1": {"type": "string", "description": "First part of the answer."},
                "answer2": {"type": "string", "description": "Second part of the answer."},
            }

            def forward(self, answer1: str, answer2: str) -> str:
                # [FUNCTION DOC]
                # HELPER: Joins the two answers with " and ".
                # Inputs: answer1 (str), answer2 (str).
                # Outputs: combined str.
                # Exceptions: none raised directly.
                # Side effects: none.
                return answer1 + " and " + answer2

        model = MagicMock()
        model.generate.return_value = ChatMessage(
            role=MessageRole.ASSISTANT,
            content=None,
            tool_calls=[
                ChatMessageToolCall(
                    id="call_0",
                    type="function",
                    function=ChatMessageToolCallFunction(
                        name="final_answer", arguments={"answer1": "1", "answer2": "2"}
                    ),
                ),
            ],
        )
        agent = ToolCallingAgent(tools=[test_tool, CustomFinalAnswerToolWithCustomInputs()], model=model)
        answer = agent.run("Fake task.")
        assert answer == "1 and 2"
        assert agent.memory.steps[-1].model_output_message.tool_calls[0].function.name == "final_answer"

    @pytest.mark.parametrize(
        "test_case",
        [
            # Case 0: Single valid tool call
            {
                "tool_calls": [
                    ChatMessageToolCall(
                        id="call_1",
                        type="function",
                        function=ChatMessageToolCallFunction(name="test_tool", arguments={"input": "test_value"}),
                    )
                ],
                "expected_observations": "Processed: test_value",
                "expected_final_outputs": ["Processed: test_value"],
                "expected_error": None,
            },
            # Case 1: Multiple tool calls
            {
                "tool_calls": [
                    ChatMessageToolCall(
                        id="call_1",
                        type="function",
                        function=ChatMessageToolCallFunction(name="test_tool", arguments={"input": "value1"}),
                    ),
                    ChatMessageToolCall(
                        id="call_2",
                        type="function",
                        function=ChatMessageToolCallFunction(name="test_tool", arguments={"input": "value2"}),
                    ),
                ],
                "expected_observations": "Processed: value1\nProcessed: value2",
                "expected_final_outputs": ["Processed: value1", "Processed: value2"],
                "expected_error": None,
            },
            # Case 2: Invalid tool name
            {
                "tool_calls": [
                    ChatMessageToolCall(
                        id="call_1",
                        type="function",
                        function=ChatMessageToolCallFunction(name="nonexistent_tool", arguments={"input": "test"}),
                    )
                ],
                "expected_error": AgentToolExecutionError,
            },
            # Case 3: Tool execution error
            {
                "tool_calls": [
                    ChatMessageToolCall(
                        id="call_1",
                        type="function",
                        function=ChatMessageToolCallFunction(name="test_tool", arguments={"input": "error"}),
                    )
                ],
                "expected_error": AgentToolExecutionError,
            },
            # Case 4: Empty tool calls list
            {
                "tool_calls": [],
                "expected_observations": "",
                "expected_final_outputs": [],
                "expected_error": None,
            },
            # Case 5: Final answer call
            {
                "tool_calls": [
                    ChatMessageToolCall(
                        id="call_1",
                        type="function",
                        function=ChatMessageToolCallFunction(
                            name="final_answer", arguments={"answer": "This is the final answer"}
                        ),
                    )
                ],
                "expected_observations": "This is the final answer",
                "expected_final_outputs": ["This is the final answer"],
                "expected_error": None,
            },
            # Case 6: Invalid arguments
            {
                "tool_calls": [
                    ChatMessageToolCall(
                        id="call_1",
                        type="function",
                        function=ChatMessageToolCallFunction(name="test_tool", arguments={"wrong_param": "value"}),
                    )
                ],
                "expected_error": AgentToolCallError,
            },
        ],
    )
    def test_process_tool_calls(self, test_case, test_tool):
        # [FUNCTION DOC]
        # Checks process_tool_calls yields ToolOutputs, sets observations and tool_calls on the step, and
        #   raises for bad tool names, tool errors or bad arguments.
        # Inputs: self, test_case (dict with tool_calls, expected_observations, expected_final_outputs,
        #   expected_error), test_tool (fixture), parametrized.
        # Outputs: none.
        # Exceptions: AssertionError on failure.
        # Side effects: mutates the ActionStep it creates.
        # Create a ToolCallingAgent instance with the test tool
        agent = ToolCallingAgent(tools=[test_tool], model=MagicMock())
        # Create chat message with the specified tool calls for process_tool_calls
        chat_message = ChatMessage(role=MessageRole.ASSISTANT, content="", tool_calls=test_case["tool_calls"])
        # Create a memory step for process_tool_calls
        memory_step = ActionStep(step_number=10, timing="mock_timing", model_output="")

        # Process tool calls
        if test_case["expected_error"]:
            with pytest.raises(test_case["expected_error"]):
                list(agent.process_tool_calls(chat_message, memory_step))
        else:
            final_outputs = list(agent.process_tool_calls(chat_message, memory_step))
            assert memory_step.model_output == ""
            assert memory_step.observations == test_case["expected_observations"]
            assert [
                final_output.output for final_output in final_outputs if isinstance(final_output, ToolOutput)
            ] == test_case["expected_final_outputs"]
            # Verify memory step tool calls were updated correctly
            if test_case["tool_calls"]:
                assert memory_step.tool_calls == [
                    ToolCall(name=tool_call.function.name, arguments=tool_call.function.arguments, id=tool_call.id)
                    for tool_call in test_case["tool_calls"]
                ]


class TestCodeAgent:
    # [CLASS DOC]
    # Test group for CodeAgent-specific behaviour.
    # Inputs: none (pytest-managed).
    # Outputs: none.
    # Exceptions: none raised directly.
    # Side effects: none.
    def test_code_agent_instructions(self):
        # [FUNCTION DOC]
        # Checks instructions are stored and rendered into the CodeAgent system prompt, with and without
        #   structured outputs.
        # Inputs: self.
        # Outputs: none.
        # Exceptions: AssertionError on failure.
        # Side effects: none.
        agent = CodeAgent(tools=[], model=MagicMock(), instructions="Test instructions")
        assert agent.instructions == "Test instructions"
        assert "Test instructions" in agent.system_prompt

        agent = CodeAgent(
            tools=[], model=MagicMock(), instructions="Test instructions", use_structured_outputs_internally=True
        )
        assert agent.instructions == "Test instructions"
        assert "Test instructions" in agent.system_prompt

    @pytest.mark.parametrize("provide_run_summary", [False, True])
    def test_call_with_provide_run_summary(self, provide_run_summary):
        # [FUNCTION DOC]
        # Checks calling a managed agent returns the report wrapper and appends a summary of work when
        #   provide_run_summary is set.
        # Inputs: self, provide_run_summary (bool, parametrized).
        # Outputs: none.
        # Exceptions: AssertionError on failure.
        # Side effects: replaces agent.run and write_memory_to_messages with mocks.
        agent = CodeAgent(tools=[], model=MagicMock(), provide_run_summary=provide_run_summary)
        assert agent.provide_run_summary is provide_run_summary
        agent.name = "test_agent"
        agent.run = MagicMock(return_value="Test output")
        agent.write_memory_to_messages = MagicMock(
            return_value=[ChatMessage(role=MessageRole.ASSISTANT, content="Test summary")]
        )

        result = agent("Test request")
        expected_summary = "Here is the final answer from your managed agent 'test_agent':\nTest output"
        if provide_run_summary:
            expected_summary += (
                "\n\nFor more detail, find below a summary of this agent's work:\n"
                "<summary_of_work>\n\nTest summary\n---\n</summary_of_work>"
            )
        assert result == expected_summary

    def test_code_agent_image_output(self):
        # [FUNCTION DOC]
        # Checks a CodeAgent can return a PIL image produced by a tool.
        # Inputs: self.
        # Outputs: none.
        # Exceptions: AssertionError on failure.
        # Side effects: none.
        from PIL import Image

        from smolagents import tool

        @tool
        def image_generation_tool():
            # [FUNCTION DOC]
            # HELPER: Nested @tool returning a 100x100 red PIL image.
            # Inputs: none.
            # Outputs: PIL.Image.Image.
            # Exceptions: none raised directly.
            # Side effects: none.
            """Generate an image"""
            return Image.new("RGB", (100, 100), color="red")

        agent = CodeAgent(tools=[image_generation_tool], model=FakeCodeModelImageGeneration(), verbosity_level=1)
        output = agent.run("Make me an image from the latest trend on google trends.")
        assert isinstance(output, Image.Image)

    def test_errors_logging(self):
        # [FUNCTION DOC]
        # Checks error logs show square brackets literally (not interpreted as rich markup).
        # Inputs: self.
        # Outputs: none.
        # Exceptions: AssertionError on failure.
        # Side effects: captures console output.
        class FakeCodeModel(Model):
            # [CLASS DOC]
            # HELPER: Local fake Model returning code that indexes a list out of range.
            # Inputs: none.
            # Outputs: see generate.
            # Exceptions: none raised directly.
            # Side effects: none.
            def generate(self, messages, stop_sequences=None):
                # [FUNCTION DOC]
                # HELPER: Returns code setting secret to 3 and indexing a two-item list with it.
                # Inputs: messages, stop_sequences (ignored).
                # Outputs: assistant ChatMessage.
                # Exceptions: none raised directly.
                # Side effects: none.
                return ChatMessage(role=MessageRole.ASSISTANT, content="<code>\nsecret=3;['1', '2'][secret]\n</code>")

        agent = CodeAgent(tools=[], model=FakeCodeModel(), verbosity_level=1)

        with agent.logger.console.capture() as capture:
            agent.run("Test request")
        # Verify [secret] is rendered literally, not interpreted as Rich markup (which would
        # inject ANSI reset/re-apply codes like \x1b[0m\x1b[1;31m in place of the brackets).
        assert "[secret]" in capture.get()

    def test_missing_import_triggers_advice_in_error_log(self):
        # [FUNCTION DOC]
        # Checks an unauthorized import error log advises using additional_authorized_imports.
        # Inputs: self.
        # Outputs: none.
        # Exceptions: AssertionError on failure.
        # Side effects: captures console output.
        # Set explicit verbosity level to 1 to override the default verbosity level of -1 set in CI fixture
        agent = CodeAgent(tools=[], model=FakeCodeModelImport(), verbosity_level=1)

        with agent.logger.console.capture() as capture:
            agent.run("Count to 3")
        str_output = capture.get()
        assert "`additional_authorized_imports`" in str_output.replace("\n", "")

    def test_errors_show_offending_line_and_error(self):
        # [FUNCTION DOC]
        # Checks a runtime error in generated code names the offending line and ValueError, and the run
        #   recovers.
        # Inputs: self.
        # Outputs: none.
        # Exceptions: AssertionError on failure.
        # Side effects: none.
        agent = CodeAgent(tools=[PythonInterpreterTool()], model=FakeCodeModelError())
        output = agent.run("What is 2 multiplied by 3.6452?")
        assert isinstance(output, AgentText)
        assert output == "got an error"
        assert "Code execution failed at line 'error_function()'" in str(agent.memory.steps[1].error)
        assert "ValueError" in str(agent.memory.steps)

    def test_error_saves_previous_print_outputs(self):
        # [FUNCTION DOC]
        # Checks prints executed before an error are kept in the step observations.
        # Inputs: self.
        # Outputs: none.
        # Exceptions: AssertionError on failure.
        # Side effects: none.
        agent = CodeAgent(tools=[PythonInterpreterTool()], model=FakeCodeModelError())
        agent.run("What is 2 multiplied by 3.6452?")
        assert "Flag!" in str(agent.memory.steps[1].observations)

    def test_syntax_error_show_offending_lines(self):
        # [FUNCTION DOC]
        # Checks a syntax error shows the offending lines and the code_action is stored verbatim.
        # Inputs: self.
        # Outputs: none.
        # Exceptions: AssertionError on failure.
        # Side effects: none.
        agent = CodeAgent(tools=[PythonInterpreterTool()], model=FakeCodeModelSyntaxError())
        output = agent.run("What is 2 multiplied by 3.6452?")
        assert isinstance(output, AgentText)
        assert output == "got an error"
        assert '    print("Failing due to unexpected indent")' in str(agent.memory.steps)
        assert isinstance(agent.memory.steps[-2], ActionStep)
        assert agent.memory.steps[-2].code_action == dedent("""a = 2
b = a * 2
    print("Failing due to unexpected indent")
print("Ok, calculation done!")""")

    def test_end_code_appending(self):
        # [FUNCTION DOC]
        # Checks the agent normalizes model outputs so recorded model_output and message content end with the
        #   closing code tag.
        # Inputs: self.
        # Outputs: none.
        # Exceptions: AssertionError on failure.
        # Side effects: none.
        # KNOWN ISSUE: the first assertion checks for a legacy end_code marker that the fake output never
        #   contained, so it is trivially true.
        # Checking original output message
        orig_output = FakeCodeModelNoReturn().generate([])
        assert not orig_output.content.endswith("<end_code>")

        # Checking the step output
        agent = CodeAgent(
            tools=[PythonInterpreterTool()],
            model=FakeCodeModelNoReturn(),
            max_steps=1,
        )
        answer = agent.run("What is 2 multiplied by 3.6452?")
        assert answer

        memory_steps = agent.memory.steps
        actions_steps = [s for s in memory_steps if isinstance(s, ActionStep)]

        outputs = [s.model_output for s in actions_steps if s.model_output]
        assert outputs
        assert all(o.endswith("</code>") for o in outputs)

        messages = [s.model_output_message for s in actions_steps if s.model_output_message]
        assert messages
        assert all(m.content.endswith("</code>") for m in messages)

    @pytest.mark.skip(
        reason="Test is not properly implemented (GH-1255) because fake_tools should have the same name. "
    )
    def test_change_tools_after_init(self):
        # [FUNCTION DOC]
        # Skipped duplicate of the tools-replacement test for CodeAgent.
        # Inputs: self.
        # Outputs: none; would assert "2CUSTOM".
        # Exceptions: AssertionError on failure (if run).
        # Side effects: mutates agent.tools.
        # KNOWN ISSUE: skipped as broken (GH-1255): the replacement tool keeps a different name.
        from smolagents import tool

        @tool
        def fake_tool_1() -> str:
            # [FUNCTION DOC]
            # HELPER: Nested @tool returning "1".
            # Inputs: none.
            # Outputs: "1".
            # Exceptions: none raised directly.
            # Side effects: none.
            """Fake tool"""
            return "1"

        @tool
        def fake_tool_2() -> str:
            # [FUNCTION DOC]
            # HELPER: Nested @tool returning "2".
            # Inputs: none.
            # Outputs: "2".
            # Exceptions: none raised directly.
            # Side effects: none.
            """Fake tool"""
            return "2"

        class FakeCodeModel(Model):
            # [CLASS DOC]
            # HELPER: Local fake Model returning code that passes fake_tool_1() to final_answer.
            # Inputs: none.
            # Outputs: see generate.
            # Exceptions: none raised directly.
            # Side effects: none.
            def generate(self, messages, stop_sequences=None):
                # [FUNCTION DOC]
                # HELPER: Returns the fixed code block.
                # Inputs: messages, stop_sequences (ignored).
                # Outputs: assistant ChatMessage.
                # Exceptions: none raised directly.
                # Side effects: none.
                return ChatMessage(role=MessageRole.ASSISTANT, content="<code>\nfinal_answer(fake_tool_1())\n</code>")

        agent = CodeAgent(tools=[fake_tool_1], model=FakeCodeModel())

        agent.tools["final_answer"] = CustomFinalAnswerTool()
        agent.tools["fake_tool_1"] = fake_tool_2

        answer = agent.run("Fake task.")
        assert answer == "2CUSTOM"

    def test_local_python_executor_with_custom_functions(self):
        # [FUNCTION DOC]
        # Checks executor_kwargs additional_functions are registered as static tools of the local executor.
        # Inputs: self.
        # Outputs: none.
        # Exceptions: AssertionError on failure.
        # Side effects: none.
        model = MagicMock()
        model.generate.return_value = ChatMessage(
            role=MessageRole.ASSISTANT,
            content="",
            tool_calls=None,
            raw="",
            token_usage=None,
        )
        agent = CodeAgent(tools=[], model=model, executor_kwargs={"additional_functions": {"open": open}})
        agent.run("Test run")
        assert "open" in agent.python_executor.static_tools

    @pytest.mark.parametrize("agent_dict_version", ["v1.9", "v1.10", "v1.20"])
    def test_from_folder(self, agent_dict_version, get_agent_dict):
        # [FUNCTION DOC]
        # Checks CodeAgent.from_folder loads agent.json across serialization versions with patched Path and
        #   MODEL_REGISTRY.
        # Inputs: self, agent_dict_version (str, parametrized), get_agent_dict (fixture returning the dict for
        #   a version).
        # Outputs: none.
        # Exceptions: AssertionError on failure.
        # Side effects: temporarily patches smolagents.agents.Path and MODEL_REGISTRY.
        agent_dict = get_agent_dict(agent_dict_version)
        mock_model_class = MagicMock()
        mock_model_instance = MagicMock()
        mock_model_instance.model_id = "Qwen/Qwen2.5-Coder-32B-Instruct"
        mock_model_class.from_dict.return_value = mock_model_instance

        with (
            patch("smolagents.agents.Path") as mock_path,
            patch.dict("smolagents.models.MODEL_REGISTRY", {"InferenceClientModel": mock_model_class}),
        ):
            import json

            mock_path.return_value.__truediv__.return_value.read_text.return_value = json.dumps(agent_dict)
            agent = CodeAgent.from_folder("ignored_dummy_folder")
        assert isinstance(agent, CodeAgent)
        assert agent.name == "test_agent"
        assert agent.description == "dummy description"
        assert agent.max_steps == 10
        assert agent.planning_interval == 2
        assert agent.additional_authorized_imports == ["pandas"]
        assert "pandas" in agent.authorized_imports
        assert agent.executor_type == "local"
        assert agent.executor_kwargs == {}
        assert agent.max_print_outputs_length is None
        assert agent.managed_agents == {}
        assert set(agent.tools.keys()) == {"final_answer"}
        assert agent.model == mock_model_instance
        assert mock_model_class.from_dict.call_args.args[0]["model_id"] == "Qwen/Qwen2.5-Coder-32B-Instruct"
        assert agent.model.model_id == "Qwen/Qwen2.5-Coder-32B-Instruct"
        assert agent.logger.level == 2
        assert agent.prompt_templates["system_prompt"] == "dummy system prompt"

    def test_from_dict(self):
        # [FUNCTION DOC]
        # Checks CodeAgent.from_dict applies CodeAgent-specific fields, defaults when missing, and kwargs
        #   overrides.
        # Inputs: self.
        # Outputs: none.
        # Exceptions: AssertionError on failure.
        # Side effects: temporarily patches MODEL_REGISTRY; executes tool code from the dict.
        # Create a test agent dictionary
        agent_dict = {
            "model": {"class": "InferenceClientModel", "data": {"model_id": "Qwen/Qwen2.5-Coder-32B-Instruct"}},
            "tools": [
                {
                    "name": "valid_tool_function",
                    "code": 'from smolagents import Tool\nfrom typing import Any, Optional\n\nclass SimpleTool(Tool):\n    name = "valid_tool_function"\n    description = "A valid tool function."\n    inputs = {"input":{"type":"string","description":"Input string."}}\n    output_type = "string"\n\n    def forward(self, input: str) -> str:\n        """A valid tool function.\n\n        Args:\n            input (str): Input string.\n        """\n        return input.upper()',
                    "requirements": {"smolagents"},
                }
            ],
            "managed_agents": {},
            "prompt_templates": EMPTY_PROMPT_TEMPLATES,
            "max_steps": 15,
            "verbosity_level": 2,
            "use_structured_output": False,
            "planning_interval": 3,
            "name": "test_code_agent",
            "description": "Test code agent description",
            "authorized_imports": ["pandas", "numpy"],
            "executor_type": "local",
            "executor_kwargs": {"max_print_outputs_length": 10_000},
            "max_print_outputs_length": 1000,
        }

        # Call from_dict
        mock_model_class = MagicMock()
        mock_model_instance = MagicMock()
        mock_model_class.from_dict.return_value = mock_model_instance

        with patch.dict("smolagents.models.MODEL_REGISTRY", {"InferenceClientModel": mock_model_class}):
            agent = CodeAgent.from_dict(agent_dict)

        # Verify the agent was created correctly with CodeAgent-specific parameters
        assert agent.model == mock_model_instance
        assert agent.additional_authorized_imports == ["pandas", "numpy"]
        assert agent.executor_type == "local"
        assert agent.executor_kwargs == {"max_print_outputs_length": 10_000}
        assert agent.max_print_outputs_length == 1000

        # Test with missing optional parameters
        minimal_agent_dict = {
            "model": {"class": "InferenceClientModel", "data": {"model_id": "Qwen/Qwen2.5-Coder-32B-Instruct"}},
            "tools": [],
            "managed_agents": {},
        }

        with patch.dict("smolagents.models.MODEL_REGISTRY", {"InferenceClientModel": mock_model_class}):
            agent = CodeAgent.from_dict(minimal_agent_dict)
        # Verify defaults are used
        assert agent.max_steps == 20  # default from MultiStepAgent.__init__

        # Test overriding with kwargs
        with patch.dict("smolagents.models.MODEL_REGISTRY", {"InferenceClientModel": mock_model_class}):
            agent = CodeAgent.from_dict(
                agent_dict,
                additional_authorized_imports=["requests"],
                executor_kwargs={"max_print_outputs_length": 5_000},
            )
        assert agent.additional_authorized_imports == ["requests"]
        assert agent.executor_kwargs == {"max_print_outputs_length": 5_000}

    def test_custom_final_answer_with_custom_inputs(self):
        # [FUNCTION DOC]
        # Checks a custom final-answer tool with two inputs works from CodeAgent code.
        # Inputs: self.
        # Outputs: none; asserts answer "1CUSTOM2".
        # Exceptions: AssertionError on failure.
        # Side effects: none.
        class CustomFinalAnswerToolWithCustomInputs(FinalAnswerTool):
            # [CLASS DOC]
            # HELPER: FinalAnswerTool with inputs answer1 and answer2 joined by "CUSTOM".
            # Inputs: inherits FinalAnswerTool constructor.
            # Outputs: tool instance.
            # Exceptions: none raised directly.
            # Side effects: none.
            inputs = {
                "answer1": {"type": "string", "description": "First part of the answer."},
                "answer2": {"type": "string", "description": "Second part of the answer."},
            }

            def forward(self, answer1: str, answer2: str) -> str:
                # [FUNCTION DOC]
                # HELPER: Joins answer1 and answer2 with "CUSTOM".
                # Inputs: answer1 (str), answer2 (str).
                # Outputs: str.
                # Exceptions: none raised directly.
                # Side effects: none.
                return answer1 + "CUSTOM" + answer2

        model = MagicMock()
        model.generate.return_value = ChatMessage(
            role=MessageRole.ASSISTANT, content="<code>\nfinal_answer(answer1='1', answer2='2')\n</code>"
        )
        agent = CodeAgent(tools=[CustomFinalAnswerToolWithCustomInputs()], model=model)
        answer = agent.run("Fake task.")
        assert answer == "1CUSTOM2"

    def test_use_structured_outputs_internally(self):
        # [FUNCTION DOC]
        # Checks with structured outputs the code is read from the JSON "code" field and yielded as the first
        #   ToolCall's arguments.
        # Inputs: self.
        # Outputs: none.
        # Exceptions: AssertionError on failure.
        # Side effects: calls the private _step_stream generator once.
        expected_code = "print('Hello, world!')"
        model = MagicMock()
        # mock structured output generation
        model.generate.return_value = ChatMessage(
            role=MessageRole.ASSISTANT,
            content=json.dumps({"thought": "LLM-generated thought", "code": expected_code}),
        )
        agent = CodeAgent(
            tools=[], model=model, use_structured_outputs_internally=True
        )  # Use structured outputs internally
        tool_call: ToolCall = next(
            agent._step_stream(ActionStep(step_number=1, timing="mock_timing", model_output=""))
        )
        assert tool_call.arguments == expected_code


class TestMultiAgents:
    # [CLASS DOC]
    # Test group for manager/managed multi-agent setups.
    # Inputs: none (pytest-managed).
    # Outputs: none.
    # Exceptions: none raised directly.
    # Side effects: none.
    def test_multiagents_save(self, tmp_path):
        # [FUNCTION DOC]
        # Checks saving a manager agent with managed agents writes the expected folder structure and
        #   from_folder reloads it with kwargs overrides.
        # Inputs: self, tmp_path (pytest temp dir).
        # Outputs: none.
        # Exceptions: AssertionError on failure.
        # Side effects: writes files into tmp_path.
        model = InferenceClientModel(model_id="Qwen/Qwen2.5-Coder-32B-Instruct", max_tokens=2096, temperature=0.5)

        web_agent = ToolCallingAgent(
            model=model,
            tools=[DuckDuckGoSearchTool(max_results=2), VisitWebpageTool()],
            name="web_agent",
            description="does web searches",
        )
        code_agent = CodeAgent(model=model, tools=[], name="useless", description="does nothing in particular")

        agent = CodeAgent(
            model=model,
            tools=[],
            additional_authorized_imports=["pandas", "datetime"],
            managed_agents=[web_agent, code_agent],
            max_print_outputs_length=1000,
            executor_type="local",
            executor_kwargs={"max_print_outputs_length": 10_000},
        )
        agent.save(tmp_path)

        expected_structure = {
            "managed_agents": {
                "useless": {"tools": {"files": ["final_answer.py"]}, "files": ["agent.json", "prompts.yaml"]},
                "web_agent": {
                    "tools": {"files": ["final_answer.py", "visit_webpage.py", "web_search.py"]},
                    "files": ["agent.json", "prompts.yaml"],
                },
            },
            "tools": {"files": ["final_answer.py"]},
            "files": ["app.py", "requirements.txt", "agent.json", "prompts.yaml"],
        }

        def verify_structure(current_path: Path, structure: dict):
            # [FUNCTION DOC]
            # HELPER: Recursively asserts that directories and files in a nested structure dict exist under a
            #   path.
            # Inputs: current_path (Path), structure (dict where key "files" lists file names and other keys
            #   are subdirectory names mapping to nested dicts).
            # Outputs: None.
            # Exceptions: AssertionError for a missing or wrongly-typed entry.
            # Side effects: reads the filesystem.
            for dir_name, contents in structure.items():
                if dir_name != "files":
                    # For directories, verify they exist and recurse into them
                    dir_path = current_path / dir_name
                    assert dir_path.exists(), f"Directory {dir_path} does not exist"
                    assert dir_path.is_dir(), f"{dir_path} is not a directory"
                    verify_structure(dir_path, contents)
                else:
                    # For files, verify each exists in the current path
                    for file_name in contents:
                        file_path = current_path / file_name
                        assert file_path.exists(), f"File {file_path} does not exist"
                        assert file_path.is_file(), f"{file_path} is not a file"

        verify_structure(tmp_path, expected_structure)

        # Test that re-loaded agents work as expected.
        agent2 = CodeAgent.from_folder(tmp_path, planning_interval=5)
        assert agent2.planning_interval == 5  # Check that kwargs are used
        assert set(agent2.authorized_imports) == set(["pandas", "datetime"] + BASE_BUILTIN_MODULES)
        assert agent2.max_print_outputs_length == 1000
        assert agent2.executor_type == "local"
        assert agent2.executor_kwargs == {"max_print_outputs_length": 10_000}
        assert (
            agent2.managed_agents["web_agent"].tools["web_search"].max_results == 10
        )  # For now tool init parameters are forgotten
        assert agent2.model.kwargs["temperature"] == pytest.approx(0.5)

    def test_multiagents(self):
        # [FUNCTION DOC]
        # Checks CodeAgent and ToolCallingAgent managers can delegate to a managed ToolCallingAgent, its
        #   output is logged, and visualize prints a tree.
        # Inputs: self.
        # Outputs: none.
        # Exceptions: AssertionError on failure.
        # Side effects: captures console output.
        class FakeModelMultiagentsManagerAgent(Model):
            # [CLASS DOC]
            # HELPER: Fake manager Model that first delegates to search_agent then returns "Final report.",
            #   via tool calls or code depending on mode.
            # Inputs: none (model_id "fake_model").
            # Outputs: see generate.
            # Exceptions: see generate.
            # Side effects: none.
            model_id = "fake_model"

            def generate(
                self,
                messages,
                stop_sequences=None,
                tools_to_call_from=None,
            ):
                # [FUNCTION DOC]
                # HELPER: Returns delegation then final answer, as tool calls when tools_to_call_from is
                #   given, else as code.
                # Inputs: messages (list), stop_sequences (ignored), tools_to_call_from (list or None).
                # Outputs: with fewer than 3 messages, a search_agent call with the question; afterwards a
                #   final_answer with "Final report.".
                # Exceptions: AssertionError if the later-stage messages lack the managed agent's report.
                # Side effects: none.
                if tools_to_call_from is not None:
                    if len(messages) < 3:
                        return ChatMessage(
                            role=MessageRole.ASSISTANT,
                            content="",
                            tool_calls=[
                                ChatMessageToolCall(
                                    id="call_0",
                                    type="function",
                                    function=ChatMessageToolCallFunction(
                                        name="search_agent",
                                        arguments="Who is the current US president?",
                                    ),
                                )
                            ],
                        )
                    else:
                        assert "Report on the current US president" in str(messages)
                        return ChatMessage(
                            role=MessageRole.ASSISTANT,
                            content="",
                            tool_calls=[
                                ChatMessageToolCall(
                                    id="call_0",
                                    type="function",
                                    function=ChatMessageToolCallFunction(
                                        name="final_answer", arguments="Final report."
                                    ),
                                )
                            ],
                        )
                else:
                    if len(messages) < 3:
                        return ChatMessage(
                            role=MessageRole.ASSISTANT,
                            content="""
Thought: Let's call our search agent.
<code>
result = search_agent("Who is the current US president?")
</code>
""",
                        )
                    else:
                        assert "Report on the current US president" in str(messages)
                        return ChatMessage(
                            role=MessageRole.ASSISTANT,
                            content="""
Thought: Let's return the report.
<code>
final_answer("Final report.")
</code>
""",
                        )

        manager_model = FakeModelMultiagentsManagerAgent()

        class FakeModelMultiagentsManagedAgent(Model):
            # [CLASS DOC]
            # HELPER: Fake managed-agent Model that immediately returns a final_answer report and content
            #   containing FLAG1.
            # Inputs: none (model_id "fake_model").
            # Outputs: see generate.
            # Exceptions: none raised directly.
            # Side effects: none.
            model_id = "fake_model"

            def generate(
                self,
                messages,
                tools_to_call_from=None,
                stop_sequences=None,
            ):
                # [FUNCTION DOC]
                # HELPER: Returns content "Here is the secret content: FLAG1" with a final_answer call "Report
                #   on the current US president".
                # Inputs: messages, tools_to_call_from, stop_sequences (ignored).
                # Outputs: assistant ChatMessage.
                # Exceptions: none raised directly.
                # Side effects: none.
                return ChatMessage(
                    role=MessageRole.ASSISTANT,
                    content="Here is the secret content: FLAG1",
                    tool_calls=[
                        ChatMessageToolCall(
                            id="call_0",
                            type="function",
                            function=ChatMessageToolCallFunction(
                                name="final_answer",
                                arguments="Report on the current US president",
                            ),
                        )
                    ],
                )

        managed_model = FakeModelMultiagentsManagedAgent()

        web_agent = ToolCallingAgent(
            tools=[],
            model=managed_model,
            max_steps=10,
            name="search_agent",
            description="Runs web searches for you. Give it your request as an argument. Make the request as detailed as needed, you can ask for thorough reports",
            verbosity_level=2,
        )

        manager_code_agent = CodeAgent(
            tools=[],
            model=manager_model,
            managed_agents=[web_agent],
            additional_authorized_imports=["time", "numpy", "pandas"],
        )

        report = manager_code_agent.run("Fake question.")
        assert report == "Final report."

        manager_toolcalling_agent = ToolCallingAgent(
            tools=[],
            model=manager_model,
            managed_agents=[web_agent],
        )

        with web_agent.logger.console.capture() as capture:
            report = manager_toolcalling_agent.run("Fake question.")
        assert report == "Final report."
        assert "FLAG1" in capture.get()  # Check that managed agent's output is properly logged

        # Test that visualization works
        with manager_toolcalling_agent.logger.console.capture() as capture:
            manager_toolcalling_agent.visualize()
        assert "├──" in capture.get()


@pytest.fixture
def prompt_templates():
    # [FUNCTION DOC]
    # HELPER: pytest fixture with minimal custom prompt templates.
    # Inputs: none.
    # Outputs: dict with system_prompt, managed_agent task/report, planning templates and final_answer
    #   pre/post messages.
    # Exceptions: none raised directly.
    # Side effects: none.
    return {
        "system_prompt": "This is a test system prompt.",
        "managed_agent": {"task": "Task for {{name}}: {{task}}", "report": "Report for {{name}}: {{final_answer}}"},
        "planning": {
            "initial_plan": "The plan.",
            "update_plan_pre_messages": "custom",
            "update_plan_post_messages": "custom",
        },
        "final_answer": {"pre_messages": "custom", "post_messages": "custom"},
    }


@pytest.mark.parametrize(
    "arguments",
    [
        {},
        {"arg": "bar"},
        {None: None},
        [1, 2, 3],
    ],
)
def test_tool_calling_agents_raises_tool_call_error_being_invoked_with_wrong_arguments(arguments):
    # [FUNCTION DOC]
    # Checks execute_tool_call raises AgentToolCallError for arguments not matching the tool signature.
    # Inputs: arguments (dict or list, parametrized).
    # Outputs: none.
    # Exceptions: AssertionError on failure.
    # Side effects: none.
    @tool
    def _sample_tool(prompt: str) -> str:
        # [FUNCTION DOC]
        # HELPER: Nested @tool echoing its prompt.
        # Inputs: prompt (str).
        # Outputs: the same string.
        # Exceptions: none raised directly.
        # Side effects: none.
        """Tool that returns same string
        Args:
            prompt: The string to return
        Returns:
            The same string
        """

        return prompt

    agent = ToolCallingAgent(model=FakeToolCallModel(), tools=[_sample_tool])
    with pytest.raises(AgentToolCallError):
        agent.execute_tool_call(_sample_tool.name, arguments)


def test_tool_calling_agents_raises_agent_execution_error_when_tool_raises():
    # [FUNCTION DOC]
    # Checks execute_tool_call wraps an exception raised inside a tool in AgentExecutionError.
    # Inputs: none.
    # Outputs: none.
    # Exceptions: AssertionError on failure.
    # Side effects: none.
    @tool
    def _sample_tool(_: str) -> float:
        # [FUNCTION DOC]
        # HELPER: Nested @tool that always divides by zero.
        # Inputs: _ (str; ignored).
        # Outputs: never returns.
        # Exceptions: ZeroDivisionError always.
        # Side effects: none.
        """Tool that fails

        Args:
            _: The pointless string
        Returns:
            Some number
        """

        return 1 / 0

    agent = ToolCallingAgent(model=FakeToolCallModel(), tools=[_sample_tool])
    with pytest.raises(AgentExecutionError):
        agent.execute_tool_call(_sample_tool.name, "sample")
