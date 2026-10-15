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
# Purpose: Tests for monitoring: token accounting of agents via Monitor, gradio streaming output counts,
#   memory replay with ChatMessages, and AgentLogger.log_task robustness.
# Dependencies: unittest, pytest, PIL, rich, smolagents agents, models, memory, monitoring, stream_to_gradio.
# Side effects: tests run agents with fake models and render to recorded consoles.

import unittest

import PIL.Image
import pytest
from rich.console import Console

from smolagents import (
    CodeAgent,
    ToolCallingAgent,
    stream_to_gradio,
)
from smolagents.memory import ActionStep, AgentMemory
from smolagents.models import (
    ChatMessage,
    ChatMessageToolCall,
    ChatMessageToolCallFunction,
    MessageRole,
    Model,
)
from smolagents.monitoring import AgentLogger, TokenUsage


class FakeLLMModel(Model):
    # [CLASS DOC]
    # HELPER: Fake Model returning a final_answer tool call when tools are provided, or a code block calling
    #   final_answer otherwise, each with 10 input and 20 output tokens.
    # Inputs: Model constructor args.
    # Exceptions: none raised directly.
    # Side effects: none.
    def generate(self, prompt, tools_to_call_from=None, **kwargs):
        # [FUNCTION DOC]
        # HELPER: Returns the canned assistant ChatMessage described in the class doc.
        # Inputs: prompt (ignored), tools_to_call_from (list or None), **kwargs (ignored).
        # Outputs: ChatMessage.
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
                token_usage=TokenUsage(input_tokens=10, output_tokens=20),
            )
        else:
            return ChatMessage(
                role=MessageRole.ASSISTANT,
                content="""<code>
final_answer('This is the final answer.')
</code>""",
                token_usage=TokenUsage(input_tokens=10, output_tokens=20),
            )


class MonitoringTester(unittest.TestCase):
    # [CLASS DOC]
    # unittest TestCase for agent metrics and gradio streaming outputs.
    # Exceptions: assertion failures on test failure.
    # Side effects: runs agents with fake models.
    def test_code_agent_metrics_max_steps(self):
        # [FUNCTION DOC]
        # Checks token totals when a malformed answer exhausts max steps: one step plus the final forced
        #   answer, totalling 20 input and 40 output tokens.
        # Inputs: none.
        # Outputs: none.
        # Exceptions: AssertionError on failure.
        # Side effects: console output from the agent.
        class FakeLLMModelMalformedAnswer(Model):
            # [CLASS DOC]
            # HELPER: Fake model always returning "Malformed answer" with 10/20 token usage.
            # Exceptions: none raised directly.
            # Side effects: none.
            def generate(self, prompt, **kwargs):
                # [FUNCTION DOC]
                # HELPER: Returns the malformed assistant message.
                # Inputs: prompt (ignored), **kwargs (ignored).
                # Outputs: ChatMessage.
                # Exceptions: none raised directly.
                # Side effects: none.
                return ChatMessage(
                    role=MessageRole.ASSISTANT,
                    content="Malformed answer",
                    token_usage=TokenUsage(input_tokens=10, output_tokens=20),
                )

        agent = CodeAgent(
            tools=[],
            model=FakeLLMModelMalformedAnswer(),
            max_steps=1,
        )

        agent.run("Fake task")

        self.assertEqual(agent.monitor.total_input_token_count, 20)
        self.assertEqual(agent.monitor.total_output_token_count, 40)

    def test_code_agent_metrics_generation_error(self):
        # [FUNCTION DOC]
        # Checks that an exception from the model propagates out of agent.run with its message.
        # Inputs: none.
        # Outputs: none.
        # Exceptions: AssertionError on failure.
        # Side effects: console output.
        class FakeLLMModelGenerationException(Model):
            # [CLASS DOC]
            # HELPER: Fake model whose generation always fails.
            # Exceptions: generate raises Exception.
            # Side effects: none.
            def generate(self, prompt, **kwargs):
                # [FUNCTION DOC]
                # HELPER: Always raises a generic Exception with "Cannot generate".
                # Inputs: prompt, **kwargs (ignored).
                # Outputs: none.
                # Exceptions: Exception always.
                # Side effects: none.
                raise Exception("Cannot generate")

        agent = CodeAgent(
            tools=[],
            model=FakeLLMModelGenerationException(),
            max_steps=1,
        )
        with pytest.raises(Exception) as e:
            agent.run("Fake task")
        assert "Cannot generate" in str(e.value)

    def test_streaming_agent_text_output(self):
        # [FUNCTION DOC]
        # Checks stream_to_gradio with planning yields 11 outputs, the second containing the code block and
        #   the last containing the final answer text.
        # Inputs: none.
        # Outputs: none.
        # Exceptions: AssertionError on failure.
        # Side effects: runs the agent.
        agent = CodeAgent(
            tools=[],
            model=FakeLLMModel(),
            max_steps=1,
            planning_interval=2,
        )

        # Use stream_to_gradio to capture the output
        outputs = list(stream_to_gradio(agent, task="Test task"))

        self.assertEqual(len(outputs), 11)
        plan_message = outputs[1]
        self.assertEqual(plan_message.role, "assistant")
        self.assertIn("<code>", plan_message.content)
        final_message = outputs[-1]
        self.assertEqual(final_message.role, "assistant")
        self.assertIn("This is the final answer.", final_message.content)

    def test_streaming_agent_image_output(self):
        # [FUNCTION DOC]
        # Checks a ToolCallingAgent whose final answer is an image passed through additional args streams 7
        #   outputs, the last being an image/png file message.
        # Inputs: none.
        # Outputs: none.
        # Exceptions: AssertionError on failure.
        # Side effects: writes a temp PNG.
        class FakeLLMModelImage(Model):
            # [CLASS DOC]
            # HELPER: Fake model calling final_answer with the "image" state variable name.
            # Exceptions: none raised directly.
            # Side effects: none.
            def generate(self, prompt, **kwargs):
                # [FUNCTION DOC]
                # HELPER: Returns an assistant message with a final_answer tool call whose answer is "image".
                # Inputs: prompt, **kwargs (ignored).
                # Outputs: ChatMessage.
                # Exceptions: none raised directly.
                # Side effects: none.
                return ChatMessage(
                    role=MessageRole.ASSISTANT,
                    content="I will call the final_answer tool.",
                    tool_calls=[
                        ChatMessageToolCall(
                            id="fake_id",
                            type="function",
                            function=ChatMessageToolCallFunction(name="final_answer", arguments={"answer": "image"}),
                        )
                    ],
                )

        agent = ToolCallingAgent(
            tools=[],
            model=FakeLLMModelImage(),
            max_steps=1,
            verbosity_level=100,
        )

        # Use stream_to_gradio to capture the output
        outputs = list(
            stream_to_gradio(
                agent,
                task="Test task",
                additional_args=dict(image=PIL.Image.new("RGB", (100, 100))),
            )
        )

        self.assertEqual(len(outputs), 7)
        final_message = outputs[-1]
        self.assertEqual(final_message.role, "assistant")
        self.assertIsInstance(final_message.content, dict)
        self.assertEqual(final_message.content["mime_type"], "image/png")

    def test_streaming_with_agent_error(self):
        # [FUNCTION DOC]
        # Checks that a malformed model output still streams 11 outputs, the last containing the malformed
        #   text as final answer.
        # Inputs: none.
        # Outputs: none.
        # Exceptions: AssertionError on failure.
        # Side effects: runs the agent.
        class DummyModel(Model):
            # [CLASS DOC]
            # HELPER: Fake model always returning "Malformed call" without token usage.
            # Exceptions: none raised directly.
            # Side effects: none.
            def generate(self, prompt, **kwargs):
                # [FUNCTION DOC]
                # HELPER: Returns the malformed assistant message.
                # Inputs: prompt, **kwargs (ignored).
                # Outputs: ChatMessage.
                # Exceptions: none raised directly.
                # Side effects: none.
                return ChatMessage(role=MessageRole.ASSISTANT, content="Malformed call")

        agent = CodeAgent(
            tools=[],
            model=DummyModel(),
            max_steps=1,
        )

        # Use stream_to_gradio to capture the output
        outputs = list(stream_to_gradio(agent, task="Test task"))

        self.assertEqual(len(outputs), 11)
        final_message = outputs[-1]
        self.assertEqual(final_message.role, "assistant")
        self.assertIn("Malformed call", final_message.content)


@pytest.mark.parametrize("agent_class", [CodeAgent, ToolCallingAgent])
def test_code_agent_metrics(agent_class):
    # [FUNCTION DOC]
    # Parametrized check that a one-step successful run of CodeAgent or ToolCallingAgent records 10 input and
    #   20 output tokens.
    # Inputs: agent_class (CodeAgent or ToolCallingAgent).
    # Outputs: none.
    # Exceptions: AssertionError on failure.
    # Side effects: runs the agent.
    agent = agent_class(
        tools=[],
        model=FakeLLMModel(),
        max_steps=1,
    )
    agent.run("Fake task")

    assert agent.monitor.total_input_token_count == 10
    assert agent.monitor.total_output_token_count == 20


class ReplayTester(unittest.TestCase):
    # [CLASS DOC]
    # Regression tests for AgentMemory.replay.
    # Exceptions: assertion failures on test failure.
    # Side effects: console output.
    def test_replay_with_chatmessage(self):
        # [FUNCTION DOC]
        # Checks detailed replay of a step holding ChatMessage inputs does not raise TypeError.
        # Inputs: none.
        # Outputs: none.
        # Exceptions: AssertionError (via fail) on TypeError.
        # Side effects: console output.
        # KNOWN ISSUE: builds the ActionStep with an integer timing instead of a Timing object.
        """Regression test for dict(message) to message.dict() fix"""
        logger = AgentLogger()
        memory = AgentMemory(system_prompt="test")
        step = ActionStep(step_number=1, timing=0)
        step.model_input_messages = [ChatMessage(role=MessageRole.USER, content="Hello")]
        memory.steps.append(step)

        try:
            memory.replay(logger, detailed=True)
        except TypeError as e:
            self.fail(f"Replay raised an error: {e}")


class AgentLoggerLogTaskTester(unittest.TestCase):
    # [CLASS DOC]
    # Tests that AgentLogger.log_task safely renders arbitrary content and subtitles.
    # Exceptions: assertion failures on test failure.
    # Side effects: renders to recorded consoles.
    def test_logger_log_task_does_not_crash_on_stray_markup_or_control_chars(self):
        # [FUNCTION DOC]
        # Checks bytes content with stray markup and control characters renders literally, with control chars
        #   shown as escape sequences.
        # Inputs: none.
        # Outputs: none.
        # Exceptions: AssertionError on failure.
        # Side effects: none outside the recorded console.
        """
        Rich Panels parse `title`/`subtitle` as markup when passed as strings.
        `AgentLogger.log_task()` must be resilient to arbitrary content/subtitle strings
        (e.g. tool logs, binary-ish payloads, or stray bracket sequences).
        """
        console = Console(record=True, width=120, highlight=False)
        logger = AgentLogger(console=console)

        # These inputs would crash Rich markup parsing if passed through as markup strings.
        content = b"hello [/bad]\x00\x1b world [bold]bold[/bold]"
        subtitle = "sub[/bad]title"

        logger.log_task(content=content, subtitle=subtitle, title=None)

        rendered = console.export_text()
        self.assertIn("hello [/bad]", rendered)
        # Control chars are made visible as escape sequences.
        self.assertIn("\\x00", rendered)
        self.assertIn("\\x1b", rendered)
        self.assertIn("sub[/bad]title", rendered)
        self.assertIn("bold", rendered)

    def test_logger_log_task_accepts_non_string_payloads(self):
        # [FUNCTION DOC]
        # Checks dict content and subtitle are rendered without errors.
        # Inputs: none.
        # Outputs: none.
        # Exceptions: AssertionError on failure.
        # Side effects: none outside the recorded console.
        console = Console(record=True, width=120, highlight=False)
        logger = AgentLogger(console=console)

        logger.log_task(content={"k": ["v", 1]}, subtitle={"also": "dict"}, title="Run")
        rendered = console.export_text()
        self.assertIn("k", rendered)
        self.assertIn("also", rendered)
