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
# Purpose: Unit tests for smolagents.gradio_ui: GradioUI file uploads, stream_to_gradio streaming and
#   parameter forwarding, and pull_messages_from_step conversion of each step type.
# Dependencies: pytest, unittest, unittest.mock, gradio (via the module under test), smolagents agent_types,
#   memory, models, monitoring.
# Side effects: tests create and delete temporary files/directories.

import os
import shutil
import tempfile
import unittest
from unittest.mock import Mock, patch

import pytest

from smolagents.agent_types import AgentAudio, AgentImage, AgentText
from smolagents.gradio_ui import GradioUI, pull_messages_from_step, stream_to_gradio
from smolagents.memory import ActionStep, FinalAnswerStep, PlanningStep, ToolCall
from smolagents.models import ChatMessageStreamDelta
from smolagents.monitoring import Timing, TokenUsage


class GradioUITester(unittest.TestCase):
    # [CLASS DOC]
    # unittest TestCase exercising GradioUI.upload_file with a mock agent and a temporary upload folder.
    # Attributes: temp_dir (upload folder), mock_agent (Mock), ui (GradioUI), allowed_types (default extension
    #   list).
    # Exceptions: assertion failures on test failure.
    # Side effects: creates/removes a temp directory per test.
    def setUp(self):
        # [FUNCTION DOC]
        # HELPER: Creates a temp directory, a Mock agent, a GradioUI using that folder, and the default
        #   allowed types list.
        # Inputs: none.
        # Outputs: none.
        # Exceptions: none raised directly.
        # Side effects: creates a temp directory; sets instance attributes.
        """Initialize test environment"""
        self.temp_dir = tempfile.mkdtemp()
        self.mock_agent = Mock()
        self.ui = GradioUI(agent=self.mock_agent, file_upload_folder=self.temp_dir)
        self.allowed_types = [".pdf", ".docx", ".txt"]

    def tearDown(self):
        # [FUNCTION DOC]
        # HELPER: Recursively deletes the temp upload directory.
        # Inputs: none.
        # Outputs: none.
        # Exceptions: OSError if removal fails.
        # Side effects: filesystem deletion.
        """Clean up test environment"""
        shutil.rmtree(self.temp_dir)

    def test_upload_file_default_types(self):
        # [FUNCTION DOC]
        # Checks that pdf, docx and txt files are accepted by default: status says uploaded, log has one entry
        #   and the file exists in the upload folder.
        # Inputs: none.
        # Outputs: none.
        # Exceptions: AssertionError on failure.
        # Side effects: creates temp files and copies them.
        """Test default allowed file types"""
        default_types = [".pdf", ".docx", ".txt"]
        for file_type in default_types:
            with tempfile.NamedTemporaryFile(suffix=file_type) as temp_file:
                mock_file = Mock()
                mock_file.name = temp_file.name

                textbox, uploads_log = self.ui.upload_file(mock_file, [])

                self.assertIn("File uploaded:", textbox.value)
                self.assertEqual(len(uploads_log), 1)
                self.assertTrue(os.path.exists(os.path.join(self.temp_dir, os.path.basename(temp_file.name))))

    def test_upload_file_default_types_disallowed(self):
        # [FUNCTION DOC]
        # Checks that exe, sh, py and jpg files are rejected with "File type disallowed" and the log stays
        #   empty.
        # Inputs: none.
        # Outputs: none.
        # Exceptions: AssertionError on failure.
        # Side effects: creates temp files.
        """Test default disallowed file types"""
        disallowed_types = [".exe", ".sh", ".py", ".jpg"]
        for file_type in disallowed_types:
            with tempfile.NamedTemporaryFile(suffix=file_type) as temp_file:
                mock_file = Mock()
                mock_file.name = temp_file.name

                textbox, uploads_log = self.ui.upload_file(mock_file, [])

                self.assertEqual(textbox.value, "File type disallowed")
                self.assertEqual(len(uploads_log), 0)

    def test_upload_file_success(self):
        # [FUNCTION DOC]
        # Checks a txt upload succeeds, the file is copied, and the logged path equals the destination in the
        #   upload folder.
        # Inputs: none.
        # Outputs: none.
        # Exceptions: AssertionError on failure.
        # Side effects: creates and copies a temp file.
        """Test successful file upload scenario"""
        with tempfile.NamedTemporaryFile(suffix=".txt") as temp_file:
            mock_file = Mock()
            mock_file.name = temp_file.name

            textbox, uploads_log = self.ui.upload_file(mock_file, [])

            self.assertIn("File uploaded:", textbox.value)
            self.assertEqual(len(uploads_log), 1)
            self.assertTrue(os.path.exists(os.path.join(self.temp_dir, os.path.basename(temp_file.name))))
            self.assertEqual(uploads_log[0], os.path.join(self.temp_dir, os.path.basename(temp_file.name)))

    def test_upload_file_none(self):
        # [FUNCTION DOC]
        # Checks that a None file returns "No file uploaded" and an unchanged empty log.
        # Inputs: none.
        # Outputs: none.
        # Exceptions: AssertionError on failure.
        # Side effects: none.
        """Test scenario when no file is selected"""
        textbox, uploads_log = self.ui.upload_file(None, [])

        self.assertEqual(textbox.value, "No file uploaded")
        self.assertEqual(len(uploads_log), 0)

    def test_upload_file_invalid_type(self):
        # [FUNCTION DOC]
        # Checks that an exe file is rejected with "File type disallowed".
        # Inputs: none.
        # Outputs: none.
        # Exceptions: AssertionError on failure.
        # Side effects: creates a temp file.
        """Test disallowed file type"""
        with tempfile.NamedTemporaryFile(suffix=".exe") as temp_file:
            mock_file = Mock()
            mock_file.name = temp_file.name

            textbox, uploads_log = self.ui.upload_file(mock_file, [])

            self.assertEqual(textbox.value, "File type disallowed")
            self.assertEqual(len(uploads_log), 0)

    def test_upload_file_special_chars(self):
        # [FUNCTION DOC]
        # Checks that special characters in the filename are replaced with underscores in the saved path (copy
        #   is patched out).
        # Inputs: none.
        # Outputs: none.
        # Exceptions: AssertionError on failure.
        # Side effects: creates and removes a specially-named file in the temp directory.
        """Test scenario with special characters in filename"""
        with tempfile.NamedTemporaryFile(suffix=".txt") as temp_file:
            # Create a new temporary file with special characters
            special_char_name = os.path.join(os.path.dirname(temp_file.name), "test@#$%^&*.txt")
            shutil.copy(temp_file.name, special_char_name)
            try:
                mock_file = Mock()
                mock_file.name = special_char_name

                with patch("shutil.copy"):
                    textbox, uploads_log = self.ui.upload_file(mock_file, [])

                    self.assertIn("File uploaded:", textbox.value)
                    self.assertEqual(len(uploads_log), 1)
                    self.assertIn("test_____", uploads_log[0])
            finally:
                # Clean up the special character file
                if os.path.exists(special_char_name):
                    os.remove(special_char_name)

    def test_upload_file_custom_types(self):
        # [FUNCTION DOC]
        # Checks that a csv file is accepted when the allowed types list is set to csv only.
        # Inputs: none.
        # Outputs: none.
        # Exceptions: AssertionError on failure.
        # Side effects: creates and copies a temp file.
        """Test custom allowed file types"""
        with tempfile.NamedTemporaryFile(suffix=".csv") as temp_file:
            mock_file = Mock()
            mock_file.name = temp_file.name

            textbox, uploads_log = self.ui.upload_file(mock_file, [], allowed_file_types=[".csv"])

            self.assertIn("File uploaded:", textbox.value)
            self.assertEqual(len(uploads_log), 1)


class TestStreamToGradio:
    # [CLASS DOC]
    # pytest test class for stream_to_gradio using mock agents whose run returns a fixed list of events.
    # Exceptions: assertion failures on test failure.
    # Side effects: none.
    """Tests for the stream_to_gradio function."""

    @patch("smolagents.gradio_ui.pull_messages_from_step")
    def test_stream_to_gradio_memory_step(self, mock_pull_messages):
        # [FUNCTION DOC]
        # Checks that a step event is passed to pull_messages_from_step (patched) once and its messages are
        #   yielded.
        # Inputs: mock_pull_messages (patched function).
        # Outputs: none.
        # Exceptions: AssertionError on failure.
        # Side effects: none.
        """Test streaming a memory step"""
        # Create mock agent and memory step
        mock_agent = Mock()
        mock_agent.run = Mock(return_value=[Mock(spec=ActionStep)])
        mock_agent.model = Mock()
        # Mock the pull_messages_from_step function to return some messages
        mock_message = Mock()
        mock_pull_messages.return_value = [mock_message]
        # Call stream_to_gradio
        result = list(stream_to_gradio(mock_agent, "test task"))
        # Verify that pull_messages_from_step was called and the message was yielded
        mock_pull_messages.assert_called_once()
        assert result == [mock_message]

    def test_stream_to_gradio_stream_delta(self):
        # [FUNCTION DOC]
        # Checks that a single stream delta yields its text.
        # Inputs: none.
        # Outputs: none.
        # Exceptions: AssertionError on failure.
        # Side effects: none.
        """Test streaming a ChatMessageStreamDelta"""
        # Create mock agent and stream delta
        mock_agent = Mock()
        mock_delta = ChatMessageStreamDelta(content="Hello")
        mock_agent.run = Mock(return_value=[mock_delta])
        mock_agent.model = Mock()
        # Call stream_to_gradio
        result = list(stream_to_gradio(mock_agent, "test task"))
        # Verify that the content was yielded
        assert result == ["Hello"]

    def test_stream_to_gradio_multiple_deltas(self):
        # [FUNCTION DOC]
        # Checks that consecutive deltas yield the progressively accumulated text.
        # Inputs: none.
        # Outputs: none.
        # Exceptions: AssertionError on failure.
        # Side effects: none.
        """Test streaming multiple ChatMessageStreamDeltas"""
        # Create mock agent and stream deltas
        mock_agent = Mock()
        mock_delta1 = ChatMessageStreamDelta(content="Hello")
        mock_delta2 = ChatMessageStreamDelta(content=" world")
        mock_agent.run = Mock(return_value=[mock_delta1, mock_delta2])
        mock_agent.model = Mock()
        # Call stream_to_gradio
        result = list(stream_to_gradio(mock_agent, "test task"))
        # Verify that the content was accumulated and yielded
        assert result == ["Hello", "Hello world"]

    @pytest.mark.parametrize(
        "task,task_images,reset_memory,additional_args",
        [
            ("simple task", None, False, None),
            ("task with images", ["image1.png", "image2.png"], False, None),
            ("task with reset", None, True, None),
            ("task with args", None, False, {"arg1": "value1"}),
            ("complex task", ["image.png"], True, {"arg1": "value1", "arg2": "value2"}),
        ],
    )
    def test_stream_to_gradio_parameters(self, task, task_images, reset_memory, additional_args):
        # [FUNCTION DOC]
        # Parametrized check that task, images, reset flag and additional args are forwarded to agent.run with
        #   streaming enabled.
        # Inputs: task (str), task_images (list or None), reset_memory (bool), additional_args (dict or None).
        # Outputs: none.
        # Exceptions: AssertionError on failure.
        # Side effects: none.
        """Test that stream_to_gradio passes parameters correctly to agent.run"""
        # Create mock agent
        mock_agent = Mock()
        mock_agent.run = Mock(return_value=[])
        # Call stream_to_gradio
        list(
            stream_to_gradio(
                mock_agent,
                task=task,
                task_images=task_images,
                reset_agent_memory=reset_memory,
                additional_args=additional_args,
            )
        )
        # Verify that agent.run was called with the right parameters
        mock_agent.run.assert_called_once_with(
            task, images=task_images, stream=True, reset=reset_memory, additional_args=additional_args
        )


class TestPullMessagesFromStep:
    # [CLASS DOC]
    # pytest test class for pull_messages_from_step across action, planning, final answer and unsupported
    #   steps.
    # Exceptions: assertion failures on test failure.
    # Side effects: none beyond temp files from image handling.
    def test_action_step_basic(
        self,
    ):
        # [FUNCTION DOC]
        # Checks an action step with model output and logs yields 5 messages: step header, output, logs,
        #   footnote with tokens and 2.5s duration, separator.
        # Inputs: none.
        # Outputs: none.
        # Exceptions: AssertionError on failure.
        # Side effects: none.
        """Test basic ActionStep processing."""
        step = ActionStep(
            step_number=1,
            model_output="This is the model output",
            observations="Some execution logs",
            error=None,
            timing=Timing(start_time=1.0, end_time=3.5),
            token_usage=TokenUsage(input_tokens=100, output_tokens=50),
        )
        messages = list(pull_messages_from_step(step))
        assert len(messages) == 5  # step number, model_output, logs, footnote, divider
        for message, expected_content in zip(
            messages,
            [
                "**Step 1**",
                "This is the model output",
                "execution logs",
                "Input tokens: 100 | Output tokens: 50 | Duration: 2.5",
                "-----",
            ],
        ):
            assert expected_content in message.content

    def test_action_step_with_tool_calls(self):
        # [FUNCTION DOC]
        # Checks a tool call with an "answer" dict argument shows the answer as content and the tool name in
        #   the title.
        # Inputs: none.
        # Outputs: none.
        # Exceptions: AssertionError on failure.
        # Side effects: none.
        """Test ActionStep with tool calls."""
        step = ActionStep(
            step_number=2,
            tool_calls=[ToolCall(name="test_tool", arguments={"answer": "Test answer"}, id="tool_call_1")],
            observations="Tool execution logs",
            timing=Timing(start_time=1.0, end_time=2.5),
            token_usage=TokenUsage(input_tokens=100, output_tokens=50),
        )
        messages = list(pull_messages_from_step(step))
        assert len(messages) == 5  # step, tool call, logs, footnote, divider
        assert messages[1].content == "Test answer"
        assert "Used tool test_tool" in messages[1].metadata["title"]

    @pytest.mark.parametrize(
        "tool_name, args, expected",
        [
            ("python_interpreter", "print('Hello')", "```python\nprint('Hello')\n```"),
            ("regular_tool", {"key": "value"}, "{'key': 'value'}"),
            ("string_args_tool", "simple string", "simple string"),
        ],
    )
    def test_action_step_tool_call_formats(self, tool_name, args, expected):
        # [FUNCTION DOC]
        # Parametrized check of tool call rendering: python_interpreter code is fenced, dicts without "answer"
        #   are stringified, strings pass through.
        # Inputs: tool_name (str), args (str or dict), expected (str).
        # Outputs: none.
        # Exceptions: AssertionError on failure; StopIteration if no tool message is found.
        # Side effects: none.
        """Test different formats of tool calls."""
        tool_call = Mock()
        tool_call.name = tool_name
        tool_call.arguments = args
        step = ActionStep(
            step_number=1,
            tool_calls=[tool_call],
            timing=Timing(start_time=1.0, end_time=2.5),
            token_usage=TokenUsage(input_tokens=100, output_tokens=50),
        )
        messages = list(pull_messages_from_step(step))
        tool_message = next(
            msg
            for msg in messages
            if msg.role == "assistant" and msg.metadata and msg.metadata.get("title", "").startswith("🛠️")
        )
        assert expected in tool_message.content

    def test_action_step_with_error(self):
        # [FUNCTION DOC]
        # Checks that an error on the step produces a message containing the error text.
        # Inputs: none.
        # Outputs: none.
        # Exceptions: AssertionError on failure.
        # Side effects: none.
        # KNOWN ISSUE: passes a plain string as the error although the field is typed as AgentError.
        """Test ActionStep with error."""
        step = ActionStep(
            step_number=3,
            error="This is an error message",
            timing=Timing(start_time=1.0, end_time=2.0),
            token_usage=TokenUsage(input_tokens=100, output_tokens=200),
        )
        messages = list(pull_messages_from_step(step))
        error_message = next((m for m in messages if "error" in str(m.content).lower()), None)
        assert error_message is not None
        assert "This is an error message" in error_message.content

    def test_action_step_with_images(self):
        # [FUNCTION DOC]
        # Checks that each observation image yields an image message using the path from a patched AgentImage.
        # Inputs: none.
        # Outputs: none.
        # Exceptions: AssertionError on failure.
        # Side effects: none.
        """Test ActionStep with observation images."""
        step = ActionStep(
            step_number=4,
            observations_images=["image1.png", "image2.jpg"],
            token_usage=TokenUsage(input_tokens=100, output_tokens=200),
            timing=Timing(start_time=1.0, end_time=2.0),
        )
        with patch("smolagents.gradio_ui.AgentImage") as mock_agent_image:
            mock_agent_image.return_value.to_string.side_effect = lambda: "path/to/image.png"
            messages = list(pull_messages_from_step(step))
            image_messages = [m for m in messages if "image" in str(m).lower()]
            assert len(image_messages) == 2
            assert "path/to/image.png" in str(image_messages[0])

    @pytest.mark.parametrize(
        "skip_model_outputs, expected_messages_length, token_usage",
        [(False, 4, TokenUsage(input_tokens=80, output_tokens=30)), (True, 2, None)],
    )
    def test_planning_step(self, skip_model_outputs, expected_messages_length, token_usage):
        # [FUNCTION DOC]
        # Parametrized check of planning step message count (4 normally, 2 when skipping outputs) and
        #   contents, including absence of token info when usage is None.
        # Inputs: skip_model_outputs (bool), expected_messages_length (int), token_usage (TokenUsage or None).
        # Outputs: none.
        # Exceptions: AssertionError on failure.
        # Side effects: none.
        # KNOWN ISSUE: the final no-token assertion checks only the last message (the separator), so it is
        #   vacuous.
        """Test PlanningStep processing."""
        step = PlanningStep(
            plan="1. First step\n2. Second step",
            model_input_messages=Mock(),
            model_output_message=Mock(),
            token_usage=token_usage,
            timing=Timing(start_time=1.0, end_time=2.0),
        )
        messages = list(pull_messages_from_step(step, skip_model_outputs=skip_model_outputs))
        assert len(messages) == expected_messages_length  # [header, plan,] footnote, divider
        expected_contents = [
            "**Planning step**",
            "1. First step\n2. Second step",
            "Input tokens: 80 | Output tokens: 30" if token_usage else "",
            "-----",
        ]
        for message, expected_content in zip(messages, expected_contents[-expected_messages_length:]):
            assert expected_content in message.content

        if not token_usage:
            assert "Input tokens: 80 | Output tokens: 30" not in message.content

    @pytest.mark.parametrize(
        "answer_type, answer_value, expected_content",
        [
            (AgentText, "This is a text answer", "**Final answer:**\nThis is a text answer\n"),
            (lambda: "Plain string", "Plain string", "**Final answer:** Plain string"),
        ],
    )
    def test_final_answer_step(self, answer_type, answer_value, expected_content):
        # [FUNCTION DOC]
        # Parametrized check that AgentText and plain-string final answers are formatted with the "Final
        #   answer" prefix as expected.
        # Inputs: answer_type (class or factory), answer_value (str), expected_content (str).
        # Outputs: none.
        # Exceptions: AssertionError on failure.
        # Side effects: none.
        """Test FinalAnswerStep with different answer types."""
        try:
            final_answer = answer_type()
        except TypeError:
            with patch.object(answer_type, "to_string", return_value=answer_value):
                final_answer = answer_type(answer_value)
        step = FinalAnswerStep(
            output=final_answer,
        )
        messages = list(pull_messages_from_step(step))
        assert len(messages) == 1
        assert messages[0].content == expected_content

    def test_final_answer_step_image(self):
        # [FUNCTION DOC]
        # Checks an AgentImage final answer yields a file dict with the path and image/png mime.
        # Inputs: none.
        # Outputs: none.
        # Exceptions: AssertionError on failure.
        # Side effects: none.
        """Test FinalAnswerStep with image answer."""
        with patch.object(AgentImage, "to_string", return_value="path/to/image.png"):
            step = FinalAnswerStep(output=AgentImage("path/to/image.png"))
            messages = list(pull_messages_from_step(step))
            assert len(messages) == 1
            assert messages[0].content["path"] == "path/to/image.png"
            assert messages[0].content["mime_type"] == "image/png"

    def test_final_answer_step_audio(self):
        # [FUNCTION DOC]
        # Checks an AgentAudio final answer yields a file dict with the path and audio/wav mime (requires
        #   soundfile and torch).
        # Inputs: none.
        # Outputs: none.
        # Exceptions: AssertionError on failure; ModuleNotFoundError if audio extras are missing.
        # Side effects: none.
        """Test FinalAnswerStep with audio answer."""
        with patch.object(AgentAudio, "to_string", return_value="path/to/audio.wav"):
            step = FinalAnswerStep(output=AgentAudio("path/to/audio.wav"))
            messages = list(pull_messages_from_step(step))
            assert len(messages) == 1
            assert messages[0].content["path"] == "path/to/audio.wav"
            assert messages[0].content["mime_type"] == "audio/wav"

    def test_unsupported_step_type(self):
        # [FUNCTION DOC]
        # Checks that a non-step object raises ValueError mentioning an unsupported step type.
        # Inputs: none.
        # Outputs: none.
        # Exceptions: AssertionError on failure.
        # Side effects: none.
        """Test handling of unsupported step types."""

        class UnsupportedStep(Mock):
            # [CLASS DOC]
            # HELPER: Empty Mock subclass used as an unknown step type.
            # Inputs: Mock constructor args.
            # Exceptions: none raised directly.
            # Side effects: none.
            pass

        step = UnsupportedStep()
        with pytest.raises(ValueError, match="Unsupported step type"):
            list(pull_messages_from_step(step))
