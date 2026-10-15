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
# Purpose: Tests for smolagents.default_tools: webpage visiting, DuckDuckGo kwargs passthrough, the Python
#   interpreter tool (outputs, agent types, imports, timeouts), SpeechToTextTool construction, the Exa engine of
#   WebSearchTool (mocked HTTP), and WikipediaSearchTool (live).
# Dependencies: pytest, unittest, unittest.mock.patch, ToolTesterMixin from tests.test_tools, the require_run_all
#   marker, ExecutionTimeoutError and the agent type mapping.
# Side effects: several tests hit the network (huggingface.co, DuckDuckGo, Wikipedia); SpeechToTextTool tests
#   require transformers.
import unittest
from unittest.mock import patch

import pytest

from smolagents.agent_types import _AGENT_TYPE_MAPPING
from smolagents.default_tools import (
    DuckDuckGoSearchTool,
    PythonInterpreterTool,
    SpeechToTextTool,
    VisitWebpageTool,
    WebSearchTool,
    WikipediaSearchTool,
)
from smolagents.local_python_executor import ExecutionTimeoutError

from .test_tools import ToolTesterMixin
from .utils.markers import require_run_all


class DefaultToolTests(unittest.TestCase):
    # [CLASS DOC]
    # unittest TestCase grouping live network tests for VisitWebpageTool and DuckDuckGoSearchTool.
    # Constructor args: standard unittest. No custom attributes.
    def test_visit_webpage(self):
        # [FUNCTION DOC]
        # Visits the Hugging Face homepage and checks the markdown contains the site title.
        # Inputs: self. Outputs: none.
        # Exceptions: AssertionError on failure.
        # Side effects: network request to huggingface.co.
        # KNOWN ISSUE: passes the whole arguments dict as a single positional argument; this relies on Tool's call
        #   handling unpacking a lone dict into keyword arguments. Depends on live page content.
        arguments = {"url": "https://huggingface.co/"}
        result = VisitWebpageTool()(arguments)
        assert isinstance(result, str)
        assert "Hugging Face – The AI community building the future" in result

    @require_run_all
    def test_ddgs_with_kwargs(self):
        # [FUNCTION DOC]
        # Checks that extra kwargs (timeout) reach the DDGS client and a search returns a string. Runs only when the
        #   run-all marker is enabled.
        # Inputs: self. Outputs: none. Exceptions: AssertionError on failure; search errors propagate.
        # Side effects: network request to DuckDuckGo.
        result = DuckDuckGoSearchTool(timeout=20)("DeepSeek parent company")
        assert isinstance(result, str)


class TestPythonInterpreterTool(ToolTesterMixin):
    # [CLASS DOC]
    # pytest test class for PythonInterpreterTool, reusing generic checks from ToolTesterMixin.
    # Key attributes: tool (a PythonInterpreterTool allowing numpy, set up before each test).
    def setup_method(self):
        # [FUNCTION DOC]
        # HELPER: Creates a PythonInterpreterTool with numpy authorized and calls its setup method.
        # Inputs: self. Outputs: none. Exceptions: none raised directly. Side effects: sets self.tool.
        self.tool = PythonInterpreterTool(authorized_imports=["numpy"])
        self.tool.setup()

    def test_exact_match_arg(self):
        # [FUNCTION DOC]
        # Checks that a positional code argument evaluating to 4.0 yields the exact formatted output with empty stdout.
        # Inputs: self. Outputs: none. Exceptions: AssertionError on mismatch. Side effects: none.
        result = self.tool("(2 / 2) * 4")
        assert result == "Stdout:\n\nOutput: 4.0"

    def test_exact_match_kwarg(self):
        # [FUNCTION DOC]
        # Same as test_exact_match_arg but passing the code as a keyword argument.
        # Inputs: self. Outputs: none. Exceptions: AssertionError on mismatch. Side effects: none.
        result = self.tool(code="(2 / 2) * 4")
        assert result == "Stdout:\n\nOutput: 4.0"

    def test_agent_type_output(self):
        # [FUNCTION DOC]
        # Checks that with input/output sanitization enabled the result is an instance of the agent type mapped from the
        #   tool's output type.
        # Inputs: self. Outputs: none. Exceptions: AssertionError on failure. Side effects: none.
        inputs = ["2 * 2"]
        output = self.tool(*inputs, sanitize_inputs_outputs=True)
        output_type = _AGENT_TYPE_MAPPING[self.tool.output_type]
        assert isinstance(output, output_type)

    def test_agent_types_inputs(self):
        # [FUNCTION DOC]
        # Wraps each input into the agent type mapped from the declared input type, then calls the tool with sanitization
        #   and checks the output agent type.
        # Inputs: self. Outputs: none. Exceptions: AssertionError on failure. Side effects: none.
        # KNOWN ISSUE: the wrapped inputs list is built but never used; the tool is called with the raw inputs, so the
        #   agent-typed input path is not actually tested.
        inputs = ["2 * 2"]
        _inputs = []

        for _input, expected_input in zip(inputs, self.tool.inputs.values()):
            input_type = expected_input["type"]
            if isinstance(input_type, list):
                _inputs.append([_AGENT_TYPE_MAPPING[_input_type](_input) for _input_type in input_type])
            else:
                _inputs.append(_AGENT_TYPE_MAPPING[input_type](_input))

        # Should not raise an error
        output = self.tool(*inputs, sanitize_inputs_outputs=True)
        output_type = _AGENT_TYPE_MAPPING[self.tool.output_type]
        assert isinstance(output, output_type)

    def test_imports_work(self):
        # [FUNCTION DOC]
        # Checks that importing an authorized module (numpy) does not produce a "not allowed" message.
        # Inputs: self. Outputs: none. Exceptions: AssertionError on failure. Side effects: imports numpy.
        result = self.tool("import numpy as np")
        assert "import from numpy is not allowed" not in result.lower()

    def test_unauthorized_imports_fail(self):
        # [FUNCTION DOC]
        # Checks that importing an unauthorized module (sympy) raises an exception whose text mentions sympy.
        # Inputs: self. Outputs: none. Exceptions: AssertionError on failure. Side effects: none.
        # KNOWN ISSUE: inspects the string form of the pytest ExceptionInfo rather than of the exception value.
        with pytest.raises(Exception) as e:
            self.tool("import sympy as sp")
        assert "sympy" in str(e).lower()

    def test_custom_timeout(self):
        # [FUNCTION DOC]
        # Checks that a 1-second timeout makes code sleeping 2 seconds raise ExecutionTimeoutError with the expected
        #   message.
        # Inputs: self. Outputs: none. Exceptions: AssertionError on failure. Side effects: sleeps about 1-2 seconds.
        """Test that PythonInterpreterTool respects custom timeout."""
        tool = PythonInterpreterTool(authorized_imports=["time"], timeout_seconds=1)
        tool.setup()

        # Code that sleeps for 2 seconds should timeout with 1-second limit
        code = """
import time
time.sleep(2)
"""
        with pytest.raises(ExecutionTimeoutError, match="Code execution exceeded the maximum execution time"):
            tool(code)

    def test_disabled_timeout(self):
        # [FUNCTION DOC]
        # Checks that with the timeout disabled, code sleeping half a second completes and its last value appears in the
        #   output.
        # Inputs: self. Outputs: none. Exceptions: AssertionError on failure. Side effects: sleeps half a second.
        """Test that PythonInterpreterTool can disable timeout."""
        tool = PythonInterpreterTool(authorized_imports=["time"], timeout_seconds=None)
        tool.setup()

        # Code should complete even without timeout
        code = """
import time
time.sleep(0.5)
result = "completed"
"""
        result = tool(code)
        assert "completed" in result


class TestSpeechToTextTool:
    # [CLASS DOC]
    # pytest tests checking SpeechToTextTool wires the Whisper processor and model classes. Requires transformers.
    def test_new_instance(self):
        # [FUNCTION DOC]
        # Creates a SpeechToTextTool with defaults and checks the processor and model classes are the Whisper ones.
        # Inputs: self. Outputs: none. Exceptions: AssertionError or ImportError. Side effects: none (model not loaded).
        from transformers.models.whisper import WhisperForConditionalGeneration, WhisperProcessor

        tool = SpeechToTextTool()
        assert tool is not None
        assert tool.pre_processor_class == WhisperProcessor
        assert tool.model_class == WhisperForConditionalGeneration

    def test_initialization(self):
        # [FUNCTION DOC]
        # Same checks as test_new_instance but with an explicit (dummy) model id.
        # Inputs: self. Outputs: none. Exceptions: AssertionError or ImportError. Side effects: none (model not loaded).
        from transformers.models.whisper import WhisperForConditionalGeneration, WhisperProcessor

        tool = SpeechToTextTool(model="dummy_model_id")
        assert tool is not None
        assert tool.pre_processor_class == WhisperProcessor
        assert tool.model_class == WhisperForConditionalGeneration


class TestWebSearchToolExa:
    # [CLASS DOC]
    # Tests of WebSearchTool's Exa engine with the environment and requests.post patched (no real network).
    """Tests for the Exa engine in WebSearchTool."""

    def test_exa_missing_api_key(self):
        # [FUNCTION DOC]
        # Checks that calling the tool with an empty environment raises ValueError mentioning EXA_API_KEY.
        # Inputs: self. Outputs: none. Exceptions: AssertionError on failure.
        # Side effects: temporarily clears os.environ within the patch context.
        tool = WebSearchTool(engine="exa")
        with patch.dict("os.environ", {}, clear=True):
            with pytest.raises(ValueError, match="EXA_API_KEY"):
                tool("test query")

    def test_exa_search_results(self):
        # [FUNCTION DOC]
        # Mocks two Exa results and checks the markdown output (heading, links, joined highlights) and the request
        #   headers and JSON payload (integration header, numResults equal to max_results, highlights content).
        # Inputs: self. Outputs: none. Exceptions: AssertionError on failure.
        # Side effects: patches os.environ and requests.post temporarily.
        mock_response_data = {
            "results": [
                {
                    "title": "Exa AI",
                    "url": "https://exa.ai",
                    "highlights": ["Exa is a search engine for AI."],
                },
                {
                    "title": "Hugging Face",
                    "url": "https://huggingface.co",
                    "highlights": ["The AI community", "building the future."],
                },
            ]
        }
        tool = WebSearchTool(engine="exa", max_results=2)
        with patch.dict("os.environ", {"EXA_API_KEY": "test-key"}):
            with patch("requests.post") as mock_post:
                mock_post.return_value.json.return_value = mock_response_data
                mock_post.return_value.raise_for_status = lambda: None
                result = tool("test query")

        assert "## Search Results" in result
        assert "[Exa AI](https://exa.ai)" in result
        assert "[Hugging Face](https://huggingface.co)" in result
        assert "Exa is a search engine for AI." in result
        assert "The AI community building the future." in result

        # Verify the API was called with correct headers and payload
        call_kwargs = mock_post.call_args
        assert call_kwargs.kwargs["headers"]["x-exa-integration"] == "smolagents"
        assert call_kwargs.kwargs["json"]["numResults"] == 2
        assert call_kwargs.kwargs["json"]["contents"] == {"highlights": True}

    def test_exa_no_results(self):
        # [FUNCTION DOC]
        # Checks that an empty Exa result list makes the tool raise the "No results found" exception.
        # Inputs: self. Outputs: none. Exceptions: AssertionError on failure. Side effects: temporary patches.
        tool = WebSearchTool(engine="exa")
        with patch.dict("os.environ", {"EXA_API_KEY": "test-key"}):
            with patch("requests.post") as mock_post:
                mock_post.return_value.json.return_value = {"results": []}
                mock_post.return_value.raise_for_status = lambda: None
                with pytest.raises(Exception, match="No results found"):
                    tool("obscure query")

    def test_exa_empty_highlights(self):
        # [FUNCTION DOC]
        # Checks that a result without a highlights key is still rendered as a markdown link.
        # Inputs: self. Outputs: none. Exceptions: AssertionError on failure. Side effects: temporary patches.
        mock_response_data = {
            "results": [
                {
                    "title": "No Highlights Page",
                    "url": "https://example.com",
                }
            ]
        }
        tool = WebSearchTool(engine="exa")
        with patch.dict("os.environ", {"EXA_API_KEY": "test-key"}):
            with patch("requests.post") as mock_post:
                mock_post.return_value.json.return_value = mock_response_data
                mock_post.return_value.raise_for_status = lambda: None
                result = tool("test")

        assert "[No Highlights Page](https://example.com)" in result

    def test_exa_null_highlights(self):
        # [FUNCTION DOC]
        # Checks that a result whose highlights value is null is still rendered as a markdown link.
        # Inputs: self. Outputs: none. Exceptions: AssertionError on failure. Side effects: temporary patches.
        mock_response_data = {
            "results": [
                {
                    "title": "Null Highlights Page",
                    "url": "https://example.com",
                    "highlights": None,
                }
            ]
        }
        tool = WebSearchTool(engine="exa")
        with patch.dict("os.environ", {"EXA_API_KEY": "test-key"}):
            with patch("requests.post") as mock_post:
                mock_post.return_value.json.return_value = mock_response_data
                mock_post.return_value.raise_for_status = lambda: None
                result = tool("test")

        assert "[Null Highlights Page](https://example.com)" in result


@pytest.mark.parametrize(
    "language, content_type, extract_format, query",
    [
        ("en", "summary", "HTML", "Python_(programming_language)"),  # English, Summary Mode, HTML format
        ("en", "text", "WIKI", "Python_(programming_language)"),  # English, Full Text Mode, WIKI format
        ("es", "summary", "HTML", "Python_(lenguaje_de_programación)"),  # Spanish, Summary Mode, HTML format
        ("es", "text", "WIKI", "Python_(lenguaje_de_programación)"),  # Spanish, Full Text Mode, WIKI format
    ],
)
def test_wikipedia_search(language, content_type, extract_format, query):
    # [FUNCTION DOC]
    # Parametrized live test of WikipediaSearchTool over English/Spanish, summary/text, HTML/WIKI combinations.
    # Inputs: language (str), content_type (str), extract_format (str), query (str page title), from parametrize.
    # Outputs: none. Checks the result is a string with the page and read-more markers, and that summaries have fewer
    #   than 1000 words while full texts have more.
    # Exceptions: AssertionError on failure.
    # Side effects: network requests to Wikipedia.
    tool = WikipediaSearchTool(
        user_agent="TestAgent (test@example.com)",
        language=language,
        content_type=content_type,
        extract_format=extract_format,
    )

    result = tool.forward(query)

    assert isinstance(result, str), "Output should be a string"
    assert "✅ **Wikipedia Page:**" in result, "Response should contain Wikipedia page title"
    assert "🔗 **Read more:**" in result, "Response should contain Wikipedia page URL"

    if content_type == "summary":
        assert len(result.split()) < 1000, "Summary mode should return a shorter text"
    if content_type == "text":
        assert len(result.split()) > 1000, "Full text mode should return a longer text"
