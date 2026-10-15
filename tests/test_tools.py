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
# Pytest suite for smolagents.tools: Tool attribute/signature validation, prompt rendering (code and tool-
#   calling), the tool decorator (schema extraction, nullable/Literal/array/any handling, source extraction with
#   extra decorators and multiline signatures), serialization (to_dict, from_dict, save),
#   ToolCollection.from_mcp (mocked and real MCP servers over stdio, streamable HTTP and SSE),
#   launch_gradio_demo, and validate_tool_arguments.
# Inputs: depends on pytest (plus shared fixtures from conftest such as no_input_tool, example_tool,
#   boolean_default_tool_class, optional_input_tool_function and pytest-datadir's shared_datadir), mcp, numpy,
#   PIL, unittest.mock, smolagents.agent_types and smolagents.tools, and the require_run_all marker.
# Outputs: defines ToolTesterMixin (reusable checks for tool test classes), test classes TestTool,
#   TestToolDecorator, TestToolCollection, module fixtures for MCP mocking, and module-level test functions.
# Exceptions: none raised at import time beyond missing dependencies.
# Side effects: none at import; some tests write to tmp_path, spawn subprocesses and bind local port 8000.
import inspect
import os
import warnings
from textwrap import dedent
from typing import Any, Literal
from unittest.mock import MagicMock, patch

import mcp
import numpy as np
import PIL.Image
import pytest

from smolagents.agent_types import _AGENT_TYPE_MAPPING
from smolagents.tools import AUTHORIZED_TYPES, Tool, ToolCollection, launch_gradio_demo, tool, validate_tool_arguments

from .utils.markers import require_run_all


class ToolTesterMixin:
    # [CLASS DOC]
    # Mixin with generic checks reusable by test classes of concrete tools; the host class must set self.tool.
    # Inputs: requires a tool attribute (Tool instance) on the test class.
    # Outputs: contributes test methods and the create_inputs fixture.
    # Exceptions: none raised directly.
    # Side effects: none.
    def test_inputs_output(self):
        # [FUNCTION DOC]
        # Checks the tool has inputs (dict) and output_type, every input spec has a type within
        #   AUTHORIZED_TYPES and a string description, and output_type is authorized.
        # Inputs: self (with self.tool).
        # Outputs: none (assertions).
        # Exceptions: AssertionError on failure.
        # Side effects: none.
        # KNOWN ISSUE: list-valued input types (allowed by Tool validation) would fail the membership check
        #   here.
        assert hasattr(self.tool, "inputs")
        assert hasattr(self.tool, "output_type")

        inputs = self.tool.inputs
        assert isinstance(inputs, dict)

        for _, input_spec in inputs.items():
            assert "type" in input_spec
            assert "description" in input_spec
            assert input_spec["type"] in AUTHORIZED_TYPES
            assert isinstance(input_spec["description"], str)

        output_type = self.tool.output_type
        assert output_type in AUTHORIZED_TYPES

    def test_common_attributes(self):
        # [FUNCTION DOC]
        # Checks the tool exposes description, name, inputs and output_type attributes.
        # Inputs: self (with self.tool).
        # Outputs: none.
        # Exceptions: AssertionError on failure.
        # Side effects: none.
        assert hasattr(self.tool, "description")
        assert hasattr(self.tool, "name")
        assert hasattr(self.tool, "inputs")
        assert hasattr(self.tool, "output_type")

    def test_agent_type_output(self, create_inputs):
        # [FUNCTION DOC]
        # Calls the tool with generated sample inputs and sanitization enabled, and checks the output is an
        #   instance of the agent type mapped from output_type (skipped when output_type is any).
        # Inputs: self (with self.tool), create_inputs fixture.
        # Outputs: none.
        # Exceptions: AssertionError on failure; tool errors propagate.
        # Side effects: whatever the tool does.
        inputs = create_inputs(self.tool.inputs)
        output = self.tool(**inputs, sanitize_inputs_outputs=True)
        if self.tool.output_type != "any":
            agent_type = _AGENT_TYPE_MAPPING[self.tool.output_type]
            assert isinstance(output, agent_type)

    @pytest.fixture
    def create_inputs(self, shared_datadir):
        # [FUNCTION DOC]
        # HELPER: Fixture returning a factory that builds sample inputs for a tool's inputs spec.
        # Inputs: shared_datadir (pytest-datadir path containing a sample PNG).
        # Outputs: the factory function.
        # Exceptions: none raised directly.
        # Side effects: none.
        def _create_inputs(tool_inputs: dict[str, dict[str | type, str]]) -> dict[str, Any]:
            # [FUNCTION DOC]
            # HELPER: Builds a dict of sample values: string gives a text, image gives the sample PNG resized
            #   to 512x512, audio gives a numpy array of 3000 ones.
            # Inputs: tool_inputs (dict of input name to spec dict with a type).
            # Outputs: dict of input name to sample value.
            # Exceptions: ValueError for any other type.
            # Side effects: opens an image file from disk.
            inputs = {}

            for input_name, input_desc in tool_inputs.items():
                input_type = input_desc["type"]

                if input_type == "string":
                    inputs[input_name] = "Text input"
                elif input_type == "image":
                    inputs[input_name] = PIL.Image.open(shared_datadir / "000000039769.png").resize((512, 512))
                elif input_type == "audio":
                    inputs[input_name] = np.ones(3000)
                else:
                    raise ValueError(f"Invalid type requested: {input_type}")

            return inputs

        return _create_inputs


class TestTool:
    # [CLASS DOC]
    # Test class for Tool construction, validation, prompt rendering, the tool decorator and serialization.
    # Inputs: uses pytest fixtures (tmp_path, request, conftest tool fixtures).
    # Outputs: test methods.
    # Exceptions: none raised directly.
    # Side effects: none beyond individual tests.
    @pytest.mark.parametrize(
        "type_value, should_raise_error, error_contains",
        [
            # Valid cases
            ("string", False, None),
            (["string", "number"], False, None),
            # Invalid cases
            ("invalid_type", ValueError, "must be one of"),
            (["string", "invalid_type"], ValueError, "must be one of"),
            ([123, "string"], TypeError, "when type is a list, all elements must be strings"),
            (123, TypeError, "must be a string or list of strings"),
        ],
    )
    def test_tool_input_type_validation(self, type_value, should_raise_error, error_contains):
        # [FUNCTION DOC]
        # Parametrized test: defining a tool with a given inputs type value either succeeds (string or list of
        #   authorized strings) or raises the expected error with the expected message (ValueError for
        #   unauthorized types, TypeError for non-string list elements or non-string/non-list types).
        # Inputs: type_value, should_raise_error (False or exception class), error_contains (str or None).
        # Outputs: none.
        # Exceptions: AssertionError on failure.
        # Side effects: none.
        """Test the validation of the type property in tool inputs."""

        # Define a tool class with the test type value
        def create_tool():
            # [FUNCTION DOC]
            # HELPER: Defines a Tool subclass using the parametrized type value and returns an instance of it.
            # Inputs: none (closes over type_value).
            # Outputs: a Tool instance.
            # Exceptions: validation errors raised at instantiation.
            # Side effects: none.
            class TestTool(Tool):
                # [CLASS DOC]
                # HELPER: Test-local Tool subclass; single text input whose type comes from the parametrized
                #   value.
                # Inputs: no constructor args (uses class attributes name, description, inputs, output_type).
                # Outputs: instance created via the base Tool constructor with post-init validation.
                # Exceptions: validation errors at instantiation, as exercised by the enclosing test.
                # Side effects: none.
                name = "test_tool"
                description = "A tool for testing type validation"
                inputs = {"text": {"type": type_value, "description": "Some input"}}
                output_type = "string"

                def forward(self, text) -> str:
                    # [FUNCTION DOC]
                    # HELPER: forward implementation of a test-local tool; returns text unchanged.
                    # Inputs: self and the parameters in its signature.
                    # Outputs: as returned by its body.
                    # Exceptions: none raised directly.
                    # Side effects: none.
                    return text

            return TestTool()

        # Check if we expect this to raise an exception
        if should_raise_error:
            with pytest.raises(should_raise_error) as exc_info:
                create_tool()
            # Verify the error message contains expected text
            assert error_contains in str(exc_info.value)
        else:
            # Should not raise an exception
            tool = create_tool()
            assert isinstance(tool, Tool)

    @pytest.mark.parametrize(
        "tool_fixture, expected_output",
        [
            ("no_input_tool", 'def no_input_tool() -> string:\n    """Tool with no inputs\n    """'),
            (
                "single_input_tool",
                'def single_input_tool(text: string) -> string:\n    """Tool with one input\n\n    Args:\n        text: Input text\n    """',
            ),
            (
                "multi_input_tool",
                'def multi_input_tool(text: string, count: integer) -> object:\n    """Tool with multiple inputs\n\n    Args:\n        text: Text input\n        count: Number count\n    """',
            ),
            (
                "multiline_description_tool",
                'def multiline_description_tool(input: string) -> string:\n    """This is a tool with\n    multiple lines\n    in the description\n\n    Args:\n        input: Some input\n    """',
            ),
        ],
    )
    def test_tool_to_code_prompt_output_format(self, tool_fixture, expected_output, request):
        # [FUNCTION DOC]
        # Parametrized test: to_code_prompt of conftest tools (no input, single input, multiple inputs,
        #   multiline description) equals the exact expected stub text.
        # Inputs: tool_fixture (fixture name), expected_output (str), request.
        # Outputs: none.
        # Exceptions: AssertionError on mismatch.
        # Side effects: none.
        """Test that to_code_prompt generates properly formatted and indented output."""
        tool = request.getfixturevalue(tool_fixture)
        code_prompt = tool.to_code_prompt()
        assert code_prompt == expected_output

    @pytest.mark.parametrize(
        "tool_fixture, expected_output",
        [
            (
                "no_input_tool",
                "no_input_tool: Tool with no inputs\n    Takes inputs: {}\n    Returns an output of type: string",
            ),
            (
                "single_input_tool",
                "single_input_tool: Tool with one input\n    Takes inputs: {'text': {'type': 'string', 'description': 'Input text'}}\n    Returns an output of type: string",
            ),
            (
                "multi_input_tool",
                "multi_input_tool: Tool with multiple inputs\n    Takes inputs: {'text': {'type': 'string', 'description': 'Text input'}, 'count': {'type': 'integer', 'description': 'Number count'}}\n    Returns an output of type: object",
            ),
            (
                "multiline_description_tool",
                "multiline_description_tool: This is a tool with\nmultiple lines\nin the description\n    Takes inputs: {'input': {'type': 'string', 'description': 'Some input'}}\n    Returns an output of type: string",
            ),
        ],
    )
    def test_tool_to_tool_calling_prompt_output_format(self, tool_fixture, expected_output, request):
        # [FUNCTION DOC]
        # Parametrized test: to_tool_calling_prompt of the same conftest tools equals the exact expected text.
        # Inputs: tool_fixture, expected_output, request.
        # Outputs: none.
        # Exceptions: AssertionError on mismatch.
        # Side effects: none.
        """Test that to_tool_calling_prompt generates properly formatted output."""
        tool = request.getfixturevalue(tool_fixture)
        tool_calling_prompt = tool.to_tool_calling_prompt()
        assert tool_calling_prompt == expected_output

    def test_tool_init_with_decorator(self):
        # [FUNCTION DOC]
        # Checks that the tool decorator maps a float return hint to output_type number.
        # Inputs: self.
        # Outputs: none.
        # Exceptions: AssertionError on failure.
        # Side effects: none.
        @tool
        def coolfunc(a: str, b: int) -> float:
            # [FUNCTION DOC]
            # HELPER: Function decorated with the tool decorator (or passed to it) inside a test; two args,
            #   float return hint.
            # Inputs: see its type-hinted parameters.
            # Outputs: per its body; only its schema/metadata is usually examined.
            # Exceptions: none raised directly.
            # KNOWN ISSUE: the body returns a tuple despite the float hint (harmless, never called).
            # Side effects: none.
            """Cool function

            Args:
                a: The first argument
                b: The second one
            """
            return b + 2, a

        assert coolfunc.output_type == "number"

    def test_tool_init_vanilla(self):
        # [FUNCTION DOC]
        # Checks a hand-written Tool subclass instantiates and keeps its inputs keys.
        # Inputs: self.
        # Outputs: none.
        # Exceptions: AssertionError on failure.
        # Side effects: none.
        class HFModelDownloadsTool(Tool):
            # [CLASS DOC]
            # HELPER: Test-local Tool subclass; one string task input, string output.
            # Inputs: no constructor args (uses class attributes name, description, inputs, output_type).
            # Outputs: instance created via the base Tool constructor with post-init validation.
            # Exceptions: validation errors at instantiation, as exercised by the enclosing test.
            # Side effects: none.
            name = "model_download_counter"
            description = """
            This is a tool that returns the most downloaded model of a given task on the Hugging Face Hub.
            It returns the name of the checkpoint."""

            inputs = {
                "task": {
                    "type": "string",
                    "description": "the task category (such as text-classification, depth-estimation, etc)",
                }
            }
            output_type = "string"

            def forward(self, task: str) -> str:
                # [FUNCTION DOC]
                # HELPER: forward implementation of a test-local tool; returns a constant string.
                # Inputs: self and the parameters in its signature.
                # Outputs: as returned by its body.
                # Exceptions: none raised directly.
                # Side effects: none.
                return "best model"

        tool = HFModelDownloadsTool()
        assert list(tool.inputs.keys())[0] == "task"

    def test_tool_init_decorator_raises_issues(self):
        # [FUNCTION DOC]
        # Checks the tool decorator raises when the return hint is missing and when the docstring lacks an
        #   argument description, with the corresponding messages.
        # Inputs: self.
        # Outputs: none.
        # Exceptions: AssertionError on failure.
        # Side effects: none.
        with pytest.raises(Exception) as e:

            @tool
            def coolfunc(a: str, b: int):
                # [FUNCTION DOC]
                # HELPER: Function decorated with the tool decorator (or passed to it) inside a test; no
                #   return hint, so decoration is expected to fail.
                # Inputs: see its type-hinted parameters.
                # Outputs: per its body; only its schema/metadata is usually examined.
                # Exceptions: none raised directly.
                # Side effects: none.
                """Cool function

                Args:
                    a: The first argument
                    b: The second one
                """
                return a + b

            assert coolfunc.output_type == "number"
        assert "Tool return type not found" in str(e)

        with pytest.raises(Exception) as e:

            @tool
            def coolfunc(a: str, b: int) -> int:
                # [FUNCTION DOC]
                # HELPER: Function decorated with the tool decorator (or passed to it) inside a test;
                #   docstring misses argument b, so decoration is expected to fail.
                # Inputs: see its type-hinted parameters.
                # Outputs: per its body; only its schema/metadata is usually examined.
                # Exceptions: none raised directly.
                # Side effects: none.
                """Cool function

                Args:
                    a: The first argument
                """
                return b + a

            assert coolfunc.output_type == "number"
        assert "docstring has no description for the argument" in str(e)

    def test_saving_tool_raises_error_imports_outside_function(self, tmp_path):
        # [FUNCTION DOC]
        # Checks saving fails (error mentions np) both for a decorated function and for a Tool subclass whose
        #   forward uses a module imported outside the function.
        # Inputs: self, tmp_path.
        # Outputs: none.
        # Exceptions: AssertionError on failure.
        # Side effects: may write files under tmp_path.
        with pytest.raises(Exception) as e:
            import numpy as np

            @tool
            def get_current_time() -> str:
                # [FUNCTION DOC]
                # HELPER: Function decorated with the tool decorator (or passed to it) inside a test; uses
                #   numpy imported outside its body.
                # Inputs: see its type-hinted parameters.
                # Outputs: per its body; only its schema/metadata is usually examined.
                # Exceptions: none raised directly.
                # Side effects: none.
                """
                Gets the current time.
                """
                return str(np.random.random())

            get_current_time.save(tmp_path)

        assert "np" in str(e)

        # Also test with classic definition
        with pytest.raises(Exception) as e:

            class GetCurrentTimeTool(Tool):
                # [CLASS DOC]
                # HELPER: Test-local Tool subclass; no inputs, forward uses an outer numpy import.
                # Inputs: no constructor args (uses class attributes name, description, inputs, output_type).
                # Outputs: instance created via the base Tool constructor with post-init validation.
                # Exceptions: validation errors at instantiation, as exercised by the enclosing test.
                # Side effects: none.
                name = "get_current_time_tool"
                description = "Gets the current time"
                inputs = {}
                output_type = "string"

                def forward(self):
                    # [FUNCTION DOC]
                    # HELPER: forward implementation of a test-local tool; returns a random number as string
                    #   using outer numpy.
                    # Inputs: self and the parameters in its signature.
                    # Outputs: as returned by its body.
                    # Exceptions: none raised directly.
                    # Side effects: none.
                    return str(np.random.random())

            get_current_time = GetCurrentTimeTool()
            get_current_time.save(tmp_path)

        assert "np" in str(e)

    def test_tool_definition_raises_no_error_imports_in_function(self):
        # [FUNCTION DOC]
        # Checks that defining a decorated function and a Tool subclass that import inside their bodies raises
        #   nothing.
        # Inputs: self.
        # Outputs: none.
        # Exceptions: errors propagate on failure.
        # Side effects: none.
        # KNOWN ISSUE: the Tool subclass is defined but never instantiated, so its validation is not
        #   exercised.
        @tool
        def get_current_time() -> str:
            # [FUNCTION DOC]
            # HELPER: Function decorated with the tool decorator (or passed to it) inside a test; imports
            #   datetime inside its body.
            # Inputs: see its type-hinted parameters.
            # Outputs: per its body; only its schema/metadata is usually examined.
            # Exceptions: none raised directly.
            # Side effects: none.
            """
            Gets the current time.
            """
            from datetime import datetime

            return str(datetime.now())

        class GetCurrentTimeTool(Tool):
            # [CLASS DOC]
            # HELPER: Test-local Tool subclass; no inputs, forward imports datetime locally.
            # Inputs: no constructor args (uses class attributes name, description, inputs, output_type).
            # Outputs: instance created via the base Tool constructor with post-init validation.
            # Exceptions: validation errors at instantiation, as exercised by the enclosing test.
            # Side effects: none.
            name = "get_current_time_tool"
            description = "Gets the current time"
            inputs = {}
            output_type = "string"

            def forward(self):
                # [FUNCTION DOC]
                # HELPER: forward implementation of a test-local tool; returns the current time as string.
                # Inputs: self and the parameters in its signature.
                # Outputs: as returned by its body.
                # Exceptions: none raised directly.
                # Side effects: none.
                from datetime import datetime

                return str(datetime.now())

    def test_tool_to_dict_allows_no_arg_in_init(self):
        # [FUNCTION DOC]
        # Checks to_dict fails for a tool whose constructor has a required parameter, and succeeds when all
        #   constructor parameters have defaults.
        # Inputs: self.
        # Outputs: none.
        # Exceptions: AssertionError on failure.
        # Side effects: none.
        # KNOWN ISSUE: the test name says allows no arg while the docstring says cannot be saved; also the
        #   second instance is still named fail_tool.
        """Test that a tool cannot be saved with required args in init"""

        class FailTool(Tool):
            # [CLASS DOC]
            # HELPER: Test-local Tool subclass; constructor requires a url argument.
            # Inputs: no constructor args (uses class attributes name, description, inputs, output_type).
            # Outputs: instance created via the base Tool constructor with post-init validation.
            # Exceptions: validation errors at instantiation, as exercised by the enclosing test.
            # Side effects: none.
            name = "specific"
            description = "test description"
            inputs = {"string_input": {"type": "string", "description": "input description"}}
            output_type = "string"

            def __init__(self, url):
                # [FUNCTION DOC]
                # HELPER: Calls the base constructor and stores url.
                # Inputs: url (required).
                # Outputs: none.
                # Exceptions: none raised directly.
                # Side effects: sets self.url.
                # KNOWN ISSUE: passes self explicitly to the parent constructor (harmless, base ignores extra
                #   args).
                super().__init__(self)
                self.url = url

            def forward(self, string_input: str) -> str:
                # [FUNCTION DOC]
                # HELPER: forward implementation of a test-local tool; returns url concatenated with the
                #   input.
                # Inputs: self and the parameters in its signature.
                # Outputs: as returned by its body.
                # Exceptions: none raised directly.
                # Side effects: none.
                return self.url + string_input

        fail_tool = FailTool("dummy_url")
        with pytest.raises(Exception) as e:
            fail_tool.to_dict()
        assert "Parameters in __init__ must have default values, found required parameters" in str(e)

        class PassTool(Tool):
            # [CLASS DOC]
            # HELPER: Test-local Tool subclass; constructor url parameter has a default.
            # Inputs: no constructor args (uses class attributes name, description, inputs, output_type).
            # Outputs: instance created via the base Tool constructor with post-init validation.
            # Exceptions: validation errors at instantiation, as exercised by the enclosing test.
            # Side effects: none.
            name = "specific"
            description = "test description"
            inputs = {"string_input": {"type": "string", "description": "input description"}}
            output_type = "string"

            def __init__(self, url: str | None = "none"):
                # [FUNCTION DOC]
                # HELPER: Calls the base constructor and stores url.
                # Inputs: url (str or None, default none string).
                # Outputs: none.
                # Exceptions: none raised directly.
                # Side effects: sets self.url.
                # KNOWN ISSUE: passes self explicitly to the parent constructor (harmless, base ignores extra
                #   args).
                super().__init__(self)
                self.url = url

            def forward(self, string_input: str) -> str:
                # [FUNCTION DOC]
                # HELPER: forward implementation of a test-local tool; returns url concatenated with the
                #   input.
                # Inputs: self and the parameters in its signature.
                # Outputs: as returned by its body.
                # Exceptions: none raised directly.
                # Side effects: none.
                return self.url + string_input

        fail_tool = PassTool()
        fail_tool.to_dict()

    def test_saving_tool_allows_no_imports_from_outside_methods(self, tmp_path):
        # [FUNCTION DOC]
        # Checks save fails with an undefined-name error when a helper method uses an outer import, and
        #   succeeds when the import is inside the method.
        # Inputs: self, tmp_path.
        # Outputs: none.
        # Exceptions: AssertionError on failure.
        # Side effects: writes tool files under tmp_path.
        # Test that using imports from outside functions fails
        import numpy as np

        class FailTool(Tool):
            # [CLASS DOC]
            # HELPER: Test-local Tool subclass; helper method uses an outer numpy import.
            # Inputs: no constructor args (uses class attributes name, description, inputs, output_type).
            # Outputs: instance created via the base Tool constructor with post-init validation.
            # Exceptions: validation errors at instantiation, as exercised by the enclosing test.
            # Side effects: none.
            name = "specific"
            description = "test description"
            inputs = {"string_input": {"type": "string", "description": "input description"}}
            output_type = "string"

            def useless_method(self):
                # [FUNCTION DOC]
                # HELPER: Stores a random number on self.client (outer numpy) and returns an empty string.
                # Inputs: self.
                # Outputs: empty string.
                # Exceptions: none raised directly.
                # Side effects: sets self.client.
                self.client = np.random.random()
                return ""

            def forward(self, string_input):
                # [FUNCTION DOC]
                # HELPER: forward implementation of a test-local tool; returns the helper result concatenated
                #   with the input.
                # Inputs: self and the parameters in its signature.
                # Outputs: as returned by its body.
                # Exceptions: none raised directly.
                # Side effects: none.
                return self.useless_method() + string_input

        fail_tool = FailTool()
        with pytest.raises(Exception) as e:
            fail_tool.save(tmp_path)
        assert "'np' is undefined" in str(e)

        # Test that putting these imports inside functions works
        class SuccessTool(Tool):
            # [CLASS DOC]
            # HELPER: Test-local Tool subclass; helper method imports numpy locally.
            # Inputs: no constructor args (uses class attributes name, description, inputs, output_type).
            # Outputs: instance created via the base Tool constructor with post-init validation.
            # Exceptions: validation errors at instantiation, as exercised by the enclosing test.
            # Side effects: none.
            name = "specific"
            description = "test description"
            inputs = {"string_input": {"type": "string", "description": "input description"}}
            output_type = "string"

            def useless_method(self):
                # [FUNCTION DOC]
                # HELPER: Imports numpy locally, stores a random number on self.client and returns an empty
                #   string.
                # Inputs: self.
                # Outputs: empty string.
                # Exceptions: none raised directly.
                # Side effects: sets self.client.
                import numpy as np

                self.client = np.random.random()
                return ""

            def forward(self, string_input):
                # [FUNCTION DOC]
                # HELPER: forward implementation of a test-local tool; returns the helper result concatenated
                #   with the input.
                # Inputs: self and the parameters in its signature.
                # Outputs: as returned by its body.
                # Exceptions: none raised directly.
                # Side effects: none.
                return self.useless_method() + string_input

        success_tool = SuccessTool()
        success_tool.save(tmp_path)

    def test_tool_missing_class_attributes_raises_error(self):
        # [FUNCTION DOC]
        # Checks instantiating a Tool subclass without output_type raises with a message requiring
        #   output_type.
        # Inputs: self.
        # Outputs: none.
        # Exceptions: AssertionError on failure.
        # Side effects: none.
        with pytest.raises(Exception) as e:

            class GetWeatherTool(Tool):
                # [CLASS DOC]
                # HELPER: Test-local Tool subclass; missing output_type on purpose.
                # Inputs: no constructor args (uses class attributes name, description, inputs, output_type).
                # Outputs: instance created via the base Tool constructor with post-init validation.
                # Exceptions: validation errors at instantiation, as exercised by the enclosing test.
                # Side effects: none.
                name = "get_weather"
                description = "Get weather in the next days at given location."
                inputs = {
                    "location": {"type": "string", "description": "the location"},
                    "celsius": {
                        "type": "string",
                        "description": "the temperature type",
                    },
                }

                def forward(self, location: str, celsius: bool | None = False) -> str:
                    # [FUNCTION DOC]
                    # HELPER: forward implementation of a test-local tool; returns a constant weather string.
                    # Inputs: self and the parameters in its signature.
                    # Outputs: as returned by its body.
                    # Exceptions: none raised directly.
                    # Side effects: none.
                    return "The weather is UNGODLY with torrential rains and temperatures below -10°C"

            GetWeatherTool()
        assert "You must set an attribute output_type" in str(e)

    def test_tool_from_decorator_optional_args(self):
        # [FUNCTION DOC]
        # Checks the tool decorator marks a parameter with an Optional hint and default as nullable, and a
        #   plain required one as not nullable.
        # Inputs: self.
        # Outputs: none.
        # Exceptions: AssertionError on failure.
        # Side effects: none.
        @tool
        def get_weather(location: str, celsius: bool | None = False) -> str:
            # [FUNCTION DOC]
            # HELPER: Function decorated with the tool decorator (or passed to it) inside a test; location
            #   plus optional celsius flag.
            # Inputs: see its type-hinted parameters.
            # Outputs: per its body; only its schema/metadata is usually examined.
            # Exceptions: none raised directly.
            # Side effects: none.
            """
            Get weather in the next days at given location.
            Secretly this tool does not care about the location, it hates the weather everywhere.

            Args:
                location: the location
                celsius: the temperature type
            """
            return "The weather is UNGODLY with torrential rains and temperatures below -10°C"

        assert "nullable" in get_weather.inputs["celsius"]
        assert get_weather.inputs["celsius"]["nullable"]
        assert "nullable" not in get_weather.inputs["location"]

    def test_tool_mismatching_nullable_args_raises_error(self):
        # [FUNCTION DOC]
        # Checks that mismatches between the inputs nullable flag and forward's signature (Optional hint,
        #   default value, or nullable input with a required parameter) raise an error mentioning Nullable.
        # Inputs: self.
        # Outputs: none.
        # Exceptions: AssertionError on failure.
        # Side effects: none.
        with pytest.raises(Exception) as e:

            class GetWeatherTool(Tool):
                # [CLASS DOC]
                # HELPER: Test-local Tool subclass; forward has an Optional celsius but inputs lack nullable.
                # Inputs: no constructor args (uses class attributes name, description, inputs, output_type).
                # Outputs: instance created via the base Tool constructor with post-init validation.
                # Exceptions: validation errors at instantiation, as exercised by the enclosing test.
                # Side effects: none.
                name = "get_weather"
                description = "Get weather in the next days at given location."
                inputs = {
                    "location": {"type": "string", "description": "the location"},
                    "celsius": {
                        "type": "string",
                        "description": "the temperature type",
                    },
                }
                output_type = "string"

                def forward(self, location: str, celsius: bool | None = False) -> str:
                    # [FUNCTION DOC]
                    # HELPER: forward implementation of a test-local tool; returns a constant weather string.
                    # Inputs: self and the parameters in its signature.
                    # Outputs: as returned by its body.
                    # Exceptions: none raised directly.
                    # Side effects: none.
                    return "The weather is UNGODLY with torrential rains and temperatures below -10°C"

            GetWeatherTool()
        assert "Nullable" in str(e)

        with pytest.raises(Exception) as e:

            class GetWeatherTool2(Tool):
                # [CLASS DOC]
                # HELPER: Test-local Tool subclass; forward celsius has a default but inputs lack nullable.
                # Inputs: no constructor args (uses class attributes name, description, inputs, output_type).
                # Outputs: instance created via the base Tool constructor with post-init validation.
                # Exceptions: validation errors at instantiation, as exercised by the enclosing test.
                # Side effects: none.
                name = "get_weather"
                description = "Get weather in the next days at given location."
                inputs = {
                    "location": {"type": "string", "description": "the location"},
                    "celsius": {
                        "type": "string",
                        "description": "the temperature type",
                    },
                }
                output_type = "string"

                def forward(self, location: str, celsius: bool = False) -> str:
                    # [FUNCTION DOC]
                    # HELPER: forward implementation of a test-local tool; returns a constant weather string.
                    # Inputs: self and the parameters in its signature.
                    # Outputs: as returned by its body.
                    # Exceptions: none raised directly.
                    # Side effects: none.
                    return "The weather is UNGODLY with torrential rains and temperatures below -10°C"

            GetWeatherTool2()
        assert "Nullable" in str(e)

        with pytest.raises(Exception) as e:

            class GetWeatherTool3(Tool):
                # [CLASS DOC]
                # HELPER: Test-local Tool subclass; inputs mark celsius nullable but forward requires it.
                # Inputs: no constructor args (uses class attributes name, description, inputs, output_type).
                # Outputs: instance created via the base Tool constructor with post-init validation.
                # Exceptions: validation errors at instantiation, as exercised by the enclosing test.
                # Side effects: none.
                name = "get_weather"
                description = "Get weather in the next days at given location."
                inputs = {
                    "location": {"type": "string", "description": "the location"},
                    "celsius": {
                        "type": "string",
                        "description": "the temperature type",
                        "nullable": True,
                    },
                }
                output_type = "string"

                def forward(self, location, celsius: str) -> str:
                    # [FUNCTION DOC]
                    # HELPER: forward implementation of a test-local tool; returns a constant weather string.
                    # Inputs: self and the parameters in its signature.
                    # Outputs: as returned by its body.
                    # Exceptions: none raised directly.
                    # Side effects: none.
                    return "The weather is UNGODLY with torrential rains and temperatures below -10°C"

            GetWeatherTool3()
        assert "Nullable" in str(e)

    def test_tool_default_parameters_is_nullable(self):
        # [FUNCTION DOC]
        # Checks a decorated function parameter with a default value is marked nullable.
        # Inputs: self.
        # Outputs: none.
        # Exceptions: AssertionError on failure.
        # Side effects: none.
        @tool
        def get_weather(location: str, celsius: bool = False) -> str:
            # [FUNCTION DOC]
            # HELPER: Function decorated with the tool decorator (or passed to it) inside a test; celsius has
            #   a default value.
            # Inputs: see its type-hinted parameters.
            # Outputs: per its body; only its schema/metadata is usually examined.
            # Exceptions: none raised directly.
            # Side effects: none.
            """
            Get weather in the next days at given location.

            Args:
                location: The location to get the weather for.
                celsius: is the temperature given in celsius?
            """
            return "The weather is UNGODLY with torrential rains and temperatures below -10°C"

        assert get_weather.inputs["celsius"]["nullable"]

    def test_tool_supports_any_none(self, tmp_path):
        # [FUNCTION DOC]
        # Checks a decorated function with an Any parameter and None return saves successfully and maps to
        #   input type any and output type null.
        # Inputs: self, tmp_path.
        # Outputs: none.
        # Exceptions: AssertionError on failure.
        # Side effects: writes tool files under tmp_path.
        @tool
        def get_weather(location: Any) -> None:
            # [FUNCTION DOC]
            # HELPER: Function decorated with the tool decorator (or passed to it) inside a test; Any
            #   parameter, None return.
            # Inputs: see its type-hinted parameters.
            # Outputs: per its body; only its schema/metadata is usually examined.
            # Exceptions: none raised directly.
            # Side effects: none.
            """
            Get weather in the next days at given location.

            Args:
                location: The location to get the weather for.
            """
            return

        get_weather.save(tmp_path)
        assert get_weather.inputs["location"]["type"] == "any"
        assert get_weather.output_type == "null"

    def test_tool_supports_array(self):
        # [FUNCTION DOC]
        # Checks list and optional tuple parameters map to input type array.
        # Inputs: self.
        # Outputs: none.
        # Exceptions: AssertionError on failure.
        # Side effects: none.
        @tool
        def get_weather(locations: list[str], months: tuple[str, str] | None = None) -> dict[str, float]:
            # [FUNCTION DOC]
            # HELPER: Function decorated with the tool decorator (or passed to it) inside a test; list and
            #   optional tuple parameters, dict return hint.
            # Inputs: see its type-hinted parameters.
            # Outputs: per its body; only its schema/metadata is usually examined.
            # Exceptions: none raised directly.
            # Side effects: none.
            """
            Get weather in the next days at given locations.

            Args:
                locations: The locations to get the weather for.
                months: The months to get the weather for
            """
            return

        assert get_weather.inputs["locations"]["type"] == "array"
        assert get_weather.inputs["months"]["type"] == "array"

    def test_tool_supports_string_literal(self):
        # [FUNCTION DOC]
        # Checks a string Literal parameter maps to type string with the literal values as enum.
        # Inputs: self.
        # Outputs: none.
        # Exceptions: AssertionError on failure.
        # Side effects: none.
        @tool
        def get_weather(unit: Literal["celsius", "fahrenheit"] = "celsius") -> None:
            # [FUNCTION DOC]
            # HELPER: Function decorated with the tool decorator (or passed to it) inside a test; string
            #   Literal unit parameter with default.
            # Inputs: see its type-hinted parameters.
            # Outputs: per its body; only its schema/metadata is usually examined.
            # Exceptions: none raised directly.
            # Side effects: none.
            """
            Get weather in the next days at given location.

            Args:
                unit: The unit of temperature
            """
            return

        assert get_weather.inputs["unit"]["type"] == "string"
        assert get_weather.inputs["unit"]["enum"] == ["celsius", "fahrenheit"]

    def test_tool_supports_numeric_literal(self):
        # [FUNCTION DOC]
        # Checks an integer Literal parameter maps to type integer with the literal values as enum.
        # Inputs: self.
        # Outputs: none.
        # Exceptions: AssertionError on failure.
        # Side effects: none.
        @tool
        def get_choice(choice: Literal[1, 2, 3]) -> None:
            # [FUNCTION DOC]
            # HELPER: Function decorated with the tool decorator (or passed to it) inside a test; integer
            #   Literal choice parameter.
            # Inputs: see its type-hinted parameters.
            # Outputs: per its body; only its schema/metadata is usually examined.
            # Exceptions: none raised directly.
            # Side effects: none.
            """
            Get choice based on the provided numeric literal.

            Args:
                choice: The numeric choice to be made.
            """
            return

        assert get_choice.inputs["choice"]["type"] == "integer"
        assert get_choice.inputs["choice"]["enum"] == [1, 2, 3]

    def test_tool_supports_nullable_literal(self):
        # [FUNCTION DOC]
        # Checks a Literal including None maps to type integer, nullable True, and an enum without None.
        # Inputs: self.
        # Outputs: none.
        # Exceptions: AssertionError on failure.
        # Side effects: none.
        @tool
        def get_choice(choice: Literal[1, 2, 3, None]) -> None:
            # [FUNCTION DOC]
            # HELPER: Function decorated with the tool decorator (or passed to it) inside a test; Literal
            #   choice parameter including None.
            # Inputs: see its type-hinted parameters.
            # Outputs: per its body; only its schema/metadata is usually examined.
            # Exceptions: none raised directly.
            # Side effects: none.
            """
            Get choice based on the provided value.

            Args:
                choice: The numeric choice to be made.
            """
            return

        assert get_choice.inputs["choice"]["type"] == "integer"
        assert get_choice.inputs["choice"]["nullable"] is True
        assert get_choice.inputs["choice"]["enum"] == [1, 2, 3]

    def test_saving_tool_produces_valid_python_code_with_multiline_description(self, tmp_path):
        # [FUNCTION DOC]
        # Checks that saving a decorated tool with a multiline description writes a tool.py that compiles.
        # Inputs: self, tmp_path.
        # Outputs: none.
        # Exceptions: SyntaxError or AssertionError on failure.
        # Side effects: writes and reads files under tmp_path.
        @tool
        def get_weather(location: Any) -> None:
            # [FUNCTION DOC]
            # HELPER: Function decorated with the tool decorator (or passed to it) inside a test; two-line
            #   description.
            # Inputs: see its type-hinted parameters.
            # Outputs: per its body; only its schema/metadata is usually examined.
            # Exceptions: none raised directly.
            # Side effects: none.
            """
            Get weather in the next days at given location.
            And works pretty well.

            Args:
                location: The location to get the weather for.
            """
            return

        get_weather.save(tmp_path)
        with open(os.path.join(tmp_path, "tool.py"), "r", encoding="utf-8") as f:
            source_code = f.read()
            compile(source_code, f.name, "exec")

    @pytest.mark.parametrize("fixture_name", ["boolean_default_tool_class", "boolean_default_tool_function"])
    def test_to_dict_boolean_default_input(self, fixture_name, request):
        # [FUNCTION DOC]
        # Parametrized over class- and function-based conftest tools: to_dict code keeps a boolean default
        #   annotation and marks the input nullable.
        # Inputs: fixture_name, request.
        # Outputs: none.
        # Exceptions: AssertionError on failure.
        # Side effects: none.
        """Test that boolean input parameter with default value is correctly represented in to_dict output"""
        tool = request.getfixturevalue(fixture_name)
        result = tool.to_dict()
        # Check that the boolean default annotation is preserved
        assert "flag: bool = False" in result["code"]
        # Check nullable attribute is set for the parameter with default value
        assert "'nullable': True" in result["code"]

    @pytest.mark.parametrize("fixture_name", ["optional_input_tool_class", "optional_input_tool_function"])
    def test_to_dict_optional_input(self, fixture_name, request):
        # [FUNCTION DOC]
        # Parametrized over class- and function-based conftest tools: to_dict code keeps an Optional string
        #   annotation with None default and marks the input nullable.
        # Inputs: fixture_name, request.
        # Outputs: none.
        # Exceptions: AssertionError on failure.
        # Side effects: none.
        """Test that Optional/nullable input parameter is correctly represented in to_dict output"""
        tool = request.getfixturevalue(fixture_name)
        result = tool.to_dict()
        # Check the Optional type annotation is preserved
        assert "optional_text: str | None = None" in result["code"]
        # Check that the input is marked as nullable in the code
        assert "'nullable': True" in result["code"]

    def test_from_dict_roundtrip(self, example_tool):
        # [FUNCTION DOC]
        # Checks to_dict then from_dict recreates a tool with identical name, description, inputs,
        #   output_type, and that calling it uppercases the input.
        # Inputs: self, example_tool (conftest fixture).
        # Outputs: none.
        # Exceptions: AssertionError on failure.
        # Side effects: executes generated code.
        # Convert to dict
        tool_dict = example_tool.to_dict()
        # Create from dict
        recreated_tool = Tool.from_dict(tool_dict)
        # Verify properties
        assert recreated_tool.name == example_tool.name
        assert recreated_tool.description == example_tool.description
        assert recreated_tool.inputs == example_tool.inputs
        assert recreated_tool.output_type == example_tool.output_type
        # Verify functionality
        test_input = "Hello, world!"
        assert recreated_tool(test_input) == test_input.upper()

    def test_tool_from_dict_invalid(self):
        # [FUNCTION DOC]
        # Checks from_dict raises ValueError mentioning the missing code key.
        # Inputs: self.
        # Outputs: none.
        # Exceptions: AssertionError on failure.
        # Side effects: none.
        # Missing code key
        with pytest.raises(ValueError) as e:
            Tool.from_dict({"name": "invalid_tool"})
        assert "must contain 'code' key" in str(e)

    def test_tool_decorator_preserves_original_function(self):
        # [FUNCTION DOC]
        # Checks applying tool to a function leaves the function's signature, name and docstring unchanged,
        #   while the tool's forward signature starts with self.
        # Inputs: self.
        # Outputs: none.
        # Exceptions: AssertionError on failure.
        # Side effects: none.
        # Define a test function with type hints and docstring
        def test_function(items: list[str]) -> str:
            # [FUNCTION DOC]
            # HELPER: Plain function passed to tool in the test; joins a list of strings with comma and space.
            # Inputs: items (list of str).
            # Outputs: the joined string.
            # Exceptions: TypeError if items are not strings.
            # Side effects: none.
            """Join a list of strings.
            Args:
                items: A list of strings to join
            Returns:
                The joined string
            """
            return ", ".join(items)

        # Store original function signature, name, and source
        original_signature = inspect.signature(test_function)
        original_name = test_function.__name__
        original_docstring = test_function.__doc__

        # Create a tool from the function
        test_tool = tool(test_function)

        # Check that the original function is unchanged
        assert original_signature == inspect.signature(test_function)
        assert original_name == test_function.__name__
        assert original_docstring == test_function.__doc__

        # Verify that the tool's forward method has a different signature (it has 'self')
        tool_forward_sig = inspect.signature(test_tool.forward)
        assert list(tool_forward_sig.parameters.keys())[0] == "self"

        # Original function should not have 'self' parameter
        assert "self" not in original_signature.parameters

    def test_tool_with_union_type_return(self):
        # [FUNCTION DOC]
        # Checks a union return hint yields a Tool with output_type any.
        # Inputs: self.
        # Outputs: none.
        # Exceptions: AssertionError on failure.
        # Side effects: none.
        @tool
        def union_type_return_tool_function(param: int) -> str | bool:
            # [FUNCTION DOC]
            # HELPER: Function decorated with the tool decorator (or passed to it) inside a test; returns str
            #   or bool.
            # Inputs: see its type-hinted parameters.
            # Outputs: per its body; only its schema/metadata is usually examined.
            # Exceptions: none raised directly.
            # Side effects: none.
            """
            Tool with output union type.

            Args:
                param: Input parameter.
            """
            return str(param) if param > 0 else False

        assert isinstance(union_type_return_tool_function, Tool)
        assert union_type_return_tool_function.output_type == "any"


class TestToolDecorator:
    # [CLASS DOC]
    # Tests of the tool decorator's source extraction (stored forward source) with extra decorators and
    #   multiline signatures.
    # Inputs: none.
    # Outputs: test methods.
    # Exceptions: none raised directly.
    # Side effects: none.
    def test_tool_decorator_source_extraction_with_multiple_decorators(self):
        # [FUNCTION DOC]
        # Checks that a function with an extra decorator warns about other decorators, works as a tool, and
        #   its stored forward source has the forward header and body but no decorator lines or original def
        #   line.
        # Inputs: self.
        # Outputs: none.
        # Exceptions: AssertionError on failure.
        # Side effects: none.
        """Test that @tool correctly extracts source code with multiple decorators."""

        def dummy_decorator(func):
            # [FUNCTION DOC]
            # HELPER: Identity decorator used to add a non-tool decorator.
            # Inputs: func.
            # Outputs: func unchanged.
            # Exceptions: none raised directly.
            # Side effects: none.
            return func

        with pytest.warns(UserWarning, match="has decorators other than @tool"):

            @tool
            @dummy_decorator
            def multi_decorator_tool(text: str) -> str:
                # [FUNCTION DOC]
                # HELPER: Function decorated with the tool decorator (or passed to it) inside a test;
                #   uppercases text; also decorated with dummy_decorator.
                # Inputs: see its type-hinted parameters.
                # Outputs: per its body; only its schema/metadata is usually examined.
                # Exceptions: none raised directly.
                # Side effects: none.
                """Tool with multiple decorators.

                Args:
                    text: Input text
                """
                return text.upper()

        # Verify the tool works
        assert isinstance(multi_decorator_tool, Tool)
        assert multi_decorator_tool.name == "multi_decorator_tool"
        assert multi_decorator_tool("hello") == "HELLO"

        # Verify the source code extraction is correct
        forward_source = multi_decorator_tool.forward.__source__
        assert "def forward(self, text: str) -> str:" in forward_source
        assert "return text.upper()" in forward_source
        # Should not contain decorator lines
        assert "@tool" not in forward_source
        assert "@dummy_decorator" not in forward_source
        # Should not contain definition line
        assert "def multi_decorator_tool" not in forward_source

    def test_tool_decorator_source_extraction_with_multiline_signature(self):
        # [FUNCTION DOC]
        # Checks a multiline-signature function produces no warnings, works as a tool, and its stored forward
        #   source has a single-line forward header and body but no remnants of the original multiline
        #   signature.
        # Inputs: self.
        # Outputs: none.
        # Exceptions: AssertionError on failure (warnings turned into errors).
        # Side effects: none.
        """Test that @tool correctly extracts source code with multiline function signatures."""

        with warnings.catch_warnings():
            warnings.simplefilter("error")

            @tool
            def multiline_signature_tool(
                text: str,
                count: int = 1,
                uppercase: bool = False,
                multiline_parameter_1: int = 1_000,
                multiline_parameter_2: int = 2_000,
            ) -> str:
                # [FUNCTION DOC]
                # HELPER: Function decorated with the tool decorator (or passed to it) inside a test; repeats
                #   text count times, optionally uppercased.
                # Inputs: see its type-hinted parameters.
                # Outputs: per its body; only its schema/metadata is usually examined.
                # Exceptions: none raised directly.
                # Side effects: none.
                """Tool with multiline signature.

                Args:
                    text: Input text
                    count: Number of repetitions
                    uppercase: Whether to convert to uppercase
                    multiline_parameter_1: Dummy parameter
                    multiline_parameter_2: Dummy parameter
                """
                result = text * count
                return result.upper() if uppercase else result

        # Verify the tool works
        assert isinstance(multiline_signature_tool, Tool)
        assert multiline_signature_tool.name == "multiline_signature_tool"
        assert multiline_signature_tool("hello", 2, True) == "HELLOHELLO"

        # Verify the source code extraction is correct
        forward_source = multiline_signature_tool.forward.__source__
        assert (
            "def forward(self, text: str, count: int=1, uppercase: bool=False, multiline_parameter_1: int=1000, multiline_parameter_2: int=2000) -> str:"
            in forward_source
            or "def forward(self, text: str, count: int = 1, uppercase: bool = False, multiline_parameter_1: int = 1000, multiline_parameter_2: int = 2000) -> str:"
            in forward_source
        )
        assert "result = text * count" in forward_source
        assert "return result.upper() if uppercase else result" in forward_source
        # Should not contain the original multiline function definition
        assert "def multiline_signature_tool(" not in forward_source
        # Should not contain leftover lines from the original multiline function definition
        assert "            count: int = 1," not in forward_source
        assert "            count: int=1," not in forward_source

    def test_tool_decorator_source_extraction_with_multiple_decorators_and_multiline(self):
        # [FUNCTION DOC]
        # Checks the combination of two extra decorators and a multiline signature: warning emitted, tool
        #   works, stored forward source is clean of decorators and multiline signature remnants.
        # Inputs: self.
        # Outputs: none.
        # Exceptions: AssertionError on failure.
        # Side effects: none.
        """Test that @tool works with both multiple decorators and multiline signatures."""

        def dummy_decorator_1(func):
            # [FUNCTION DOC]
            # HELPER: Identity decorator.
            # Inputs: func.
            # Outputs: func unchanged.
            # Exceptions: none raised directly.
            # Side effects: none.
            return func

        def dummy_decorator_2(func):
            # [FUNCTION DOC]
            # HELPER: Identity decorator.
            # Inputs: func.
            # Outputs: func unchanged.
            # Exceptions: none raised directly.
            # Side effects: none.
            return func

        with pytest.warns(UserWarning, match="has decorators other than @tool"):

            @tool
            @dummy_decorator_1
            @dummy_decorator_2
            def complex_tool(
                text: str,
                multiplier: int = 2,
                separator: str = " ",
                multiline_parameter_1: int = 1_000,
                multiline_parameter_2: int = 2_000,
            ) -> str:
                # [FUNCTION DOC]
                # HELPER: Function decorated with the tool decorator (or passed to it) inside a test; joins
                #   multiplier copies of text with separator; has two extra decorators.
                # Inputs: see its type-hinted parameters.
                # Outputs: per its body; only its schema/metadata is usually examined.
                # Exceptions: none raised directly.
                # Side effects: none.
                """Complex tool with multiple decorators and multiline signature.

                Args:
                    text: Input text
                    multiplier: How many times to repeat
                    separator: What to use between repetitions
                    multiline_parameter_1: Dummy parameter
                    multiline_parameter_2: Dummy parameter
                """
                parts = [text] * multiplier
                return separator.join(parts)

        # Verify the tool works
        assert isinstance(complex_tool, Tool)
        assert complex_tool.name == "complex_tool"
        assert complex_tool("hello", 3, "-") == "hello-hello-hello"

        # Verify the source code extraction is correct
        forward_source = complex_tool.forward.__source__
        assert (
            "def forward(self, text: str, multiplier: int=2, separator: str=' ', multiline_parameter_1: int=1000, multiline_parameter_2: int=2000) -> str:"
            in forward_source
            or "def forward(self, text: str, multiplier: int = 2, separator: str = ' ', multiline_parameter_1: int = 1000, multiline_parameter_2: int = 2000) -> str:"
            in forward_source
        )
        assert "parts = [text] * multiplier" in forward_source
        assert "return separator.join(parts)" in forward_source
        # Should not contain any decorator lines
        assert "@tool" not in forward_source
        assert "@dummy_decorator_1" not in forward_source
        assert "@dummy_decorator_2" not in forward_source
        # Should not contain leftover lines from the original multiline function definition
        assert "            multiplier: int = 2," not in forward_source
        assert "            multiplier: int=2," not in forward_source


@pytest.fixture
def mock_server_parameters():
    # [FUNCTION DOC]
    # HELPER: Fixture providing a MagicMock standing in for MCP server parameters.
    # Inputs: none.
    # Outputs: a MagicMock.
    # Exceptions: none raised directly.
    # Side effects: none.
    return MagicMock()


@pytest.fixture
def mock_mcp_adapt():
    # [FUNCTION DOC]
    # HELPER: Fixture patching mcpadapt.core.MCPAdapt so its context manager yields two fake tool names.
    # Inputs: none.
    # Outputs: yields the patch mock.
    # Exceptions: none raised directly.
    # Side effects: patches MCPAdapt for the test duration.
    with patch("mcpadapt.core.MCPAdapt") as mock:
        mock.return_value.__enter__.return_value = ["tool1", "tool2"]
        mock.return_value.__exit__.return_value = None
        yield mock


@pytest.fixture
def mock_smolagents_adapter():
    # [FUNCTION DOC]
    # HELPER: Fixture patching mcpadapt.smolagents_adapter.SmolAgentsAdapter.
    # Inputs: none.
    # Outputs: yields the patch mock.
    # Exceptions: none raised directly.
    # Side effects: patches SmolAgentsAdapter for the test duration.
    with patch("mcpadapt.smolagents_adapter.SmolAgentsAdapter") as mock:
        yield mock


# Ignore FutureWarning about structured_output default value change: this test intentionally uses default behavior
@pytest.mark.filterwarnings("ignore:.*structured_output:FutureWarning")
class TestToolCollection:
    # [CLASS DOC]
    # Tests for ToolCollection.from_mcp with mocked and real MCP servers; FutureWarnings about
    #   structured_output are ignored.
    # Inputs: none.
    # Outputs: test methods.
    # Exceptions: none raised directly.
    # Side effects: see individual tests.
    def test_from_mcp(self, mock_server_parameters, mock_mcp_adapt, mock_smolagents_adapter):
        # [FUNCTION DOC]
        # Checks from_mcp with mocked MCPAdapt yields a ToolCollection containing the two fake tools.
        # Inputs: self, mock_server_parameters, mock_mcp_adapt, mock_smolagents_adapter.
        # Outputs: none.
        # Exceptions: AssertionError on failure.
        # Side effects: none (patched).
        with ToolCollection.from_mcp(mock_server_parameters, trust_remote_code=True) as tool_collection:
            assert isinstance(tool_collection, ToolCollection)
            assert len(tool_collection.tools) == 2
            assert "tool1" in tool_collection.tools
            assert "tool2" in tool_collection.tools

    @require_run_all
    def test_integration_from_mcp(self):
        # [FUNCTION DOC]
        # Integration test (only with the run-all marker): starts a stdio FastMCP echo server via python and
        #   checks the single loaded tool echoes its input.
        # Inputs: self.
        # Outputs: none.
        # Exceptions: AssertionError on failure.
        # Side effects: spawns a subprocess.
        # define the most simple mcp server with one tool that echoes the input text
        mcp_server_script = dedent("""\
            from mcp.server.fastmcp import FastMCP

            mcp = FastMCP("Echo Server")

            @mcp.tool()
            def echo_tool(text: str) -> str:
                return text

            mcp.run()
        """).strip()

        mcp_server_params = mcp.StdioServerParameters(
            command="python",
            args=["-c", mcp_server_script],
        )

        with ToolCollection.from_mcp(mcp_server_params, trust_remote_code=True) as tool_collection:
            assert len(tool_collection.tools) == 1, "Expected 1 tool"
            assert tool_collection.tools[0].name == "echo_tool", "Expected tool name to be 'echo_tool'"
            assert tool_collection.tools[0](text="Hello") == "Hello", "Expected tool to echo the input text"

    def test_integration_from_mcp_with_streamable_http(self):
        # [FUNCTION DOC]
        # Starts a FastMCP echo server on 127.0.0.1:8000 with streamable HTTP in a subprocess, waits one
        #   second, loads it via from_mcp and checks the echo tool; kills the server afterwards.
        # Inputs: self.
        # Outputs: none.
        # Exceptions: AssertionError on failure.
        # Side effects: spawns a subprocess, binds port 8000, sleeps.
        # KNOWN ISSUE: relies on a fixed one-second sleep and a fixed port, so it can be flaky; the comment
        #   mentions SSE although the transport is streamable HTTP.
        import subprocess
        import time

        # define the most simple mcp server with one tool that echoes the input text
        mcp_server_script = dedent("""\
            from mcp.server.fastmcp import FastMCP

            mcp = FastMCP("Echo Server", host="127.0.0.1", port=8000)

            @mcp.tool()
            def echo_tool(text: str) -> str:
                return text

            mcp.run(transport="streamable-http")
        """).strip()

        # start the SSE mcp server in a subprocess
        server_process = subprocess.Popen(
            ["python", "-c", mcp_server_script],
        )

        # wait for the server to start
        time.sleep(1)

        try:
            with ToolCollection.from_mcp(
                {"url": "http://127.0.0.1:8000/mcp", "transport": "streamable-http"}, trust_remote_code=True
            ) as tool_collection:
                assert len(tool_collection.tools) == 1, "Expected 1 tool"
                assert tool_collection.tools[0].name == "echo_tool", "Expected tool name to be 'echo_tool'"
                assert tool_collection.tools[0](text="Hello") == "Hello", "Expected tool to echo the input text"
        finally:
            # clean up the process when test is done
            server_process.kill()
            server_process.wait()

    def test_integration_from_mcp_with_sse(self):
        # [FUNCTION DOC]
        # Same as the streamable HTTP test but with the legacy SSE transport at the sse endpoint.
        # Inputs: self.
        # Outputs: none.
        # Exceptions: AssertionError on failure.
        # Side effects: spawns a subprocess, binds port 8000, sleeps.
        # KNOWN ISSUE: relies on a fixed one-second sleep and a fixed port, so it can be flaky.
        import subprocess
        import time

        # define the most simple mcp server with one tool that echoes the input text
        mcp_server_script = dedent("""\
            from mcp.server.fastmcp import FastMCP

            mcp = FastMCP("Echo Server", host="127.0.0.1", port=8000)

            @mcp.tool()
            def echo_tool(text: str) -> str:
                return text

            mcp.run("sse")
        """).strip()

        # start the SSE mcp server in a subprocess
        server_process = subprocess.Popen(
            ["python", "-c", mcp_server_script],
        )

        # wait for the server to start
        time.sleep(1)

        try:
            with ToolCollection.from_mcp(
                {"url": "http://127.0.0.1:8000/sse", "transport": "sse"}, trust_remote_code=True
            ) as tool_collection:
                assert len(tool_collection.tools) == 1, "Expected 1 tool"
                assert tool_collection.tools[0].name == "echo_tool", "Expected tool name to be 'echo_tool'"
                assert tool_collection.tools[0](text="Hello") == "Hello", "Expected tool to echo the input text"
        finally:
            # clean up the process when test is done
            server_process.kill()
            server_process.wait()


@pytest.mark.parametrize("tool_fixture_name", ["boolean_default_tool_class"])
def test_launch_gradio_demo_does_not_raise(tool_fixture_name, request):
    # [FUNCTION DOC]
    # Checks launch_gradio_demo builds the interface and calls launch once (launch patched) for a conftest
    #   tool.
    # Inputs: tool_fixture_name, request.
    # Outputs: none.
    # Exceptions: AssertionError on failure.
    # Side effects: none (launch patched); requires gradio.
    tool = request.getfixturevalue(tool_fixture_name)
    with patch("gradio.Interface.launch") as mock_launch:
        launch_gradio_demo(tool)
    assert mock_launch.call_count == 1


@pytest.mark.parametrize(
    "tool_input_type, expected_input, expects_error",
    [
        (bool, True, False),
        (str, "b", False),
        (int, 1, False),
        (float, 1, False),
        (list, ["a", "b"], False),
        (list[str], ["a", "b"], False),
        (dict[str, str], {"a": "b"}, False),
        (dict[str, str], "b", True),
        (bool, "b", True),
        (str | int, "a", False),
        (str | int, 1, False),
        (str | int, None, True),
        (str | int, True, True),
    ],
)
def test_validate_tool_arguments(tool_input_type, expected_input, expects_error):
    # [FUNCTION DOC]
    # Parametrized test: builds a one-argument tool with the given type hint and checks
    #   validate_tool_arguments accepts or rejects (ValueError or TypeError) the given value, including int-to-
    #   float coercion and unions.
    # Inputs: tool_input_type, expected_input, expects_error.
    # Outputs: none.
    # Exceptions: AssertionError on failure.
    # Side effects: none.
    @tool
    def test_tool(argument_a: tool_input_type) -> str:
        # [FUNCTION DOC]
        # HELPER: Function decorated with the tool decorator (or passed to it) inside a test; one argument
        #   typed with the parametrized hint.
        # Inputs: see its type-hinted parameters.
        # Outputs: per its body; only its schema/metadata is usually examined.
        # Exceptions: none raised directly.
        # KNOWN ISSUE: returns the argument despite the str return hint (never called).
        # Side effects: none.
        """Fake tool

        Args:
            argument_a: The input
        """
        return argument_a

    if expects_error:
        with pytest.raises((ValueError, TypeError)):
            validate_tool_arguments(test_tool, {"argument_a": expected_input})

    else:
        # Should not raise any exception
        validate_tool_arguments(test_tool, {"argument_a": expected_input})


@pytest.mark.parametrize(
    "scenario, type_hint, default, input_value, expected_error_message",
    [
        # Required parameters (no default)
        # - Valid input
        ("required_unsupported_none", str, ..., "text", None),
        # - None not allowed
        ("required_unsupported_none", str, ..., None, "Argument param has type 'null' but should be 'string'"),
        # - Missing required parameter is not allowed
        ("required_unsupported_none", str, ..., ..., "Argument param is required"),
        #
        # Required parameters but supports None
        # - Valid input
        ("required_supported_none", str | None, ..., "text", None),
        # - None allowed
        ("required_supported_none", str | None, ..., None, None),
        # - Missing required parameter is not allowed
        # TODO: Fix this test case: property is marked as nullable because it can be None, but it can't be missing because it is required
        # ("required_supported_none", str | None, ..., ..., "Argument param is required"),
        pytest.param(
            "required_supported_none",
            str | None,
            ...,
            ...,
            "Argument param is required",
            marks=pytest.mark.skip(reason="TODO: Fix this test case"),
        ),
        #
        # Optional parameters (has default, doesn't support None)
        # - Valid input
        ("optional_unsupported_none", str, "default", "text", None),
        # - None not allowed
        # TODO: Fix this test case: property is marked as nullable because it has a default value, but it can't be None
        # ("optional_unsupported_none", str, "default", None, "Argument param has type 'null' but should be 'string'"),
        pytest.param(
            "optional_unsupported_none",
            str,
            "default",
            None,
            "Argument param has type 'null' but should be 'string'",
            marks=pytest.mark.skip(reason="TODO: Fix this test case"),
        ),
        # - Missing optional parameter is allowed
        ("optional_unsupported_none", str, "default", ..., None),
        #
        # Optional and supports None parameters with string default
        # - Valid input
        ("optional_supported_none_str_default", str | None, "default", "text", None),
        # - None allowed
        ("optional_supported_none_str_default", str | None, "default", None, None),
        # - Missing optional parameter is allowed
        ("optional_supported_none_str_default", str | None, "default", ..., None),
        #
        # Optional and supports None parameters with None default
        # - Valid input
        ("optional_supported_none_none_default", str | None, None, "text", None),
        # - None allowed
        ("optional_supported_none_none_default", str | None, None, None, None),
        # - Missing optional parameter is allowed
        ("optional_supported_none_none_default", str | None, None, ..., None),
    ],
)
def test_validate_tool_arguments_nullable(scenario, type_hint, default, input_value, expected_error_message):
    # [FUNCTION DOC]
    # Parametrized test of validate_tool_arguments for required/optional and None-accepting parameters: builds
    #   a tool with or without a default, then validates an input dict (empty when the value is Ellipsis) and
    #   checks for the expected error message or no error. Two cases are skipped as known TODOs.
    # Inputs: scenario (label), type_hint, default (Ellipsis for none), input_value (Ellipsis for missing),
    #   expected_error_message (str or None).
    # Outputs: none.
    # Exceptions: AssertionError on failure.
    # Side effects: none.
    """Test validation of tool arguments with focus on nullable properties: optional (with default value) and supporting None value."""

    # Create a tool with the appropriate signature
    if default is ...:  # Using Ellipsis to indicate no default value

        @tool
        def test_tool(param: type_hint) -> str:
            # [FUNCTION DOC]
            # HELPER: Function decorated with the tool decorator (or passed to it) inside a test; required
            #   param with the parametrized hint.
            # Inputs: see its type-hinted parameters.
            # Outputs: per its body; only its schema/metadata is usually examined.
            # Exceptions: none raised directly.
            # Side effects: none.
            """Test tool.

            Args:
                param: Input param
            """
            return str(param) if param is not None else "NULL"
    else:

        @tool
        def test_tool(param: type_hint = default) -> str:
            # [FUNCTION DOC]
            # HELPER: Function decorated with the tool decorator (or passed to it) inside a test; param with
            #   the parametrized hint and default.
            # Inputs: see its type-hinted parameters.
            # Outputs: per its body; only its schema/metadata is usually examined.
            # Exceptions: none raised directly.
            # Side effects: none.
            """Test tool.

            Args:
                param: Input param.
            """
            return str(param) if param is not None else "NULL"

    # Test with the input dictionary
    input_dict = {"param": input_value} if input_value is not ... else {}

    if expected_error_message:
        with pytest.raises((ValueError, TypeError), match=expected_error_message):
            validate_tool_arguments(test_tool, input_dict)
    else:
        # Should not raise any exception
        validate_tool_arguments(test_tool, input_dict)
