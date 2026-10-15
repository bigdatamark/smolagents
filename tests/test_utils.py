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
# Purpose: Tests for smolagents.utils: code-blob and JSON-blob parsing, get_source (regular and IPython
#   fallbacks), instance_to_source, end-to-end Tool.save output for class and function tools (regular and IPython),
#   is_valid_name, and the agent Gradio app template.
# Fixtures/data: ValidTool and valid_tool_function sample tools with their expected generated sources
#   (VALID_TOOL_SOURCE, VALID_TOOL_FUNCTION_SOURCE), and the ipython_shell fixture.
# Dependencies: pytest, unittest, IPython InteractiveShell, smolagents Tool and tool decorator.
# Side effects: tests write files into pytest tmp_path directories and mutate the global IPython shell instance.
# Note: methods of sample tools whose source is compared verbatim (ValidTool methods, valid_tool_function, the
#   nested TestTool.forward and test_tool, and the one-line TestClass/test_func) intentionally carry no doc block,
#   because a comment inside their bodies would change the extracted source and break the assertions.
import inspect
import os
import textwrap
import unittest

import pytest
from IPython.core.interactiveshell import InteractiveShell

from smolagents import Tool
from smolagents.tools import tool
from smolagents.utils import (
    create_agent_gradio_app_template,
    get_source,
    instance_to_source,
    is_valid_name,
    parse_code_blobs,
    parse_json_blob,
)


class ValidTool(Tool):
    # [CLASS DOC]
    # HELPER: Sample Tool subclass used as a fixture for instance_to_source. Its methods' source is compared
    #   verbatim against VALID_TOOL_SOURCE.
    # Class attributes: name "valid_tool", description, inputs (one string input named "input"), output_type
    #   "string", simple_attr (str) and dict_attr (dict) extra attributes. Constructor arg: optional_param
    #   (default "default") stored as param. forward returns the input uppercased.
    name = "valid_tool"
    description = "A valid tool"
    inputs = {"input": {"type": "string", "description": "input"}}
    output_type = "string"
    simple_attr = "string"
    dict_attr = {"key": "value"}

    def __init__(self, optional_param="default"):
        super().__init__()
        self.param = optional_param

    def forward(self, input: str) -> str:
        return input.upper()


@tool
def valid_tool_function(input: str) -> str:
    """A valid tool function.

    Args:
        input (str): Input string.
    """
    return input.upper()


VALID_TOOL_SOURCE = """\
from smolagents.tools import Tool

class ValidTool(Tool):
    name = "valid_tool"
    description = "A valid tool"
    inputs = {'input': {'type': 'string', 'description': 'input'}}
    output_type = "string"
    simple_attr = "string"
    dict_attr = {'key': 'value'}

    def __init__(self, optional_param="default"):
        super().__init__()
        self.param = optional_param

    def forward(self, input: str) -> str:
        return input.upper()
"""

VALID_TOOL_FUNCTION_SOURCE = '''\
from smolagents.tools import Tool

class SimpleTool(Tool):
    name = "valid_tool_function"
    description = "A valid tool function."
    inputs = {'input': {'type': 'string', 'description': 'Input string.'}}
    output_type = "string"

    def __init__(self):
        self.is_initialized = True

    def forward(self, input: str) -> str:
        """A valid tool function.

        Args:
            input (str): Input string.
        """
        return input.upper()
'''


class AgentTextTests(unittest.TestCase):
    # [CLASS DOC]
    # unittest TestCase for parse_code_blobs.
    def test_parse_code_blobs(self):
        # [FUNCTION DOC]
        # Checks parse_code_blobs: invalid text raises ValueError; code within custom tags is extracted; pure Python text
        #   is returned as-is; whitespace after the opening tag is stripped; markdown python fences are used as fallback.
        # Inputs: self. Outputs: none. Exceptions: AssertionError on failure. Side effects: none.
        with pytest.raises(ValueError):
            parse_code_blobs("Wrong blob!", ("<code>", "</code>"))

        # Parsing mardkwon with code blobs should work
        output = parse_code_blobs(
            """
Here is how to solve the problem:
<code>
import numpy as np
</code>
""",
            ("<code>", "</code>"),
        )
        assert output == "import numpy as np"

        # Parsing pure python code blobs should work
        code_blob = "import numpy as np"
        output = parse_code_blobs(code_blob, ("```python", "```"))
        assert output == code_blob

        # Allow whitespaces after header
        output = parse_code_blobs("<code>    \ncode_a\n</code>", ("<code>", "</code>"))
        assert output == "code_a"

        # Parsing markdown with code blobs should work
        output = parse_code_blobs(
            """
Here is how to solve the problem:
```python
import numpy as np
```
""",
            ("<code>", "</code>"),
        )
        assert output == "import numpy as np"

    def test_multiple_code_blobs(self):
        # [FUNCTION DOC]
        # Checks that several tagged code blocks are joined with blank lines in order.
        # Inputs: self. Outputs: none. Exceptions: AssertionError on failure. Side effects: none.
        test_input = "<code>\nFoo\n</code>\n\n<code>\ncode_a\n</code>\n\n<code>\ncode_b\n</code>"
        result = parse_code_blobs(test_input, ("<code>", "</code>"))
        assert result == "Foo\n\ncode_a\n\ncode_b"


@pytest.fixture(scope="function")
def ipython_shell():
    # [FUNCTION DOC]
    # HELPER: Function-scoped pytest fixture yielding the singleton IPython InteractiveShell, reset before and after
    #   the test.
    # Inputs: none. Yields: the shell instance.
    # Exceptions: none raised directly. Side effects: clears the shell namespace and history state twice.
    """Reset IPython shell before and after each test."""
    shell = InteractiveShell.instance()
    shell.reset()  # Clean before test
    yield shell
    shell.reset()  # Clean after test


@pytest.mark.parametrize(
    "obj_name, code_blob",
    [
        ("test_func", "def test_func():\n    return 42"),
        ("TestClass", "class TestClass:\n    ..."),
    ],
)
def test_get_source_ipython(ipython_shell, obj_name, code_blob):
    # [FUNCTION DOC]
    # Parametrized test: defines a function or class by running a cell in IPython, then checks get_source returns
    #   exactly that cell's code via the history fallback.
    # Inputs: ipython_shell (fixture), obj_name (str), code_blob (str).
    # Outputs: none. Exceptions: AssertionError on failure. Side effects: runs code in the IPython shell.
    ipython_shell.run_cell(code_blob, store_history=True)
    obj = ipython_shell.user_ns[obj_name]
    assert get_source(obj) == code_blob


def test_get_source_standard_class():
    # [FUNCTION DOC]
    # Checks get_source on a locally defined one-line class matches the expected text and inspect's dedented output.
    # Inputs: none. Outputs: none. Exceptions: AssertionError on failure. Side effects: none.
    class TestClass: ...

    source = get_source(TestClass)
    assert source == "class TestClass: ..."
    assert source == textwrap.dedent(inspect.getsource(TestClass)).strip()


def test_get_source_standard_function():
    # [FUNCTION DOC]
    # Checks get_source on a locally defined one-line function matches the expected text and inspect's output.
    # Inputs: none. Outputs: none. Exceptions: AssertionError on failure. Side effects: none.
    def test_func(): ...

    source = get_source(test_func)
    assert source == "def test_func(): ..."
    assert source == textwrap.dedent(inspect.getsource(test_func)).strip()


def test_get_source_ipython_errors_empty_cells(ipython_shell):
    # [FUNCTION DOC]
    # Checks that when the IPython history is empty, get_source raises ValueError "No code cells found".
    # Inputs: ipython_shell (fixture). Outputs: none. Exceptions: AssertionError on failure.
    # Side effects: overwrites the shell's input history list and runs a cell.
    # KNOWN ISSUE: relies on run_cell with store_history appending to the replaced "In" list in a way that still
    #   leaves only blank content; the test depends on IPython history internals.
    test_code = textwrap.dedent("""class TestClass:\n    ...""").strip()
    ipython_shell.user_ns["In"] = [""]
    ipython_shell.run_cell(test_code, store_history=True)
    with pytest.raises(ValueError, match="No code cells found in IPython session"):
        get_source(ipython_shell.user_ns["TestClass"])


def test_get_source_ipython_errors_definition_not_found(ipython_shell):
    # [FUNCTION DOC]
    # Checks that when the history lacks the class definition, get_source raises the "Could not find source code"
    #   ValueError.
    # Inputs: ipython_shell (fixture). Outputs: none. Exceptions: AssertionError on failure.
    # Side effects: overwrites the shell's input history list and runs a cell.
    test_code = textwrap.dedent("""class TestClass:\n    ...""").strip()
    ipython_shell.user_ns["In"] = ["", "print('No class definition here')"]
    ipython_shell.run_cell(test_code, store_history=True)
    with pytest.raises(ValueError, match="Could not find source code for TestClass in IPython history"):
        get_source(ipython_shell.user_ns["TestClass"])


def test_get_source_ipython_errors_type_error():
    # [FUNCTION DOC]
    # Checks that get_source raises TypeError for a non-callable (None).
    # Inputs: none. Outputs: none. Exceptions: AssertionError on failure. Side effects: none.
    with pytest.raises(TypeError, match="Expected class or callable"):
        get_source(None)


@pytest.mark.parametrize(
    "tool, expected_tool_source", [(ValidTool(), VALID_TOOL_SOURCE), (valid_tool_function, VALID_TOOL_FUNCTION_SOURCE)]
)
def test_instance_to_source(tool, expected_tool_source):
    # [FUNCTION DOC]
    # Parametrized test comparing instance_to_source output (with Tool as base) for a class tool and a function tool
    #   against expected source strings.
    # Inputs: tool (Tool instance), expected_tool_source (str). Outputs: none.
    # Exceptions: AssertionError on mismatch. Side effects: none.
    tool_source = instance_to_source(tool, base_cls=Tool)
    assert tool_source == expected_tool_source


def test_e2e_class_tool_save(tmp_path):
    # [FUNCTION DOC]
    # Saves a locally defined class tool with a Gradio app and checks the produced files: tool.py content (including
    #   the generated init), requirements (IPython and smolagents) and app.py.
    # Inputs: tmp_path (pytest fixture). Outputs: none. Exceptions: AssertionError on failure.
    # Side effects: writes files into tmp_path.
    class TestTool(Tool):
        # [CLASS DOC]
        # HELPER: Sample class tool named "test_tool" with one string input "task"; forward imports IPython (to test
        #   requirement detection) and returns the task. Its forward source is compared verbatim.
        name = "test_tool"
        description = "Test tool description"
        inputs = {
            "task": {
                "type": "string",
                "description": "tool input",
            }
        }
        output_type = "string"

        def forward(self, task: str):
            import IPython  # noqa: F401

            return task

    test_tool = TestTool()
    test_tool.save(tmp_path, make_gradio_app=True)
    assert set(os.listdir(tmp_path)) == {"requirements.txt", "app.py", "tool.py"}
    assert (tmp_path / "tool.py").read_text() == textwrap.dedent(
        """\
        from typing import Any, Optional
        from smolagents.tools import Tool
        import IPython

        class TestTool(Tool):
            name = "test_tool"
            description = "Test tool description"
            inputs = {'task': {'type': 'string', 'description': 'tool input'}}
            output_type = "string"

            def forward(self, task: str):
                import IPython  # noqa: F401

                return task

            def __init__(self, *args, **kwargs):
                self.is_initialized = False
        """
    )
    requirements = set((tmp_path / "requirements.txt").read_text().split())
    assert requirements == {"IPython", "smolagents"}
    assert (tmp_path / "app.py").read_text() == textwrap.dedent(
        """\
        from smolagents import launch_gradio_demo
        from tool import TestTool

        tool = TestTool()
        launch_gradio_demo(tool)
        """
    )


def test_e2e_ipython_class_tool_save(tmp_path):
    # [FUNCTION DOC]
    # Same as test_e2e_class_tool_save but defines and saves the tool inside an IPython cell, exercising the IPython
    #   source fallback.
    # Inputs: tmp_path (pytest fixture). Outputs: none. Exceptions: AssertionError on failure.
    # Side effects: runs code in the global IPython shell; writes files into tmp_path.
    shell = InteractiveShell.instance()
    code_blob = textwrap.dedent(
        f"""\
        from smolagents.tools import Tool
        class TestTool(Tool):
            name = "test_tool"
            description = "Test tool description"
            inputs = {{"task": {{"type": "string",
                    "description": "tool input",
                }}
            }}
            output_type = "string"

            def forward(self, task: str):
                import IPython  # noqa: F401

                return task
        TestTool().save("{tmp_path}", make_gradio_app=True)
        """
    )
    assert shell.run_cell(code_blob, store_history=True).success
    assert set(os.listdir(tmp_path)) == {"requirements.txt", "app.py", "tool.py"}
    assert (tmp_path / "tool.py").read_text() == textwrap.dedent(
        """\
        from typing import Any, Optional
        from smolagents.tools import Tool
        import IPython

        class TestTool(Tool):
            name = "test_tool"
            description = "Test tool description"
            inputs = {'task': {'type': 'string', 'description': 'tool input'}}
            output_type = "string"

            def forward(self, task: str):
                import IPython  # noqa: F401

                return task

            def __init__(self, *args, **kwargs):
                self.is_initialized = False
        """
    )
    requirements = set((tmp_path / "requirements.txt").read_text().split())
    assert requirements == {"IPython", "smolagents"}
    assert (tmp_path / "app.py").read_text() == textwrap.dedent(
        """\
        from smolagents import launch_gradio_demo
        from tool import TestTool

        tool = TestTool()
        launch_gradio_demo(tool)
        """
    )


def test_e2e_function_tool_save(tmp_path):
    # [FUNCTION DOC]
    # Saves a function tool created with the tool decorator and checks tool.py (a generated SimpleTool class),
    #   requirements and app.py.
    # Inputs: tmp_path (pytest fixture). Outputs: none. Exceptions: AssertionError on failure.
    # Side effects: writes files into tmp_path.
    # KNOWN ISSUE: as the inline FIXME notes, IPython imported inside the function is not detected as a requirement,
    #   and the test asserts the current (incomplete) behavior.
    @tool
    def test_tool(task: str) -> str:
        """
        Test tool description

        Args:
            task: tool input
        """
        import IPython  # noqa: F401

        return task

    test_tool.save(tmp_path, make_gradio_app=True)
    assert set(os.listdir(tmp_path)) == {"requirements.txt", "app.py", "tool.py"}
    assert (tmp_path / "tool.py").read_text() == textwrap.dedent(
        """\
        from smolagents import Tool
        from typing import Any, Optional

        class SimpleTool(Tool):
            name = "test_tool"
            description = "Test tool description"
            inputs = {'task': {'type': 'string', 'description': 'tool input'}}
            output_type = "string"

            def forward(self, task: str) -> str:
                \"""
                Test tool description

                Args:
                    task: tool input
                \"""
                import IPython  # noqa: F401

                return task"""
    )
    requirements = set((tmp_path / "requirements.txt").read_text().split())
    assert requirements == {"smolagents"}  # FIXME: IPython should be in the requirements
    assert (tmp_path / "app.py").read_text() == textwrap.dedent(
        """\
        from smolagents import launch_gradio_demo
        from tool import SimpleTool

        tool = SimpleTool()
        launch_gradio_demo(tool)
        """
    )


def test_e2e_ipython_function_tool_save(tmp_path):
    # [FUNCTION DOC]
    # Same as test_e2e_function_tool_save but defines and saves the function tool inside an IPython cell.
    # Inputs: tmp_path (pytest fixture). Outputs: none. Exceptions: AssertionError on failure.
    # Side effects: runs code in the global IPython shell; writes files into tmp_path.
    # KNOWN ISSUE: same missing IPython requirement as test_e2e_function_tool_save.
    shell = InteractiveShell.instance()
    code_blob = textwrap.dedent(
        f"""
        from smolagents import tool

        @tool
        def test_tool(task: str) -> str:
            \"""
            Test tool description

            Args:
                task: tool input
            \"""
            import IPython  # noqa: F401

            return task

        test_tool.save("{tmp_path}", make_gradio_app=True)
        """
    )
    assert shell.run_cell(code_blob, store_history=True).success
    assert set(os.listdir(tmp_path)) == {"requirements.txt", "app.py", "tool.py"}
    assert (tmp_path / "tool.py").read_text() == textwrap.dedent(
        """\
        from smolagents import Tool
        from typing import Any, Optional

        class SimpleTool(Tool):
            name = "test_tool"
            description = "Test tool description"
            inputs = {'task': {'type': 'string', 'description': 'tool input'}}
            output_type = "string"

            def forward(self, task: str) -> str:
                \"""
                Test tool description

                Args:
                    task: tool input
                \"""
                import IPython  # noqa: F401

                return task"""
    )
    requirements = set((tmp_path / "requirements.txt").read_text().split())
    assert requirements == {"smolagents"}  # FIXME: IPython should be in the requirements
    assert (tmp_path / "app.py").read_text() == textwrap.dedent(
        """\
        from smolagents import launch_gradio_demo
        from tool import SimpleTool

        tool = SimpleTool()
        launch_gradio_demo(tool)
        """
    )


@pytest.mark.parametrize(
    "raw_json, expected_data, expected_blob",
    [
        (
            """{}""",
            {},
            "",
        ),
        (
            """Text{}""",
            {},
            "Text",
        ),
        (
            """{"simple": "json"}""",
            {"simple": "json"},
            "",
        ),
        (
            """With text here{"simple": "json"}""",
            {"simple": "json"},
            "With text here",
        ),
        (
            """{"simple": "json"}With text after""",
            {"simple": "json"},
            "",
        ),
        (
            """With text before{"simple": "json"}And text after""",
            {"simple": "json"},
            "With text before",
        ),
    ],
)
def test_parse_json_blob_with_valid_json(raw_json, expected_data, expected_blob):
    # [FUNCTION DOC]
    # Parametrized test checking parse_json_blob returns the parsed dict and the text before the JSON (text after is
    #   dropped).
    # Inputs: raw_json (str), expected_data (dict), expected_blob (str). Outputs: none.
    # Exceptions: AssertionError on failure. Side effects: none.
    data, blob = parse_json_blob(raw_json)

    assert data == expected_data
    assert blob == expected_blob


@pytest.mark.parametrize(
    "raw_json",
    [
        """simple": "json"}""",
        """With text here"simple": "json"}""",
        """{"simple": ""json"}With text after""",
        """{"simple": "json"With text after""",
        "}}",
    ],
)
def test_parse_json_blob_with_invalid_json(raw_json):
    # [FUNCTION DOC]
    # Parametrized test checking parse_json_blob raises for malformed or missing JSON.
    # Inputs: raw_json (str). Outputs: none. Exceptions: AssertionError if nothing is raised. Side effects: none.
    with pytest.raises(Exception):
        parse_json_blob(raw_json)


@pytest.mark.parametrize(
    "name,expected",
    [
        # Valid identifiers
        ("valid_name", True),
        ("ValidName", True),
        ("valid123", True),
        ("_private", True),
        # Invalid identifiers
        ("", False),
        ("123invalid", False),
        ("invalid-name", False),
        ("invalid name", False),
        ("invalid.name", False),
        # Python keywords
        ("if", False),
        ("for", False),
        ("class", False),
        ("return", False),
        # Non-string inputs
        (123, False),
        (None, False),
        ([], False),
        ({}, False),
    ],
)
def test_is_valid_name(name, expected):
    # [FUNCTION DOC]
    # Parametrized test of is_valid_name over valid identifiers, invalid strings, keywords and non-string values.
    # Inputs: name (any), expected (bool). Outputs: none. Exceptions: AssertionError on mismatch. Side effects: none.
    """Test the is_valid_name function with various inputs."""
    assert is_valid_name(name) is expected


def test_agent_gradio_app_template_excludes_class_keyword():
    # [FUNCTION DOC]
    # Renders the agent Gradio app template with an agent_dict containing a "class" key and checks that it is not
    #   emitted as a keyword argument, that other attributes are, and that the result is valid Python.
    # Inputs: none. Outputs: none. Exceptions: AssertionError or pytest failure on syntax error. Side effects: none.
    """Test that the AGENT_GRADIO_APP_TEMPLATE excludes 'class' from agent kwargs."""

    # Mock agent_dict with 'class' key that should be excluded
    agent_dict = {
        "model": {"class": "CodeAgent", "data": {}},
        "class": "CodeAgent",  # This should be excluded to prevent SyntaxError
        "some_valid_attr": "value",
        "tools": [],
        "managed_agents": {},
        "requirements": [],
        "prompt_templates": {},
    }

    template = create_agent_gradio_app_template()
    result = template.render(
        agent_name="test_agent",
        class_name="CodeAgent",
        agent_dict=agent_dict,
        tools={},
        managed_agents={},
        managed_agent_relative_path="",
    )

    # Should contain valid attribute but not 'class='  as a keyword argument
    assert "some_valid_attr='value'," in result
    assert "class=" not in result

    # Verify the generated code is syntactically valid Python
    import ast

    try:
        ast.parse(result)
    except SyntaxError as e:
        pytest.fail(f"Generated app.py contains syntax error: {e}")
