# [MODULE DOC]
# Description: Tests for validate_tool_attributes and MethodChecker using built-in tools and purpose-built
#   valid/invalid Tool classes.
# Inputs: depends on pytest, ast, default tools, MethodChecker, validate_tool_attributes, Tool and tool.
# Outputs: test classes/functions and the constant UNDEFINED_VARIABLE (a string used as an undefined name in
#   sample tools).
# Exceptions: none raised directly.
# Side effects: instantiates MultipleAssignmentsTool at import (in a parametrize list).
import ast
from textwrap import dedent

import pytest

from smolagents.default_tools import (
    DuckDuckGoSearchTool,
    GoogleSearchTool,
    SpeechToTextTool,
    VisitWebpageTool,
    WebSearchTool,
)
from smolagents.tool_validation import MethodChecker, validate_tool_attributes
from smolagents.tools import Tool, tool


UNDEFINED_VARIABLE = "undefined_variable"


@pytest.mark.parametrize(
    "tool_class", [DuckDuckGoSearchTool, GoogleSearchTool, SpeechToTextTool, VisitWebpageTool, WebSearchTool]
)
def test_validate_tool_attributes_with_default_tools(tool_class):
    # [FUNCTION DOC]
    # Description: Checks each default tool class passes validation.
    # Inputs: tool_class (parametrized).
    # Outputs: none (passes or fails via assertions).
    # Exceptions: AssertionError on failure.
    # Side effects: none.
    assert validate_tool_attributes(tool_class) is None, f"failed for {tool_class.name} tool"


class ValidTool(Tool):
    # [CLASS DOC]
    # Description: HELPER: Sample tool with string/dict class attributes and an __init__ with a literal
    #   default; must validate.
    # Inputs: optional_param (default 'default').
    # Outputs: instance.
    # Exceptions: none raised directly.
    # Side effects: none.
    name = "valid_tool"
    description = "A valid tool"
    inputs = {"input": {"type": "string", "description": "input"}}
    output_type = "string"
    simple_attr = "string"
    dict_attr = {"key": "value"}

    def __init__(self, optional_param="default"):
        # [FUNCTION DOC]
        # Description: HELPER: Calls Tool init and stores the param.
        # Inputs: optional_param (default 'default').
        # Outputs: None.
        # Exceptions: none raised directly.
        # Side effects: sets self.param.
        super().__init__()
        self.param = optional_param

    def forward(self, input: str) -> str:
        # [FUNCTION DOC]
        # Description: HELPER: Upper-cases the input.
        # Inputs: input (str).
        # Outputs: str.
        # Exceptions: none raised directly.
        # Side effects: none.
        return input.upper()


@tool
def valid_tool_function(input: str) -> str:
    # [FUNCTION DOC]
    # Description: HELPER: Decorator-based sample tool that upper-cases its input.
    # Inputs: input (str).
    # Outputs: str.
    # Exceptions: none raised directly.
    # Side effects: none.
    """A valid tool function.

    Args:
        input (str): Input string.
    """
    return input.upper()


@pytest.mark.parametrize("tool_class", [ValidTool, valid_tool_function.__class__])
def test_validate_tool_attributes_valid(tool_class):
    # [FUNCTION DOC]
    # Description: Checks ValidTool and the class generated for valid_tool_function pass.
    # Inputs: tool_class (parametrized).
    # Outputs: none (passes or fails via assertions).
    # Exceptions: AssertionError on failure.
    # Side effects: none.
    assert validate_tool_attributes(tool_class) is None


class InvalidToolName(Tool):
    # [CLASS DOC]
    # Description: HELPER: Sample tool whose name contains spaces (invalid identifier).
    # Inputs: none.
    # Outputs: instance.
    # Exceptions: none raised directly.
    # Side effects: none.
    name = "invalid tool name"
    description = "Tool with invalid name"
    inputs = {"input": {"type": "string", "description": "input"}}
    output_type = "string"

    def __init__(self):
        # [FUNCTION DOC]
        # Description: HELPER: Calls Tool init.
        # Inputs: none.
        # Outputs: None.
        # Exceptions: none raised directly.
        # Side effects: none.
        super().__init__()

    def forward(self, input: str) -> str:
        # [FUNCTION DOC]
        # Description: HELPER: Returns its input.
        # Inputs: input (str).
        # Outputs: str.
        # Exceptions: none raised directly.
        # Side effects: none.
        return input


class InvalidToolComplexAttrs(Tool):
    # [CLASS DOC]
    # Description: HELPER: Sample tool with a list-comprehension class attribute (complex).
    # Inputs: none.
    # Outputs: instance.
    # Exceptions: none raised directly.
    # Side effects: none.
    name = "invalid_tool"
    description = "Tool with complex class attributes"
    inputs = {"input": {"type": "string", "description": "input"}}
    output_type = "string"
    complex_attr = [x for x in range(3)]  # Complex class attribute

    def __init__(self):
        # [FUNCTION DOC]
        # Description: HELPER: Calls Tool init.
        # Inputs: none.
        # Outputs: None.
        # Exceptions: none raised directly.
        # Side effects: none.
        super().__init__()

    def forward(self, input: str) -> str:
        # [FUNCTION DOC]
        # Description: HELPER: Returns its input.
        # Inputs: input (str).
        # Outputs: str.
        # Exceptions: none raised directly.
        # Side effects: none.
        return input


class InvalidToolRequiredParams(Tool):
    # [CLASS DOC]
    # Description: HELPER: Sample tool whose __init__ has a parameter without default.
    # Inputs: required_param; kwarg1 (default 1).
    # Outputs: instance.
    # Exceptions: none raised directly.
    # Side effects: none.
    name = "invalid_tool"
    description = "Tool with required params"
    inputs = {"input": {"type": "string", "description": "input"}}
    output_type = "string"

    def __init__(self, required_param, kwarg1=1):  # No default value
        # [FUNCTION DOC]
        # Description: HELPER: Calls Tool init and stores the required param.
        # Inputs: required_param; kwarg1 (default 1, unused).
        # Outputs: None.
        # Exceptions: none raised directly.
        # Side effects: sets self.param.
        super().__init__()
        self.param = required_param

    def forward(self, input: str) -> str:
        # [FUNCTION DOC]
        # Description: HELPER: Returns its input.
        # Inputs: input (str).
        # Outputs: str.
        # Exceptions: none raised directly.
        # Side effects: none.
        return input


class InvalidToolNonLiteralDefaultParam(Tool):
    # [CLASS DOC]
    # Description: HELPER: Sample tool whose __init__ default is a variable reference (non-literal).
    # Inputs: default_param (default the module constant).
    # Outputs: instance.
    # Exceptions: none raised directly.
    # Side effects: none.
    name = "invalid_tool"
    description = "Tool with non-literal default parameter value"
    inputs = {"input": {"type": "string", "description": "input"}}
    output_type = "string"

    def __init__(self, default_param=UNDEFINED_VARIABLE):  # UNDEFINED_VARIABLE as default is non-literal
        # [FUNCTION DOC]
        # Description: HELPER: Calls Tool init and stores default_param.
        # Inputs: default_param.
        # Outputs: None.
        # Exceptions: none raised directly.
        # Side effects: sets self.default_param.
        super().__init__()
        self.default_param = default_param

    def forward(self, input: str) -> str:
        # [FUNCTION DOC]
        # Description: HELPER: Returns its input.
        # Inputs: input (str).
        # Outputs: str.
        # Exceptions: none raised directly.
        # Side effects: none.
        return input


class InvalidToolUndefinedNames(Tool):
    # [CLASS DOC]
    # Description: HELPER: Sample tool whose forward references a module-level name not available to the tool.
    # Inputs: none.
    # Outputs: instance.
    # Exceptions: none raised directly.
    # Side effects: none.
    name = "invalid_tool"
    description = "Tool with undefined names"
    inputs = {"input": {"type": "string", "description": "input"}}
    output_type = "string"

    def forward(self, input: str) -> str:
        # [FUNCTION DOC]
        # Description: HELPER: Returns the module constant UNDEFINED_VARIABLE (seen as undefined by
        #   validation).
        # Inputs: input (str, unused).
        # Outputs: str.
        # Exceptions: none raised directly.
        # Side effects: none.
        return UNDEFINED_VARIABLE  # Undefined name


@pytest.mark.parametrize(
    "tool_class, expected_error",
    [
        (
            InvalidToolName,
            "Class attribute 'name' must be a valid Python identifier and not a reserved keyword, found 'invalid tool name'",
        ),
        (InvalidToolComplexAttrs, "Complex attributes should be defined in __init__, not as class attributes"),
        (InvalidToolRequiredParams, "Parameters in __init__ must have default values, found required parameters"),
        (
            InvalidToolNonLiteralDefaultParam,
            "Parameters in __init__ must have literal default values, found non-literal defaults",
        ),
        (InvalidToolUndefinedNames, "Name 'UNDEFINED_VARIABLE' is undefined"),
    ],
)
def test_validate_tool_attributes_exceptions(tool_class, expected_error):
    # [FUNCTION DOC]
    # Description: Checks each invalid sample tool raises ValueError whose message matches the expected error
    #   text.
    # Inputs: tool_class, expected_error (parametrized).
    # Outputs: none (passes or fails via assertions).
    # Exceptions: AssertionError / pytest failure if no matching ValueError.
    # Side effects: none.
    with pytest.raises(ValueError, match=expected_error):
        validate_tool_attributes(tool_class)


class MultipleAssignmentsTool(Tool):
    # [CLASS DOC]
    # Description: HELPER: Sample tool whose forward uses tuple-unpacking assignment.
    # Inputs: none.
    # Outputs: instance.
    # Exceptions: none raised directly.
    # Side effects: none.
    name = "multiple_assignments_tool"
    description = "Tool with multiple assignments"
    inputs = {"input": {"type": "string", "description": "input"}}
    output_type = "string"

    def __init__(self):
        # [FUNCTION DOC]
        # Description: HELPER: Calls Tool init.
        # Inputs: none.
        # Outputs: None.
        # Exceptions: none raised directly.
        # Side effects: none.
        super().__init__()

    def forward(self, input: str) -> str:
        # [FUNCTION DOC]
        # Description: HELPER: Returns the concatenation of two unpacked strings (always 12).
        # Inputs: input (str, unused).
        # Outputs: str 12.
        # Exceptions: none raised directly.
        # Side effects: none.
        a, b = "1", "2"
        return a + b


def test_validate_tool_attributes_multiple_assignments():
    # [FUNCTION DOC]
    # Description: Checks tuple-unpacking names count as defined (no error raised).
    # Inputs: none.
    # Outputs: none (passes or fails via assertions).
    # Exceptions: ValueError would propagate on failure.
    # Side effects: none.
    validate_tool_attributes(MultipleAssignmentsTool)


@tool
def tool_function_with_multiple_assignments(input: str) -> str:
    # [FUNCTION DOC]
    # Description: HELPER: Decorator-based sample tool using tuple unpacking; returns the upper-cased input
    #   followed by 12.
    # Inputs: input (str).
    # Outputs: str.
    # Exceptions: none raised directly.
    # Side effects: none.
    """A valid tool function.

    Args:
        input (str): Input string.
    """
    a, b = "1", "2"
    return input.upper() + a + b


@pytest.mark.parametrize("tool_instance", [MultipleAssignmentsTool(), tool_function_with_multiple_assignments])
def test_tool_to_dict_validation_with_multiple_assignments(tool_instance):
    # [FUNCTION DOC]
    # Description: Checks to_dict (which validates the tool) succeeds for class and function tools with tuple
    #   unpacking.
    # Inputs: tool_instance (parametrized).
    # Outputs: none (passes or fails via assertions).
    # Exceptions: exceptions from to_dict propagate as failures.
    # Side effects: none.
    tool_instance.to_dict()


class TestMethodChecker:
    # [CLASS DOC]
    # Description: Direct tests of MethodChecker.
    # Inputs: none.
    # Outputs: test outcomes.
    # Exceptions: AssertionError on failure.
    # Side effects: none.
    def test_multiple_assignments(self):
        # [FUNCTION DOC]
        # Description: Visits a parsed function using tuple unpacking and checks no errors are recorded.
        # Inputs: self.
        # Outputs: none (passes or fails via assertions).
        # Exceptions: AssertionError on failure.
        # Side effects: none.
        source_code = dedent(
            """
            def forward(self) -> str:
                a, b = "1", "2"
                return a + b
            """
        )
        method_checker = MethodChecker(set())
        method_checker.visit(ast.parse(source_code))
        assert method_checker.errors == []
