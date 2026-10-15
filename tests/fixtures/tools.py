# [MODULE DOC]
# Description: HELPER: Fixture module providing ready-made Tool instances (class-based and decorator-based)
#   with various input shapes for tool tests.
# Inputs: depends on pytest and smolagents.tools (Tool, tool).
# Outputs: fixtures test_tool, no_input_tool, single_input_tool, multi_input_tool, multiline_description_tool,
#   example_tool, boolean_default_tool_class, boolean_default_tool_function, optional_input_tool_class,
#   optional_input_tool_function.
# Exceptions: none raised directly.
# Side effects: none.
import pytest

from smolagents.tools import Tool, tool


@pytest.fixture
def test_tool():
    # [FUNCTION DOC]
    # Description: HELPER: Fixture that defines and instantiates TestTool, a string-in string-out tool.
    # Inputs: none.
    # Outputs: a TestTool instance.
    # Exceptions: none raised directly.
    # Side effects: none.
    class TestTool(Tool):
        # [CLASS DOC]
        # Description: HELPER: Test Tool subclass: named test_tool.
        # Inputs: no constructor args; class attributes name, description, inputs ({inputs}) and output_type
        #   ({otype}).
        # Outputs: tool instance.
        # Exceptions: none raised directly.
        # Side effects: none.
        name = "test_tool"
        description = "A test tool"
        inputs = {"input": {"type": "string", "description": "Input value"}}
        output_type = "string"

        def forward(self, input):
            # [FUNCTION DOC]
            # Description: HELPER: Returns 'Processed: ' followed by the input, or fails on the input 'error'.
            # Inputs: input (str).
            # Outputs: str.
            # Exceptions: ValueError with message Tool execution error when input equals error.
            # Side effects: none.
            if input == "error":
                raise ValueError("Tool execution error")
            return f"Processed: {input}"

    return TestTool()


@pytest.fixture
def no_input_tool():
    # [FUNCTION DOC]
    # Description: HELPER: Fixture that defines and instantiates NoInputTool, a tool without inputs.
    # Inputs: none.
    # Outputs: a NoInputTool instance.
    # Exceptions: none raised directly.
    # Side effects: none.
    class NoInputTool(Tool):
        # [CLASS DOC]
        # Description: HELPER: Test Tool subclass: named no_input_tool.
        # Inputs: no constructor args; class attributes name, description, inputs ({inputs}) and output_type
        #   ({otype}).
        # Outputs: tool instance.
        # Exceptions: none raised directly.
        # Side effects: none.
        name = "no_input_tool"
        description = "Tool with no inputs"
        inputs = {}
        output_type = "string"

        def forward(self):
            # [FUNCTION DOC]
            # Description: HELPER: Returns the constant string test.
            # Inputs: none.
            # Outputs: str test.
            # Exceptions: none raised directly.
            # Side effects: none.
            return "test"

    return NoInputTool()


@pytest.fixture
def single_input_tool():
    # [FUNCTION DOC]
    # Description: HELPER: Fixture that defines and instantiates SingleInputTool, a tool with one string
    #   input.
    # Inputs: none.
    # Outputs: a SingleInputTool instance.
    # Exceptions: none raised directly.
    # Side effects: none.
    class SingleInputTool(Tool):
        # [CLASS DOC]
        # Description: HELPER: Test Tool subclass: named single_input_tool.
        # Inputs: no constructor args; class attributes name, description, inputs ({inputs}) and output_type
        #   ({otype}).
        # Outputs: tool instance.
        # Exceptions: none raised directly.
        # Side effects: none.
        name = "single_input_tool"
        description = "Tool with one input"
        inputs = {"text": {"type": "string", "description": "Input text"}}
        output_type = "string"

        def forward(self, text):
            # [FUNCTION DOC]
            # Description: HELPER: Ignores its input and returns test.
            # Inputs: text (str).
            # Outputs: str test.
            # Exceptions: none raised directly.
            # Side effects: none.
            return "test"

    return SingleInputTool()


@pytest.fixture
def multi_input_tool():
    # [FUNCTION DOC]
    # Description: HELPER: Fixture that defines and instantiates MultiInputTool, a tool with a string and an
    #   integer input.
    # Inputs: none.
    # Outputs: a MultiInputTool instance.
    # Exceptions: none raised directly.
    # Side effects: none.
    class MultiInputTool(Tool):
        # [CLASS DOC]
        # Description: HELPER: Test Tool subclass: named multi_input_tool.
        # Inputs: no constructor args; class attributes name, description, inputs ({inputs}) and output_type
        #   ({otype}).
        # Outputs: tool instance.
        # Exceptions: none raised directly.
        # Side effects: none.
        name = "multi_input_tool"
        description = "Tool with multiple inputs"
        inputs = {
            "text": {"type": "string", "description": "Text input"},
            "count": {"type": "integer", "description": "Number count"},
        }
        output_type = "object"

        def forward(self, text, count):
            # [FUNCTION DOC]
            # Description: HELPER: Ignores its inputs and returns test.
            # Inputs: text (str), count (int).
            # Outputs: str test.
            # Exceptions: none raised directly.
            # Side effects: none.
            return "test"

    return MultiInputTool()


@pytest.fixture
def multiline_description_tool():
    # [FUNCTION DOC]
    # Description: HELPER: Fixture that defines and instantiates MultilineDescriptionTool, a tool whose
    #   description spans three lines.
    # Inputs: none.
    # Outputs: a MultilineDescriptionTool instance.
    # Exceptions: none raised directly.
    # Side effects: none.
    class MultilineDescriptionTool(Tool):
        # [CLASS DOC]
        # Description: HELPER: Test Tool subclass: named multiline_description_tool with a newline-separated
        #   description.
        # Inputs: no constructor args; class attributes name, description, inputs ({inputs}) and output_type
        #   ({otype}).
        # Outputs: tool instance.
        # Exceptions: none raised directly.
        # Side effects: none.
        name = "multiline_description_tool"
        description = "This is a tool with\nmultiple lines\nin the description"
        inputs = {"input": {"type": "string", "description": "Some input"}}
        output_type = "string"

        def forward(self, input):
            # [FUNCTION DOC]
            # Description: HELPER: Ignores its input and returns test.
            # Inputs: input (str).
            # Outputs: str test.
            # Exceptions: none raised directly.
            # Side effects: none.
            return "test"

    return MultilineDescriptionTool()


@pytest.fixture
def example_tool():
    @tool
    # [FUNCTION DOC]
    # Description: HELPER: Fixture that builds a function tool via the tool decorator.
    # Inputs: none.
    # Outputs: the tool object named valid_tool_function.
    # Exceptions: none raised directly.
    # Side effects: none.
    def valid_tool_function(input: str) -> str:
        # [FUNCTION DOC]
        # Description: HELPER: Tool function that upper-cases its input.
        # Inputs: input (str).
        # Outputs: str, upper-cased input.
        # Exceptions: none raised directly.
        # Side effects: none.
        """A valid tool function.

        Args:
            input (str): Input string.
        """
        return input.upper()

    return valid_tool_function


@pytest.fixture
def boolean_default_tool_class():
    # [FUNCTION DOC]
    # Description: HELPER: Fixture that defines and instantiates BooleanDefaultTool, a tool with a nullable
    #   boolean flag defaulting to False.
    # Inputs: none.
    # Outputs: a BooleanDefaultTool instance.
    # Exceptions: none raised directly.
    # Side effects: none.
    class BooleanDefaultTool(Tool):
        # [CLASS DOC]
        # Description: HELPER: Test Tool subclass: named boolean_default_tool.
        # Inputs: no constructor args; class attributes name, description, inputs ({inputs}) and output_type
        #   ({otype}).
        # Outputs: tool instance.
        # Exceptions: none raised directly.
        # Side effects: none.
        name = "boolean_default_tool"
        description = "A tool with a boolean default parameter"
        inputs = {
            "text": {"type": "string", "description": "Input text"},
            "flag": {"type": "boolean", "description": "Boolean flag with default value", "nullable": True},
        }
        output_type = "string"

        def forward(self, text: str, flag: bool = False) -> str:
            # [FUNCTION DOC]
            # Description: HELPER: Formats text and flag into one string.
            # Inputs: text (str), flag (bool, default False).
            # Outputs: str of the form Text: <text>, Flag: <flag>.
            # Exceptions: none raised directly.
            # Side effects: none.
            return f"Text: {text}, Flag: {flag}"

    return BooleanDefaultTool()


@pytest.fixture
def boolean_default_tool_function():
    @tool
    # [FUNCTION DOC]
    # Description: HELPER: Fixture that builds the decorator-based equivalent of BooleanDefaultTool.
    # Inputs: none.
    # Outputs: the tool object named boolean_default_tool.
    # Exceptions: none raised directly.
    # Side effects: none.
    def boolean_default_tool(text: str, flag: bool = False) -> str:
        # [FUNCTION DOC]
        # Description: HELPER: Formats text and flag into one string.
        # Inputs: text (str), flag (bool, default False).
        # Outputs: str of the form Text: <text>, Flag: <flag>.
        # Exceptions: none raised directly.
        # Side effects: none.
        """
        A tool with a boolean default parameter.

        Args:
            text: Input text
            flag: Boolean flag with default value
        """
        return f"Text: {text}, Flag: {flag}"

    return boolean_default_tool


@pytest.fixture
def optional_input_tool_class():
    # [FUNCTION DOC]
    # Description: HELPER: Fixture that defines and instantiates OptionalInputTool, a tool with a required and
    #   an optional (nullable) text.
    # Inputs: none.
    # Outputs: an OptionalInputTool instance.
    # Exceptions: none raised directly.
    # Side effects: none.
    class OptionalInputTool(Tool):
        # [CLASS DOC]
        # Description: HELPER: Test Tool subclass: named optional_input_tool.
        # Inputs: no constructor args; class attributes name, description, inputs ({inputs}) and output_type
        #   ({otype}).
        # Outputs: tool instance.
        # Exceptions: none raised directly.
        # Side effects: none.
        name = "optional_input_tool"
        description = "A tool with an optional input parameter"
        inputs = {
            "required_text": {"type": "string", "description": "Required input text"},
            "optional_text": {"type": "string", "description": "Optional input text", "nullable": True},
        }
        output_type = "string"

        def forward(self, required_text: str, optional_text: str | None = None) -> str:
            # [FUNCTION DOC]
            # Description: HELPER: Returns required_text joined with optional_text by ' + ' when optional_text
            #   is truthy, otherwise required_text alone.
            # Inputs: required_text (str); optional_text (str or None, default None).
            # Outputs: str.
            # Exceptions: none raised directly.
            # Side effects: none.
            if optional_text:
                return f"{required_text} + {optional_text}"
            return required_text

    return OptionalInputTool()


@pytest.fixture
def optional_input_tool_function():
    @tool
    # [FUNCTION DOC]
    # Description: HELPER: Fixture that builds the decorator-based equivalent of OptionalInputTool.
    # Inputs: none.
    # Outputs: the tool object named optional_input_tool.
    # Exceptions: none raised directly.
    # Side effects: none.
    def optional_input_tool(required_text: str, optional_text: str | None = None) -> str:
        # [FUNCTION DOC]
        # Description: HELPER: Returns required_text joined with optional_text by ' + ' when optional_text is
        #   truthy, otherwise required_text alone.
        # Inputs: required_text (str); optional_text (str or None, default None).
        # Outputs: str.
        # Exceptions: none raised directly.
        # Side effects: none.
        """
        A tool with an optional input parameter.

        Args:
            required_text: Required input text
            optional_text: Optional input text
        """
        if optional_text:
            return f"{required_text} + {optional_text}"
        return required_text

    return optional_input_tool
