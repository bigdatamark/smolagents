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
# What: pytest tests for smolagents._function_type_hints_utils: get_json_schema (types, unions, optional args,
#   enum choices, docstring parsing quirks, error cases) and get_imports.
# Inputs: depends on pytest, typing.Any and DocstringParsingException, get_imports, get_json_schema.
# Outputs: pytest fixtures returning sample functions, plus test classes TestGetJsonSchema and TestGetCode.
# Exceptions: assertion failures on regressions.
# Side effects: none.
from typing import Any

import pytest

from smolagents._function_type_hints_utils import DocstringParsingException, get_imports, get_json_schema


@pytest.fixture
def valid_func():
    # [FUNCTION DOC]
    # HELPER: pytest fixture returning a fully documented function with hints and a Returns block.
    # Inputs: none.
    # Outputs: the nested function (multiply).
    # Exceptions: none raised directly.
    # Side effects: none.
    """A well-formed function with docstring, type hints, and return block."""

    def multiply(x: int, y: float) -> float:
        # [FUNCTION DOC]
        # HELPER: sample function used only as get_json_schema input; x int, y float, documented args and a
        #   Returns description.
        # Inputs: x (int), y (float).
        # Outputs: the product of x and y (float).
        # Exceptions: none raised directly.
        # Side effects: none.
        """
        Multiplies two numbers.

        Args:
            x: The first number.
            y: The second number.
        Returns:
            Product of x and y.
        """
        return x * y

    return multiply


@pytest.fixture
def no_docstring_func():
    # [FUNCTION DOC]
    # HELPER: pytest fixture returning a hinted function without any docstring.
    # Inputs: none.
    # Outputs: the nested function (sample).
    # Exceptions: none raised directly.
    # Side effects: none.
    """Function with no docstring."""

    def sample(x: int):
        # [FUNCTION DOC]
        # HELPER: sample function used only as get_json_schema input; deliberately has no docstring.
        # Inputs: x (int).
        # Outputs: x unchanged.
        # Exceptions: none raised directly.
        # Side effects: none.
        return x

    return sample


@pytest.fixture
def missing_arg_doc_func():
    # [FUNCTION DOC]
    # HELPER: pytest fixture returning a function whose docstring omits the description for argument y.
    # Inputs: none.
    # Outputs: the nested function (add).
    # Exceptions: none raised directly.
    # Side effects: none.
    """Function with docstring but missing an argument description."""

    def add(x: int, y: int):
        # [FUNCTION DOC]
        # HELPER: sample function used only as get_json_schema input; docstring documents x only.
        # Inputs: x (int), y (int).
        # Outputs: x plus y.
        # Exceptions: none raised directly.
        # Side effects: none.
        """
        Adds two numbers.

        Args:
            x: The first number.
        """
        return x + y

    return add


@pytest.fixture
def bad_return_func():
    # [FUNCTION DOC]
    # HELPER: pytest fixture returning a function with an optional string argument and no Returns block
    #   (allowed).
    # Inputs: none.
    # Outputs: the nested function (do_nothing).
    # Exceptions: none raised directly.
    # Side effects: none.
    """Function docstring with missing return description (allowed)."""

    def do_nothing(x: str | None = None):
        # [FUNCTION DOC]
        # HELPER: sample function used only as get_json_schema input; optional nullable string with default
        #   None, no return annotation.
        # Inputs: x (str or None, default None).
        # Outputs: None.
        # Exceptions: none raised directly.
        # Side effects: none.
        """
        Does nothing.

        Args:
            x: Some optional string.
        """
        pass

    return do_nothing


@pytest.fixture
def complex_types_func():
    # [FUNCTION DOC]
    # HELPER: pytest fixture returning a function using list, dict and two-element tuple hints.
    # Inputs: none.
    # Outputs: the nested function (process_data).
    # Exceptions: none raised directly.
    # Side effects: none.
    def process_data(items: list[str], config: dict[str, float], point: tuple[int, int]) -> dict:
        # [FUNCTION DOC]
        # HELPER: sample function used only as get_json_schema input; generic container hints.
        # Inputs: items (list of str), config (dict str->float), point (tuple of two ints).
        # Outputs: a constant dict with key 'result' set to True.
        # Exceptions: none raised directly.
        # Side effects: none.
        """
        Process some data.

        Args:
            items: List of items to process.
            config: Configuration parameters.
            point: A position as (x,y).

        Returns:
            Processed data result.
        """
        return {"result": True}

    return process_data


@pytest.fixture
def optional_types_func():
    # [FUNCTION DOC]
    # HELPER: pytest fixture returning a function with one required and one optional (int or None) argument.
    # Inputs: none.
    # Outputs: the nested function (process_with_optional).
    # Exceptions: none raised directly.
    # Side effects: none.
    def process_with_optional(required_arg: str, optional_arg: int | None = None) -> str:
        # [FUNCTION DOC]
        # HELPER: sample function used only as get_json_schema input; required and optional argument.
        # Inputs: required_arg (str), optional_arg (int or None, default None).
        # Outputs: the constant string 'processed'.
        # Exceptions: none raised directly.
        # Side effects: none.
        """
        Process with optional argument.

        Args:
            required_arg: A required string argument.
            optional_arg: An optional integer argument.

        Returns:
            Processing result.
        """
        return "processed"

    return process_with_optional


@pytest.fixture
def enum_choices_func():
    # [FUNCTION DOC]
    # HELPER: pytest fixture returning a function whose argument description ends with a choices block.
    # Inputs: none.
    # Outputs: the nested function (select_color).
    # Exceptions: none raised directly.
    # Side effects: none.
    def select_color(color: str) -> str:
        # [FUNCTION DOC]
        # HELPER: sample function used only as get_json_schema input; argument description has a '(choices:
        #   [...])' suffix.
        # Inputs: color (str).
        # Outputs: color unchanged.
        # Exceptions: none raised directly.
        # Side effects: none.
        """
        Select a color.

        Args:
            color: The color to select (choices: ["red", "green", "blue"])

        Returns:
            Selected color.
        """
        return color

    return select_color


@pytest.fixture
def union_types_func():
    # [FUNCTION DOC]
    # HELPER: pytest fixture returning a function with union-typed argument and return.
    # Inputs: none.
    # Outputs: the nested function (process_union).
    # Exceptions: none raised directly.
    # Side effects: none.
    def process_union(value: int | str) -> bool | str:
        # [FUNCTION DOC]
        # HELPER: sample function used only as get_json_schema input; int-or-str argument and bool-or-str
        #   return.
        # Inputs: value (int or str).
        # Outputs: True for ints, otherwise the string 'string result'.
        # Exceptions: none raised directly.
        # Side effects: none.
        """
        Process a value that can be either int or string.

        Args:
            value: An integer or string value.

        Returns:
            Processing result.
        """
        return True if isinstance(value, int) else "string result"

    return process_union


@pytest.fixture
def nested_types_func():
    # [FUNCTION DOC]
    # HELPER: pytest fixture returning a function with a list-of-dicts argument.
    # Inputs: none.
    # Outputs: the nested function (process_nested_data).
    # Exceptions: none raised directly.
    # Side effects: none.
    def process_nested_data(data: list[dict[str, Any]]) -> list[str]:
        # [FUNCTION DOC]
        # HELPER: sample function used only as get_json_schema input; nested generic hint.
        # Inputs: data (list of dicts str->Any).
        # Outputs: a constant one-element list of str.
        # Exceptions: none raised directly.
        # Side effects: none.
        """
        Process nested data structure.

        Args:
            data: List of dictionaries to process.

        Returns:
            List of processed results.
        """
        return ["result"]

    return process_nested_data


@pytest.fixture
def typed_docstring_func():
    # [FUNCTION DOC]
    # HELPER: pytest fixture returning a function whose docstring includes parenthesised types for args and a
    #   typed Returns.
    # Inputs: none.
    # Outputs: the nested function (calculate).
    # Exceptions: none raised directly.
    # Side effects: none.
    def calculate(x: int, y: float) -> float:
        # [FUNCTION DOC]
        # HELPER: sample function used only as get_json_schema input; docstring types agree with the hints.
        # Inputs: x (int), y (float).
        # Outputs: x times y.
        # Exceptions: none raised directly.
        # Side effects: none.
        """
        Calculate something.

        Args:
            x (int): An integer parameter with type in docstring.
            y (float): A float parameter with type in docstring.

        Returns:
            float: The calculated result.
        """
        return x * y

    return calculate


@pytest.fixture
def mismatched_types_func():
    # [FUNCTION DOC]
    # HELPER: pytest fixture returning a function whose docstring types contradict its hints.
    # Inputs: none.
    # Outputs: the nested function (convert).
    # Exceptions: none raised directly.
    # Side effects: none.
    def convert(value: int) -> str:
        # [FUNCTION DOC]
        # HELPER: sample function used only as get_json_schema input; docstring claims str/int while hints say
        #   int/str.
        # Inputs: value (int).
        # Outputs: the string form of value.
        # Exceptions: none raised directly.
        # Side effects: none.
        """
        Convert a value.

        Args:
            value (str): A string value (type mismatch with hint).

        Returns:
            int: Converted value (type mismatch with hint).
        """
        return str(value)

    return convert


@pytest.fixture
def complex_docstring_types_func():
    # [FUNCTION DOC]
    # HELPER: pytest fixture returning a function with nested generic hints and matching typed docstring
    #   entries.
    # Inputs: none.
    # Outputs: the nested function (process).
    # Exceptions: none raised directly.
    # Side effects: none.
    def process(data: dict[str, list[int]]) -> list[dict[str, Any]]:
        # [FUNCTION DOC]
        # HELPER: sample function used only as get_json_schema input; dict of str->list of int argument.
        # Inputs: data (dict str->list of int).
        # Outputs: a one-element list holding a dict comprehension result (dict mapping each key to the sum of
        #   its list).
        # Exceptions: none raised directly.
        # Side effects: none.
        """
        Process complex data.

        Args:
            data (Dict[str, List[int]]): Nested structure with types.

        Returns:
            List[Dict[str, Any]]: Processed results with types.
        """
        return [{"result": sum(v) for k, v in data.items()}]

    return process


@pytest.fixture
def keywords_in_description_func():
    # [FUNCTION DOC]
    # HELPER: pytest fixture returning a function whose description mentions 'Args:' and 'Returns:' inline.
    # Inputs: none.
    # Outputs: the nested function (process).
    # Exceptions: none raised directly.
    # Side effects: none.
    def process(value: str) -> str:
        # [FUNCTION DOC]
        # HELPER: sample function used only as get_json_schema input; description contains section keywords
        #   mid-line.
        # Inputs: value (str).
        # Outputs: value upper-cased.
        # Exceptions: none raised directly.
        # Side effects: none.
        """
        Function with Args: or Returns: keywords in its description.

        Args:
            value: A string value.

        Returns:
            str: Processed value.
        """
        return value.upper()

    return process


class TestGetJsonSchema:
    # [CLASS DOC]
    # What: test group for get_json_schema covering full schema shape, error cases, type mapping, optional,
    #   enum, union, nested types and docstring-type handling.
    # Inputs: none (pytest collects it; tests pull fixtures by name or via the request fixture).
    # Outputs: n/a.
    # Exceptions: none raised directly.
    # Side effects: none.
    def test_get_json_schema_example(self):
        # [FUNCTION DOC]
        # What: checks the exact schema for a function with an int and an optional three-element tuple
        #   (prefixItems, nullable, required only x, return type null).
        # Inputs: self.
        # Outputs: None (pytest test).
        # Exceptions: AssertionError (or pytest failure) if the behavior regresses.
        # Side effects: none.
        def fn(x: int, y: tuple[str, str, float] | None = None) -> None:
            # [FUNCTION DOC]
            # HELPER: sample function used only as get_json_schema input; x int, y optional tuple of (str,
            #   str, float); docstring has no blank line before Args.
            # Inputs: x (int), y (tuple or None, default None).
            # Outputs: None.
            # Exceptions: none raised directly.
            # Side effects: none.
            """
            Test function
            Args:
                x: The first input
                y: The second input
            """
            pass

        schema = get_json_schema(fn)
        expected_schema = {
            "name": "fn",
            "description": "Test function",
            "parameters": {
                "type": "object",
                "properties": {
                    "x": {"type": "integer", "description": "The first input"},
                    "y": {
                        "type": "array",
                        "description": "The second input",
                        "nullable": True,
                        "prefixItems": [{"type": "string"}, {"type": "string"}, {"type": "number"}],
                    },
                },
                "required": ["x"],
            },
            "return": {"type": "null"},
        }
        assert schema["function"]["parameters"]["properties"]["y"] == expected_schema["parameters"]["properties"]["y"]
        assert schema["function"] == expected_schema

    @pytest.mark.parametrize(
        "fixture_name,should_fail",
        [
            ("valid_func", False),
            # ('no_docstring_func', True),
            # ('missing_arg_doc_func', True),
            ("bad_return_func", False),
        ],
    )
    def test_get_json_schema(self, request, fixture_name, should_fail):
        # [FUNCTION DOC]
        # What: for fixtures that should succeed, checks get_json_schema returns the function envelope with
        #   parameters.
        # Inputs: self, request (pytest request), fixture_name (str), should_fail (bool, unused).
        # Outputs: None (pytest test).
        # Exceptions: AssertionError (or pytest failure) if the behavior regresses.
        # Side effects: none.
        # KNOWN ISSUE: the should_fail parameter is never used; the parametrization keeps commented-out cases.
        func = request.getfixturevalue(fixture_name)
        schema = get_json_schema(func)
        assert schema["type"] == "function"
        assert "function" in schema
        assert "parameters" in schema["function"]

    @pytest.mark.parametrize(
        "fixture_name,should_fail",
        [
            # ('valid_func', False),
            ("no_docstring_func", True),
            ("missing_arg_doc_func", True),
            # ('bad_return_func', False),
        ],
    )
    def test_get_json_schema_raises(self, request, fixture_name, should_fail):
        # [FUNCTION DOC]
        # What: for fixtures without docstring or with an undocumented argument, checks
        #   DocstringParsingException is raised.
        # Inputs: self, request, fixture_name (str), should_fail (bool, unused).
        # Outputs: None (pytest test).
        # Exceptions: AssertionError (or pytest failure) if the behavior regresses.
        # Side effects: none.
        # KNOWN ISSUE: the should_fail parameter is never used.
        func = request.getfixturevalue(fixture_name)
        with pytest.raises(DocstringParsingException):
            get_json_schema(func)

    @pytest.mark.parametrize(
        "fixture_name,expected_properties",
        [
            ("valid_func", {"x": "integer", "y": "number"}),
            ("bad_return_func", {"x": "string"}),
        ],
    )
    def test_property_types(self, request, fixture_name, expected_properties):
        # [FUNCTION DOC]
        # What: checks each named property maps to the expected JSON type.
        # Inputs: self, request, fixture_name (str), expected_properties (dict name -> type string).
        # Outputs: None (pytest test).
        # Exceptions: AssertionError (or pytest failure) if the behavior regresses.
        # Side effects: none.
        """Test that property types are correctly mapped."""
        func = request.getfixturevalue(fixture_name)
        schema = get_json_schema(func)

        properties = schema["function"]["parameters"]["properties"]
        for prop_name, expected_type in expected_properties.items():
            assert properties[prop_name]["type"] == expected_type

    def test_schema_basic_structure(self, valid_func):
        # [FUNCTION DOC]
        # What: checks the full structure of the valid_func schema including required list and return
        #   description.
        # Inputs: self, valid_func (fixture).
        # Outputs: None (pytest test).
        # Exceptions: AssertionError (or pytest failure) if the behavior regresses.
        # Side effects: none.
        """Test that basic schema structure is correct."""
        schema = get_json_schema(valid_func)
        # Check schema type
        assert schema["type"] == "function"
        assert "function" in schema
        # Check function schema
        function_schema = schema["function"]
        assert function_schema["name"] == "multiply"
        assert "description" in function_schema
        assert function_schema["description"] == "Multiplies two numbers."
        # Check parameters schema
        assert "parameters" in function_schema
        params = function_schema["parameters"]
        assert params["type"] == "object"
        assert "properties" in params
        assert "required" in params
        assert set(params["required"]) == {"x", "y"}
        properties = params["properties"]
        assert properties["x"]["type"] == "integer"
        assert properties["y"]["type"] == "number"
        # Check return schema
        assert "return" in function_schema
        return_schema = function_schema["return"]
        assert return_schema["type"] == "number"
        assert return_schema["description"] == "Product of x and y."

    def test_complex_types(self, complex_types_func):
        # [FUNCTION DOC]
        # What: checks list -> array, dict -> object and tuple -> array with two integer prefixItems.
        # Inputs: self, complex_types_func (fixture).
        # Outputs: None (pytest test).
        # Exceptions: AssertionError (or pytest failure) if the behavior regresses.
        # Side effects: none.
        """Test schema generation for complex types."""
        schema = get_json_schema(complex_types_func)
        properties = schema["function"]["parameters"]["properties"]
        # Check list type
        assert properties["items"]["type"] == "array"
        # Check dict type
        assert properties["config"]["type"] == "object"
        # Check tuple type
        assert properties["point"]["type"] == "array"
        assert len(properties["point"]["prefixItems"]) == 2
        assert properties["point"]["prefixItems"][0]["type"] == "integer"
        assert properties["point"]["prefixItems"][1]["type"] == "integer"

    def test_optional_types(self, optional_types_func):
        # [FUNCTION DOC]
        # What: checks the optional argument is not required, is nullable and keeps integer type.
        # Inputs: self, optional_types_func (fixture).
        # Outputs: None (pytest test).
        # Exceptions: AssertionError (or pytest failure) if the behavior regresses.
        # Side effects: none.
        """Test schema generation for optional arguments."""
        schema = get_json_schema(optional_types_func)
        params = schema["function"]["parameters"]
        # Required argument should be in required list
        assert "required_arg" in params["required"]
        # Optional argument should not be in required list
        assert "optional_arg" not in params["required"]
        # Optional argument should be nullable
        assert params["properties"]["optional_arg"]["nullable"] is True
        assert params["properties"]["optional_arg"]["type"] == "integer"

    def test_enum_choices(self, enum_choices_func):
        # [FUNCTION DOC]
        # What: checks a choices suffix becomes an enum list.
        # Inputs: self, enum_choices_func (fixture).
        # Outputs: None (pytest test).
        # Exceptions: AssertionError (or pytest failure) if the behavior regresses.
        # Side effects: none.
        """Test schema generation for enum choices in docstring."""
        schema = get_json_schema(enum_choices_func)
        color_prop = schema["function"]["parameters"]["properties"]["color"]
        assert "enum" in color_prop
        assert color_prop["enum"] == ["red", "green", "blue"]

    def test_union_types(self, union_types_func):
        # [FUNCTION DOC]
        # What: checks a basic-type union argument becomes a two-element type list and a union return becomes
        #   'any'.
        # Inputs: self, union_types_func (fixture).
        # Outputs: None (pytest test).
        # Exceptions: AssertionError (or pytest failure) if the behavior regresses.
        # Side effects: none.
        """Test schema generation for union types."""
        schema = get_json_schema(union_types_func)
        value_prop = schema["function"]["parameters"]["properties"]["value"]
        return_prop = schema["function"]["return"]
        # Check union in parameter
        assert len(value_prop["type"]) == 2
        # Check union in return type: should be converted to "any"
        assert return_prop["type"] == "any"

    def test_nested_types(self, nested_types_func):
        # [FUNCTION DOC]
        # What: checks a list-of-dicts argument is typed as array.
        # Inputs: self, nested_types_func (fixture).
        # Outputs: None (pytest test).
        # Exceptions: AssertionError (or pytest failure) if the behavior regresses.
        # Side effects: none.
        """Test schema generation for nested complex types."""
        schema = get_json_schema(nested_types_func)
        data_prop = schema["function"]["parameters"]["properties"]["data"]
        assert data_prop["type"] == "array"

    def test_typed_docstring_parsing(self, typed_docstring_func):
        # [FUNCTION DOC]
        # What: checks hint types win over docstring types and that parenthesised types / return type prefix
        #   are stripped from descriptions.
        # Inputs: self, typed_docstring_func (fixture).
        # Outputs: None (pytest test).
        # Exceptions: AssertionError (or pytest failure) if the behavior regresses.
        # Side effects: none.
        """Test parsing of docstrings with type annotations."""
        schema = get_json_schema(typed_docstring_func)
        # Type hints should take precedence over docstring types
        assert schema["function"]["parameters"]["properties"]["x"]["type"] == "integer"
        assert schema["function"]["parameters"]["properties"]["y"]["type"] == "number"
        # Description should be extracted correctly
        assert (
            schema["function"]["parameters"]["properties"]["x"]["description"]
            == "An integer parameter with type in docstring."
        )
        assert (
            schema["function"]["parameters"]["properties"]["y"]["description"]
            == "A float parameter with type in docstring."
        )
        # Return type and description should be correct
        assert schema["function"]["return"]["type"] == "number"
        assert schema["function"]["return"]["description"] == "The calculated result."

    def test_mismatched_docstring_types(self, mismatched_types_func):
        # [FUNCTION DOC]
        # What: checks conflicting docstring types are ignored in favour of hints.
        # Inputs: self, mismatched_types_func (fixture).
        # Outputs: None (pytest test).
        # Exceptions: AssertionError (or pytest failure) if the behavior regresses.
        # Side effects: none.
        """Test that type hints take precedence over docstring types when they conflict."""
        schema = get_json_schema(mismatched_types_func)
        # Type hints should take precedence over docstring types
        assert schema["function"]["parameters"]["properties"]["value"]["type"] == "integer"
        # Return type from type hint should be used, not docstring
        assert schema["function"]["return"]["type"] == "string"

    def test_complex_docstring_types(self, complex_docstring_types_func):
        # [FUNCTION DOC]
        # What: checks nested generic hints produce object/array and that complex docstring types are stripped
        #   from descriptions.
        # Inputs: self, complex_docstring_types_func (fixture).
        # Outputs: None (pytest test).
        # Exceptions: AssertionError (or pytest failure) if the behavior regresses.
        # Side effects: none.
        """Test parsing of complex type annotations in docstrings."""
        schema = get_json_schema(complex_docstring_types_func)
        # Check that complex nested type is parsed correctly from type hints
        data_prop = schema["function"]["parameters"]["properties"]["data"]
        assert data_prop["type"] == "object"
        # Check return type
        return_prop = schema["function"]["return"]
        assert return_prop["type"] == "array"
        # Description should include the type information from docstring
        assert data_prop["description"] == "Nested structure with types."
        assert return_prop["description"] == "Processed results with types."

    @pytest.mark.parametrize(
        "fixture_name,expected_description",
        [
            ("typed_docstring_func", "An integer parameter with type in docstring."),
            ("complex_docstring_types_func", "Nested structure with types."),
        ],
    )
    def test_type_in_description_handling(self, request, fixture_name, expected_description):
        # [FUNCTION DOC]
        # What: checks the first parameter's description equals the expected cleaned text.
        # Inputs: self, request, fixture_name (str), expected_description (str).
        # Outputs: None (pytest test).
        # Exceptions: AssertionError (or pytest failure) if the behavior regresses.
        # Side effects: none.
        """Test that type information in docstrings is preserved in description."""
        func = request.getfixturevalue(fixture_name)
        schema = get_json_schema(func)
        # First parameter description should contain the expected text
        first_param_name = list(schema["function"]["parameters"]["properties"].keys())[0]
        assert schema["function"]["parameters"]["properties"][first_param_name]["description"] == expected_description

    def test_with_special_words_in_description_func(self, keywords_in_description_func):
        # [FUNCTION DOC]
        # What: checks inline 'Args:'/'Returns:' words in the description do not truncate it.
        # Inputs: self, keywords_in_description_func (fixture).
        # Outputs: None (pytest test).
        # Exceptions: AssertionError (or pytest failure) if the behavior regresses.
        # Side effects: none.
        schema = get_json_schema(keywords_in_description_func)
        assert schema["function"]["description"] == "Function with Args: or Returns: keywords in its description."


class TestGetCode:
    # [CLASS DOC]
    # What: test group for get_imports (despite the name, it does not test any get_code function).
    # Inputs: none.
    # Outputs: n/a.
    # Exceptions: none raised directly.
    # Side effects: none.
    @pytest.mark.parametrize(
        "code, expected",
        [
            (
                """
        import numpy
        import pandas
        """,
                ["numpy", "pandas"],
            ),
            # From imports
            (
                """
        from torch import nn
        from transformers import AutoModel
        """,
                ["torch", "transformers"],
            ),
            # Mixed case with nested imports
            (
                """
        import numpy as np
        from torch.nn import Linear
        import os.path
        """,
                ["numpy", "torch", "os"],
            ),
            # Try/except block (should be filtered)
            (
                """
        try:
            import torch
        except ImportError:
            pass
        import numpy
        """,
                ["numpy"],
            ),
            # Flash attention block (should be filtered)
            (
                """
        if is_flash_attn_2_available():
            from flash_attn import flash_attn_func
        import transformers
        """,
                ["transformers"],
            ),
            # Relative imports (should be excluded)
            (
                """
        from .utils import helper
        from ..models import transformer
        """,
                [],
            ),
        ],
    )
    def test_get_imports(self, code: str, expected: list[str]):
        # [FUNCTION DOC]
        # What: checks the detected package list (order-insensitive) for plain, from, aliased/dotted,
        #   try-guarded, flash-attn-guarded and relative imports.
        # Inputs: self, code (str), expected (list of str).
        # Outputs: None (pytest test).
        # Exceptions: AssertionError (or pytest failure) if the behavior regresses.
        # Side effects: none.
        assert sorted(get_imports(code)) == sorted(expected)
