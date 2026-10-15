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
# Pytest suite for smolagents.local_python_executor, the restricted AST-walking Python interpreter used by
#   CodeAgent. Covers evaluate_python_code semantics (assignments, calls, comprehensions, control flow,
#   classes, context managers, imports, f-strings, augmented assignment, deletion, operation and while-loop
#   limits), the low-level evaluators (evaluate_boolop, evaluate_condition, evaluate_delete,
#   evaluate_subscript), get_safe_module, PrintContainer, fix_final_answer_code, the timeout decorator,
#   check_import_authorized, LocalPythonExecutor behaviour and a large set of sandbox-escape (security)
#   regression tests.
# Inputs: Depends on pytest, numpy, pandas, unittest.mock.patch, smolagents.default_tools (BASE_PYTHON_TOOLS,
#   FinalAnswerTool) and the public/private names imported from smolagents.local_python_executor. Exports only
#   test functions/classes plus the add_two fake tool.
# Outputs: none (collected by pytest).
# Exceptions: none raised at import time beyond ImportError if dependencies are missing.
# Side effects: none at import. Some tests sleep (timeouts), spawn a thread, monkeypatch module globals, and a
#   few allowed-path security cases actually run os.system or a harmless shell subprocess.

import ast
import time
import types
from contextlib import nullcontext as does_not_raise
from textwrap import dedent
from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest

from smolagents.default_tools import BASE_PYTHON_TOOLS, FinalAnswerTool
from smolagents.local_python_executor import (
    DANGEROUS_FUNCTIONS,
    DANGEROUS_MODULES,
    ExecutionTimeoutError,
    InterpreterError,
    LocalPythonExecutor,
    PrintContainer,
    check_import_authorized,
    evaluate_boolop,
    evaluate_condition,
    evaluate_delete,
    evaluate_python_code,
    evaluate_subscript,
    fix_final_answer_code,
    get_safe_module,
    timeout,
)


# Fake function we will use as tool
def add_two(x):
    # [FUNCTION DOC]
    # HELPER: Fake tool passed to the interpreter as a static tool; adds 2 to its argument.
    # Inputs: x (number).
    # Outputs: x plus 2.
    # Exceptions: TypeError propagates if x is not addable to an int.
    # Side effects: none.
    return x + 2


class TestEvaluatePythonCode:
    # [CLASS DOC]
    # Test group for evaluate_python_code: runs small code strings through the interpreter with given static
    #   tools and a state dict, then checks the returned value (the value of the last evaluated statement),
    #   the mutated state (including the _operations_count counter and _print_outputs) and raised
    #   InterpreterErrors.
    # Inputs: no constructor args (pytest instantiates it); no attributes.
    # Outputs: n/a.
    # Exceptions: none raised directly.
    # Side effects: none.
    def assertDictEqualNoPrint(self, dict1, dict2):
        # [FUNCTION DOC]
        # HELPER: Asserts two state dicts are equal after dropping the _print_outputs key from both.
        # Inputs: self (the test class instance); dict1 (dict), dict2 (dict).
        # Outputs: None.
        # Exceptions: AssertionError if the filtered dicts differ.
        # Side effects: none.
        assert {k: v for k, v in dict1.items() if k != "_print_outputs"} == {
            k: v for k, v in dict2.items() if k != "_print_outputs"
        }

    def test_evaluate_assign(self):
        # [FUNCTION DOC]
        # Checks simple assignment: assigning a literal returns it and stores it with an operation counter of
        #   2; assigning from an existing state name copies the value; for two semicolon-separated assignments
        #   the result is the last assigned value (None).
        # Inputs: self (the test class instance).
        # Outputs: None.
        # Exceptions: AssertionError (pytest failure) if any check fails; none raised directly otherwise.
        # Side effects: none (only local interpreter state dicts created inside the test are mutated).
        code = "x = 3"
        state = {}
        result, _ = evaluate_python_code(code, {}, state=state)
        assert result == 3
        self.assertDictEqualNoPrint(state, {"x": 3, "_operations_count": {"counter": 2}})

        code = "x = y"
        state = {"y": 5}
        result, _ = evaluate_python_code(code, {}, state=state)
        # evaluate returns the value of the last assignment.
        assert result == 5
        self.assertDictEqualNoPrint(state, {"x": 5, "y": 5, "_operations_count": {"counter": 2}})

        code = "a=1;b=None"
        result, _ = evaluate_python_code(code, {}, state={})
        # evaluate returns the value of the last assignment.
        assert result is None

    def test_assignment_cannot_overwrite_tool(self):
        # [FUNCTION DOC]
        # Checks that assigning to a name that is a static tool (print) raises InterpreterError mentioning
        #   that it would erase the existing tool.
        # Inputs: self (the test class instance).
        # Outputs: None.
        # Exceptions: AssertionError (pytest failure) if any check fails; none raised directly otherwise.
        # Side effects: none (only local interpreter state dicts created inside the test are mutated).
        code = "print = '3'"
        with pytest.raises(InterpreterError) as e:
            evaluate_python_code(code, {"print": print}, state={})
        assert "Cannot assign to name 'print': doing this would erase the existing tool!" in str(e)

    def test_subscript_call(self):
        # [FUNCTION DOC]
        # Checks calling functions fetched by indexing a list of interpreter-defined functions; verifies both
        #   results in state (8 and 64) and that the last value (64) is returned.
        # Inputs: self (the test class instance).
        # Outputs: None.
        # Exceptions: AssertionError (pytest failure) if any check fails; none raised directly otherwise.
        # Side effects: none (only local interpreter state dicts created inside the test are mutated).
        code = """def foo(x,y):return x*y\n\ndef boo(y):\n\treturn y**3\nfun = [foo, boo]\nresult_foo = fun[0](4,2)\nresult_boo = fun[1](4)"""
        state = {}
        result, _ = evaluate_python_code(code, BASE_PYTHON_TOOLS, state=state)
        assert result == 64
        assert state["result_foo"] == 8
        assert state["result_boo"] == 64

    def test_evaluate_call(self):
        # [FUNCTION DOC]
        # Checks calling a static tool (add_two) on a state variable returns 5 and updates state with counter
        #   3; then checks that the same code without the tool raises InterpreterError 'Forbidden function
        #   evaluation'.
        # Inputs: self (the test class instance).
        # Outputs: None.
        # Exceptions: AssertionError (pytest failure) if any check fails; none raised directly otherwise.
        # Side effects: none (only local interpreter state dicts created inside the test are mutated).
        code = "y = add_two(x)"
        state = {"x": 3}
        result, _ = evaluate_python_code(code, {"add_two": add_two}, state=state)
        assert result == 5
        self.assertDictEqualNoPrint(state, {"x": 3, "y": 5, "_operations_count": {"counter": 3}})

        # Should not work without the tool
        with pytest.raises(InterpreterError, match="Forbidden function evaluation: 'add_two'"):
            evaluate_python_code(code, {}, state=state)

    @pytest.mark.parametrize(
        "code, expected_result",
        [
            # Basic **kwargs unpacking
            (
                """
def test_func(a, b=10, **kwargs):
    return a + b + sum(kwargs.values())

kwargs_dict = {'x': 5, 'y': 15}
test_func(1, **kwargs_dict)
""",
                31,  # 1 + 10 + 5 + 15
            ),
            # **kwargs with regular kwargs
            (
                """
def test_func(a, **kwargs):
    return a + sum(kwargs.values())

kwargs_dict = {'x': 5, 'y': 15}
test_func(1, b=20, **kwargs_dict)
""",
                41,  # 1 + 20 + 5 + 15
            ),
            # Multiple **kwargs unpacking
            (
                """
def test_func(**kwargs):
    return sum(kwargs.values())

dict1 = {'a': 1, 'b': 2}
dict2 = {'c': 3, 'd': 4}
test_func(**dict1, **dict2)
""",
                10,  # 1 + 2 + 3 + 4
            ),
            # **kwargs with positional args
            (
                """
def test_func(x, y, **kwargs):
    return x * y + sum(kwargs.values())

params = {'factor': 2, 'offset': 5}
test_func(3, 4, **params)
""",
                19,  # 3 * 4 + 2 + 5
            ),
            # Empty **kwargs dict
            (
                """
def test_func(a, **kwargs):
    return a + len(kwargs)

empty_dict = {}
test_func(10, **empty_dict)
""",
                10,  # 10 + 0
            ),
        ],
    )
    def test_evaluate_call_starred_kwargs(self, code, expected_result):
        # [FUNCTION DOC]
        # Parametrized: checks double-star keyword unpacking into interpreter-defined functions (with
        #   defaults, extra keyword args, multiple unpackings, positional args and an empty dict) yields the
        #   expected sum.
        # Inputs: self (the test class instance); code (str, program to run), expected_result (int).
        # Outputs: None.
        # Exceptions: AssertionError (pytest failure) if any check fails; none raised directly otherwise.
        # Side effects: none (only local interpreter state dicts created inside the test are mutated).
        result, _ = evaluate_python_code(code, {"sum": sum, "len": len}, state={})
        assert result == expected_result

    @pytest.mark.parametrize(
        "code, expected_error_message",
        [
            # Non-dict value in **kwargs
            (
                """
def test_func(**kwargs):
    return sum(kwargs.values())

not_a_dict = [1, 2, 3]
test_func(**not_a_dict)
""",
                "Cannot unpack non-dict value in **kwargs: list",
            ),
            # **kwargs with non-dict variable
            (
                """
def test_func(**kwargs):
    return kwargs

test_func(**42)
""",
                "Cannot unpack non-dict value in **kwargs: int",
            ),
            # **kwargs with None
            (
                """
def test_func(**kwargs):
    return kwargs

test_func(**None)
""",
                "Cannot unpack non-dict value in **kwargs: NoneType",
            ),
        ],
    )
    def test_evaluate_call_starred_kwargs_errors(self, code, expected_error_message):
        # [FUNCTION DOC]
        # Parametrized: checks double-star unpacking of a non-dict (list, int, None) raises InterpreterError
        #   whose message names the offending type.
        # Inputs: self (the test class instance); code (str), expected_error_message (str substring).
        # Outputs: None.
        # Exceptions: AssertionError (pytest failure) if any check fails; none raised directly otherwise.
        # Side effects: none (only local interpreter state dicts created inside the test are mutated).
        """Test that **kwargs unpacking raises appropriate errors for non-dict values."""
        with pytest.raises(InterpreterError) as exception_info:
            evaluate_python_code(code, {"sum": sum}, state={})
        assert expected_error_message in str(exception_info.value)

    def test_evaluate_class_def(self):
        # [FUNCTION DOC]
        # Checks defining a class with a docstring, constructor and method, instantiating it and calling the
        #   method returns 42 and that the instance's class docstring is preserved.
        # Inputs: self (the test class instance).
        # Outputs: None.
        # Exceptions: AssertionError (pytest failure) if any check fails; none raised directly otherwise.
        # Side effects: none (only local interpreter state dicts created inside the test are mutated).
        code = dedent('''\
            class MyClass:
                """A class with a value."""

                def __init__(self, value):
                    self.value = value

                def get_value(self):
                    return self.value

            instance = MyClass(42)
            result = instance.get_value()
        ''')
        state = {}
        result, _ = evaluate_python_code(code, {}, state=state)
        assert result == 42
        assert state["instance"].__doc__ == "A class with a value."

    def test_evaluate_class_def_with_assign_attribute_target(self):
        # [FUNCTION DOC]
        # Checks that, inside a class body, assignments to attributes of a class-level object (attribute
        #   targets) are applied; verifies the resulting tuple and that the class stored in state is a real
        #   type.
        # Inputs: self (the test class instance).
        # Outputs: None.
        # Exceptions: AssertionError (pytest failure) if any check fails; none raised directly otherwise.
        # Side effects: none (only local interpreter state dicts created inside the test are mutated).
        """
        Test evaluate_class_def function when stmt is an instance of ast.Assign with ast.Attribute target.
        """
        code = dedent("""
        class TestSubClass:
            attr1 = 1
        class TestClass:
            data = TestSubClass()
            data.attr1 = "value1"
            data.attr2 = "value2"
        result = (TestClass.data.attr1, TestClass.data.attr2)
        """)

        state = {}
        result, _ = evaluate_python_code(code, BASE_PYTHON_TOOLS, state=state)

        assert result == ("value1", "value2")
        assert isinstance(state["TestClass"], type)
        assert state["TestClass"].data.attr1 == "value1"
        assert state["TestClass"].data.attr2 == "value2"

    def test_evaluate_constant(self):
        # [FUNCTION DOC]
        # Checks a constant assignment returns 3 and leaves state with x and an operation counter of 2
        #   (duplicates test_evaluate_assign's first case).
        # Inputs: self (the test class instance).
        # Outputs: None.
        # Exceptions: AssertionError (pytest failure) if any check fails; none raised directly otherwise.
        # Side effects: none (only local interpreter state dicts created inside the test are mutated).
        code = "x = 3"
        state = {}
        result, _ = evaluate_python_code(code, {}, state=state)
        assert result == 3
        self.assertDictEqualNoPrint(state, {"x": 3, "_operations_count": {"counter": 2}})

    def test_evaluate_dict(self):
        # [FUNCTION DOC]
        # Checks a dict literal built from a state variable and a tool call evaluates correctly; counter
        #   becomes 7.
        # Inputs: self (the test class instance).
        # Outputs: None.
        # Exceptions: AssertionError (pytest failure) if any check fails; none raised directly otherwise.
        # Side effects: none (only local interpreter state dicts created inside the test are mutated).
        code = "test_dict = {'x': x, 'y': add_two(x)}"
        state = {"x": 3}
        result, _ = evaluate_python_code(code, {"add_two": add_two}, state=state)
        assert result == {"x": 3, "y": 5}
        self.assertDictEqualNoPrint(
            state, {"x": 3, "test_dict": {"x": 3, "y": 5}, "_operations_count": {"counter": 7}}
        )

    def test_evaluate_expression(self):
        # [FUNCTION DOC]
        # Checks two newline-separated assignments: result is the last value (5), state holds both, counter 4.
        # Inputs: self (the test class instance).
        # Outputs: None.
        # Exceptions: AssertionError (pytest failure) if any check fails; none raised directly otherwise.
        # Side effects: none (only local interpreter state dicts created inside the test are mutated).
        code = "x = 3\ny = 5"
        state = {}
        result, _ = evaluate_python_code(code, {}, state=state)
        # evaluate returns the value of the last assignment.
        assert result == 5
        self.assertDictEqualNoPrint(state, {"x": 3, "y": 5, "_operations_count": {"counter": 4}})

    def test_evaluate_f_string(self):
        # [FUNCTION DOC]
        # Checks a simple f-string interpolation of a state variable; counter 6.
        # Inputs: self (the test class instance).
        # Outputs: None.
        # Exceptions: AssertionError (pytest failure) if any check fails; none raised directly otherwise.
        # Side effects: none (only local interpreter state dicts created inside the test are mutated).
        code = "text = f'This is x: {x}.'"
        state = {"x": 3}
        result, _ = evaluate_python_code(code, {}, state=state)
        # evaluate returns the value of the last assignment.
        assert result == "This is x: 3."
        self.assertDictEqualNoPrint(state, {"x": 3, "text": "This is x: 3.", "_operations_count": {"counter": 6}})

    def test_evaluate_f_string_with_format(self):
        # [FUNCTION DOC]
        # Checks an f-string with a fixed float format spec rounds to two decimals; counter 8.
        # Inputs: self (the test class instance).
        # Outputs: None.
        # Exceptions: AssertionError (pytest failure) if any check fails; none raised directly otherwise.
        # Side effects: none (only local interpreter state dicts created inside the test are mutated).
        code = "text = f'This is x: {x:.2f}.'"
        state = {"x": 3.336}
        result, _ = evaluate_python_code(code, {}, state=state)
        assert result == "This is x: 3.34."
        self.assertDictEqualNoPrint(
            state, {"x": 3.336, "text": "This is x: 3.34.", "_operations_count": {"counter": 8}}
        )

    def test_evaluate_f_string_with_complex_format(self):
        # [FUNCTION DOC]
        # Checks an f-string whose format spec is itself built from nested width/precision fields; counter 14.
        # Inputs: self (the test class instance).
        # Outputs: None.
        # Exceptions: AssertionError (pytest failure) if any check fails; none raised directly otherwise.
        # Side effects: none (only local interpreter state dicts created inside the test are mutated).
        code = "text = f'This is x: {x:>{width}.{precision}f}.'"
        state = {"x": 3.336, "width": 10, "precision": 2}
        result, _ = evaluate_python_code(code, {}, state=state)
        assert result == "This is x:       3.34."
        self.assertDictEqualNoPrint(
            state,
            {
                "x": 3.336,
                "width": 10,
                "precision": 2,
                "text": "This is x:       3.34.",
                "_operations_count": {"counter": 14},
            },
        )

    def test_evaluate_if(self):
        # [FUNCTION DOC]
        # Checks an if/else assignment takes the right branch for two different input values; counter 6 each
        #   time.
        # Inputs: self (the test class instance).
        # Outputs: None.
        # Exceptions: AssertionError (pytest failure) if any check fails; none raised directly otherwise.
        # Side effects: none (only local interpreter state dicts created inside the test are mutated).
        code = "if x <= 3:\n    y = 2\nelse:\n    y = 5"
        state = {"x": 3}
        result, _ = evaluate_python_code(code, {}, state=state)
        # evaluate returns the value of the last assignment.
        assert result == 2
        self.assertDictEqualNoPrint(state, {"x": 3, "y": 2, "_operations_count": {"counter": 6}})

        state = {"x": 8}
        result, _ = evaluate_python_code(code, {}, state=state)
        # evaluate returns the value of the last assignment.
        assert result == 5
        self.assertDictEqualNoPrint(state, {"x": 8, "y": 5, "_operations_count": {"counter": 6}})

    def test_evaluate_list(self):
        # [FUNCTION DOC]
        # Checks a list literal with a tool call evaluates to [3, 5]; counter 5.
        # Inputs: self (the test class instance).
        # Outputs: None.
        # Exceptions: AssertionError (pytest failure) if any check fails; none raised directly otherwise.
        # Side effects: none (only local interpreter state dicts created inside the test are mutated).
        code = "test_list = [x, add_two(x)]"
        state = {"x": 3}
        result, _ = evaluate_python_code(code, {"add_two": add_two}, state=state)
        assert result == [3, 5]
        self.assertDictEqualNoPrint(state, {"x": 3, "test_list": [3, 5], "_operations_count": {"counter": 5}})

    def test_evaluate_name(self):
        # [FUNCTION DOC]
        # Checks assigning one state name to another copies the value; counter 2.
        # Inputs: self (the test class instance).
        # Outputs: None.
        # Exceptions: AssertionError (pytest failure) if any check fails; none raised directly otherwise.
        # Side effects: none (only local interpreter state dicts created inside the test are mutated).
        code = "y = x"
        state = {"x": 3}
        result, _ = evaluate_python_code(code, {}, state=state)
        assert result == 3
        self.assertDictEqualNoPrint(state, {"x": 3, "y": 3, "_operations_count": {"counter": 2}})

    def test_evaluate_subscript(self):
        # [FUNCTION DOC]
        # Checks list and dict subscripting after building them (counters 9 and 11) and subscript assignment
        #   into a dict combined with round and division.
        # Inputs: self (the test class instance).
        # Outputs: None.
        # Exceptions: AssertionError (pytest failure) if any check fails; none raised directly otherwise.
        # Side effects: none (only local interpreter state dicts created inside the test are mutated).
        code = "test_list = [x, add_two(x)]\ntest_list[1]"
        state = {"x": 3}
        result, _ = evaluate_python_code(code, {"add_two": add_two}, state=state)
        assert result == 5
        self.assertDictEqualNoPrint(state, {"x": 3, "test_list": [3, 5], "_operations_count": {"counter": 9}})

        code = "test_dict = {'x': x, 'y': add_two(x)}\ntest_dict['y']"
        state = {"x": 3}
        result, _ = evaluate_python_code(code, {"add_two": add_two}, state=state)
        assert result == 5
        self.assertDictEqualNoPrint(
            state, {"x": 3, "test_dict": {"x": 3, "y": 5}, "_operations_count": {"counter": 11}}
        )

        code = "vendor = {'revenue': 31000, 'rent': 50312}; vendor['ratio'] = round(vendor['revenue'] / vendor['rent'], 2)"
        state = {}
        evaluate_python_code(code, {"min": min, "print": print, "round": round}, state=state)
        assert state["vendor"] == {"revenue": 31000, "rent": 50312, "ratio": 0.62}

    def test_subscript_string_with_string_index_raises_appropriate_error(self):
        # [FUNCTION DOC]
        # Checks that iterating over a string and subscripting a character with a string key raises
        #   InterpreterError.
        # Inputs: self (the test class instance).
        # Outputs: None.
        # Exceptions: AssertionError (pytest failure) if any check fails; none raised directly otherwise.
        # Side effects: none (only local interpreter state dicts created inside the test are mutated).
        # KNOWN ISSUE: The message assertion is inside the pytest.raises block after the raising call, so it
        #   never runs; it also tests membership in the ExceptionInfo object rather than its string. Only the
        #   exception type is effectively checked.
        code = """
search_results = "[{'title': 'Paris, Ville de Paris, France Weather Forecast | AccuWeather', 'href': 'https://www.accuweather.com/en/fr/paris/623/weather-forecast/623', 'body': 'Get the latest weather forecast for Paris, Ville de Paris, France , including hourly, daily, and 10-day outlooks. AccuWeather provides you with reliable and accurate information on temperature ...'}]"
for result in search_results:
    if 'current' in result['title'].lower() or 'temperature' in result['title'].lower():
        current_weather_url = result['href']
        print(current_weather_url)
        break"""
        with pytest.raises(InterpreterError) as e:
            evaluate_python_code(code, BASE_PYTHON_TOOLS, state={})
            assert "You're trying to subscript a string with a string index" in e

    def test_evaluate_for(self):
        # [FUNCTION DOC]
        # Checks a for loop over range leaves the last values in state; counter 11.
        # Inputs: self (the test class instance).
        # Outputs: None.
        # Exceptions: AssertionError (pytest failure) if any check fails; none raised directly otherwise.
        # Side effects: none (only local interpreter state dicts created inside the test are mutated).
        code = "x = 0\nfor i in range(3):\n    x = i"
        state = {}
        result, _ = evaluate_python_code(code, {"range": range}, state=state)
        assert result == 2
        self.assertDictEqualNoPrint(state, {"x": 2, "i": 2, "_operations_count": {"counter": 11}})

    def test_evaluate_binop(self):
        # [FUNCTION DOC]
        # Checks addition of two state names returns 9; counter 4.
        # Inputs: self (the test class instance).
        # Outputs: None.
        # Exceptions: AssertionError (pytest failure) if any check fails; none raised directly otherwise.
        # Side effects: none (only local interpreter state dicts created inside the test are mutated).
        code = "y + x"
        state = {"x": 3, "y": 6}
        result, _ = evaluate_python_code(code, {}, state=state)
        assert result == 9
        self.assertDictEqualNoPrint(state, {"x": 3, "y": 6, "_operations_count": {"counter": 4}})

    def test_recursive_function(self):
        # [FUNCTION DOC]
        # Checks an interpreter-defined recursive Fibonacci function returns 8 for 6.
        # Inputs: self (the test class instance).
        # Outputs: None.
        # Exceptions: AssertionError (pytest failure) if any check fails; none raised directly otherwise.
        # Side effects: none (only local interpreter state dicts created inside the test are mutated).
        code = """
def recur_fibo(n):
    if n <= 1:
        return n
    else:
        return(recur_fibo(n-1) + recur_fibo(n-2))
recur_fibo(6)"""
        result, _ = evaluate_python_code(code, {}, state={})
        assert result == 8

    def test_max_operations(self):
        # [FUNCTION DOC]
        # Patches MAX_OPERATIONS to 100 and checks that nested loops calling a function exceed it, raising
        #   InterpreterError 'Reached the max number of operations' (i.e. the counter is shared into function
        #   calls).
        # Inputs: self (the test class instance).
        # Outputs: None.
        # Exceptions: AssertionError (pytest failure) if any check fails; none raised directly otherwise.
        # Side effects: temporarily patches the module-level MAX_OPERATIONS constant (restored by patch).
        # Check that operation counter is not reset in functions
        code = dedent(
            """
            def func(a):
                for j in range(10):
                    a += j
                return a

            for i in range(5):
                func(i)
            """
        )
        with patch("smolagents.local_python_executor.MAX_OPERATIONS", 100):
            with pytest.raises(InterpreterError) as exception_info:
                evaluate_python_code(code, {"range": range}, state={})
        assert "Reached the max number of operations" in str(exception_info.value)

    def test_operations_count(self):
        # [FUNCTION DOC]
        # Checks defining and calling a trivial function yields an operation counter of exactly 5 in state.
        # Inputs: self (the test class instance).
        # Outputs: None.
        # Exceptions: AssertionError (pytest failure) if any check fails; none raised directly otherwise.
        # Side effects: none (only local interpreter state dicts created inside the test are mutated).
        # Check that operation counter is not reset in functions
        code = dedent(
            """
            def func():
                return 0

            func()
            """
        )
        state = {}
        evaluate_python_code(code, {"range": range}, state=state)
        assert state["_operations_count"]["counter"] == 5

    def test_evaluate_string_methods(self):
        # [FUNCTION DOC]
        # Checks chained str method calls (replace then split) on a literal.
        # Inputs: self (the test class instance).
        # Outputs: None.
        # Exceptions: AssertionError (pytest failure) if any check fails; none raised directly otherwise.
        # Side effects: none (only local interpreter state dicts created inside the test are mutated).
        code = "'hello'.replace('h', 'o').split('e')"
        result, _ = evaluate_python_code(code, {}, state={})
        assert result == ["o", "llo"]

    def test_evaluate_slicing(self):
        # [FUNCTION DOC]
        # Checks chained slicing including a negative step on a string literal.
        # Inputs: self (the test class instance).
        # Outputs: None.
        # Exceptions: AssertionError (pytest failure) if any check fails; none raised directly otherwise.
        # Side effects: none (only local interpreter state dicts created inside the test are mutated).
        code = "'hello'[1:3][::-1]"
        result, _ = evaluate_python_code(code, {}, state={})
        assert result == "le"

    def test_access_attributes(self):
        # [FUNCTION DOC]
        # Checks attribute access on a host class placed into state returns its value.
        # Inputs: self (the test class instance).
        # Outputs: None.
        # Exceptions: AssertionError (pytest failure) if any check fails; none raised directly otherwise.
        # Side effects: none (only local interpreter state dicts created inside the test are mutated).
        class A:
            # [CLASS DOC]
            # HELPER: Local host-side class with one class attribute (attr equal to 2) injected into
            #   interpreter state.
            # Inputs: no constructor args; attribute attr (int 2).
            # Outputs: n/a.
            # Exceptions: none raised directly.
            # Side effects: none.
            attr = 2

        code = "A.attr"
        result, _ = evaluate_python_code(code, {}, state={"A": A})
        assert result == 2

    def test_list_comprehension(self):
        # [FUNCTION DOC]
        # Checks a filtered list comprehension inside a str join call.
        # Inputs: self (the test class instance).
        # Outputs: None.
        # Exceptions: AssertionError (pytest failure) if any check fails; none raised directly otherwise.
        # Side effects: none (only local interpreter state dicts created inside the test are mutated).
        code = "sentence = 'THESEAGULL43'\nmeaningful_sentence = '-'.join([char.lower() for char in sentence if char.isalpha()])"
        result, _ = evaluate_python_code(code, {}, state={})
        assert result == "t-h-e-s-e-a-g-u-l-l"

    def test_string_indexing(self):
        # [FUNCTION DOC]
        # Checks nested for loops with string indexing and augmented string concatenation build 'THESEAGULL'.
        # Inputs: self (the test class instance).
        # Outputs: None.
        # Exceptions: AssertionError (pytest failure) if any check fails; none raised directly otherwise.
        # Side effects: none (only local interpreter state dicts created inside the test are mutated).
        code = """text_block = [
    "THESE",
    "AGULL"
]
sentence = ""
for block in text_block:
    for col in range(len(text_block[0])):
        sentence += block[col]
        """
        result, _ = evaluate_python_code(code, {"len": len, "range": range}, state={})
        assert result == "THESEAGULL"

    def test_tuples(self):
        # [FUNCTION DOC]
        # Checks tuple indexing, tuple-swap assignment with subscript targets, and a generator-in-sum ISBN
        #   check-digit program run with print as a tool.
        # Inputs: self (the test class instance).
        # Outputs: None.
        # Exceptions: AssertionError (pytest failure) if any check fails; none raised directly otherwise.
        # Side effects: prints the computed check digits to stdout (print tool is the real print).
        # KNOWN ISSUE: Only the first snippet is asserted; the swap and ISBN snippets only check that no
        #   exception occurs.
        code = "x = (1, 2, 3)\nx[1]"
        result, _ = evaluate_python_code(code, {}, state={})
        assert result == 2

        code = """
digits, i = [1, 2, 3], 1
digits[i], digits[i + 1] = digits[i + 1], digits[i]"""
        evaluate_python_code(code, {"range": range, "print": print, "int": int}, {})

        code = """
def calculate_isbn_10_check_digit(number):
    total = sum((10 - i) * int(digit) for i, digit in enumerate(number))
    remainder = total % 11
    check_digit = 11 - remainder
    if check_digit == 10:
        return 'X'
    elif check_digit == 11:
        return '0'
    else:
        return str(check_digit)

# Given 9-digit numbers
numbers = [
    "478225952",
    "643485613",
    "739394228",
    "291726859",
    "875262394",
    "542617795",
    "031810713",
    "957007669",
    "871467426"
]

# Calculate check digits for each number
check_digits = [calculate_isbn_10_check_digit(number) for number in numbers]
print(check_digits)
"""
        state = {}
        evaluate_python_code(
            code,
            {
                "range": range,
                "print": print,
                "sum": sum,
                "enumerate": enumerate,
                "int": int,
                "str": str,
            },
            state,
        )

    def test_dictcomp(self):
        # [FUNCTION DOC]
        # Checks dict comprehensions: a simple one, one with an if filter over dict items, and one with a
        #   nested tuple target.
        # Inputs: self (the test class instance).
        # Outputs: None.
        # Exceptions: AssertionError (pytest failure) if any check fails; none raised directly otherwise.
        # Side effects: none (only local interpreter state dicts created inside the test are mutated).
        code = "x = {i: i**2 for i in range(3)}"
        result, _ = evaluate_python_code(code, {"range": range}, state={})
        assert result == {0: 0, 1: 1, 2: 4}

        code = "{num: name for num, name in {101: 'a', 102: 'b'}.items() if name not in ['a']}"
        result, _ = evaluate_python_code(code, {"print": print}, state={}, authorized_imports=["pandas"])
        assert result == {102: "b"}

        code = """
shifts = {'A': ('6:45', '8:00'), 'B': ('10:00', '11:45')}
shift_minutes = {worker: ('a', 'b') for worker, (start, end) in shifts.items()}
"""
        result, _ = evaluate_python_code(code, {}, state={})
        assert result == {"A": ("a", "b"), "B": ("a", "b")}

    def test_dictcomp_nested(self):
        # [FUNCTION DOC]
        # Checks a dict comprehension with two for clauses produces the cartesian keys.
        # Inputs: self (the test class instance).
        # Outputs: None.
        # Exceptions: AssertionError (pytest failure) if any check fails; none raised directly otherwise.
        # Side effects: none (only local interpreter state dicts created inside the test are mutated).
        code = """
simple_map = {
    (x, y): f"key_{x}_{y}"
    for x in [1, 2]
    for y in ['a', 'b']
}
"""
        result, _ = evaluate_python_code(code, {}, state={})
        assert result == {(1, "a"): "key_1_a", (1, "b"): "key_1_b", (2, "a"): "key_2_a", (2, "b"): "key_2_b"}

    def test_listcomp(self):
        # [FUNCTION DOC]
        # Checks a simple list comprehension over range.
        # Inputs: self (the test class instance).
        # Outputs: None.
        # Exceptions: AssertionError (pytest failure) if any check fails; none raised directly otherwise.
        # Side effects: none (only local interpreter state dicts created inside the test are mutated).
        code = "x = [i for i in range(3)]"
        result, _ = evaluate_python_code(code, {"range": range}, state={})
        assert result == [0, 1, 2]

    def test_listcomp_nested(self):
        # [FUNCTION DOC]
        # Checks a list comprehension with two for clauses keeps order and duplicates.
        # Inputs: self (the test class instance).
        # Outputs: None.
        # Exceptions: AssertionError (pytest failure) if any check fails; none raised directly otherwise.
        # Side effects: none (only local interpreter state dicts created inside the test are mutated).
        code = """
simple_list = [
    (x, y)
    for x in [1, 2, 1]
    for y in ['a', 'b']
]
"""
        result, _ = evaluate_python_code(code, {}, state={})
        assert result == [(1, "a"), (1, "b"), (2, "a"), (2, "b"), (1, "a"), (1, "b")]

    def test_setcomp(self):
        # [FUNCTION DOC]
        # Checks a set comprehension over a list of dicts.
        # Inputs: self (the test class instance).
        # Outputs: None.
        # Exceptions: AssertionError (pytest failure) if any check fails; none raised directly otherwise.
        # Side effects: none (only local interpreter state dicts created inside the test are mutated).
        code = "batman_times = {entry['time'] for entry in [{'time': 10}, {'time': 19}, {'time': 20}]}"
        result, _ = evaluate_python_code(code, {}, state={})
        assert result == {10, 19, 20}

    def test_setcomp_nested(self):
        # [FUNCTION DOC]
        # Checks a set comprehension with two for clauses deduplicates results.
        # Inputs: self (the test class instance).
        # Outputs: None.
        # Exceptions: AssertionError (pytest failure) if any check fails; none raised directly otherwise.
        # Side effects: none (only local interpreter state dicts created inside the test are mutated).
        code = """
simple_set = {
    (x, y)
    for x in [1, 2, 1]
    for y in ['a', 'b']
}
"""
        result, _ = evaluate_python_code(code, {}, state={})
        assert result == {(1, "a"), (1, "b"), (2, "a"), (2, "b")}

    def test_generatorexp(self):
        # [FUNCTION DOC]
        # Checks a generator expression evaluates to a real generator object, not a list.
        # Inputs: self (the test class instance).
        # Outputs: None.
        # Exceptions: AssertionError (pytest failure) if any check fails; none raised directly otherwise.
        # Side effects: none (only local interpreter state dicts created inside the test are mutated).
        code = "x = (i for i in range(3))"
        result, _ = evaluate_python_code(code, {"range": range}, state={})
        # assert not isinstance(result, list)
        assert isinstance(result, types.GeneratorType)
        assert list(result) == [0, 1, 2]

    def test_generatorexp_with_infinite_sequence(self):
        # [FUNCTION DOC]
        # Checks a generator expression over an infinite itertools.count is lazy: taking the first three even
        #   numbers with next works and the stored object is a generator.
        # Inputs: self (the test class instance).
        # Outputs: None.
        # Exceptions: AssertionError (pytest failure) if any check fails; none raised directly otherwise.
        # Side effects: none (only local interpreter state dicts created inside the test are mutated).
        """Test that generator expressions handle infinite sequences correctly without hanging."""
        code = dedent(
            """\
            import itertools

            def infinite_counter():
                return itertools.count()

            # Create a generator expression that filters an infinite sequence
            even_numbers = (x for x in infinite_counter() if x % 2 == 0)

            # Get just the first 3 even numbers
            first_three = []
            gen_iter = iter(even_numbers)
            for _ in range(3):
                first_three.append(next(gen_iter))

            result = first_three
            """
        )

        state = {}
        result, _ = evaluate_python_code(code, {"int": int, "iter": iter, "next": next, "range": range}, state=state)

        # Verify we got the expected values
        assert result == [0, 2, 4]

        # Verify it's actually a generator
        even_numbers = state["even_numbers"]
        assert isinstance(even_numbers, types.GeneratorType)

        # If this were a list, the code would hang indefinitely trying to
        # evaluate the entire infinite sequence upfront

    def test_break(self):
        # [FUNCTION DOC]
        # Checks break exits a for loop, leaving the loop variable at 5.
        # Inputs: self (the test class instance).
        # Outputs: None.
        # Exceptions: AssertionError (pytest failure) if any check fails; none raised directly otherwise.
        # Side effects: none (only local interpreter state dicts created inside the test are mutated).
        code = "for i in range(10):\n    if i == 5:\n        break\ni"
        result, _ = evaluate_python_code(code, {"range": range}, state={})
        assert result == 5

    def test_pass(self):
        # [FUNCTION DOC]
        # Checks pass inside a loop is a no-op, the loop runs to completion (last value 9).
        # Inputs: self (the test class instance).
        # Outputs: None.
        # Exceptions: AssertionError (pytest failure) if any check fails; none raised directly otherwise.
        # Side effects: none (only local interpreter state dicts created inside the test are mutated).
        code = "for i in range(10):\n    if i == 5:\n        pass\ni"
        result, _ = evaluate_python_code(code, {"range": range}, state={})
        assert result == 9

    def test_continue(self):
        # [FUNCTION DOC]
        # Checks continue skips the remainder of the loop body, for unconditional and conditional cases.
        # Inputs: self (the test class instance).
        # Outputs: None.
        # Exceptions: AssertionError (pytest failure) if any check fails; none raised directly otherwise.
        # Side effects: none (only local interpreter state dicts created inside the test are mutated).
        code = "cnt = 0\nfor i in range(10):\n    continue\n    cnt += 1\ncnt"
        result, _ = evaluate_python_code(code, {"range": range}, state={})
        assert result == 0

        code = "cnt = 0\nfor i in range(3):\n    if i == 1:\n        continue\n    cnt += 1\ncnt"
        result, _ = evaluate_python_code(code, {"range": range}, state={})
        assert result == 2

    def test_call_int(self):
        # [FUNCTION DOC]
        # Checks importing math and wrapping math.ceil in a lambda tool returns '149'.
        # Inputs: self (the test class instance).
        # Outputs: None.
        # Exceptions: AssertionError (pytest failure) if any check fails; none raised directly otherwise.
        # Side effects: none (only local interpreter state dicts created inside the test are mutated).
        code = "import math\nstr(math.ceil(149))"
        result, _ = evaluate_python_code(code, {"str": lambda x: str(x)}, state={})
        assert result == "149"

    def test_lambda(self):
        # [FUNCTION DOC]
        # Checks defining a lambda in interpreted code and calling it.
        # Inputs: self (the test class instance).
        # Outputs: None.
        # Exceptions: AssertionError (pytest failure) if any check fails; none raised directly otherwise.
        # Side effects: none (only local interpreter state dicts created inside the test are mutated).
        code = "f = lambda x: x + 2\nf(3)"
        result, _ = evaluate_python_code(code, {}, state={})
        assert result == 5

    def test_tuple_assignment(self):
        # [FUNCTION DOC]
        # Checks tuple unpacking assignment.
        # Inputs: self (the test class instance).
        # Outputs: None.
        # Exceptions: AssertionError (pytest failure) if any check fails; none raised directly otherwise.
        # Side effects: none (only local interpreter state dicts created inside the test are mutated).
        code = "a, b = 0, 1\nb"
        result, _ = evaluate_python_code(code, BASE_PYTHON_TOOLS, state={})
        assert result == 1

    def test_while(self):
        # [FUNCTION DOC]
        # Checks a while loop; then with MAX_WHILE_ITERATIONS patched to 100 checks an infinite loop raises
        #   InterpreterError; then checks a short-circuiting while condition does not index out of range.
        # Inputs: self (the test class instance).
        # Outputs: None.
        # Exceptions: AssertionError (pytest failure) if any check fails; none raised directly otherwise.
        # Side effects: temporarily patches MAX_WHILE_ITERATIONS (restored by patch).
        code = "i = 0\nwhile i < 3:\n    i += 1\ni"
        result, _ = evaluate_python_code(code, BASE_PYTHON_TOOLS, state={})
        assert result == 3

        # test infinite loop
        code = "i = 0\nwhile i < 3:\n    i -= 1\ni"
        with patch("smolagents.local_python_executor.MAX_WHILE_ITERATIONS", 100):
            with pytest.raises(InterpreterError, match=".*Maximum number of 100 iterations in While loop exceeded"):
                evaluate_python_code(code, BASE_PYTHON_TOOLS, state={})

        # test lazy evaluation
        code = dedent(
            """
            house_positions = [0, 7, 10, 15, 18, 22, 22]
            i, n, loc = 0, 7, 30
            while i < n and house_positions[i] <= loc:
                i += 1
            """
        )
        state = {}
        evaluate_python_code(code, BASE_PYTHON_TOOLS, state=state)

    def test_generator(self):
        # [FUNCTION DOC]
        # Checks a generator expression consumed by list.
        # Inputs: self (the test class instance).
        # Outputs: None.
        # Exceptions: AssertionError (pytest failure) if any check fails; none raised directly otherwise.
        # Side effects: none (only local interpreter state dicts created inside the test are mutated).
        code = "a = [1, 2, 3, 4, 5]; b = (i**2 for i in a); list(b)"
        result, _ = evaluate_python_code(code, BASE_PYTHON_TOOLS, state={})
        assert result == [1, 4, 9, 16, 25]

    def test_boolops(self):
        # [FUNCTION DOC]
        # Checks boolean operators in if/elif conditions and in expressions, including short-circuit semantics
        #   that return the deciding operand rather than a bool.
        # Inputs: self (the test class instance).
        # Outputs: None.
        # Exceptions: AssertionError (pytest failure) if any check fails; none raised directly otherwise.
        # Side effects: none (only local interpreter state dicts created inside the test are mutated).
        code = """if (not (a > b and a > c)) or d > e:
    best_city = "Brooklyn"
else:
    best_city = "Manhattan"
    best_city
    """
        result, _ = evaluate_python_code(code, BASE_PYTHON_TOOLS, state={"a": 1, "b": 2, "c": 3, "d": 4, "e": 5})
        assert result == "Brooklyn"

        code = """if d > e and a < b:
    best_city = "Brooklyn"
elif d < e and a < b:
    best_city = "Sacramento"
else:
    best_city = "Manhattan"
    best_city
    """
        result, _ = evaluate_python_code(code, BASE_PYTHON_TOOLS, state={"a": 1, "b": 2, "c": 3, "d": 4, "e": 5})
        assert result == "Sacramento"

        # Short-circuit evaluation:
        # (T and 0) or (T and T) => 0 or True => True
        code = "result = (x > 3 and y) or (z == 10 and not y)\nresult"
        result, _ = evaluate_python_code(code, BASE_PYTHON_TOOLS, state={"x": 5, "y": 0, "z": 10})
        assert result

        # (None or "") or "Found" => "" or "Found" => "Found"
        code = "result = (a or c) or b\nresult"
        result, _ = evaluate_python_code(code, BASE_PYTHON_TOOLS, state={"a": None, "b": "Found", "c": ""})
        assert result == "Found"

        # ("First" and "") or "Third" => "" or "Third" -> "Third"
        code = "result = (a and b) or c\nresult"
        result, _ = evaluate_python_code(code, BASE_PYTHON_TOOLS, state={"a": "First", "b": "", "c": "Third"})
        assert result == "Third"

    def test_if_conditions(self):
        # [FUNCTION DOC]
        # Checks print inside an if body is captured in state _print_outputs.
        # Inputs: self (the test class instance).
        # Outputs: None.
        # Exceptions: AssertionError (pytest failure) if any check fails; none raised directly otherwise.
        # Side effects: none (only local interpreter state dicts created inside the test are mutated).
        code = """char='a'
if char.isalpha():
    print('2')"""
        state = {}
        evaluate_python_code(code, BASE_PYTHON_TOOLS, state=state)
        assert state["_print_outputs"].value == "2\n"

    def test_imports(self):
        # [FUNCTION DOC]
        # Checks imports of default-authorized stdlib modules (math, random, time, re, queue, itertools, stat,
        #   statistics, unicodedata) and of numpy.random when explicitly authorized.
        # Inputs: self (the test class instance).
        # Outputs: None.
        # Exceptions: AssertionError (pytest failure) if any check fails; none raised directly otherwise.
        # Side effects: sleeps 0.1s via time.sleep inside interpreted code.
        # KNOWN ISSUE: The numpy.random cases have no assertions; they only check that no exception is raised.
        code = "import math\nmath.sqrt(4)"
        result, _ = evaluate_python_code(code, BASE_PYTHON_TOOLS, state={})
        assert result == 2.0

        code = "from random import choice, seed\nseed(12)\nchoice(['win', 'lose', 'draw'])"
        result, _ = evaluate_python_code(code, BASE_PYTHON_TOOLS, state={})
        assert result == "lose"

        code = "import time, re\ntime.sleep(0.1)"
        result, _ = evaluate_python_code(code, BASE_PYTHON_TOOLS, state={})
        assert result is None

        code = "from queue import Queue\nq = Queue()\nq.put(1)\nq.get()"
        result, _ = evaluate_python_code(code, BASE_PYTHON_TOOLS, state={})
        assert result == 1

        code = "import itertools\nlist(itertools.islice(range(10), 3))"
        result, _ = evaluate_python_code(code, BASE_PYTHON_TOOLS, state={})
        assert result == [0, 1, 2]

        code = "import re\nre.search('a', 'abc').group()"
        result, _ = evaluate_python_code(code, BASE_PYTHON_TOOLS, state={})
        assert result == "a"

        code = "import stat\nstat.S_ISREG(0o100644)"
        result, _ = evaluate_python_code(code, BASE_PYTHON_TOOLS, state={})
        assert result

        code = "import statistics\nstatistics.mean([1, 2, 3, 4, 4])"
        result, _ = evaluate_python_code(code, BASE_PYTHON_TOOLS, state={})
        assert result == 2.8

        code = "import unicodedata\nunicodedata.name('A')"
        result, _ = evaluate_python_code(code, BASE_PYTHON_TOOLS, state={})
        assert result == "LATIN CAPITAL LETTER A"

        # Test submodules are handled properly, thus not raising error
        code = "import numpy.random as rd\nrng = rd.default_rng(12345)\nrng.random()"
        result, _ = evaluate_python_code(code, BASE_PYTHON_TOOLS, state={}, authorized_imports=["numpy.random"])

        code = "from numpy.random import default_rng as d_rng\nrng = d_rng(12345)\nrng.random()"
        result, _ = evaluate_python_code(code, BASE_PYTHON_TOOLS, state={}, authorized_imports=["numpy.random"])

    def test_additional_imports(self):
        # [FUNCTION DOC]
        # Checks authorized_imports handling: exact names, 'pkg.*' wildcards covering the root and submodules,
        #   '*', and that unrelated or non-matching patterns raise InterpreterError on import of numpy.random.
        # Inputs: self (the test class instance).
        # Outputs: None.
        # Exceptions: AssertionError (pytest failure) if any check fails; none raised directly otherwise.
        # Side effects: none (only local interpreter state dicts created inside the test are mutated).
        code = "import numpy as np"
        evaluate_python_code(code, authorized_imports=["numpy"], state={})

        # Test that allowing 'numpy.*' allows numpy root package and its submodules
        code = "import numpy as np\nnp.random.default_rng(123)\nnp.array([1, 2])"
        result, _ = evaluate_python_code(code, BASE_PYTHON_TOOLS, state={}, authorized_imports=["numpy.*"])

        # Test that allowing 'numpy.*' allows importing a submodule
        code = "import numpy.random as rd\nrd.default_rng(12345)"
        result, _ = evaluate_python_code(code, BASE_PYTHON_TOOLS, state={}, authorized_imports=["numpy.*"])

        code = "import numpy.random as rd"
        evaluate_python_code(code, authorized_imports=["numpy.random"], state={})
        evaluate_python_code(code, authorized_imports=["numpy.*"], state={})
        evaluate_python_code(code, authorized_imports=["*"], state={})
        with pytest.raises(InterpreterError):
            evaluate_python_code(code, authorized_imports=["random"], state={})

        with pytest.raises(InterpreterError):
            evaluate_python_code(code, authorized_imports=["numpy.a"], state={})
        with pytest.raises(InterpreterError):
            evaluate_python_code(code, authorized_imports=["numpy.a.*"], state={})

    def test_multiple_comparators(self):
        # [FUNCTION DOC]
        # Checks chained comparisons combined with and.
        # Inputs: self (the test class instance).
        # Outputs: None.
        # Exceptions: AssertionError (pytest failure) if any check fails; none raised directly otherwise.
        # Side effects: none (only local interpreter state dicts created inside the test are mutated).
        code = "0 <= -1 < 4 and 0 <= -5 < 4"
        result, _ = evaluate_python_code(code, BASE_PYTHON_TOOLS, state={})
        assert not result

        code = "0 <= 1 < 4 and 0 <= -5 < 4"
        result, _ = evaluate_python_code(code, BASE_PYTHON_TOOLS, state={})
        assert not result

        code = "0 <= 4 < 4 and 0 <= 3 < 4"
        result, _ = evaluate_python_code(code, BASE_PYTHON_TOOLS, state={})
        assert not result

        code = "0 <= 3 < 4 and 0 <= 3 < 4"
        result, _ = evaluate_python_code(code, BASE_PYTHON_TOOLS, state={})
        assert result

    def test_print_output(self):
        # [FUNCTION DOC]
        # Checks print output accumulates in _print_outputs, including from inside interpreter functions and
        #   list comprehensions (which use copied state but share the print container).
        # Inputs: self (the test class instance).
        # Outputs: None.
        # Exceptions: AssertionError (pytest failure) if any check fails; none raised directly otherwise.
        # Side effects: none (only local interpreter state dicts created inside the test are mutated).
        code = "print('Hello world!')\nprint('Ok no one cares')"
        state = {}
        result, _ = evaluate_python_code(code, BASE_PYTHON_TOOLS, state=state)
        assert result is None
        assert state["_print_outputs"].value == "Hello world!\nOk no one cares\n"

        # Test print in function (state copy)
        code = """
print("1")
def function():
    print("2")
function()"""
        state = {}
        evaluate_python_code(code, {"print": print}, state=state)
        assert state["_print_outputs"].value == "1\n2\n"

        # Test print in list comprehension (state copy)
        code = """
print("1")
def function():
    print("2")
[function() for i in range(10)]"""
        state = {}
        evaluate_python_code(code, {"print": print, "range": range}, state=state)
        assert state["_print_outputs"].value == "1\n2\n2\n2\n2\n2\n2\n2\n2\n2\n2\n"

    def test_tuple_target_in_iterator(self):
        # [FUNCTION DOC]
        # Checks a for loop with a tuple target over a list of tuples.
        # Inputs: self (the test class instance).
        # Outputs: None.
        # Exceptions: AssertionError (pytest failure) if any check fails; none raised directly otherwise.
        # Side effects: none (only local interpreter state dicts created inside the test are mutated).
        code = "for a, b in [('Ralf Weikert', 'Austria'), ('Samuel Seungwon Lee', 'South Korea')]:res = a.split()[0]"
        result, _ = evaluate_python_code(code, BASE_PYTHON_TOOLS, state={})
        assert result == "Samuel"

    def test_classes(self):
        # [FUNCTION DOC]
        # Checks class inheritance, super() calls, method overriding, class attributes, __str__, and
        #   raising/catching an exception in a method; inspects many state values.
        # Inputs: self (the test class instance).
        # Outputs: None.
        # Exceptions: AssertionError (pytest failure) if any check fails; none raised directly otherwise.
        # Side effects: none (only local interpreter state dicts created inside the test are mutated).
        code = """
class Animal:
    species = "Generic Animal"

    def __init__(self, name, age):
        self.name = name
        self.age = age

    def sound(self):
        return "The animal makes a sound."

    def __str__(self):
        return f"{self.name}, {self.age} years old"

class Dog(Animal):
    species = "Canine"

    def __init__(self, name, age, breed):
        super().__init__(name, age)
        self.breed = breed

    def sound(self):
        return "The dog barks."

    def __str__(self):
        return f"{self.name}, {self.age} years old, {self.breed}"

class Cat(Animal):
    def sound(self):
        return "The cat meows."

    def __str__(self):
        return f"{self.name}, {self.age} years old, {self.species}"


# Testing multiple instances
dog1 = Dog("Fido", 3, "Labrador")
dog2 = Dog("Buddy", 5, "Golden Retriever")

# Testing method with built-in function
animals = [dog1, dog2, Cat("Whiskers", 2)]
num_animals = len(animals)

# Testing exceptions in methods
class ExceptionTest:
    def method_that_raises(self):
        raise ValueError("An error occurred")

try:
    exc_test = ExceptionTest()
    exc_test.method_that_raises()
except ValueError as e:
    exception_message = str(e)


# Collecting results
dog1_sound = dog1.sound()
dog1_str = str(dog1)
dog2_sound = dog2.sound()
dog2_str = str(dog2)
cat = Cat("Whiskers", 2)
cat_sound = cat.sound()
cat_str = str(cat)
    """
        state = {}
        evaluate_python_code(
            code,
            {"print": print, "len": len, "super": super, "str": str, "sum": sum},
            state=state,
        )

        # Assert results
        assert state["dog1_sound"] == "The dog barks."
        assert state["dog1_str"] == "Fido, 3 years old, Labrador"
        assert state["dog2_sound"] == "The dog barks."
        assert state["dog2_str"] == "Buddy, 5 years old, Golden Retriever"
        assert state["cat_sound"] == "The cat meows."
        assert state["cat_str"] == "Whiskers, 2 years old, Generic Animal"
        assert state["num_animals"] == 3
        assert state["exception_message"] == "An error occurred"

    def test_variable_args(self):
        # [FUNCTION DOC]
        # Checks a function with a self parameter, star-args and double-star kwargs returns 15 for positional
        #   1,2,3 and keywords 4,5.
        # Inputs: self (the test class instance).
        # Outputs: None.
        # Exceptions: AssertionError (pytest failure) if any check fails; none raised directly otherwise.
        # Side effects: none (only local interpreter state dicts created inside the test are mutated).
        # KNOWN ISSUE: In CPython the result would be 14 (the 1 binds to self, so args is (2, 3)). The
        #   interpreter's create_function puts ALL positional args into the vararg, so this test encodes non-
        #   standard semantics.
        code = """
def var_args_method(self, *args, **kwargs):
    return sum(args) + sum(kwargs.values())

var_args_method(1, 2, 3, x=4, y=5)
"""
        state = {}
        result, _ = evaluate_python_code(code, {"sum": sum}, state=state)
        assert result == 15

    def test_exceptions(self):
        # [FUNCTION DOC]
        # Checks an exception raised by an interpreter function is caught by try/except and its message
        #   stored.
        # Inputs: self (the test class instance).
        # Outputs: None.
        # Exceptions: AssertionError (pytest failure) if any check fails; none raised directly otherwise.
        # Side effects: none (only local interpreter state dicts created inside the test are mutated).
        code = """
def method_that_raises(self):
    raise ValueError("An error occurred")

try:
    method_that_raises()
except ValueError as e:
    exception_message = str(e)
    """
        state = {}
        evaluate_python_code(
            code,
            {"print": print, "len": len, "super": super, "str": str, "sum": sum},
            state=state,
        )
        assert state["exception_message"] == "An error occurred"

    def test_print(self):
        # [FUNCTION DOC]
        # Checks print of a tool call result is captured as '1' plus newline.
        # Inputs: self (the test class instance).
        # Outputs: None.
        # Exceptions: AssertionError (pytest failure) if any check fails; none raised directly otherwise.
        # Side effects: none (only local interpreter state dicts created inside the test are mutated).
        code = "print(min([1, 2, 3]))"
        state = {}
        evaluate_python_code(code, {"min": min, "print": print}, state=state)
        assert state["_print_outputs"].value == "1\n"

    def test_types_as_objects(self):
        # [FUNCTION DOC]
        # Checks that type objects assigned in code are returned as-is (not wrapped by safer_func).
        # Inputs: self (the test class instance).
        # Outputs: None.
        # Exceptions: AssertionError (pytest failure) if any check fails; none raised directly otherwise.
        # Side effects: none (only local interpreter state dicts created inside the test are mutated).
        code = "type_a = float(2); type_b = str; type_c = int"
        state = {}
        result, is_final_answer = evaluate_python_code(code, {"float": float, "str": str, "int": int}, state=state)
        # Type objects are not wrapped by safer_func
        assert not hasattr(result, "__wrapped__")
        assert result is int

    def test_tuple_id(self):
        # [FUNCTION DOC]
        # Checks a list comprehension with tuple targets over dict items returns the keys whose count is 1.
        # Inputs: self (the test class instance).
        # Outputs: None.
        # Exceptions: AssertionError (pytest failure) if any check fails; none raised directly otherwise.
        # Side effects: none (only local interpreter state dicts created inside the test are mutated).
        # KNOWN ISSUE: The code reads an undefined name (food_item_counts) while defining food_items; it only
        #   works because evaluate_name silently falls back to a difflib close match from state. The test name
        #   does not describe this.
        code = """
food_items = {"apple": 2, "banana": 3, "orange": 1, "pear": 1}
unique_food_items = [item for item, count in food_item_counts.items() if count == 1]
"""
        state = {}
        result, is_final_answer = evaluate_python_code(code, {}, state=state)
        assert result == ["orange", "pear"]

    def test_nonsimple_augassign(self):
        # [FUNCTION DOC]
        # Checks augmented assignment on a dict subscript, a list (list extension) and an instance attribute.
        # Inputs: self (the test class instance).
        # Outputs: None.
        # Exceptions: AssertionError (pytest failure) if any check fails; none raised directly otherwise.
        # Side effects: none (only local interpreter state dicts created inside the test are mutated).
        code = """
counts_dict = {'a': 0}
counts_dict['a'] += 1
counts_list = [1, 2, 3]
counts_list += [4, 5, 6]

class Counter:
    def __init__(self):
        self.count = 0

a = Counter()
a.count += 1
"""
        state = {}
        evaluate_python_code(code, {}, state=state)
        assert state["counts_dict"] == {"a": 1}
        assert state["counts_list"] == [1, 2, 3, 4, 5, 6]
        assert state["a"].count == 1

    def test_adding_int_to_list_raises_error(self):
        # [FUNCTION DOC]
        # Checks that augmented-adding an int to a list raises InterpreterError 'Cannot add non-list value 1
        #   to a list.'
        # Inputs: self (the test class instance).
        # Outputs: None.
        # Exceptions: AssertionError (pytest failure) if any check fails; none raised directly otherwise.
        # Side effects: none (only local interpreter state dicts created inside the test are mutated).
        code = """
counts = [1, 2, 3]
counts += 1"""
        with pytest.raises(InterpreterError) as e:
            evaluate_python_code(code, BASE_PYTHON_TOOLS, state={})
        assert "Cannot add non-list value 1 to a list." in str(e)

    def test_error_highlights_correct_line_of_code(self):
        # [FUNCTION DOC]
        # Checks the InterpreterError message names the exact failing source line.
        # Inputs: self (the test class instance).
        # Outputs: None.
        # Exceptions: AssertionError (pytest failure) if any check fails; none raised directly otherwise.
        # Side effects: none (only local interpreter state dicts created inside the test are mutated).
        code = """a = 1
b = 2

counts = [1, 2, 3]
counts += 1
b += 1"""
        with pytest.raises(InterpreterError) as e:
            evaluate_python_code(code, BASE_PYTHON_TOOLS, state={})
        assert "Code execution failed at line 'counts += 1" in str(e)

    def test_error_type_returned_in_function_call(self):
        # [FUNCTION DOC]
        # Checks that an exception raised inside an interpreter function surfaces as InterpreterError
        #   including the original type name and message.
        # Inputs: self (the test class instance).
        # Outputs: None.
        # Exceptions: AssertionError (pytest failure) if any check fails; none raised directly otherwise.
        # Side effects: none (only local interpreter state dicts created inside the test are mutated).
        code = """def error_function():
    raise ValueError("error")

error_function()"""
        with pytest.raises(InterpreterError) as e:
            evaluate_python_code(code)
        assert "error" in str(e)
        assert "ValueError" in str(e)

    def test_assert(self):
        # [FUNCTION DOC]
        # Checks a failing assert raises InterpreterError mentioning only the failing expression.
        # Inputs: self (the test class instance).
        # Outputs: None.
        # Exceptions: AssertionError (pytest failure) if any check fails; none raised directly otherwise.
        # Side effects: none (only local interpreter state dicts created inside the test are mutated).
        code = """
assert 1 == 1
assert 1 == 2
"""
        with pytest.raises(InterpreterError) as e:
            evaluate_python_code(code, BASE_PYTHON_TOOLS, state={})
        assert "1 == 2" in str(e) and "1 == 1" not in str(e)

    def test_with_context_manager(self):
        # [FUNCTION DOC]
        # Checks a user-defined context manager's enter/exit run and the as-target is bound (assertions are
        #   inside the interpreted code).
        # Inputs: self (the test class instance).
        # Outputs: None.
        # Exceptions: AssertionError (pytest failure) if any check fails; none raised directly otherwise.
        # Side effects: none (only local interpreter state dicts created inside the test are mutated).
        code = """
class SimpleLock:
    def __init__(self):
        self.locked = False

    def __enter__(self):
        self.locked = True
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.locked = False

lock = SimpleLock()

with lock as l:
    assert l.locked == True

assert lock.locked == False
    """
        state = {}
        tools = {}
        evaluate_python_code(code, tools, state=state)

    def test_with_context_manager_enter_returns_different_object(self):
        # [FUNCTION DOC]
        # Checks exit is called on the context manager itself, not on the value returned by enter.
        # Inputs: self (the test class instance).
        # Outputs: None.
        # Exceptions: AssertionError (pytest failure) if any check fails; none raised directly otherwise.
        # Side effects: none (only local interpreter state dicts created inside the test are mutated).
        """Test that __exit__ is called on the context manager, not the __enter__ return value."""
        code = """
class MyContextManager:
    def __init__(self):
        self.entered = False
        self.exited = False

    def __enter__(self):
        self.entered = True
        return "I am NOT the context manager"

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.exited = True
        return False

cm = MyContextManager()
with cm as val:
    assert val == "I am NOT the context manager"
    assert cm.entered == True

assert cm.exited == True
    """
        evaluate_python_code(code, {}, state={})

    def test_with_context_manager_no_as_clause_exit_called(self):
        # [FUNCTION DOC]
        # Checks exit is called for a with statement without an as clause.
        # Inputs: self (the test class instance).
        # Outputs: None.
        # Exceptions: AssertionError (pytest failure) if any check fails; none raised directly otherwise.
        # Side effects: none (only local interpreter state dicts created inside the test are mutated).
        """Test that __exit__ is called on context managers used without 'as' clause."""
        code = """
class MyContextManager:
    def __init__(self):
        self.exited = False

    def __enter__(self):
        return "not self"

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.exited = True
        return False

cm = MyContextManager()
with cm:
    pass

assert cm.exited == True
    """
        evaluate_python_code(code, {}, state={})

    def test_with_exception_suppressed_by_exit(self):
        # [FUNCTION DOC]
        # Checks an exception in the with body is suppressed when exit returns True.
        # Inputs: self (the test class instance).
        # Outputs: None.
        # Exceptions: AssertionError (pytest failure) if any check fails; none raised directly otherwise.
        # Side effects: none (only local interpreter state dicts created inside the test are mutated).
        """Test that __exit__ returning True suppresses the exception."""
        code = """
class Suppressor:
    def __init__(self):
        self.exit_called = False
    def __enter__(self):
        return self
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.exit_called = True
        return True  # suppress

cm = Suppressor()
with cm:
    raise ValueError("should be suppressed")

assert cm.exit_called == True
        """
        evaluate_python_code(code, {}, state={})

    def test_with_exception_not_suppressed_by_exit(self):
        # [FUNCTION DOC]
        # Checks that a ValueError raised in the with body propagates as ValueError when exit returns False.
        # Inputs: self (the test class instance).
        # Outputs: None.
        # Exceptions: AssertionError (pytest failure) if any check fails; none raised directly otherwise.
        # Side effects: none (only local interpreter state dicts created inside the test are mutated).
        """Test that __exit__ returning False re-raises the original exception."""
        code = """
class NonSuppressor:
    def __enter__(self):
        return self
    def __exit__(self, exc_type, exc_val, exc_tb):
        return False

with NonSuppressor():
    raise ValueError("should propagate")
        """
        with pytest.raises(ValueError, match="should propagate"):
            evaluate_python_code(code, {}, state={})

    def test_with_multiple_cms_inner_suppresses_outer_sees_no_exception(self):
        # [FUNCTION DOC]
        # Checks with two context managers: inner exit sees the ValueError and suppresses it, outer exit sees
        #   no exception.
        # Inputs: self (the test class instance).
        # Outputs: None.
        # Exceptions: AssertionError (pytest failure) if any check fails; none raised directly otherwise.
        # Side effects: none (only local interpreter state dicts created inside the test are mutated).
        """Test that when the inner CM suppresses, the outer CM's __exit__ gets (None, None, None)."""
        code = """
calls = []

class Outer:
    def __enter__(self): return self
    def __exit__(self, exc_type, exc_val, exc_tb):
        calls.append(("outer", exc_type))
        return False

class Inner:
    def __enter__(self): return self
    def __exit__(self, exc_type, exc_val, exc_tb):
        calls.append(("inner", exc_type))
        return True  # suppress

with Outer(), Inner():
    raise ValueError("suppressed by inner")

assert calls[0] == ("inner", ValueError), calls
assert calls[1] == ("outer", None), calls
        """
        evaluate_python_code(code, {}, state={})

    def test_with_multiple_cms_neither_suppresses(self):
        # [FUNCTION DOC]
        # Checks with two non-suppressing context managers the RuntimeError propagates (wrapped as
        #   InterpreterError).
        # Inputs: self (the test class instance).
        # Outputs: None.
        # Exceptions: AssertionError (pytest failure) if any check fails; none raised directly otherwise.
        # Side effects: none (only local interpreter state dicts created inside the test are mutated).
        # KNOWN ISSUE: Inconsistent with test_with_exception_not_suppressed_by_exit, which expects a raw
        #   ValueError for a single manager; here the exception arrives wrapped in InterpreterError.
        """Test that when no CM suppresses, the original exception propagates."""
        code = """
calls = []

class Recorder:
    def __init__(self, name):
        self.name = name
    def __enter__(self): return self
    def __exit__(self, exc_type, exc_val, exc_tb):
        calls.append((self.name, exc_type))
        return False

with Recorder("outer"), Recorder("inner"):
    raise RuntimeError("should propagate")
        """
        with pytest.raises(InterpreterError, match="should propagate"):
            evaluate_python_code(code, {}, state={})

    def test_with_multiple_cms_outer_suppresses(self):
        # [FUNCTION DOC]
        # Checks the outer manager can suppress an exception the inner one passed on; both exits see
        #   ValueError.
        # Inputs: self (the test class instance).
        # Outputs: None.
        # Exceptions: AssertionError (pytest failure) if any check fails; none raised directly otherwise.
        # Side effects: none (only local interpreter state dicts created inside the test are mutated).
        """Test that the outer CM can suppress after the inner CM does not."""
        code = """
calls = []

class Outer:
    def __enter__(self): return self
    def __exit__(self, exc_type, exc_val, exc_tb):
        calls.append(("outer", exc_type))
        return True  # suppress

class Inner:
    def __enter__(self): return self
    def __exit__(self, exc_type, exc_val, exc_tb):
        calls.append(("inner", exc_type))
        return False  # don't suppress

with Outer(), Inner():
    raise ValueError("suppressed by outer")

assert calls[0] == ("inner", ValueError), calls
assert calls[1] == ("outer", ValueError), calls
        """
        evaluate_python_code(code, {}, state={})

    def test_with_exit_raising_replaces_original_exception(self):
        # [FUNCTION DOC]
        # Checks an exception raised inside exit replaces the original one (surfaced as InterpreterError).
        # Inputs: self (the test class instance).
        # Outputs: None.
        # Exceptions: AssertionError (pytest failure) if any check fails; none raised directly otherwise.
        # Side effects: none (only local interpreter state dicts created inside the test are mutated).
        """Test that an exception raised inside __exit__ replaces the original."""
        code = """
class RaisingExit:
    def __enter__(self): return self
    def __exit__(self, exc_type, exc_val, exc_tb):
        raise RuntimeError("from __exit__")

with RaisingExit():
    raise ValueError("original")
        """
        with pytest.raises(InterpreterError, match="from __exit__"):
            evaluate_python_code(code, {}, state={})

    def test_default_arg_in_function(self):
        # [FUNCTION DOC]
        # Checks default parameter values and keyword overriding; also that is_final_answer is False.
        # Inputs: self (the test class instance).
        # Outputs: None.
        # Exceptions: AssertionError (pytest failure) if any check fails; none raised directly otherwise.
        # Side effects: none (only local interpreter state dicts created inside the test are mutated).
        code = """
def f(a, b=333, n=1000):
    return b + n
n = f(1, n=667)
"""
        res, is_final_answer = evaluate_python_code(code, {}, {})
        assert res == 1000
        assert not is_final_answer

    def test_set(self):
        # [FUNCTION DOC]
        # Checks set literals and difference/intersection methods.
        # Inputs: self (the test class instance).
        # Outputs: None.
        # Exceptions: AssertionError (pytest failure) if any check fails; none raised directly otherwise.
        # Side effects: none (only local interpreter state dicts created inside the test are mutated).
        code = """
S1 = {'a', 'b', 'c'}
S2 = {'b', 'c', 'd'}
S3 = S1.difference(S2)
S4 = S1.intersection(S2)
"""
        state = {}
        evaluate_python_code(code, {}, state=state)
        assert state["S3"] == {"a"}
        assert state["S4"] == {"b", "c"}

    def test_return(self):
        # [FUNCTION DOC]
        # Checks early return inside an if and a bare return yielding None.
        # Inputs: self (the test class instance).
        # Outputs: None.
        # Exceptions: AssertionError (pytest failure) if any check fails; none raised directly otherwise.
        # Side effects: none (only local interpreter state dicts created inside the test are mutated).
        # test early returns
        code = """
def add_one(n, shift):
    if True:
        return n + shift
    return n

add_one(1, 1)
"""
        state = {}
        result, is_final_answer = evaluate_python_code(
            code, {"print": print, "range": range, "ord": ord, "chr": chr}, state=state
        )
        assert result == 2

        # test returning None
        code = """
def returns_none(a):
    return

returns_none(1)
"""
        state = {}
        result, is_final_answer = evaluate_python_code(
            code, {"print": print, "range": range, "ord": ord, "chr": chr}, state=state
        )
        assert result is None

    def test_nested_for_loop(self):
        # [FUNCTION DOC]
        # Checks nested for loops with list appends then a flattening comprehension.
        # Inputs: self (the test class instance).
        # Outputs: None.
        # Exceptions: AssertionError (pytest failure) if any check fails; none raised directly otherwise.
        # Side effects: none (only local interpreter state dicts created inside the test are mutated).
        code = """
all_res = []
for i in range(10):
    subres = []
    for j in range(i):
        subres.append(j)
    all_res.append(subres)

out = [i for sublist in all_res for i in sublist]
out[:10]
"""
        state = {}
        result, is_final_answer = evaluate_python_code(code, {"print": print, "range": range}, state=state)
        assert result == [0, 0, 1, 0, 1, 2, 0, 1, 2, 3]

    def test_pandas(self):
        # [FUNCTION DOC]
        # Checks pandas usage with pandas authorized: to_numeric, boolean masking, isin with loc, groupby
        #   mean, sort_values with iloc.
        # Inputs: self (the test class instance).
        # Outputs: None.
        # Exceptions: AssertionError (pytest failure) if any check fails; none raised directly otherwise.
        # Side effects: none (only local interpreter state dicts created inside the test are mutated).
        # KNOWN ISSUE: The last snippet (loc/iloc/sort_values) has no assertion; it only checks no exception.
        code = """
import pandas as pd

df = pd.DataFrame.from_dict({'SetCount': ['5', '4', '5'], 'Quantity': [1, 0, -1]})

df['SetCount'] = pd.to_numeric(df['SetCount'], errors='coerce')

parts_with_5_set_count = df[df['SetCount'] == 5.0]
parts_with_5_set_count[['Quantity', 'SetCount']].values[1]
"""
        state = {}
        result, _ = evaluate_python_code(code, {}, state=state, authorized_imports=["pandas"])
        assert np.array_equal(result, [-1, 5])

        code = """
import pandas as pd

df = pd.DataFrame.from_dict({"AtomicNumber": [111, 104, 105], "ok": [0, 1, 2]})

# Filter the DataFrame to get only the rows with outdated atomic numbers
filtered_df = df.loc[df['AtomicNumber'].isin([104])]
"""
        result, _ = evaluate_python_code(code, {"print": print}, state={}, authorized_imports=["pandas"])
        assert np.array_equal(result.values[0], [104, 1])

        # Test groupby
        code = """import pandas as pd
data = pd.DataFrame.from_dict([
    {"Pclass": 1, "Survived": 1},
    {"Pclass": 2, "Survived": 0},
    {"Pclass": 2, "Survived": 1}
])
survival_rate_by_class = data.groupby('Pclass')['Survived'].mean()
"""
        result, _ = evaluate_python_code(code, {}, state={}, authorized_imports=["pandas"])
        assert result.values[1] == 0.5

        # Test loc and iloc
        code = """import pandas as pd
data = pd.DataFrame.from_dict([
    {"Pclass": 1, "Survived": 1},
    {"Pclass": 2, "Survived": 0},
    {"Pclass": 2, "Survived": 1}
])
survival_rate_biased = data.loc[data['Survived']==1]['Survived'].mean()
survival_rate_biased = data.loc[data['Survived']==1]['Survived'].mean()
survival_rate_sorted = data.sort_values(by='Survived', ascending=False).iloc[0]
"""
        result, _ = evaluate_python_code(code, {}, state={}, authorized_imports=["pandas"])

    def test_starred(self):
        # [FUNCTION DOC]
        # Checks star-unpacking of two tuples into a haversine function using math functions and map.
        # Inputs: self (the test class instance).
        # Outputs: None.
        # Exceptions: AssertionError (pytest failure) if any check fails; none raised directly otherwise.
        # Side effects: none (only local interpreter state dicts created inside the test are mutated).
        code = """
from math import radians, sin, cos, sqrt, atan2

def haversine(lat1, lon1, lat2, lon2):
    R = 6371000  # Radius of the Earth in meters
    lat1, lon1, lat2, lon2 = map(radians, [lat1, lon1, lat2, lon2])
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = sin(dlat / 2) ** 2 + cos(lat1) * cos(lat2) * sin(dlon / 2) ** 2
    c = 2 * atan2(sqrt(a), sqrt(1 - a))
    distance = R * c
    return distance

coords_geneva = (46.1978, 6.1342)
coords_barcelona = (41.3869, 2.1660)

distance_geneva_barcelona = haversine(*coords_geneva, *coords_barcelona)
"""
        result, _ = evaluate_python_code(code, {"print": print, "map": map}, state={}, authorized_imports=["math"])
        assert round(result, 1) == 622395.4

    def test_for(self):
        # [FUNCTION DOC]
        # Checks a for loop with nested tuple target over dict items filling a dict.
        # Inputs: self (the test class instance).
        # Outputs: None.
        # Exceptions: AssertionError (pytest failure) if any check fails; none raised directly otherwise.
        # Side effects: none (only local interpreter state dicts created inside the test are mutated).
        code = """
shifts = {
    "Worker A": ("6:45 pm", "8:00 pm"),
    "Worker B": ("10:00 am", "11:45 am")
}

shift_intervals = {}
for worker, (start, end) in shifts.items():
    shift_intervals[worker] = end
shift_intervals
"""
        result, _ = evaluate_python_code(code, {"print": print, "map": map}, state={})
        assert result == {"Worker A": "8:00 pm", "Worker B": "11:45 am"}

    def test_syntax_error_points_error(self):
        # [FUNCTION DOC]
        # Checks a syntax error is reported as InterpreterError containing 'SyntaxError' and a caret marker.
        # Inputs: self (the test class instance).
        # Outputs: None.
        # Exceptions: AssertionError (pytest failure) if any check fails; none raised directly otherwise.
        # Side effects: none (only local interpreter state dicts created inside the test are mutated).
        code = "a = ;"
        with pytest.raises(InterpreterError) as e:
            evaluate_python_code(code)
        assert "SyntaxError" in str(e)
        assert "     ^" in str(e)

    def test_close_matches_subscript(self):
        # [FUNCTION DOC]
        # Checks a missing dict key error suggests close-matching keys ('Bhutan' for 'Butan').
        # Inputs: self (the test class instance).
        # Outputs: None.
        # Exceptions: AssertionError (pytest failure) if any check fails; none raised directly otherwise.
        # Side effects: none (only local interpreter state dicts created inside the test are mutated).
        code = 'capitals = {"Czech Republic": "Prague", "Monaco": "Monaco", "Bhutan": "Thimphu"};capitals["Butan"]'
        with pytest.raises(Exception) as e:
            evaluate_python_code(code)
        assert "Maybe you meant one of these indexes instead" in str(e) and "['Bhutan']" in str(e).replace("\\", "")

    def test_dangerous_builtins_calls_are_blocked(self):
        # [FUNCTION DOC]
        # Checks importing os and reaching exec/compile via callable's __self__ both raise InterpreterError.
        # Inputs: self (the test class instance).
        # Outputs: None.
        # Exceptions: AssertionError (pytest failure) if any check fails; none raised directly otherwise.
        # Side effects: none (only local interpreter state dicts created inside the test are mutated).
        unsafe_code = "import os"
        dangerous_code = f"""
exec = callable.__self__.exec
compile = callable.__self__.compile
exec(compile('{unsafe_code}', 'no filename', 'exec'))
"""

        with pytest.raises(InterpreterError):
            evaluate_python_code(unsafe_code, static_tools=BASE_PYTHON_TOOLS)

        with pytest.raises(InterpreterError):
            evaluate_python_code(dangerous_code, static_tools=BASE_PYTHON_TOOLS)

    def test_final_answer_accepts_kwarg_answer(self):
        # [FUNCTION DOC]
        # Checks final_answer can be called with the answer keyword and its tool result (doubled) is returned.
        # Inputs: self (the test class instance).
        # Outputs: None.
        # Exceptions: AssertionError (pytest failure) if any check fails; none raised directly otherwise.
        # Side effects: none (only local interpreter state dicts created inside the test are mutated).
        code = "final_answer(answer=2)"
        result, _ = evaluate_python_code(code, {"final_answer": (lambda answer: 2 * answer)}, state={})
        assert result == 4

    def test_final_answer_not_caught_by_except_exception(self):
        # [FUNCTION DOC]
        # Checks that the final-answer signal raised by final_answer inside try is not caught by a generic
        #   except Exception; result is 1 and is_final_answer is True.
        # Inputs: self (the test class instance).
        # Outputs: None.
        # Exceptions: AssertionError (pytest failure) if any check fails; none raised directly otherwise.
        # Side effects: none (only local interpreter state dicts created inside the test are mutated).
        """Test that final_answer is not caught by generic 'except Exception' clauses.

        This test reproduces the issue from GitHub issue #1905 where agent-generated
        code with try/except Exception blocks would incorrectly catch FinalAnswerException.
        """
        code = dedent("""
            try:
                final_answer(1)
            except Exception as e:
                final_answer(2)
        """)
        result, is_final_answer = evaluate_python_code(code, {"final_answer": (lambda answer: answer)}, state={})
        # The result should be 1 (from the first final_answer call),
        # not 2 (which would happen if FinalAnswerException was caught)
        assert result == 1
        assert is_final_answer is True

    def test_dangerous_builtins_are_callable_if_explicitly_added(self):
        # [FUNCTION DOC]
        # Checks eval/exec/compile work when explicitly passed as static tools.
        # Inputs: self (the test class instance).
        # Outputs: None.
        # Exceptions: AssertionError (pytest failure) if any check fails; none raised directly otherwise.
        # Side effects: none (only local interpreter state dicts created inside the test are mutated).
        dangerous_code = dedent("""
            eval("1 + 1")
            exec(compile("1 + 1", "no filename", "exec"))
        """)
        evaluate_python_code(
            dangerous_code, static_tools={"compile": compile, "eval": eval, "exec": exec} | BASE_PYTHON_TOOLS
        )

    def test_can_import_os_if_explicitly_authorized(self):
        # [FUNCTION DOC]
        # Checks importing os and listing the current directory succeeds when os is authorized.
        # Inputs: self (the test class instance).
        # Outputs: None.
        # Exceptions: AssertionError (pytest failure) if any check fails; none raised directly otherwise.
        # Side effects: reads the current working directory listing (filesystem read).
        dangerous_code = "import os; os.listdir('./')"
        evaluate_python_code(dangerous_code, authorized_imports=["os"])

    def test_can_import_os_if_all_imports_authorized(self):
        # [FUNCTION DOC]
        # Checks importing os and listing the current directory succeeds with the '*' wildcard.
        # Inputs: self (the test class instance).
        # Outputs: None.
        # Exceptions: AssertionError (pytest failure) if any check fails; none raised directly otherwise.
        # Side effects: reads the current working directory listing (filesystem read).
        dangerous_code = "import os; os.listdir('./')"
        evaluate_python_code(dangerous_code, authorized_imports=["*"])

    @pytest.mark.filterwarnings("ignore::DeprecationWarning")
    def test_can_import_scipy_if_explicitly_authorized(self):
        # [FUNCTION DOC]
        # Checks import of scipy succeeds when authorized (DeprecationWarnings ignored).
        # Inputs: self (the test class instance).
        # Outputs: None.
        # Exceptions: AssertionError/ImportError-derived failure if scipy is missing or blocked.
        # Side effects: imports scipy into the host process.
        code = "import scipy"
        evaluate_python_code(code, authorized_imports=["scipy"])

    @pytest.mark.filterwarnings("ignore::DeprecationWarning")
    def test_can_import_sklearn_if_explicitly_authorized(self):
        # [FUNCTION DOC]
        # Checks import of sklearn succeeds when authorized (DeprecationWarnings ignored).
        # Inputs: self (the test class instance).
        # Outputs: None.
        # Exceptions: failure if sklearn is missing or blocked.
        # Side effects: imports sklearn into the host process.
        code = "import sklearn"
        evaluate_python_code(code, authorized_imports=["sklearn"])

    def test_function_def_recovers_source_code(self):
        # [FUNCTION DOC]
        # Uses LocalPythonExecutor with a FinalAnswerTool to return an interpreter-defined function and checks
        #   its __name__ and that __source__ holds the unparsed source.
        # Inputs: self (the test class instance).
        # Outputs: None.
        # Exceptions: AssertionError (pytest failure) if any check fails; none raised directly otherwise.
        # Side effects: none (only local interpreter state dicts created inside the test are mutated).
        executor = LocalPythonExecutor([])

        executor.send_tools({"final_answer": FinalAnswerTool()})

        res = executor(
            dedent(
                """
                def target_function():
                    return "Hello world"

                final_answer(target_function)
                """
            )
        ).output
        assert res.__name__ == "target_function"
        assert res.__source__ == "def target_function():\n    return 'Hello world'"

    def test_evaluate_class_def_with_pass(self):
        # [FUNCTION DOC]
        # Checks an empty (pass) class can be instantiated and given instance attributes.
        # Inputs: self (the test class instance).
        # Outputs: None.
        # Exceptions: AssertionError (pytest failure) if any check fails; none raised directly otherwise.
        # Side effects: none (only local interpreter state dicts created inside the test are mutated).
        code = dedent("""
            class TestClass:
                pass

            instance = TestClass()
            instance.attr = "value"
            result = instance.attr
        """)
        state = {}
        result, _ = evaluate_python_code(code, BASE_PYTHON_TOOLS, state=state)
        assert result == "value"

    def test_evaluate_class_def_with_ann_assign_name(self):
        # [FUNCTION DOC]
        # Checks annotated class attributes set values and __annotations__ (with unwrapped type objects).
        # Inputs: self (the test class instance).
        # Outputs: None.
        # Exceptions: AssertionError (pytest failure) if any check fails; none raised directly otherwise.
        # Side effects: none (only local interpreter state dicts created inside the test are mutated).
        """
        Test evaluate_class_def function when stmt is an instance of ast.AnnAssign with ast.Name target.

        This test verifies that annotated assignments within a class definition are correctly evaluated.
        """
        code = dedent("""
            class TestClass:
                x: int = 5
                y: str = "test"

            instance = TestClass()
            result = (instance.x, instance.y)
        """)

        state = {}
        result, _ = evaluate_python_code(code, BASE_PYTHON_TOOLS, state=state)

        assert result == (5, "test")
        assert isinstance(state["TestClass"], type)
        # Type objects are not wrapped by safer_func
        for value in state["TestClass"].__annotations__.values():
            assert not hasattr(value, "__wrapped__")
        assert state["TestClass"].__annotations__ == {"x": int, "y": str}
        assert state["TestClass"].x == 5
        assert state["TestClass"].y == "test"
        assert isinstance(state["instance"], state["TestClass"])
        assert state["instance"].x == 5
        assert state["instance"].y == "test"

    def test_evaluate_class_def_with_ann_assign_attribute(self):
        # [FUNCTION DOC]
        # Checks annotated assignment to an attribute target in a class body sets the value but only records
        #   the plain-name annotation.
        # Inputs: self (the test class instance).
        # Outputs: None.
        # Exceptions: AssertionError (pytest failure) if any check fails; none raised directly otherwise.
        # Side effects: none (only local interpreter state dicts created inside the test are mutated).
        """
        Test evaluate_class_def function when stmt is an instance of ast.AnnAssign with ast.Attribute target.

        This test ensures that class attributes using attribute notation are correctly handled.
        """
        code = dedent("""
        class TestSubClass:
            attr = 1
        class TestClass:
            data: TestSubClass = TestSubClass()
            data.attr: str = "value"

        result = TestClass.data.attr
        """)

        state = {}
        result, _ = evaluate_python_code(code, BASE_PYTHON_TOOLS, state=state)

        assert result == "value"
        assert isinstance(state["TestClass"], type)
        assert state["TestClass"].__annotations__.keys() == {"data"}
        assert isinstance(state["TestClass"].__annotations__["data"], type)
        assert state["TestClass"].__annotations__["data"].__name__ == "TestSubClass"
        assert state["TestClass"].data.attr == "value"

    def test_evaluate_class_def_with_ann_assign_subscript(self):
        # [FUNCTION DOC]
        # Checks annotated assignment to subscript and slice targets in a class body mutates the containers
        #   and only plain-name annotations are recorded.
        # Inputs: self (the test class instance).
        # Outputs: None.
        # Exceptions: AssertionError (pytest failure) if any check fails; none raised directly otherwise.
        # Side effects: none (only local interpreter state dicts created inside the test are mutated).
        """
        Test evaluate_class_def function when stmt is an instance of ast.AnnAssign with ast.Subscript target.

        This test ensures that class attributes using subscript notation are correctly handled.
        """
        code = dedent("""
        class TestClass:
            key_data: dict = {}
            key_data["key"]: str = "value"
            index_data: list = [10, 20, 30]
            index_data[0:2]: list[str] = ["a", "b"]

        result = (TestClass.key_data['key'], TestClass.index_data[1:])
        """)

        state = {}
        result, _ = evaluate_python_code(code, BASE_PYTHON_TOOLS, state=state)

        assert result == ("value", ["b", 30])
        assert isinstance(state["TestClass"], type)
        # Type objects are not wrapped by safer_func
        for value in state["TestClass"].__annotations__.values():
            assert not hasattr(value, "__wrapped__")
        assert state["TestClass"].__annotations__ == {"key_data": dict, "index_data": list}
        assert state["TestClass"].key_data == {"key": "value"}
        assert state["TestClass"].index_data == ["a", "b", 30]

    def test_evaluate_class_def_with_enum(self):
        # [FUNCTION DOC]
        # Checks an Enum subclass defined in interpreted code (enum authorized) gets proper members via the
        #   metaclass and its prepare hook.
        # Inputs: self (the test class instance).
        # Outputs: None.
        # Exceptions: AssertionError (pytest failure) if any check fails; none raised directly otherwise.
        # Side effects: none (only local interpreter state dicts created inside the test are mutated).
        """
        Test evaluate_class_def function with Enum classes.

        This test ensures that Enum classes are correctly handled by using the
        appropriate metaclass and __prepare__ method.
        """
        code = dedent("""
        from enum import Enum

        class Status(Enum):
            SUCCESS = "Success"
            FAILURE = "Failure"
            PENDING = "Pending"
            ERROR = "Error"

        status_value = Status.SUCCESS.value
        status_name = Status.SUCCESS.name
        """)

        state = {}
        result, _ = evaluate_python_code(code, BASE_PYTHON_TOOLS, state=state, authorized_imports=["enum"])

        assert state["status_value"] == "Success"
        assert state["status_name"] == "SUCCESS"
        assert isinstance(state["Status"], type)
        assert hasattr(state["Status"], "SUCCESS")
        assert state["Status"].SUCCESS.value == "Success"
        assert state["Status"].FAILURE.value == "Failure"
        assert state["Status"].PENDING.value == "Pending"
        assert state["Status"].ERROR.value == "Error"

    def test_evaluate_annassign(self):
        # [FUNCTION DOC]
        # Checks module-level annotated assignments: values set, annotation-only names stay undefined, and
        #   types are not enforced.
        # Inputs: self (the test class instance).
        # Outputs: None.
        # Exceptions: AssertionError (pytest failure) if any check fails; none raised directly otherwise.
        # Side effects: none (only local interpreter state dicts created inside the test are mutated).
        code = dedent("""\
            # Basic annotated assignment
            x: int = 42

            # Type annotations with expressions
            y: float = x / 2

            # Type annotation without assignment
            z: list

            # Type annotation with complex value
            names: list = ["Alice", "Bob", "Charlie"]

            # Type hint shouldn't restrict values at runtime
            s: str = 123  # Would be a type error in static checking, but valid at runtime

            # Access the values
            result = (x, y, names, s)
        """)
        state = {}
        evaluate_python_code(code, BASE_PYTHON_TOOLS, state=state)
        assert state["x"] == 42
        assert state["y"] == 21.0
        assert "z" not in state  # z should be not be defined
        assert state["names"] == ["Alice", "Bob", "Charlie"]
        assert state["s"] == 123  # Type hints don't restrict at runtime
        assert state["result"] == (42, 21.0, ["Alice", "Bob", "Charlie"], 123)

    @pytest.mark.parametrize(
        "code, expected_result",
        [
            (
                dedent("""\
                    x = 1
                    x += 2
                """),
                3,
            ),
            (
                dedent("""\
                    x = "a"
                    x += "b"
                """),
                "ab",
            ),
            (
                dedent("""\
                    class Custom:
                        def __init__(self, value):
                            self.value = value
                        def __iadd__(self, other):
                            self.value += other * 10
                            return self

                    x = Custom(1)
                    x += 2
                    x.value
                """),
                21,
            ),
        ],
    )
    def test_evaluate_augassign(self, code, expected_result):
        # [FUNCTION DOC]
        # Parametrized: checks augmented add for int, str and a custom class with an in-place add method.
        # Inputs: self (the test class instance); code (str), expected_result (any).
        # Outputs: None.
        # Exceptions: AssertionError (pytest failure) if any check fails; none raised directly otherwise.
        # Side effects: none (only local interpreter state dicts created inside the test are mutated).
        state = {}
        result, _ = evaluate_python_code(code, {}, state=state)
        assert result == expected_result

    @pytest.mark.parametrize(
        "operator, expected_result",
        [
            ("+=", 7),
            ("-=", 3),
            ("*=", 10),
            ("/=", 2.5),
            ("//=", 2),
            ("%=", 1),
            ("**=", 25),
            ("&=", 0),
            ("|=", 7),
            ("^=", 7),
            (">>=", 1),
            ("<<=", 20),
        ],
    )
    def test_evaluate_augassign_number(self, operator, expected_result):
        # [FUNCTION DOC]
        # Parametrized over all 12 augmented operators: checks each applied to 5 and 2 gives the expected
        #   number.
        # Inputs: self (the test class instance); operator (str augmented operator), expected_result (number).
        # Outputs: None.
        # Exceptions: AssertionError (pytest failure) if any check fails; none raised directly otherwise.
        # Side effects: none (only local interpreter state dicts created inside the test are mutated).
        code = dedent("""\
            x = 5
            x {operator} 2
        """).format(operator=operator)
        state = {}
        result, _ = evaluate_python_code(code, {}, state=state)
        assert result == expected_result

    @pytest.mark.parametrize(
        "operator, expected_result",
        [
            ("+=", 7),
            ("-=", 3),
            ("*=", 10),
            ("/=", 2.5),
            ("//=", 2),
            ("%=", 1),
            ("**=", 25),
            ("&=", 0),
            ("|=", 7),
            ("^=", 7),
            (">>=", 1),
            ("<<=", 20),
        ],
    )
    def test_evaluate_augassign_custom(self, operator, expected_result):
        # [FUNCTION DOC]
        # Parametrized over all 12 augmented operators: checks a custom class implementing the matching in-
        #   place dunder method is used, giving the expected stored value.
        # Inputs: self (the test class instance); operator (str), expected_result (number).
        # Outputs: None.
        # Exceptions: AssertionError (pytest failure) if any check fails; none raised directly otherwise.
        # Side effects: none (only local interpreter state dicts created inside the test are mutated).
        operator_names = {
            "+=": "iadd",
            "-=": "isub",
            "*=": "imul",
            "/=": "itruediv",
            "//=": "ifloordiv",
            "%=": "imod",
            "**=": "ipow",
            "&=": "iand",
            "|=": "ior",
            "^=": "ixor",
            ">>=": "irshift",
            "<<=": "ilshift",
        }
        code = dedent("""\
            class Custom:
                def __init__(self, value):
                    self.value = value
                def __{operator_name}__(self, other):
                    self.value {operator} other
                    return self

            x = Custom(5)
            x {operator} 2
            x.value
        """).format(operator=operator, operator_name=operator_names[operator])
        state = {}
        result, _ = evaluate_python_code(code, {}, state=state)
        assert result == expected_result

    @pytest.mark.parametrize(
        "code, expected_error_message",
        [
            (
                dedent("""\
                    x = 5
                    del x
                    x
                """),
                "The variable `x` is not defined",
            ),
            (
                dedent("""\
                    x = [1, 2, 3]
                    del x[2]
                    x[2]
                """),
                "IndexError: list index out of range",
            ),
            (
                dedent("""\
                    x = {"key": "value"}
                    del x["key"]
                    x["key"]
                """),
                "Could not index {} with 'key'",
            ),
            (
                dedent("""\
                    del x
                """),
                "Cannot delete name 'x': name is not defined",
            ),
        ],
    )
    def test_evaluate_delete(self, code, expected_error_message):
        # [FUNCTION DOC]
        # Parametrized: checks del of a name, list index and dict key, then a later access raises
        #   InterpreterError with the expected message; also deleting an undefined name.
        # Inputs: self (the test class instance); code (str), expected_error_message (str).
        # Outputs: None.
        # Exceptions: AssertionError (pytest failure) if any check fails; none raised directly otherwise.
        # Side effects: none (only local interpreter state dicts created inside the test are mutated).
        state = {}
        with pytest.raises(InterpreterError) as exception_info:
            evaluate_python_code(code, {}, state=state)
        assert expected_error_message in str(exception_info.value)

    def test_non_standard_comparisons(self):
        # [FUNCTION DOC]
        # Checks equality comparison returns whatever a custom eq method returns (a non-bool object) rather
        #   than coercing to bool.
        # Inputs: self (the test class instance).
        # Outputs: None.
        # Exceptions: AssertionError (pytest failure) if any check fails; none raised directly otherwise.
        # Side effects: none (only local interpreter state dicts created inside the test are mutated).
        code = dedent("""\
            class NonStdEqualsResult:
                def __init__(self, left:object, right:object):
                    self._left = left
                    self._right = right
                def __str__(self) -> str:
                    return f'{self._left} == {self._right}'

            class NonStdComparisonClass:
                def __init__(self, value: str ):
                    self._value = value
                def __str__(self):
                    return self._value
                def __eq__(self, other):
                    return NonStdEqualsResult(self, other)
            a = NonStdComparisonClass("a")
            b = NonStdComparisonClass("b")
            result = a == b
            """)
        result, _ = evaluate_python_code(code, state={})
        assert not isinstance(result, bool)
        assert str(result) == "a == b"


class TestEvaluateBoolop:
    # [CLASS DOC]
    # Test group calling evaluate_boolop directly on parsed and/or expressions.
    # Inputs: no constructor args; no attributes.
    # Outputs: n/a.
    # Exceptions: none raised directly.
    # Side effects: none.
    @pytest.mark.parametrize("a", [1, 0])
    @pytest.mark.parametrize("b", [2, 0])
    @pytest.mark.parametrize("c", [3, 0])
    def test_evaluate_boolop_and(self, a, b, c):
        # [FUNCTION DOC]
        # Parametrized over truthy/falsy a, b, c: checks evaluate_boolop on a chained and returns the same
        #   operand as Python's and.
        # Inputs: self (the test class instance); a, b, c (ints, truthy or 0).
        # Outputs: None.
        # Exceptions: AssertionError (pytest failure) if any check fails; none raised directly otherwise.
        # Side effects: none (only local interpreter state dicts created inside the test are mutated).
        boolop_ast = ast.parse("a and b and c").body[0].value
        state = {"a": a, "b": b, "c": c}
        result = evaluate_boolop(boolop_ast, state, {}, {}, [])
        assert result == (a and b and c)

    @pytest.mark.parametrize("a", [1, 0])
    @pytest.mark.parametrize("b", [2, 0])
    @pytest.mark.parametrize("c", [3, 0])
    def test_evaluate_boolop_or(self, a, b, c):
        # [FUNCTION DOC]
        # Parametrized over truthy/falsy a, b, c: checks evaluate_boolop on a chained or matches Python's or.
        # Inputs: self (the test class instance); a, b, c (ints, truthy or 0).
        # Outputs: None.
        # Exceptions: AssertionError (pytest failure) if any check fails; none raised directly otherwise.
        # Side effects: none (only local interpreter state dicts created inside the test are mutated).
        boolop_ast = ast.parse("a or b or c").body[0].value
        state = {"a": a, "b": b, "c": c}
        result = evaluate_boolop(boolop_ast, state, {}, {}, [])
        assert result == (a or b or c)


class TestEvaluateDelete:
    # [CLASS DOC]
    # Test group calling evaluate_delete directly on parsed del statements.
    # Inputs: no constructor args; no attributes.
    # Outputs: n/a.
    # Exceptions: none raised directly.
    # Side effects: none.
    @pytest.mark.parametrize(
        "code, state, expectation",
        [
            ("del x", {"x": 1}, {}),
            ("del x[1]", {"x": [1, 2, 3]}, {"x": [1, 3]}),
            ("del x['key']", {"x": {"key": "value"}}, {"x": {}}),
            ("del x", {}, InterpreterError("Cannot delete name 'x': name is not defined")),
        ],
    )
    def test_evaluate_delete(self, code, state, expectation):
        # [FUNCTION DOC]
        # Parametrized: runs evaluate_delete on a del node; if an exception is expected checks its type and
        #   message, otherwise drops the _operations_count key and compares the mutated state.
        # Inputs: self (the test class instance); code (str), state (dict, mutated), expectation (expected
        #   dict or an exception instance).
        # Outputs: None.
        # Exceptions: AssertionError (pytest failure) if any check fails; none raised directly otherwise.
        # Side effects: mutates the parametrized state dict.
        delete_node = ast.parse(code).body[0]
        if isinstance(expectation, Exception):
            with pytest.raises(type(expectation)) as exception_info:
                evaluate_delete(delete_node, state, {}, {}, [])
            assert str(expectation) in str(exception_info.value)
        else:
            evaluate_delete(delete_node, state, {}, {}, [])
            _ = state.pop("_operations_count", None)
            assert state == expectation


class TestEvaluateCondition:
    # [CLASS DOC]
    # Test group calling evaluate_condition directly on parsed comparison expressions, including pandas.
    # Inputs: no constructor args; no attributes.
    # Outputs: n/a.
    # Exceptions: none raised directly.
    # Side effects: none.
    @pytest.mark.parametrize(
        "condition, state, expected_result",
        [
            ("a == b", {"a": 1, "b": 1}, True),
            ("a == b", {"a": 1, "b": 2}, False),
            ("a != b", {"a": 1, "b": 1}, False),
            ("a != b", {"a": 1, "b": 2}, True),
            ("a < b", {"a": 1, "b": 1}, False),
            ("a < b", {"a": 1, "b": 2}, True),
            ("a < b", {"a": 2, "b": 1}, False),
            ("a <= b", {"a": 1, "b": 1}, True),
            ("a <= b", {"a": 1, "b": 2}, True),
            ("a <= b", {"a": 2, "b": 1}, False),
            ("a > b", {"a": 1, "b": 1}, False),
            ("a > b", {"a": 1, "b": 2}, False),
            ("a > b", {"a": 2, "b": 1}, True),
            ("a >= b", {"a": 1, "b": 1}, True),
            ("a >= b", {"a": 1, "b": 2}, False),
            ("a >= b", {"a": 2, "b": 1}, True),
            ("a is b", {"a": 1, "b": 1}, True),
            ("a is b", {"a": 1, "b": 2}, False),
            ("a is not b", {"a": 1, "b": 1}, False),
            ("a is not b", {"a": 1, "b": 2}, True),
            ("a in b", {"a": 1, "b": [1, 2, 3]}, True),
            ("a in b", {"a": 4, "b": [1, 2, 3]}, False),
            ("a not in b", {"a": 1, "b": [1, 2, 3]}, False),
            ("a not in b", {"a": 4, "b": [1, 2, 3]}, True),
            # Chained conditions:
            ("a == b == c", {"a": 1, "b": 1, "c": 1}, True),
            ("a == b == c", {"a": 1, "b": 2, "c": 1}, False),
            ("a == b < c", {"a": 2, "b": 2, "c": 2}, False),
            ("a == b < c", {"a": 0, "b": 0, "c": 1}, True),
        ],
    )
    def test_evaluate_condition(self, condition, state, expected_result):
        # [FUNCTION DOC]
        # Parametrized: checks every comparison operator (equality, ordering, is, in and their negations) and
        #   chained comparisons against expected booleans.
        # Inputs: self (the test class instance); condition (str), state (dict), expected_result (bool).
        # Outputs: None.
        # Exceptions: AssertionError (pytest failure) if any check fails; none raised directly otherwise.
        # Side effects: none (only local interpreter state dicts created inside the test are mutated).
        condition_ast = ast.parse(condition, mode="eval").body
        result = evaluate_condition(condition_ast, state, {}, {}, [])
        assert result == expected_result

    @pytest.mark.parametrize(
        "condition, state, expected_result",
        [
            ("a == b", {"a": pd.Series([1, 2, 3]), "b": pd.Series([2, 2, 2])}, pd.Series([False, True, False])),
            ("a != b", {"a": pd.Series([1, 2, 3]), "b": pd.Series([2, 2, 2])}, pd.Series([True, False, True])),
            ("a < b", {"a": pd.Series([1, 2, 3]), "b": pd.Series([2, 2, 2])}, pd.Series([True, False, False])),
            ("a <= b", {"a": pd.Series([1, 2, 3]), "b": pd.Series([2, 2, 2])}, pd.Series([True, True, False])),
            ("a > b", {"a": pd.Series([1, 2, 3]), "b": pd.Series([2, 2, 2])}, pd.Series([False, False, True])),
            ("a >= b", {"a": pd.Series([1, 2, 3]), "b": pd.Series([2, 2, 2])}, pd.Series([False, True, True])),
            (
                "a == b",
                {"a": pd.DataFrame({"x": [1, 2], "y": [3, 4]}), "b": pd.DataFrame({"x": [1, 2], "y": [3, 5]})},
                pd.DataFrame({"x": [True, True], "y": [True, False]}),
            ),
            (
                "a != b",
                {"a": pd.DataFrame({"x": [1, 2], "y": [3, 4]}), "b": pd.DataFrame({"x": [1, 2], "y": [3, 5]})},
                pd.DataFrame({"x": [False, False], "y": [False, True]}),
            ),
            (
                "a < b",
                {"a": pd.DataFrame({"x": [1, 2], "y": [3, 4]}), "b": pd.DataFrame({"x": [2, 2], "y": [2, 2]})},
                pd.DataFrame({"x": [True, False], "y": [False, False]}),
            ),
            (
                "a <= b",
                {"a": pd.DataFrame({"x": [1, 2], "y": [3, 4]}), "b": pd.DataFrame({"x": [2, 2], "y": [2, 2]})},
                pd.DataFrame({"x": [True, True], "y": [False, False]}),
            ),
            (
                "a > b",
                {"a": pd.DataFrame({"x": [1, 2], "y": [3, 4]}), "b": pd.DataFrame({"x": [2, 2], "y": [2, 2]})},
                pd.DataFrame({"x": [False, False], "y": [True, True]}),
            ),
            (
                "a >= b",
                {"a": pd.DataFrame({"x": [1, 2], "y": [3, 4]}), "b": pd.DataFrame({"x": [2, 2], "y": [2, 2]})},
                pd.DataFrame({"x": [False, True], "y": [True, True]}),
            ),
        ],
    )
    def test_evaluate_condition_with_pandas(self, condition, state, expected_result):
        # [FUNCTION DOC]
        # Parametrized: checks comparisons of pandas Series/DataFrames return elementwise results, compared
        #   with pandas testing helpers.
        # Inputs: self (the test class instance); condition (str), state (dict of pandas objects),
        #   expected_result (Series or DataFrame).
        # Outputs: None.
        # Exceptions: AssertionError (pytest failure) if any check fails; none raised directly otherwise.
        # Side effects: none (only local interpreter state dicts created inside the test are mutated).
        condition_ast = ast.parse(condition, mode="eval").body
        result = evaluate_condition(condition_ast, state, {}, {}, [])
        if isinstance(result, pd.Series):
            pd.testing.assert_series_equal(result, expected_result)
        else:
            pd.testing.assert_frame_equal(result, expected_result)

    @pytest.mark.parametrize(
        "condition, state, expected_exception",
        [
            # Chained conditions:
            (
                "a == b == c",
                {
                    "a": pd.Series([1, 2, 3]),
                    "b": pd.Series([2, 2, 2]),
                    "c": pd.Series([3, 3, 3]),
                },
                ValueError(
                    "The truth value of a Series is ambiguous. Use a.empty, a.bool(), a.item(), a.any() or a.all()."
                ),
            ),
            (
                "a == b == c",
                {
                    "a": pd.DataFrame({"x": [1, 2], "y": [3, 4]}),
                    "b": pd.DataFrame({"x": [2, 2], "y": [2, 2]}),
                    "c": pd.DataFrame({"x": [3, 3], "y": [3, 3]}),
                },
                ValueError(
                    "The truth value of a DataFrame is ambiguous. Use a.empty, a.bool(), a.item(), a.any() or a.all()."
                ),
            ),
        ],
    )
    def test_evaluate_condition_with_pandas_exceptions(self, condition, state, expected_exception):
        # [FUNCTION DOC]
        # Parametrized: checks chained comparisons of Series/DataFrames raise ValueError about ambiguous truth
        #   value.
        # Inputs: self (the test class instance); condition (str), state (dict), expected_exception (exception
        #   instance).
        # Outputs: None.
        # Exceptions: AssertionError (pytest failure) if any check fails; none raised directly otherwise.
        # Side effects: none (only local interpreter state dicts created inside the test are mutated).
        condition_ast = ast.parse(condition, mode="eval").body
        with pytest.raises(type(expected_exception)) as exception_info:
            _ = evaluate_condition(condition_ast, state, {}, {}, [])
        assert str(expected_exception) in str(exception_info.value)


class TestEvaluateSubscript:
    # [CLASS DOC]
    # Test group calling evaluate_subscript directly on parsed subscript expressions.
    # Inputs: no constructor args; no attributes.
    # Outputs: n/a.
    # Exceptions: none raised directly.
    # Side effects: none.
    @pytest.mark.parametrize(
        "subscript, state, expected_result",
        [
            ("dct[1]", {"dct": {1: 11, 2: 22}}, 11),
            ("dct[2]", {"dct": {1: "a", 2: "b"}}, "b"),
            ("dct['b']", {"dct": {"a": 1, "b": 2}}, 2),
            ("dct['a']", {"dct": {"a": "aa", "b": "bb"}}, "aa"),
            ("dct[1, 2]", {"dct": {(1, 2): 3}}, 3),  # tuple-index
            ("dct['a']['b']", {"dct": {"a": {"b": 1}}}, 1),  # nested
            ("lst[0]", {"lst": [1, 2, 3]}, 1),
            ("lst[-1]", {"lst": [1, 2, 3]}, 3),
            ("lst[1:3]", {"lst": [1, 2, 3, 4]}, [2, 3]),
            ("lst[:]", {"lst": [1, 2, 3]}, [1, 2, 3]),
            ("lst[::2]", {"lst": [1, 2, 3, 4]}, [1, 3]),
            ("lst[::-1]", {"lst": [1, 2, 3]}, [3, 2, 1]),
            ("tup[1]", {"tup": (1, 2, 3)}, 2),
            ("tup[-1]", {"tup": (1, 2, 3)}, 3),
            ("tup[1:3]", {"tup": (1, 2, 3, 4)}, (2, 3)),
            ("tup[:]", {"tup": (1, 2, 3)}, (1, 2, 3)),
            ("tup[::2]", {"tup": (1, 2, 3, 4)}, (1, 3)),
            ("tup[::-1]", {"tup": (1, 2, 3)}, (3, 2, 1)),
            ("st[1]", {"str": "abc"}, "b"),
            ("st[-1]", {"str": "abc"}, "c"),
            ("st[1:3]", {"str": "abcd"}, "bc"),
            ("st[:]", {"str": "abc"}, "abc"),
            ("st[::2]", {"str": "abcd"}, "ac"),
            ("st[::-1]", {"str": "abc"}, "cba"),
            ("arr[1]", {"arr": np.array([1, 2, 3])}, 2),
            ("arr[1:3]", {"arr": np.array([1, 2, 3, 4])}, np.array([2, 3])),
            ("arr[:]", {"arr": np.array([1, 2, 3])}, np.array([1, 2, 3])),
            ("arr[::2]", {"arr": np.array([1, 2, 3, 4])}, np.array([1, 3])),
            ("arr[::-1]", {"arr": np.array([1, 2, 3])}, np.array([3, 2, 1])),
            ("arr[1, 2]", {"arr": np.array([[1, 2, 3], [4, 5, 6]])}, 6),
            ("ser[1]", {"ser": pd.Series([1, 2, 3])}, 2),
            ("ser.loc[1]", {"ser": pd.Series([1, 2, 3])}, 2),
            ("ser.loc[1]", {"ser": pd.Series([1, 2, 3], index=[2, 3, 1])}, 3),
            ("ser.iloc[1]", {"ser": pd.Series([1, 2, 3])}, 2),
            ("ser.iloc[1]", {"ser": pd.Series([1, 2, 3], index=[2, 3, 1])}, 2),
            ("ser.at[1]", {"ser": pd.Series([1, 2, 3])}, 2),
            ("ser.at[1]", {"ser": pd.Series([1, 2, 3], index=[2, 3, 1])}, 3),
            ("ser.iat[1]", {"ser": pd.Series([1, 2, 3])}, 2),
            ("ser.iat[1]", {"ser": pd.Series([1, 2, 3], index=[2, 3, 1])}, 2),
            ("ser[1:3]", {"ser": pd.Series([1, 2, 3, 4])}, pd.Series([2, 3], index=[1, 2])),
            ("ser[:]", {"ser": pd.Series([1, 2, 3])}, pd.Series([1, 2, 3])),
            ("ser[::2]", {"ser": pd.Series([1, 2, 3, 4])}, pd.Series([1, 3], index=[0, 2])),
            ("ser[::-1]", {"ser": pd.Series([1, 2, 3])}, pd.Series([3, 2, 1], index=[2, 1, 0])),
            ("df['y'][1]", {"df": pd.DataFrame({"x": [1, 2], "y": [3, 4]})}, 4),
            ("df['y'][5]", {"df": pd.DataFrame({"x": [1, 2], "y": [3, 4]}, index=[5, 6])}, 3),
            ("df.loc[1, 'y']", {"df": pd.DataFrame({"x": [1, 2], "y": [3, 4]})}, 4),
            ("df.loc[5, 'y']", {"df": pd.DataFrame({"x": [1, 2], "y": [3, 4]}, index=[5, 6])}, 3),
            ("df.iloc[1, 1]", {"df": pd.DataFrame({"x": [1, 2], "y": [3, 4]})}, 4),
            ("df.iloc[1, 1]", {"df": pd.DataFrame({"x": [1, 2], "y": [3, 4]}, index=[5, 6])}, 4),
            ("df.at[1, 'y']", {"df": pd.DataFrame({"x": [1, 2], "y": [3, 4]})}, 4),
            ("df.at[5, 'y']", {"df": pd.DataFrame({"x": [1, 2], "y": [3, 4]}, index=[5, 6])}, 3),
            ("df.iat[1, 1]", {"df": pd.DataFrame({"x": [1, 2], "y": [3, 4]})}, 4),
            ("df.iat[1, 1]", {"df": pd.DataFrame({"x": [1, 2], "y": [3, 4]}, index=[5, 6])}, 4),
        ],
    )
    def test_evaluate_subscript(self, subscript, state, expected_result):
        # [FUNCTION DOC]
        # Parametrized: checks indexing/slicing of dicts (incl. tuple keys, nested), lists, tuples, strings,
        #   numpy arrays, pandas Series (incl. loc/iloc/at/iat) and DataFrames; falls back to an all() check
        #   when equality is elementwise (ValueError).
        # Inputs: self (the test class instance); subscript (str), state (dict), expected_result (any).
        # Outputs: None.
        # Exceptions: AssertionError (pytest failure) if any check fails; none raised directly otherwise.
        # Side effects: none (only local interpreter state dicts created inside the test are mutated).
        # KNOWN ISSUE: The string cases subscript a name 'st' but the state key is 'str'; they pass only
        #   through evaluate_name's difflib close-match fallback, not by direct lookup.
        subscript_ast = ast.parse(subscript).body[0].value
        result = evaluate_subscript(subscript_ast, state, {}, {}, [])
        try:
            assert result == expected_result
        except ValueError:
            assert (result == expected_result).all()

    @pytest.mark.parametrize(
        "subscript, state, expected_error_message",
        [
            ("dct['a']", {"dct": {}}, "KeyError: 'a'"),
            ("dct[0]", {"dct": {}}, "KeyError: 0"),
            ("dct['c']", {"dct": {"a": 1, "b": 2}}, "KeyError: 'c'"),
            ("dct[1, 2, 3]", {"dct": {(1, 2): 3}}, "KeyError: (1, 2, 3)"),
            ("lst[0]", {"lst": []}, "IndexError: list index out of range"),
            ("lst[3]", {"lst": [1, 2, 3]}, "IndexError: list index out of range"),
            ("lst[-4]", {"lst": [1, 2, 3]}, "IndexError: list index out of range"),
            ("value[0]", {"value": 1}, "TypeError: 'int' object is not subscriptable"),
        ],
    )
    def test_evaluate_subscript_error(self, subscript, state, expected_error_message):
        # [FUNCTION DOC]
        # Parametrized: checks missing keys, out-of-range indices and non-subscriptable values raise
        #   InterpreterError 'Could not index' with the underlying error text.
        # Inputs: self (the test class instance); subscript (str), state (dict), expected_error_message (str).
        # Outputs: None.
        # Exceptions: AssertionError (pytest failure) if any check fails; none raised directly otherwise.
        # Side effects: none (only local interpreter state dicts created inside the test are mutated).
        subscript_ast = ast.parse(subscript).body[0].value
        with pytest.raises(InterpreterError, match="Could not index") as exception_info:
            _ = evaluate_subscript(subscript_ast, state, {}, {}, [])
        assert expected_error_message in str(exception_info.value)

    @pytest.mark.parametrize(
        "subscriptable_class, expectation",
        [
            (True, 20),
            (False, InterpreterError("TypeError: 'Custom' object is not subscriptable")),
        ],
    )
    def test_evaluate_subscript_with_custom_class(self, subscriptable_class, expectation):
        # [FUNCTION DOC]
        # Parametrized: defines a local class with or without getitem, and checks subscripting returns key
        #   times 10 or raises InterpreterError mentioning 'not subscriptable'.
        # Inputs: self (the test class instance); subscriptable_class (bool), expectation (int or exception
        #   instance).
        # Outputs: None.
        # Exceptions: AssertionError (pytest failure) if any check fails; none raised directly otherwise.
        # Side effects: none (only local interpreter state dicts created inside the test are mutated).
        if subscriptable_class:

            class Custom:
                # [CLASS DOC]
                # HELPER: Local class used as a subscript target. Two variants share this name: one with a
                #   getitem method (subscriptable) and one empty class (not subscriptable).
                # Inputs: no constructor args; no attributes.
                # Outputs: n/a.
                # Exceptions: none raised directly.
                # Side effects: none.
                def __getitem__(self, key):
                    # [FUNCTION DOC]
                    # HELPER: Returns the key multiplied by 10.
                    # Inputs: self; key (number).
                    # Outputs: key times 10.
                    # Exceptions: TypeError propagates if key does not support multiplication.
                    # Side effects: none.
                    return key * 10
        else:

            class Custom:
                # [CLASS DOC]
                # HELPER: Local class used as a subscript target. Two variants share this name: one with a
                #   getitem method (subscriptable) and one empty class (not subscriptable).
                # Inputs: no constructor args; no attributes.
                # Outputs: n/a.
                # Exceptions: none raised directly.
                # Side effects: none.
                pass

        state = {"obj": Custom()}
        subscript = "obj[2]"
        subscript_ast = ast.parse(subscript).body[0].value
        if isinstance(expectation, Exception):
            with pytest.raises(type(expectation), match="Could not index") as exception_info:
                evaluate_subscript(subscript_ast, state, {}, {}, [])
            assert "TypeError: 'Custom' object is not subscriptable" in str(exception_info.value)
        else:
            result = evaluate_subscript(subscript_ast, state, {}, {}, [])
            assert result == expectation


def test_get_safe_module_handle_lazy_imports():
    # [FUNCTION DOC]
    # Checks get_safe_module skips attributes whose access raises ImportError (lazy imports) while copying
    #   normal attributes.
    # Inputs: none.
    # Outputs: None.
    # Exceptions: AssertionError (pytest failure) if any check fails; none raised directly otherwise.
    # Side effects: none (only local interpreter state dicts created inside the test are mutated).
    class FakeModule(types.ModuleType):
        # [CLASS DOC]
        # HELPER: Module subclass simulating a lazy-import module: lists 'lazy_attribute' in dir but raises
        #   ImportError on access.
        # Inputs: constructor arg name (str module name); attribute non_lazy_attribute set to 'ok'.
        # Outputs: n/a.
        # Exceptions: none raised directly.
        # Side effects: none.
        def __init__(self, name):
            # [FUNCTION DOC]
            # HELPER: Initializes the module with the given name and sets non_lazy_attribute to 'ok'.
            # Inputs: self; name (str).
            # Outputs: None.
            # Exceptions: none raised directly.
            # Side effects: sets an attribute on self.
            super().__init__(name)
            self.non_lazy_attribute = "ok"

        def __getattr__(self, name):
            # [FUNCTION DOC]
            # HELPER: Fallback attribute lookup: raises ImportError for 'lazy_attribute', otherwise defers to
            #   the parent.
            # Inputs: self; name (str attribute name).
            # Outputs: never returns normally in practice.
            # Exceptions: ImportError for 'lazy_attribute'; AttributeError for anything else.
            # Side effects: none.
            # KNOWN ISSUE: ModuleType defines no getattr fallback, so the super call itself raises
            #   AttributeError (about the super object) rather than a normal missing-attribute error; harmless
            #   here.
            if name == "lazy_attribute":
                raise ImportError("lazy import failure")
            return super().__getattr__(name)

        def __dir__(self):
            # [FUNCTION DOC]
            # HELPER: Returns the normal dir listing plus 'lazy_attribute'.
            # Inputs: self.
            # Outputs: list of attribute names.
            # Exceptions: none raised directly.
            # Side effects: none.
            return super().__dir__() + ["lazy_attribute"]

    fake_module = FakeModule("fake_module")
    safe_module = get_safe_module(fake_module, authorized_imports=set())
    assert not hasattr(safe_module, "lazy_attribute")
    assert getattr(safe_module, "non_lazy_attribute") == "ok"


class TestPrintContainer:
    # [CLASS DOC]
    # Test group for PrintContainer, the string accumulator used for captured print output.
    # Inputs: no constructor args; no attributes.
    # Outputs: n/a.
    # Exceptions: none raised directly.
    # Side effects: none.
    def test_initial_value(self):
        # [FUNCTION DOC]
        # Checks a new PrintContainer has an empty value.
        # Inputs: self (the test class instance).
        # Outputs: None.
        # Exceptions: AssertionError (pytest failure) if any check fails; none raised directly otherwise.
        # Side effects: none (only local interpreter state dicts created inside the test are mutated).
        pc = PrintContainer()
        assert pc.value == ""

    def test_append(self):
        # [FUNCTION DOC]
        # Checks append adds text to value.
        # Inputs: self (the test class instance).
        # Outputs: None.
        # Exceptions: AssertionError (pytest failure) if any check fails; none raised directly otherwise.
        # Side effects: none (only local interpreter state dicts created inside the test are mutated).
        pc = PrintContainer()
        pc.append("Hello")
        assert pc.value == "Hello"

    def test_iadd(self):
        # [FUNCTION DOC]
        # Checks in-place add appends text and returns the container.
        # Inputs: self (the test class instance).
        # Outputs: None.
        # Exceptions: AssertionError (pytest failure) if any check fails; none raised directly otherwise.
        # Side effects: none (only local interpreter state dicts created inside the test are mutated).
        pc = PrintContainer()
        pc += "World"
        assert pc.value == "World"

    def test_str(self):
        # [FUNCTION DOC]
        # Checks str returns the accumulated text.
        # Inputs: self (the test class instance).
        # Outputs: None.
        # Exceptions: AssertionError (pytest failure) if any check fails; none raised directly otherwise.
        # Side effects: none (only local interpreter state dicts created inside the test are mutated).
        pc = PrintContainer()
        pc.append("Hello")
        assert str(pc) == "Hello"

    def test_repr(self):
        # [FUNCTION DOC]
        # Checks repr wraps the text as PrintContainer(text).
        # Inputs: self (the test class instance).
        # Outputs: None.
        # Exceptions: AssertionError (pytest failure) if any check fails; none raised directly otherwise.
        # Side effects: none (only local interpreter state dicts created inside the test are mutated).
        pc = PrintContainer()
        pc.append("Hello")
        assert repr(pc) == "PrintContainer(Hello)"

    def test_len(self):
        # [FUNCTION DOC]
        # Checks len is the length of the accumulated text.
        # Inputs: self (the test class instance).
        # Outputs: None.
        # Exceptions: AssertionError (pytest failure) if any check fails; none raised directly otherwise.
        # Side effects: none (only local interpreter state dicts created inside the test are mutated).
        pc = PrintContainer()
        pc.append("Hello")
        assert len(pc) == 5


def test_fix_final_answer_code():
    # [FUNCTION DOC]
    # Checks fix_final_answer_code renames variables named final_answer to final_answer_variable (in
    #   assignments and uses, including inside functions and after semicolons) while leaving calls and
    #   attribute assignments untouched.
    # Inputs: none.
    # Outputs: None.
    # Exceptions: AssertionError (pytest failure) if any check fails; none raised directly otherwise.
    # Side effects: none (only local interpreter state dicts created inside the test are mutated).
    test_cases = [
        (
            "final_answer = 3.21\nfinal_answer(final_answer)",
            "final_answer_variable = 3.21\nfinal_answer(final_answer_variable)",
        ),
        (
            "x = final_answer(5)\nfinal_answer = x + 1\nfinal_answer(final_answer)",
            "x = final_answer(5)\nfinal_answer_variable = x + 1\nfinal_answer(final_answer_variable)",
        ),
        (
            "def func():\n    final_answer = 42\n    return final_answer(final_answer)",
            "def func():\n    final_answer_variable = 42\n    return final_answer(final_answer_variable)",
        ),
        (
            "final_answer(5)  # Should not change function calls",
            "final_answer(5)  # Should not change function calls",
        ),
        (
            "obj.final_answer = 5  # Should not change object attributes",
            "obj.final_answer = 5  # Should not change object attributes",
        ),
        (
            "final_answer=3.21;final_answer(final_answer)",
            "final_answer_variable=3.21;final_answer(final_answer_variable)",
        ),
    ]

    for i, (input_code, expected) in enumerate(test_cases, 1):
        result = fix_final_answer_code(input_code)
        assert result == expected, f"""
Test case {i} failed:
Input:    {input_code}
Expected: {expected}
Got:      {result}
"""


class TestTimeout:
    # [CLASS DOC]
    # Test group for the timeout decorator and the timeout_seconds option of evaluate_python_code and
    #   LocalPythonExecutor.
    # Inputs: no constructor args; no attributes.
    # Outputs: n/a.
    # Exceptions: none raised directly.
    # Side effects: tests sleep for up to a few seconds.
    """Test the timeout mechanism for code execution."""

    def test_timeout_decorator_completes_within_limit(self):
        # [FUNCTION DOC]
        # Checks a function decorated with a 2s timeout that sleeps 0.1s returns its value.
        # Inputs: self (the test class instance).
        # Outputs: None.
        # Exceptions: AssertionError (pytest failure) if any check fails; none raised directly otherwise.
        # Side effects: sleeps (wall clock delay).
        """Test that code completing within the timeout limit works correctly."""

        @timeout(2)
        def short_task():
            # [FUNCTION DOC]
            # HELPER: Sleeps 0.1s and returns 'completed'.
            # Inputs: none.
            # Outputs: the string 'completed'.
            # Exceptions: ExecutionTimeoutError via the decorator if it overruns.
            # Side effects: sleeps (wall clock delay).
            time.sleep(0.1)
            return "completed"

        assert short_task() == "completed"

    def test_timeout_decorator_raises_error_when_exceeded(self):
        # [FUNCTION DOC]
        # Checks a function decorated with a 1s timeout that sleeps 2s raises ExecutionTimeoutError.
        # Inputs: self (the test class instance).
        # Outputs: None.
        # Exceptions: AssertionError (pytest failure) if any check fails; none raised directly otherwise.
        # Side effects: sleeps (wall clock delay).
        """Test that code exceeding the timeout limit raises ExecutionTimeoutError."""

        @timeout(1)
        def long_task():
            # [FUNCTION DOC]
            # HELPER: Sleeps 2s and would return a sentinel string.
            # Inputs: none.
            # Outputs: the string 'should not complete' (never reached under the 1s timeout).
            # Exceptions: ExecutionTimeoutError via the decorator.
            # Side effects: sleeps (wall clock delay).
            time.sleep(2)
            return "should not complete"

        with pytest.raises(ExecutionTimeoutError, match="Code execution exceeded the maximum execution time"):
            long_task()

    def test_evaluate_python_code_with_timeout_completes(self):
        # [FUNCTION DOC]
        # Checks quick code completes under the default timeout and returns 4.
        # Inputs: self (the test class instance).
        # Outputs: None.
        # Exceptions: AssertionError (pytest failure) if any check fails; none raised directly otherwise.
        # Side effects: none (only local interpreter state dicts created inside the test are mutated).
        # KNOWN ISSUE: Does not exercise the timeout at all; it is a plain evaluation check.
        """Test that evaluate_python_code completes within timeout for quick code."""
        code = "result = 2 + 2"
        result, is_final = evaluate_python_code(code)
        assert result == 4

    def test_evaluate_python_code_with_timeout_raises(self):
        # [FUNCTION DOC]
        # Checks code sleeping 3s with a 2s timeout raises ExecutionTimeoutError.
        # Inputs: self (the test class instance).
        # Outputs: None.
        # Exceptions: AssertionError (pytest failure) if any check fails; none raised directly otherwise.
        # Side effects: sleeps (wall clock delay).
        """Test that evaluate_python_code raises timeout error for long-running code."""
        # Use a short custom timeout (2 seconds) with longer sleep (3 seconds) to test quickly
        code = """
import time
time.sleep(3)
result = "should not complete"
"""
        with pytest.raises(ExecutionTimeoutError, match="Code execution exceeded the maximum execution time"):
            evaluate_python_code(code, authorized_imports=["time"], timeout_seconds=2)

    def test_timeout_works_in_thread(self):
        # [FUNCTION DOC]
        # Runs evaluation in a non-main thread and checks both normal execution and a timeout work there; the
        #   worker records results in a shared dict which is asserted after join (10s).
        # Inputs: self (the test class instance).
        # Outputs: None.
        # Exceptions: AssertionError (pytest failure) if any check fails; none raised directly otherwise.
        # Side effects: starts and joins a background thread; sleeps.
        """Test that timeout mechanism works when called from a non-main thread.

        This verifies the fix for the issue where signal-based timeouts failed
        in threads (signals only work in the main thread).
        """
        import threading

        result = {"success": False, "error": None}

        def run_in_thread():
            # [FUNCTION DOC]
            # HELPER: Thread target: evaluates quick code, then code that should time out, recording success
            #   or an error description in the enclosing result dict.
            # Inputs: none (closes over result).
            # Outputs: None.
            # Exceptions: none escape (all exceptions are caught and recorded).
            # Side effects: mutates the enclosing result dict; sleeps.
            try:
                # Quick code should work
                code = "result = 42"
                res, _ = evaluate_python_code(code)
                assert res == 42

                # Timeout should still work in thread - use short timeout for fast test
                timeout_code = """
import time
time.sleep(3)
"""
                try:
                    evaluate_python_code(timeout_code, authorized_imports=["time"], timeout_seconds=2)
                    result["error"] = "Code should have timed out but didn't"
                except ExecutionTimeoutError:
                    result["success"] = True
            except Exception as e:
                result["error"] = f"{type(e).__name__}: {e}"

        thread = threading.Thread(target=run_in_thread)
        thread.start()
        thread.join(timeout=10)

        assert not thread.is_alive(), "Thread should have completed"
        assert result["error"] is None, f"Error in thread: {result['error']}"
        assert result["success"], "Timeout should have been raised in thread"

    def test_custom_timeout_value(self):
        # [FUNCTION DOC]
        # Checks a 1s timeout_seconds triggers ExecutionTimeoutError mentioning 1 for a 2s sleep.
        # Inputs: self (the test class instance).
        # Outputs: None.
        # Exceptions: AssertionError (pytest failure) if any check fails; none raised directly otherwise.
        # Side effects: sleeps (wall clock delay).
        """Test that a custom timeout value can be specified."""
        # Code that sleeps for 2 seconds should timeout with 1-second limit
        code = """
import time
time.sleep(2)
"""
        with pytest.raises(ExecutionTimeoutError, match="Code execution exceeded the maximum execution time of 1"):
            evaluate_python_code(code, authorized_imports=["time"], timeout_seconds=1)

    def test_longer_timeout_value(self):
        # [FUNCTION DOC]
        # Checks a 5s timeout lets a 2s sleep complete.
        # Inputs: self (the test class instance).
        # Outputs: None.
        # Exceptions: AssertionError (pytest failure) if any check fails; none raised directly otherwise.
        # Side effects: sleeps (wall clock delay).
        """Test that a longer custom timeout value allows longer execution."""
        # Code that sleeps for 2 seconds should complete with 5-second limit
        code = """
import time
time.sleep(2)
result = "completed"
"""
        result, is_final = evaluate_python_code(code, authorized_imports=["time"], timeout_seconds=5)
        assert result == "completed"

    def test_disabled_timeout(self):
        # [FUNCTION DOC]
        # Checks timeout_seconds None disables the timeout.
        # Inputs: self (the test class instance).
        # Outputs: None.
        # Exceptions: AssertionError (pytest failure) if any check fails; none raised directly otherwise.
        # Side effects: sleeps (wall clock delay).
        """Test that timeout can be disabled by setting it to None."""
        # Even slow code should complete when timeout is disabled
        # Using a shorter sleep to keep test fast, but demonstrating None works
        code = """
import time
time.sleep(0.5)
result = "completed without timeout"
"""
        result, is_final = evaluate_python_code(code, authorized_imports=["time"], timeout_seconds=None)
        assert result == "completed without timeout"

    def test_local_executor_custom_timeout(self):
        # [FUNCTION DOC]
        # Checks LocalPythonExecutor with timeout_seconds 1 raises ExecutionTimeoutError for a 2s sleep.
        # Inputs: self (the test class instance).
        # Outputs: None.
        # Exceptions: AssertionError (pytest failure) if any check fails; none raised directly otherwise.
        # Side effects: sleeps (wall clock delay).
        """Test that LocalPythonExecutor respects custom timeout."""
        executor = LocalPythonExecutor(additional_authorized_imports=["time"], timeout_seconds=1)
        executor.send_tools({})

        # Code that sleeps for 2 seconds should timeout with 1-second executor limit
        code = """
import time
time.sleep(2)
"""
        with pytest.raises(ExecutionTimeoutError, match="Code execution exceeded the maximum execution time of 1"):
            executor(code)

    def test_local_executor_disabled_timeout(self):
        # [FUNCTION DOC]
        # Checks LocalPythonExecutor with timeout disabled returns the last value of slow code.
        # Inputs: self (the test class instance).
        # Outputs: None.
        # Exceptions: AssertionError (pytest failure) if any check fails; none raised directly otherwise.
        # Side effects: sleeps (wall clock delay).
        """Test that LocalPythonExecutor can disable timeout."""
        executor = LocalPythonExecutor(additional_authorized_imports=["time"], timeout_seconds=None)
        executor.send_tools({})

        # Code should complete even without timeout
        code = """
import time
time.sleep(0.5)
result = "completed"
"""
        output = executor(code)
        assert output.output == "completed"


@pytest.mark.parametrize(
    "module,authorized_imports,expected",
    [
        ("os", ["other", "*"], True),
        ("AnyModule", ["*"], True),
        ("os", ["os"], True),
        ("AnyModule", ["AnyModule"], True),
        ("Module.os", ["Module"], False),
        ("Module.os", ["Module", "Module.os"], True),
        ("os.path", ["os.*"], True),
        ("os", ["os.path"], True),
    ],
)
def test_check_import_authorized(module: str, authorized_imports: list[str], expected: bool):
    # [FUNCTION DOC]
    # Parametrized: checks check_import_authorized for wildcards, exact names, parent-only authorization (not
    #   enough for a submodule) and 'pkg.*' patterns; also a submodule authorization allowing its parent.
    # Inputs: module (str), authorized_imports (list of str), expected (bool).
    # Outputs: None.
    # Exceptions: AssertionError (pytest failure) if any check fails; none raised directly otherwise.
    # Side effects: none (only local interpreter state dicts created inside the test are mutated).
    assert check_import_authorized(module, authorized_imports) == expected


class TestLocalPythonExecutor:
    # [CLASS DOC]
    # Test group for LocalPythonExecutor construction and calling (state, tools, assignments, isinstance).
    # Inputs: no constructor args; no attributes.
    # Outputs: n/a.
    # Exceptions: none raised directly.
    # Side effects: none.
    @pytest.mark.parametrize(
        "additional_authorized_imports, should_raise",
        [
            # Valid imports
            (["math"], None),
            (["math", "os"], None),  # Multiple valid imports
            ([], None),  # Empty list of imports
            (["*"], None),  # Wildcard allows all imports
            (["os.*"], None),  # Submodule wildcard
            # Invalid imports
            (["i_do_not_exist"], True),  # Non-existent module
            (["math", "i_do_not_exist"], True),  # Mix of valid and invalid
            (["i_do_not_exist.*"], True),  # Non-existent module with wildcard
        ],
    )
    def test_additional_authorized_imports_are_installed(self, additional_authorized_imports, should_raise):
        # [FUNCTION DOC]
        # Parametrized: checks constructing LocalPythonExecutor raises InterpreterError 'Non-installed
        #   authorized modules' when an authorized module (with or without wildcard) is not installed, and
        #   succeeds otherwise.
        # Inputs: self (the test class instance); additional_authorized_imports (list of str), should_raise
        #   (True or None).
        # Outputs: None.
        # Exceptions: AssertionError (pytest failure) if any check fails; none raised directly otherwise.
        # Side effects: none (only local interpreter state dicts created inside the test are mutated).
        expectation = (
            pytest.raises(InterpreterError, match="Non-installed authorized modules")
            if should_raise
            else does_not_raise()
        )
        with expectation:
            LocalPythonExecutor(additional_authorized_imports=additional_authorized_imports)

    def test_state_name(self):
        # [FUNCTION DOC]
        # Checks the executor state defines __name__ as '__main__'.
        # Inputs: self (the test class instance).
        # Outputs: None.
        # Exceptions: AssertionError (pytest failure) if any check fails; none raised directly otherwise.
        # Side effects: none (only local interpreter state dicts created inside the test are mutated).
        executor = LocalPythonExecutor(additional_authorized_imports=[])
        assert executor.state.get("__name__") == "__main__"

    @pytest.mark.parametrize(
        "code",
        [
            "d = {'func': lambda x: x + 10}; func = d['func']; func(1)",
            "d = {'func': lambda x: x + 10}; d['func'](1)",
        ],
    )
    def test_call_from_dict(self, code):
        # [FUNCTION DOC]
        # Parametrized: checks calling a lambda stored in a dict, directly or via a variable, returns 11.
        # Inputs: self (the test class instance); code (str).
        # Outputs: None.
        # Exceptions: AssertionError (pytest failure) if any check fails; none raised directly otherwise.
        # Side effects: none (only local interpreter state dicts created inside the test are mutated).
        executor = LocalPythonExecutor([])
        result = executor(code).output
        assert result == 11

    @pytest.mark.parametrize(
        "code",
        [
            "a = b = 1; a",
            "a = b = 1; b",
            "a, b = c, d = 1, 1; a",
            "a, b = c, d = 1, 1; b",
            "a, b = c, d = 1, 1; c",
            "a, b = c, d = {1, 2}; a",
            "a, b = c, d = {1, 2}; c",
            "a, b = c, d = {1: 10, 2: 20}; a",
            "a, b = c, d = {1: 10, 2: 20}; c",
            "a = b = (lambda: 1)(); b",
            "a = b = (lambda: 1)(); lambda x: 10; b",
            "a = b = (lambda x: lambda y: x + y)(0)(1); b",
            dedent("""
            def foo():
                return 1;
            a = b = foo(); b"""),
            dedent("""
            def foo(*args, **kwargs):
                return sum(args)
            a = b = foo(1,-1,1); b"""),
            "a, b = 1, 2; a, b = b, a; b",
        ],
    )
    def test_chained_assignments(self, code):
        # [FUNCTION DOC]
        # Parametrized: checks chained assignments (multiple targets, tuple unpacking from tuples/sets/dicts,
        #   lambdas, function calls, swaps) all yield 1.
        # Inputs: self (the test class instance); code (str).
        # Outputs: None.
        # Exceptions: AssertionError (pytest failure) if any check fails; none raised directly otherwise.
        # Side effects: none (only local interpreter state dicts created inside the test are mutated).
        executor = LocalPythonExecutor([])
        executor.send_tools({})
        result = executor(code).output
        assert result == 1

    def test_evaluate_assign_error(self):
        # [FUNCTION DOC]
        # Checks unpacking three values into two names raises InterpreterError 'Cannot unpack tuple of wrong
        #   size'.
        # Inputs: self (the test class instance).
        # Outputs: None.
        # Exceptions: AssertionError (pytest failure) if any check fails; none raised directly otherwise.
        # Side effects: none (only local interpreter state dicts created inside the test are mutated).
        code = "a, b = 1, 2, 3; a"
        executor = LocalPythonExecutor([])
        with pytest.raises(InterpreterError, match=".*Cannot unpack tuple of wrong size"):
            executor(code)

    def test_function_def_recovers_source_code(self):
        # [FUNCTION DOC]
        # Checks a function returned via final_answer carries its name and unparsed __source__.
        # Inputs: self (the test class instance).
        # Outputs: None.
        # Exceptions: AssertionError (pytest failure) if any check fails; none raised directly otherwise.
        # Side effects: none (only local interpreter state dicts created inside the test are mutated).
        # KNOWN ISSUE: Exact duplicate of TestEvaluatePythonCode.test_function_def_recovers_source_code.
        executor = LocalPythonExecutor([])
        executor.send_tools({"final_answer": FinalAnswerTool()})
        res = executor(
            dedent(
                """
                def target_function():
                    return "Hello world"

                final_answer(target_function)
                """
            )
        ).output
        assert res.__name__ == "target_function"
        assert res.__source__ == "def target_function():\n    return 'Hello world'"

    @pytest.mark.parametrize(
        "code, expected_result",
        [("isinstance(5, int)", True), ("isinstance('foo', str)", True), ("isinstance(5, str)", False)],
    )
    def test_isinstance_builtin_type(self, code, expected_result):
        # [FUNCTION DOC]
        # Parametrized: checks isinstance with builtin types returns the exact bool singleton.
        # Inputs: self (the test class instance); code (str), expected_result (bool).
        # Outputs: None.
        # Exceptions: AssertionError (pytest failure) if any check fails; none raised directly otherwise.
        # Side effects: none (only local interpreter state dicts created inside the test are mutated).
        executor = LocalPythonExecutor([])
        executor.send_tools({})
        result = executor(code).output
        assert result is expected_result


class TestLocalPythonExecutorSecurity:
    # [CLASS DOC]
    # Test group of sandbox-escape regression tests: imports, dangerous functions/modules, access through sys,
    #   importlib, submodules, indirect getattr, frames/tracebacks, subclass walking and dunder access/calls.
    #   Most cases expect a specific InterpreterError; 'None' cases check the path works once explicitly
    #   allowed.
    # Inputs: no constructor args; no attributes.
    # Outputs: n/a.
    # Exceptions: none raised directly.
    # Side effects: allowed-path cases execute real os.system(':') or spawn a harmless 'sh -c :' subprocess.
    @pytest.mark.parametrize(
        "additional_authorized_imports, expected_error",
        [([], InterpreterError("Import of os is not allowed")), (["os"], None)],
    )
    def test_vulnerability_import(self, additional_authorized_imports, expected_error):
        # [FUNCTION DOC]
        # Checks importing os is blocked unless os is authorized.
        # Inputs: self (the test class instance); additional_authorized_imports (list of str), expected_error
        #   (InterpreterError instance or None).
        # Outputs: None.
        # Exceptions: AssertionError (pytest failure) if any check fails; none raised directly otherwise.
        # Side effects: none (only local interpreter state dicts created inside the test are mutated).
        executor = LocalPythonExecutor(additional_authorized_imports)
        with (
            pytest.raises(type(expected_error), match=f".*{expected_error}")
            if isinstance(expected_error, Exception)
            else does_not_raise()
        ):
            executor("import os")

    @pytest.mark.parametrize(
        "additional_authorized_imports, expected_error",
        [([], InterpreterError("Import of builtins is not allowed")), (["builtins"], None)],
    )
    def test_vulnerability_builtins(self, additional_authorized_imports, expected_error):
        # [FUNCTION DOC]
        # Checks importing builtins is blocked unless authorized.
        # Inputs: self (the test class instance); additional_authorized_imports (list of str), expected_error
        #   (InterpreterError instance or None).
        # Outputs: None.
        # Exceptions: AssertionError (pytest failure) if any check fails; none raised directly otherwise.
        # Side effects: none (only local interpreter state dicts created inside the test are mutated).
        executor = LocalPythonExecutor(additional_authorized_imports)
        with (
            pytest.raises(type(expected_error), match=f".*{expected_error}")
            if isinstance(expected_error, Exception)
            else does_not_raise()
        ):
            executor("import builtins")

    @pytest.mark.parametrize(
        "additional_authorized_imports, expected_error",
        [([], InterpreterError("Import of builtins is not allowed")), (["builtins"], None)],
    )
    def test_vulnerability_builtins_safe_functions(self, additional_authorized_imports, expected_error):
        # [FUNCTION DOC]
        # Checks calling builtins.print through an imported builtins module is blocked unless builtins is
        #   authorized.
        # Inputs: self (the test class instance); additional_authorized_imports (list of str), expected_error
        #   (InterpreterError instance or None).
        # Outputs: None.
        # Exceptions: AssertionError (pytest failure) if any check fails; none raised directly otherwise.
        # Side effects: may print 1 to stdout when allowed.
        executor = LocalPythonExecutor(additional_authorized_imports)
        with (
            pytest.raises(type(expected_error), match=f".*{expected_error}")
            if isinstance(expected_error, Exception)
            else does_not_raise()
        ):
            executor("import builtins; builtins.print(1)")

    @pytest.mark.parametrize(
        "additional_authorized_imports, additional_tools, expected_error",
        [
            ([], [], InterpreterError("Import of builtins is not allowed")),
            (["builtins"], [], InterpreterError("Forbidden access to function: exec")),
            (["builtins"], ["exec"], None),
        ],
    )
    def test_vulnerability_builtins_dangerous_functions(
        self, additional_authorized_imports, additional_tools, expected_error
    ):
        # [FUNCTION DOC]
        # Checks accessing builtins.exec requires both authorizing builtins and sending exec as a tool.
        # Inputs: self (the test class instance); additional_authorized_imports (list of str),
        #   additional_tools (list of tool names to send), expected_error (InterpreterError instance or None).
        # Outputs: None.
        # Exceptions: AssertionError (pytest failure) if any check fails; none raised directly otherwise.
        # Side effects: none (only local interpreter state dicts created inside the test are mutated).
        executor = LocalPythonExecutor(additional_authorized_imports)
        if additional_tools:
            from builtins import exec

            executor.send_tools({"exec": exec})
        with (
            pytest.raises(type(expected_error), match=f".*{expected_error}")
            if isinstance(expected_error, Exception)
            else does_not_raise()
        ):
            executor("import builtins; builtins.exec")

    @pytest.mark.parametrize(
        "additional_authorized_imports, additional_tools, expected_error",
        [
            ([], [], InterpreterError("Import of os is not allowed")),
            (["os"], [], InterpreterError("Forbidden access to function: popen")),
            (["os"], ["popen"], None),
        ],
    )
    def test_vulnerability_dangerous_functions(self, additional_authorized_imports, additional_tools, expected_error):
        # [FUNCTION DOC]
        # Checks accessing os.popen requires authorizing os and sending popen as a tool.
        # Inputs: self (the test class instance); additional_authorized_imports (list of str),
        #   additional_tools (list of tool names to send), expected_error (InterpreterError instance or None).
        # Outputs: None.
        # Exceptions: AssertionError (pytest failure) if any check fails; none raised directly otherwise.
        # Side effects: none (only local interpreter state dicts created inside the test are mutated).
        executor = LocalPythonExecutor(additional_authorized_imports)
        if additional_tools:
            from os import popen

            executor.send_tools({"popen": popen})
        with (
            pytest.raises(type(expected_error), match=f".*{expected_error}")
            if isinstance(expected_error, Exception)
            else does_not_raise()
        ):
            executor("import os; os.popen")

    @pytest.mark.parametrize("dangerous_function", DANGEROUS_FUNCTIONS)
    def test_vulnerability_for_all_dangerous_functions(self, dangerous_function):
        # [FUNCTION DOC]
        # Parametrized over DANGEROUS_FUNCTIONS: with the owning module authorized, checks accessing the
        #   function raises 'Forbidden access to function' (or 'dunder attribute' for dunder names); skips
        #   uninstalled modules.
        # Inputs: self (the test class instance); dangerous_function (str dotted path).
        # Outputs: None.
        # Exceptions: AssertionError (pytest failure) if any check fails; none raised directly otherwise.
        #   pytest.skip if the module is missing.
        # Side effects: none (only local interpreter state dicts created inside the test are mutated).
        dangerous_module_name, dangerous_function_name = dangerous_function.rsplit(".", 1)
        # Skip test if module is not installed: posix module is not installed on Windows
        pytest.importorskip(dangerous_module_name)
        executor = LocalPythonExecutor([dangerous_module_name])
        if "__" in dangerous_function_name:
            error_match = f".*Forbidden access to dunder attribute: {dangerous_function_name}"
        else:
            error_match = f".*Forbidden access to function: {dangerous_function_name}.*"
        with pytest.raises(InterpreterError, match=error_match):
            executor(f"import {dangerous_module_name}; {dangerous_function}")

    @pytest.mark.parametrize(
        "additional_authorized_imports, expected_error",
        [
            ([], InterpreterError("Import of sys is not allowed")),
            (["sys"], InterpreterError("Forbidden access to module: os")),
            (["sys", "os"], None),
        ],
    )
    def test_vulnerability_via_sys(self, additional_authorized_imports, expected_error):
        # [FUNCTION DOC]
        # Checks reaching os through sys.modules is blocked unless both sys and os are authorized.
        # Inputs: self (the test class instance); additional_authorized_imports (list of str), expected_error
        #   (InterpreterError instance or None).
        # Outputs: None.
        # Exceptions: AssertionError (pytest failure) if any check fails; none raised directly otherwise.
        # Side effects: when the case expects no error, runs os.system with a no-op command in a shell.
        executor = LocalPythonExecutor(additional_authorized_imports)
        with (
            pytest.raises(type(expected_error), match=f".*{expected_error}")
            if isinstance(expected_error, Exception)
            else does_not_raise()
        ):
            executor(
                dedent(
                    """
                    import sys
                    sys.modules["os"].system(":")
                    """
                )
            )

    @pytest.mark.parametrize("dangerous_module", DANGEROUS_MODULES)
    def test_vulnerability_via_sys_for_all_dangerous_modules(self, dangerous_module):
        # [FUNCTION DOC]
        # Parametrized over DANGEROUS_MODULES: with sys authorized, checks fetching the module from
        #   sys.modules raises 'Forbidden access to module'; skips sys itself and modules not loaded.
        # Inputs: self (the test class instance); dangerous_module (str).
        # Outputs: None.
        # Exceptions: AssertionError (pytest failure) if any check fails; none raised directly otherwise.
        #   pytest.skip when not applicable.
        # Side effects: none (only local interpreter state dicts created inside the test are mutated).
        import sys

        if dangerous_module not in sys.modules or dangerous_module == "sys":
            pytest.skip("module not present in sys.modules")
        executor = LocalPythonExecutor(["sys"])
        with pytest.raises(InterpreterError) as exception_info:
            executor(
                dedent(
                    f"""
                    import sys
                    sys.modules["{dangerous_module}"]
                    """
                )
            )
        assert f"Forbidden access to module: {dangerous_module}" in str(exception_info.value)

    @pytest.mark.parametrize(
        "additional_authorized_imports, expected_error",
        [(["importlib"], InterpreterError("Forbidden access to module: os")), (["importlib", "os"], None)],
    )
    def test_vulnerability_via_importlib(self, additional_authorized_imports, expected_error):
        # [FUNCTION DOC]
        # Checks importlib.import_module of os is blocked unless os is authorized too.
        # Inputs: self (the test class instance); additional_authorized_imports (list of str), expected_error
        #   (InterpreterError instance or None).
        # Outputs: None.
        # Exceptions: AssertionError (pytest failure) if any check fails; none raised directly otherwise.
        # Side effects: when the case expects no error, runs os.system with a no-op command in a shell.
        executor = LocalPythonExecutor(additional_authorized_imports)
        with (
            pytest.raises(type(expected_error), match=f".*{expected_error}")
            if isinstance(expected_error, Exception)
            else does_not_raise()
        ):
            executor(
                dedent(
                    """
                    import importlib
                    importlib.import_module("os").system(":")
                    """
                )
            )

    @pytest.mark.parametrize(
        "code, additional_authorized_imports, expected_error",
        [
            # os submodule
            (
                "import queue; queue.threading._os.system(':')",
                [],
                InterpreterError("Forbidden access to module: threading"),
            ),
            (
                "import queue; queue.threading._os.system(':')",
                ["threading"],
                InterpreterError("Forbidden access to module: os"),
            ),
            ("import random; random._os.system(':')", [], InterpreterError("Forbidden access to module: os")),
            (
                "import random; random.__dict__['_os'].system(':')",
                [],
                InterpreterError("Forbidden access to dunder attribute: __dict__"),
            ),
            (
                "import doctest; doctest.inspect.os.system(':')",
                ["doctest"],
                InterpreterError("Forbidden access to module: inspect"),
            ),
            (
                "import doctest; doctest.inspect.os.system(':')",
                ["doctest", "inspect"],
                InterpreterError("Forbidden access to module: os"),
            ),
            # subprocess submodule
            (
                "import asyncio; asyncio.base_events.events.subprocess",
                ["asyncio"],
                InterpreterError("Forbidden access to module: asyncio.base_events"),
            ),
            (
                "import asyncio; asyncio.base_events.events.subprocess",
                ["asyncio", "asyncio.base_events"],
                InterpreterError("Forbidden access to module: asyncio.events"),
            ),
            (
                "import asyncio; asyncio.base_events.events.subprocess",
                ["asyncio", "asyncio.base_events", "asyncio.base_events.events"],
                InterpreterError("Forbidden access to module: asyncio.events"),
            ),
            # sys submodule
            (
                "import queue; queue.threading._sys.modules['os'].system(':')",
                [],
                InterpreterError("Forbidden access to module: threading"),
            ),
            (
                "import queue; queue.threading._sys.modules['os'].system(':')",
                ["threading"],
                InterpreterError("Forbidden access to module: sys"),
            ),
            ("import warnings; warnings.sys", ["warnings"], InterpreterError("Forbidden access to module: sys")),
            # Allowed
            ("import pandas; pandas.io", ["pandas", "pandas.io"], None),
        ],
    )
    def test_vulnerability_via_submodules(self, code, additional_authorized_imports, expected_error):
        # [FUNCTION DOC]
        # Parametrized: checks reaching os, subprocess or sys through module attributes of authorized modules
        #   (queue, random, doctest, asyncio, warnings) is blocked at the first unauthorized module; pandas.io
        #   is allowed.
        # Inputs: self (the test class instance); code (str), additional_authorized_imports (list of str),
        #   expected_error (InterpreterError or None).
        # Outputs: None.
        # Exceptions: AssertionError (pytest failure) if any check fails; none raised directly otherwise.
        # Side effects: none (only local interpreter state dicts created inside the test are mutated).
        executor = LocalPythonExecutor(additional_authorized_imports)
        with (
            pytest.raises(type(expected_error), match=f".*{expected_error}")
            if isinstance(expected_error, Exception)
            else does_not_raise()
        ):
            executor(code)

    @pytest.mark.parametrize(
        "code, additional_authorized_imports, expected_error",
        [
            # Using filter with functools.partial
            (
                dedent(
                    """
                    import functools
                    import warnings
                    list(filter(functools.partial(getattr, warnings), ["sys"]))
                    """
                ),
                ["warnings", "functools"],
                InterpreterError("Forbidden access to module: sys"),
            ),
            # Using map
            (
                dedent(
                    """
                    import warnings
                    list(map(getattr, [warnings], ["sys"]))
                    """
                ),
                ["warnings"],
                InterpreterError("Forbidden access to module: sys"),
            ),
            # Using map with functools.partial
            (
                dedent(
                    """
                    import functools
                    import warnings
                    list(map(functools.partial(getattr, warnings), ["sys"]))
                    """
                ),
                ["warnings", "functools"],
                InterpreterError("Forbidden access to module: sys"),
            ),
        ],
    )
    def test_vulnerability_via_submodules_through_indirect_attribute_access(
        self, code, additional_authorized_imports, expected_error
    ):
        # [FUNCTION DOC]
        # Parametrized: checks getattr used indirectly via filter/map/functools.partial cannot fetch
        #   warnings.sys.
        # Inputs: self (the test class instance); code (str), additional_authorized_imports (list of str),
        #   expected_error (InterpreterError).
        # Outputs: None.
        # Exceptions: AssertionError (pytest failure) if any check fails; none raised directly otherwise.
        # Side effects: none (only local interpreter state dicts created inside the test are mutated).
        # warnings.sys
        executor = LocalPythonExecutor(additional_authorized_imports)
        executor.send_tools({})
        with pytest.raises(type(expected_error), match=f".*{expected_error}"):
            executor(code)

    @pytest.mark.parametrize(
        "additional_authorized_imports, additional_tools, expected_error",
        [
            ([], [], InterpreterError("Import of sys is not allowed")),
            (["sys"], [], InterpreterError("Forbidden access to module: builtins")),
            (
                ["sys", "builtins"],
                [],
                InterpreterError("Forbidden access to function: __import__"),
            ),
            (["sys", "builtins"], ["__import__"], InterpreterError("Forbidden access to module: os")),
            (["sys", "builtins", "os"], ["__import__"], None),
        ],
    )
    def test_vulnerability_builtins_via_sys(self, additional_authorized_imports, additional_tools, expected_error):
        # [FUNCTION DOC]
        # Checks reaching __import__ through a frame's builtins via sys is blocked at each layer (sys,
        #   builtins module, __import__ function, os module) until all are allowed.
        # Inputs: self (the test class instance); additional_authorized_imports (list of str),
        #   additional_tools (list of tool names to send), expected_error (InterpreterError instance or None).
        # Outputs: None.
        # Exceptions: AssertionError (pytest failure) if any check fails; none raised directly otherwise.
        # Side effects: when the case expects no error, runs os.system with a no-op command in a shell.
        executor = LocalPythonExecutor(additional_authorized_imports)
        if additional_tools:
            from builtins import __import__

            executor.send_tools({"__import__": __import__})
        with (
            pytest.raises(type(expected_error), match=f".*{expected_error}")
            if isinstance(expected_error, Exception)
            else does_not_raise()
        ):
            executor(
                dedent(
                    """
                    import sys
                    builtins = sys._getframe().f_builtins
                    builtins_import = builtins["__import__"]
                    os_module = builtins_import("os")
                    os_module.system(":")
                    """
                )
            )

    @pytest.mark.parametrize("patch_builtin_import_module", [False, True])  # builtins_import.__module__ = None
    @pytest.mark.parametrize(
        "additional_authorized_imports, additional_tools, expected_error",
        [
            ([], [], InterpreterError("Forbidden access to dunder attribute: __traceback__")),
            (
                ["builtins", "os"],
                ["__import__"],
                InterpreterError("Forbidden access to dunder attribute: __traceback__"),
            ),
        ],
    )
    def test_vulnerability_builtins_via_traceback(
        self, patch_builtin_import_module, additional_authorized_imports, additional_tools, expected_error, monkeypatch
    ):
        # [FUNCTION DOC]
        # Checks reaching builtins through an exception's __traceback__ frame is blocked at the dunder access,
        #   optionally with builtins.__import__.__module__ monkeypatched to None.
        # Inputs: self (the test class instance); patch_builtin_import_module (bool),
        #   additional_authorized_imports, additional_tools, expected_error, monkeypatch (pytest fixture).
        # Outputs: None.
        # Exceptions: AssertionError (pytest failure) if any check fails; none raised directly otherwise.
        # Side effects: may monkeypatch builtins.__import__.__module__ (restored by monkeypatch).
        if patch_builtin_import_module:
            monkeypatch.setattr("builtins.__import__.__module__", None)  # inspect.getmodule(func) = None
        executor = LocalPythonExecutor(additional_authorized_imports)
        if additional_tools:
            from builtins import __import__

            executor.send_tools({"__import__": __import__})
        with (
            pytest.raises(type(expected_error), match=f".*{expected_error}")
            if isinstance(expected_error, Exception)
            else does_not_raise()
        ):
            executor(
                dedent(
                    """
                    try:
                        1 / 0
                    except Exception as e:
                        builtins = e.__traceback__.tb_frame.f_back.f_globals["__builtins__"]
                        builtins_import = builtins["__import__"]
                        os_module = builtins_import("os")
                        os_module.system(":")
                    """
                )
            )

    @pytest.mark.parametrize("patch_builtin_import_module", [False, True])  # builtins_import.__module__ = None
    @pytest.mark.parametrize(
        "additional_authorized_imports, additional_tools, expected_error",
        [
            ([], [], InterpreterError("Forbidden access to dunder attribute: __base__")),
            (["warnings"], [], InterpreterError("Forbidden access to dunder attribute: __base__")),
            (
                ["warnings", "builtins"],
                [],
                InterpreterError("Forbidden access to dunder attribute: __base__"),
            ),
            (["warnings", "builtins", "os"], [], InterpreterError("Forbidden access to dunder attribute: __base__")),
            (
                ["warnings", "builtins", "os"],
                ["__import__"],
                InterpreterError("Forbidden access to dunder attribute: __base__"),
            ),
        ],
    )
    def test_vulnerability_builtins_via_class_catch_warnings(
        self, patch_builtin_import_module, additional_authorized_imports, additional_tools, expected_error, monkeypatch
    ):
        # [FUNCTION DOC]
        # Checks walking object subclasses to catch_warnings to reach builtins is blocked at the __base__
        #   dunder access, under various authorizations and with the optional __module__ monkeypatch.
        # Inputs: self (the test class instance); patch_builtin_import_module (bool),
        #   additional_authorized_imports, additional_tools, expected_error, monkeypatch (pytest fixture).
        # Outputs: None.
        # Exceptions: AssertionError (pytest failure) if any check fails; none raised directly otherwise.
        # Side effects: may monkeypatch builtins.__import__.__module__ (restored by monkeypatch).
        # KNOWN ISSUE: The tuple-expected_error branch is dead (no such cases), and if expected_error were
        #   neither an exception nor None the expectation variable would be unbound (NameError).
        if patch_builtin_import_module:
            monkeypatch.setattr("builtins.__import__.__module__", None)  # inspect.getmodule(func) = None
        executor = LocalPythonExecutor(additional_authorized_imports)
        if additional_tools:
            from builtins import __import__

            executor.send_tools({"__import__": __import__})
        if isinstance(expected_error, tuple):  # different error depending on patch status
            expected_error = expected_error[patch_builtin_import_module]
        if isinstance(expected_error, Exception):
            expectation = pytest.raises(type(expected_error), match=f".*{expected_error}")
        elif expected_error is None:
            expectation = does_not_raise()
        with expectation:
            executor(
                dedent(
                    """
                    classes = {}.__class__.__base__.__subclasses__()
                    for cls in classes:
                        if cls.__name__ == "catch_warnings":
                            break
                    builtins = cls()._module.__builtins__
                    builtins_import = builtins["__import__"]
                    os_module = builtins_import('os')
                    os_module.system(":")
                    """
                )
            )

    @pytest.mark.filterwarnings("ignore::DeprecationWarning")
    @pytest.mark.parametrize(
        "additional_authorized_imports, expected_error",
        [
            ([], InterpreterError("Forbidden access to dunder attribute: __base__")),
            (["os"], InterpreterError("Forbidden access to dunder attribute: __base__")),
        ],
    )
    def test_vulnerability_load_module_via_builtin_importer(self, additional_authorized_imports, expected_error):
        # [FUNCTION DOC]
        # Checks walking subclasses to BuiltinImporter to load os is blocked at the __base__ dunder access.
        # Inputs: self (the test class instance); additional_authorized_imports (list of str), expected_error
        #   (InterpreterError instance or None).
        # Outputs: None.
        # Exceptions: AssertionError (pytest failure) if any check fails; none raised directly otherwise.
        # Side effects: none (only local interpreter state dicts created inside the test are mutated).
        executor = LocalPythonExecutor(additional_authorized_imports)
        with (
            pytest.raises(type(expected_error), match=f".*{expected_error}")
            if isinstance(expected_error, Exception)
            else does_not_raise()
        ):
            executor(
                dedent(
                    """
                    classes = {}.__class__.__base__.__subclasses__()
                    for cls in classes:
                        if cls.__name__ == "BuiltinImporter":
                            break
                    os_module = cls().load_module("os")
                    os_module.system(":")
                    """
                )
            )

    def test_vulnerability_class_via_subclasses(self):
        # [FUNCTION DOC]
        # Checks finding subprocess.Popen via subclass walking (loop and comprehension forms) is blocked at
        #   __base__.
        # Inputs: self (the test class instance).
        # Outputs: None.
        # Exceptions: AssertionError (pytest failure) if any check fails; none raised directly otherwise.
        # Side effects: none (only local interpreter state dicts created inside the test are mutated).
        # Subclass: subprocess.Popen
        executor = LocalPythonExecutor([])
        code = dedent(
            """
            for cls in ().__class__.__base__.__subclasses__():
                if 'Popen' in cls.__class__.__repr__(cls):
                    break
            cls(["sh", "-c", ":"]).wait()
            """
        )
        with pytest.raises(InterpreterError, match="Forbidden access to dunder attribute: __base__"):
            executor(code)

        code = dedent(
            """
            [c for c in ().__class__.__base__.__subclasses__() if "Popen" in c.__class__.__repr__(c)][0](
                ["sh", "-c", ":"]
            ).wait()
            """
        )
        with pytest.raises(InterpreterError, match="Forbidden access to dunder attribute: __base__"):
            executor(code)

    @pytest.mark.parametrize(
        "code, dunder_attribute",
        [("a = (); b = a.__class__", "__class__"), ("class A:\n    attr=1\nx = A()\nx_dict = x.__dict__", "__dict__")],
    )
    def test_vulnerability_via_dunder_access(self, code, dunder_attribute):
        # [FUNCTION DOC]
        # Parametrized: checks direct reads of __class__ and __dict__ raise 'Forbidden access to dunder
        #   attribute'.
        # Inputs: self (the test class instance); code (str), dunder_attribute (str).
        # Outputs: None.
        # Exceptions: AssertionError (pytest failure) if any check fails; none raised directly otherwise.
        # Side effects: none (only local interpreter state dicts created inside the test are mutated).
        executor = LocalPythonExecutor([])
        with pytest.raises(InterpreterError, match=f"Forbidden access to dunder attribute: {dunder_attribute}"):
            executor(code)

    def test_vulnerability_via_dunder_indirect_access(self):
        # [FUNCTION DOC]
        # Checks getattr is not available as a tool, so indirect dunder access fails with 'Forbidden function
        #   evaluation'.
        # Inputs: self (the test class instance).
        # Outputs: None.
        # Exceptions: AssertionError (pytest failure) if any check fails; none raised directly otherwise.
        # Side effects: none (only local interpreter state dicts created inside the test are mutated).
        executor = LocalPythonExecutor([])
        code = "a = (); b = getattr(a, '__class__')"
        with pytest.raises(InterpreterError, match="Forbidden function evaluation: 'getattr'"):
            executor(code)

    @pytest.mark.parametrize(
        "code, additional_allowed_dunder_methods, expected_dunder_function_error",
        [
            ("'string'.__dir__()", [], "__dir__"),
            ("'string'.__dir__()", ["__dir__"], None),
            (
                dedent(
                    """
                    def func():
                        pass
                    function_class = func.__getattribute__("__class__")
                    object_class = type.__getattribute__(function_class, '__bases__')[0]
                    for i, subclass in enumerate(object_class.__subclasses__()):
                        if 'subprocess.Popen' in str(subclass):
                            break
                    subclass(["sh", "-c", ":"]).wait()
                    """
                ),
                [],
                "__getattribute__",
            ),
            (
                dedent(
                    """
                    def func():
                        pass
                    function_class = func.__getattribute__("__class__")
                    object_class = type.__getattribute__(function_class, '__bases__')[0]
                    for i, subclass in enumerate(object_class.__subclasses__()):
                        if 'subprocess.Popen' in str(subclass):
                            break
                    subclass(["sh", "-c", ":"]).wait()
                    """
                ),
                ["__getattribute__"],
                "__subclasses__",
            ),
            (
                dedent(
                    """
                    def func():
                        pass
                    function_class = func.__getattribute__("__class__")
                    object_class = type.__getattribute__(function_class, '__bases__')[0]
                    for i, subclass in enumerate(object_class.__subclasses__()):
                        if 'subprocess.Popen' in str(subclass):
                            break
                    subclass(["sh", "-c", ":"]).wait()
                    """
                ),
                ["__getattribute__", "__subclasses__"],
                None,
            ),
        ],
    )
    def test_vulnerability_via_dunder_call(
        self, code, additional_allowed_dunder_methods, expected_dunder_function_error, monkeypatch
    ):
        # [FUNCTION DOC]
        # Parametrized: monkeypatches ALLOWED_DUNDER_METHODS with extra names and checks calls to non-allowed
        #   dunder methods (__dir__, __getattribute__, __subclasses__) raise 'Forbidden call to dunder
        #   function', or succeed once allowed.
        # Inputs: self (the test class instance); code (str), additional_allowed_dunder_methods (list of str),
        #   expected_dunder_function_error (str or None), monkeypatch (pytest fixture).
        # Outputs: None.
        # Exceptions: AssertionError (pytest failure) if any check fails; none raised directly otherwise.
        # Side effects: monkeypatches ALLOWED_DUNDER_METHODS (restored); the fully-allowed case spawns a real
        #   'sh -c :' subprocess via subprocess.Popen.
        import smolagents.local_python_executor

        monkeypatch.setattr(
            "smolagents.local_python_executor.ALLOWED_DUNDER_METHODS",
            smolagents.local_python_executor.ALLOWED_DUNDER_METHODS + additional_allowed_dunder_methods,
        )
        executor = LocalPythonExecutor([])
        executor.send_tools({})
        expectation = (
            pytest.raises(
                InterpreterError, match=f"Forbidden call to dunder function: {expected_dunder_function_error}"
            )
            if expected_dunder_function_error
            else does_not_raise()
        )
        with expectation:
            executor(code)
