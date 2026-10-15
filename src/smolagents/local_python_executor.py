#!/usr/bin/env python
# coding=utf-8

# Copyright 2024 The HuggingFace Inc. team. All rights reserved.
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
# What it does: Implements a restricted, AST-walking Python interpreter used by smolagents to run LLM-
#   generated code locally. Code is parsed with the ast module and every node is evaluated by dedicated
#   evaluate_* functions against a state dict, with limits on operation count, while-loop iterations and wall-
#   clock time, a whitelist of builtins (BASE_PYTHON_TOOLS), an import whitelist (authorized_imports, with '*'
#   wildcards), blocking of dunder attribute access and of dangerous modules/functions (DANGEROUS_MODULES,
#   DANGEROUS_FUNCTIONS). It is not a security sandbox.
# Inputs: Depends on .tools.Tool (type hint for tools) and .utils (BASE_BUILTIN_MODULES default import
#   whitelist, truncate_content for print output). Module constants: DEFAULT_MAX_LEN_OUTPUT (50000),
#   MAX_OPERATIONS (10,000,000), MAX_WHILE_ITERATIONS (1,000,000), MAX_EXECUTION_TIME_SECONDS (30),
#   ALLOWED_DUNDER_METHODS (__init__, __str__, __repr__), ERRORS (all builtin exception classes by name).
# Outputs: Public exports (__all__): evaluate_python_code and LocalPythonExecutor. Also used elsewhere:
#   InterpreterError, PythonExecutor, CodeOutput, FinalAnswerException, BASE_PYTHON_TOOLS,
#   fix_final_answer_code, check_import_authorized, get_safe_module and the timeout decorator.
# Exceptions: None raised at import time beyond import errors of its dependencies.
# Side effects: Creates a module-level logger; otherwise none at import time.
# KNOWN ISSUE: BASE_PYTHON_TOOLS exposes the raw setattr and hasattr builtins, so interpreted code can set
#   dunder attributes via setattr even though attribute reads of dunders are blocked.
import ast
import builtins
import difflib
import inspect
import logging
import math
import re
from abc import ABC, abstractmethod
from collections.abc import Callable, Generator, Mapping
from concurrent.futures import ThreadPoolExecutor
from concurrent.futures import TimeoutError as FuturesTimeoutError
from dataclasses import dataclass
from functools import wraps
from importlib import import_module
from importlib.util import find_spec
from types import BuiltinFunctionType, FunctionType, ModuleType
from typing import Any

from .tools import Tool
from .utils import BASE_BUILTIN_MODULES, truncate_content


logger = logging.getLogger(__name__)


class InterpreterError(ValueError):
    # [CLASS DOC]
    # What it does: Error type raised whenever the interpreter cannot evaluate code: syntax errors,
    #   unsupported nodes, forbidden imports/functions/dunder access, exceeded limits, and as the wrapper for
    #   runtime failures.
    # Inputs: Same constructor as ValueError (a message).
    # Outputs: n/a (exception class).
    # Exceptions: none raised directly.
    # Side effects: none.
    """
    An error raised when the interpreter cannot evaluate a Python expression, due to syntax error or unsupported
    operations.
    """

    pass


ERRORS = {
    name: getattr(builtins, name)
    for name in dir(builtins)
    if isinstance(getattr(builtins, name), type) and issubclass(getattr(builtins, name), BaseException)
}

DEFAULT_MAX_LEN_OUTPUT = 50000
MAX_OPERATIONS = 10000000
MAX_WHILE_ITERATIONS = 1000000
MAX_EXECUTION_TIME_SECONDS = 30
ALLOWED_DUNDER_METHODS = ["__init__", "__str__", "__repr__"]


def custom_print(*args):
    # [FUNCTION DOC]
    # What it does: HELPER: No-op stand-in for print registered in BASE_PYTHON_TOOLS; actual print capture is
    #   done in evaluate_call, which intercepts calls named print.
    # Inputs: any positional args (ignored).
    # Outputs: always None.
    # Exceptions: none raised directly.
    # Side effects: none.
    return None


def nodunder_getattr(obj, name, default=None):
    # [FUNCTION DOC]
    # What it does: HELPER: Restricted replacement for getattr exposed to interpreted code; refuses names that
    #   both start and end with double underscores, otherwise delegates to the builtin getattr.
    # Inputs: obj (any object), name (str attribute name), default (any, default None).
    # Outputs: the attribute value, or default if the attribute is missing.
    # Exceptions: InterpreterError for dunder names.
    # Side effects: none.
    # KNOWN ISSUE: because default always defaults to None, a missing attribute returns None instead of
    #   raising AttributeError as the builtin two-argument getattr would.
    if name.startswith("__") and name.endswith("__"):
        raise InterpreterError(f"Forbidden access to dunder attribute: {name}")
    return getattr(obj, name, default)


BASE_PYTHON_TOOLS = {
    "print": custom_print,
    "isinstance": isinstance,
    "range": range,
    "float": float,
    "int": int,
    "bool": bool,
    "str": str,
    "set": set,
    "list": list,
    "dict": dict,
    "tuple": tuple,
    "round": round,
    "ceil": math.ceil,
    "floor": math.floor,
    "log": math.log,
    "exp": math.exp,
    "sin": math.sin,
    "cos": math.cos,
    "tan": math.tan,
    "asin": math.asin,
    "acos": math.acos,
    "atan": math.atan,
    "atan2": math.atan2,
    "degrees": math.degrees,
    "radians": math.radians,
    "pow": pow,
    "sqrt": math.sqrt,
    "len": len,
    "sum": sum,
    "max": max,
    "min": min,
    "abs": abs,
    "enumerate": enumerate,
    "zip": zip,
    "reversed": reversed,
    "sorted": sorted,
    "all": all,
    "any": any,
    "map": map,
    "filter": filter,
    "ord": ord,
    "chr": chr,
    "next": next,
    "iter": iter,
    "divmod": divmod,
    "callable": callable,
    "getattr": nodunder_getattr,
    "hasattr": hasattr,
    "setattr": setattr,
    "issubclass": issubclass,
    "type": type,
    "complex": complex,
}

# Non-exhaustive list of dangerous modules that should not be imported
DANGEROUS_MODULES = [
    "builtins",
    "io",
    "multiprocessing",
    "os",
    "pathlib",
    "pty",
    "shutil",
    "socket",
    "subprocess",
    "sys",
]

DANGEROUS_FUNCTIONS = [
    "builtins.compile",
    "builtins.eval",
    "builtins.exec",
    "builtins.globals",
    "builtins.locals",
    "builtins.__import__",
    "os.popen",
    "os.system",
    "posix.system",
]


def check_safer_result(result: Any, static_tools: dict[str, Callable] = None, authorized_imports: list[str] = None):
    # [FUNCTION DOC]
    # What it does: Validates a value produced by evaluation: module objects (or module-like dicts having a
    #   truthy __spec__ key) must be authorized imports; plain/builtin functions must not match an entry of
    #   DANGEROUS_FUNCTIONS by name and module unless that name is a static tool.
    # Inputs: result (any), static_tools (dict or None), authorized_imports (list of str or None).
    # Outputs: None (returns normally if the result is considered safe).
    # Exceptions: InterpreterError for forbidden modules or functions.
    # Side effects: none.
    # KNOWN ISSUE: authorized_imports defaults to None, but if result is a module the None is passed to
    #   check_import_authorized which iterates it, raising TypeError.
    """
    Checks if a result is safer according to authorized imports and static tools.

    Args:
        result (Any): The result to check.
        static_tools (dict[str, Callable]): Dictionary of static tools.
        authorized_imports (list[str]): List of authorized imports.

    Raises:
        InterpreterError: If the result is not safe
    """
    if isinstance(result, ModuleType):
        if not check_import_authorized(result.__name__, authorized_imports):
            raise InterpreterError(f"Forbidden access to module: {result.__name__}")
    elif isinstance(result, dict) and result.get("__spec__"):
        if not check_import_authorized(result["__name__"], authorized_imports):
            raise InterpreterError(f"Forbidden access to module: {result['__name__']}")
    elif isinstance(result, (FunctionType, BuiltinFunctionType)):
        for qualified_function_name in DANGEROUS_FUNCTIONS:
            module_name, function_name = qualified_function_name.rsplit(".", 1)
            if (
                (static_tools is None or function_name not in static_tools)
                and result.__name__ == function_name
                and result.__module__ == module_name
            ):
                raise InterpreterError(f"Forbidden access to function: {function_name}")


def safer_eval(func: Callable):
    # [FUNCTION DOC]
    # What it does: Decorator for evaluate_ast-style functions: calls the wrapped evaluator and runs
    #   check_safer_result on its return value before returning it.
    # Inputs: func (callable taking expression, state, static_tools, custom_tools, authorized_imports
    #   keyword).
    # Outputs: the wrapping function (functools.wraps preserved).
    # Exceptions: none raised directly.
    # Side effects: none.
    """
    Decorator to enhance the security of an evaluation function by checking its return value.

    Args:
        func (Callable): Evaluation function to be made safer.

    Returns:
        Callable: Safer evaluation function with return value check.
    """

    @wraps(func)
    def _check_return(
        expression,
        state,
        static_tools,
        custom_tools,
        authorized_imports=BASE_BUILTIN_MODULES,
    ):
        # [FUNCTION DOC]
        # What it does: HELPER: Wrapper produced by safer_eval; evaluates then safety-checks the result.
        # Inputs: expression, state, static_tools, custom_tools, authorized_imports (default
        #   BASE_BUILTIN_MODULES).
        # Outputs: the wrapped evaluator's result.
        # Exceptions: InterpreterError from check_safer_result; anything the evaluator raises propagates.
        # Side effects: whatever the wrapped evaluator does (state mutation).
        result = func(expression, state, static_tools, custom_tools, authorized_imports=authorized_imports)
        check_safer_result(result, static_tools, authorized_imports)
        return result

    return _check_return


def safer_func(
    func: Callable,
    static_tools: dict[str, Callable] = BASE_PYTHON_TOOLS,
    authorized_imports: list[str] = BASE_BUILTIN_MODULES,
):
    # [FUNCTION DOC]
    # What it does: Wraps a callable so that its return value is passed through check_safer_result. Classes
    #   (instances of type) are returned unwrapped so they remain usable as types.
    # Inputs: func (callable), static_tools (dict, default BASE_PYTHON_TOOLS), authorized_imports (list,
    #   default BASE_BUILTIN_MODULES).
    # Outputs: the original class, or a wrapping function.
    # Exceptions: none raised directly.
    # Side effects: none.
    """
    Decorator to enhance the security of a function call by checking its return value.

    Args:
        func (Callable): Function to be made safer.
        static_tools (dict[str, Callable]): Dictionary of static tools.
        authorized_imports (list[str]): List of authorized imports.

    Returns:
        Callable: Safer function with return value check.
    """
    # If the function is a type, return it directly without wrapping
    if isinstance(func, type):
        return func

    @wraps(func)
    def _check_return(*args, **kwargs):
        # [FUNCTION DOC]
        # What it does: HELPER: Wrapper produced by safer_func; calls func with all arguments then safety-
        #   checks the result.
        # Inputs: arbitrary positional and keyword arguments forwarded to func.
        # Outputs: func's return value.
        # Exceptions: InterpreterError from check_safer_result; func's exceptions propagate.
        # Side effects: those of func.
        result = func(*args, **kwargs)
        check_safer_result(result, static_tools, authorized_imports)
        return result

    return _check_return


class PrintContainer:
    # [CLASS DOC]
    # What it does: Mutable string accumulator used as state['_print_outputs'] to collect print output of
    #   interpreted code; supports +=, len, str and repr.
    # Inputs: no constructor arguments. Attribute: value (str, initially empty).
    # Outputs: n/a (class).
    # Exceptions: none raised directly.
    # Side effects: none.
    def __init__(self):
        # [FUNCTION DOC]
        # What it does: HELPER: Initializes the accumulated text to the empty string.
        # Inputs: none.
        # Outputs: None.
        # Exceptions: none raised directly.
        # Side effects: sets self.value.
        self.value = ""

    def append(self, text):
        # [FUNCTION DOC]
        # What it does: HELPER: Appends text to the accumulated value.
        # Inputs: text (str; must be a str).
        # Outputs: self, for chaining.
        # Exceptions: TypeError if text is not a str.
        # Side effects: mutates self.value.
        self.value += text
        return self

    def __iadd__(self, other):
        # [FUNCTION DOC]
        # What it does: HELPER: In-place addition; appends the string conversion of other.
        # Inputs: other (any).
        # Outputs: self.
        # Exceptions: none raised directly.
        # Side effects: mutates self.value.
        """Implements the += operator"""
        self.value += str(other)
        return self

    def __str__(self):
        # [FUNCTION DOC]
        # What it does: HELPER: Returns the accumulated text.
        # Inputs: none.
        # Outputs: str.
        # Exceptions: none raised directly.
        # Side effects: none.
        """String representation"""
        return self.value

    def __repr__(self):
        # [FUNCTION DOC]
        # What it does: HELPER: Debug representation in the form PrintContainer(<text>).
        # Inputs: none.
        # Outputs: str.
        # Exceptions: none raised directly.
        # Side effects: none.
        """Representation for debugging"""
        return f"PrintContainer({self.value})"

    def __len__(self):
        # [FUNCTION DOC]
        # What it does: HELPER: Length of the accumulated text.
        # Inputs: none.
        # Outputs: int.
        # Exceptions: none raised directly.
        # Side effects: none.
        """Implements len() function support"""
        return len(self.value)


class BreakException(Exception):
    # [CLASS DOC]
    # What it does: HELPER: Control-flow signal raised by evaluate_ast for a break statement and caught by the
    #   for/while loop evaluators.
    # Inputs: none.
    # Outputs: n/a.
    # Exceptions: none raised directly.
    # Side effects: none.
    pass


class ContinueException(Exception):
    # [CLASS DOC]
    # What it does: HELPER: Control-flow signal raised for a continue statement and caught by loop evaluators,
    #   which stop the current iteration's body.
    # Inputs: none.
    # Outputs: n/a.
    # Exceptions: none raised directly.
    # Side effects: none.
    pass


class ReturnException(Exception):
    # [CLASS DOC]
    # What it does: HELPER: Control-flow signal raised for a return statement, carrying the returned value;
    #   caught by functions created via create_function.
    # Inputs: value (any) stored as attribute value.
    # Outputs: n/a.
    # Exceptions: none raised directly.
    # Side effects: none.
    def __init__(self, value):
        # [FUNCTION DOC]
        # What it does: HELPER: Stores the returned value.
        # Inputs: value (any).
        # Outputs: None.
        # Exceptions: none raised directly.
        # Side effects: sets self.value; does not call the base Exception constructor, so args stays empty.
        self.value = value


class ExecutionTimeoutError(Exception):
    # [CLASS DOC]
    # What it does: Raised by the timeout decorator when execution exceeds the allowed wall-clock time.
    # Inputs: a message.
    # Outputs: n/a.
    # Exceptions: none raised directly.
    # Side effects: none.
    """Exception raised when code execution exceeds the maximum allowed time."""

    pass


def timeout(timeout_seconds: int):
    # [FUNCTION DOC]
    # What it does: Decorator factory limiting a function's wall-clock time: each call runs the function in a
    #   fresh single-worker ThreadPoolExecutor and waits for the result up to timeout_seconds.
    # Inputs: timeout_seconds (int seconds).
    # Outputs: a decorator that wraps a function.
    # Exceptions: the wrapper raises ExecutionTimeoutError on timeout; the function's own exceptions
    #   propagate.
    # Side effects: spawns a worker thread per call.
    # KNOWN ISSUE: the docstring says callers receive a TimeoutError and can continue, but the executor is
    #   used as a context manager whose exit waits for the worker thread to finish, so the caller actually
    #   blocks until the timed-out function completes before ExecutionTimeoutError is seen; an infinite loop
    #   thus is only stopped by the operation/iteration limits.
    """
    Decorator to limit the execution time of a function using threading.

    This implementation is cross-platform (works on Windows) and thread-safe (works when
    called from any thread, not just the main thread), unlike signal-based approaches.

    Args:
        timeout_seconds (`int`): Maximum time in seconds allowed for function execution.

    Raises:
        ExecutionTimeoutError: If the function execution exceeds the timeout period.

    Note:
        If a timeout occurs, the thread running the function cannot be forcefully killed
        in Python, so it will continue running in the background until completion. However,
        the caller will receive a TimeoutError and can continue execution.
    """

    def decorator(func):
        @wraps(func)
        # [FUNCTION DOC]
        # What it does: HELPER: The decorator returned by timeout; wraps func with the time-limited wrapper.
        # Inputs: func (callable).
        # Outputs: wrapper function.
        # Exceptions: none raised directly.
        # Side effects: none.
        def wrapper(*args, **kwargs):
            # Create a new ThreadPoolExecutor for each call to avoid threading issues
            # [FUNCTION DOC]
            # What it does: HELPER: Submits func to a new thread pool and waits up to timeout_seconds for its
            #   result.
            # Inputs: arbitrary args/kwargs forwarded to func.
            # Outputs: func's result.
            # Exceptions: ExecutionTimeoutError on timeout; func's exceptions re-raised.
            # Side effects: creates and shuts down a thread pool (blocking until the thread ends).
            with ThreadPoolExecutor(max_workers=1) as executor:
                future = executor.submit(func, *args, **kwargs)
                try:
                    result = future.result(timeout=timeout_seconds)
                    return result
                except FuturesTimeoutError:
                    raise ExecutionTimeoutError(
                        f"Code execution exceeded the maximum execution time of {timeout_seconds} seconds"
                    )

        return wrapper

    return decorator


def get_iterable(obj):
    # [FUNCTION DOC]
    # What it does: HELPER: Returns a list as-is, converts any object having __iter__ into a list.
    # Inputs: obj (any).
    # Outputs: list.
    # Exceptions: InterpreterError if obj is not iterable.
    # Side effects: consumes iterators passed in.
    if isinstance(obj, list):
        return obj
    elif hasattr(obj, "__iter__"):
        return list(obj)
    else:
        raise InterpreterError("Object is not iterable")


def fix_final_answer_code(code: str) -> str:
    # [FUNCTION DOC]
    # What it does: Rewrites LLM code that assigns to the name final_answer (which would shadow the tool):
    #   when the code both calls final_answer( and assigns to the bare name, every assignment target and every
    #   non-call use of the bare name is renamed to final_answer_variable, leaving calls and attribute
    #   accesses untouched. Otherwise code is returned unchanged.
    # Inputs: code (str).
    # Outputs: possibly rewritten code (str).
    # Exceptions: none raised directly.
    # Side effects: none.
    # KNOWN ISSUE: the regex rewrite is textual, so occurrences inside strings or comments are renamed as
    #   well, and a keyword argument such as final_answer= in a call is also rewritten.
    """
    Sometimes an LLM can try to assign a variable to final_answer, which would break the final_answer() tool.
    This function fixes this behaviour by replacing variable assignments to final_answer with final_answer_variable,
    while preserving function calls to final_answer().
    """
    # First, find if there's a direct assignment to final_answer
    # Use word boundary and negative lookbehind to ensure it's not an object attribute
    assignment_pattern = r"(?<!\.)(?<!\w)\bfinal_answer\s*="
    if "final_answer(" not in code or not re.search(assignment_pattern, code):
        # If final_answer tool is not called in this blob, then doing the replacement is hazardous because it could false the model's memory for next steps.
        # Let's not modify the code and leave the subsequent assignment error happen.
        return code

    # Pattern for replacing variable assignments
    # Looks for 'final_answer' followed by '=' with optional whitespace
    # Negative lookbehind ensures we don't match object attributes
    assignment_regex = r"(?<!\.)(?<!\w)(\bfinal_answer)(\s*=)"
    code = re.sub(assignment_regex, r"final_answer_variable\2", code)

    # Pattern for replacing variable usage but not function calls
    # Negative lookahead (?!\s*\() ensures we don't match function calls
    # Negative lookbehind (?<!\.|\w) ensures we don't match object methods or other variables
    variable_regex = r"(?<!\.)(?<!\w)(\bfinal_answer\b)(?!\s*\()"
    code = re.sub(variable_regex, "final_answer_variable", code)
    return code


def build_import_tree(authorized_imports: list[str]) -> dict[str, Any]:
    # [FUNCTION DOC]
    # What it does: HELPER: Builds a nested dict trie from dotted import paths, one level per dotted
    #   component.
    # Inputs: authorized_imports (list of str, may include '*' components).
    # Outputs: dict mapping component names to child dicts.
    # Exceptions: none raised directly.
    # Side effects: none.
    tree = {}
    for import_path in authorized_imports:
        parts = import_path.split(".")
        current = tree
        for part in parts:
            if part not in current:
                current[part] = {}
            current = current[part]
    return tree


def check_import_authorized(import_to_check: str, authorized_imports: list[str]) -> bool:
    # [FUNCTION DOC]
    # What it does: Decides whether a dotted module path is importable: walks the trie of authorized imports
    #   component by component; returns True as soon as the current level contains '*', False if a component
    #   is missing, True if all components were matched.
    # Inputs: import_to_check (dotted str), authorized_imports (list of str).
    # Outputs: bool.
    # Exceptions: AttributeError if import_to_check is None (e.g. relative imports).
    # Side effects: none.
    # KNOWN ISSUE: authorizing a submodule (e.g. 'a.b') implicitly authorizes its parent package 'a', while
    #   authorizing 'a' does not authorize 'a.b' unless 'a.*' is listed.
    current_node = build_import_tree(authorized_imports)
    for part in import_to_check.split("."):
        if "*" in current_node:
            return True
        if part not in current_node:
            return False
        current_node = current_node[part]
    return True


def evaluate_attribute(
    expression: ast.Attribute,
    state: dict[str, Any],
    static_tools: dict[str, Callable],
    custom_tools: dict[str, Callable],
    authorized_imports: list[str],
) -> Any:
    # [FUNCTION DOC]
    # What it does: Evaluates attribute access: rejects dunder attribute names, evaluates the object then
    #   returns the attribute.
    # Inputs: expression/node (the AST node to evaluate), state (dict of interpreter variables, mutated by
    #   assignments), static_tools (dict of non-overwritable callables), custom_tools (dict of overwritable
    #   callables), authorized_imports (list of importable module names/patterns).
    # Outputs: the attribute value.
    # Exceptions: InterpreterError for dunder names; AttributeError if missing.
    # Side effects: none beyond evaluating the sub-expression.
    if expression.attr.startswith("__") and expression.attr.endswith("__"):
        raise InterpreterError(f"Forbidden access to dunder attribute: {expression.attr}")
    value = evaluate_ast(expression.value, state, static_tools, custom_tools, authorized_imports)
    return getattr(value, expression.attr)


def evaluate_unaryop(
    expression: ast.UnaryOp,
    state: dict[str, Any],
    static_tools: dict[str, Callable],
    custom_tools: dict[str, Callable],
    authorized_imports: list[str],
) -> Any:
    # [FUNCTION DOC]
    # What it does: Evaluates unary operators: negation, unary plus, logical not and bitwise invert.
    # Inputs: expression/node (the AST node to evaluate), state (dict of interpreter variables, mutated by
    #   assignments), static_tools (dict of non-overwritable callables), custom_tools (dict of overwritable
    #   callables), authorized_imports (list of importable module names/patterns).
    # Outputs: the operation result.
    # Exceptions: InterpreterError for other unary operators; operand errors propagate.
    # Side effects: none beyond operand evaluation.
    operand = evaluate_ast(expression.operand, state, static_tools, custom_tools, authorized_imports)
    if isinstance(expression.op, ast.USub):
        return -operand
    elif isinstance(expression.op, ast.UAdd):
        return operand
    elif isinstance(expression.op, ast.Not):
        return not operand
    elif isinstance(expression.op, ast.Invert):
        return ~operand
    else:
        raise InterpreterError(f"Unary operation {expression.op.__class__.__name__} is not supported.")


def evaluate_lambda(
    lambda_expression: ast.Lambda,
    state: dict[str, Any],
    static_tools: dict[str, Callable],
    custom_tools: dict[str, Callable],
    authorized_imports: list[str],
) -> Callable:
    # [FUNCTION DOC]
    # What it does: Builds a Python callable for a lambda expression; calling it copies the current state,
    #   binds positional parameters by position and evaluates the body in that copy.
    # Inputs: expression/node (the AST node to evaluate), state (dict of interpreter variables, mutated by
    #   assignments), static_tools (dict of non-overwritable callables), custom_tools (dict of overwritable
    #   callables), authorized_imports (list of importable module names/patterns).
    # Outputs: a callable.
    # Exceptions: none raised directly.
    # Side effects: none at creation.
    # KNOWN ISSUE: default values, keyword arguments, *args and **kwargs are ignored; missing or extra
    #   positional arguments are silently dropped instead of raising TypeError.
    args = [arg.arg for arg in lambda_expression.args.args]

    def lambda_func(*values: Any) -> Any:
        # [FUNCTION DOC]
        # What it does: HELPER: The callable returned for a lambda; evaluates the body with parameters bound
        #   in a shallow copy of the enclosing state.
        # Inputs: values (positional arguments).
        # Outputs: the body's value.
        # Exceptions: errors from evaluating the body propagate.
        # Side effects: none on the enclosing state's bindings (mutation of shared objects still possible).
        new_state = state.copy()
        for arg, value in zip(args, values):
            new_state[arg] = value
        return evaluate_ast(
            lambda_expression.body,
            new_state,
            static_tools,
            custom_tools,
            authorized_imports,
        )

    return lambda_func


def evaluate_while(
    while_loop: ast.While,
    state: dict[str, Any],
    static_tools: dict[str, Callable],
    custom_tools: dict[str, Callable],
    authorized_imports: list[str],
) -> None:
    # [FUNCTION DOC]
    # What it does: Executes a while loop: re-evaluates the test, runs the body statements, handles break
    #   (returns) and continue (skips remaining body), and counts iterations.
    # Inputs: expression/node (the AST node to evaluate), state (dict of interpreter variables, mutated by
    #   assignments), static_tools (dict of non-overwritable callables), custom_tools (dict of overwritable
    #   callables), authorized_imports (list of importable module names/patterns).
    # Outputs: None.
    # Exceptions: InterpreterError when more than MAX_WHILE_ITERATIONS iterations run.
    # Side effects: mutates state via the body.
    # KNOWN ISSUE: the else clause of a while loop is ignored.
    iterations = 0
    while evaluate_ast(while_loop.test, state, static_tools, custom_tools, authorized_imports):
        for node in while_loop.body:
            try:
                evaluate_ast(node, state, static_tools, custom_tools, authorized_imports)
            except BreakException:
                return None
            except ContinueException:
                break
        iterations += 1
        if iterations > MAX_WHILE_ITERATIONS:
            raise InterpreterError(f"Maximum number of {MAX_WHILE_ITERATIONS} iterations in While loop exceeded")
    return None


def create_function(
    func_def: ast.FunctionDef,
    state: dict[str, Any],
    static_tools: dict[str, Callable],
    custom_tools: dict[str, Callable],
    authorized_imports: list[str],
) -> Callable:
    # [FUNCTION DOC]
    # What it does: Builds a Python callable from a FunctionDef node. On each call it copies the current
    #   state, evaluates default values, binds positional args, keyword args, the vararg (as the full args
    #   tuple) and kwarg (the kwargs dict), fills missing defaults, sets self and __class__ when the first
    #   parameter is named self, then runs the body until a ReturnException. Functions named __init__ always
    #   return None. The callable carries __ast__, __source__ and __name__.
    # Inputs: expression/node (the AST node to evaluate), state (dict of interpreter variables, mutated by
    #   assignments), static_tools (dict of non-overwritable callables), custom_tools (dict of overwritable
    #   callables), authorized_imports (list of importable module names/patterns).
    # Outputs: a callable.
    # Exceptions: none raised directly.
    # Side effects: none at creation.
    # KNOWN ISSUE: the *args parameter receives all positional args (including ones bound to named
    #   parameters); keyword-only arguments and their defaults are ignored; defaults are re-evaluated at every
    #   call instead of once at definition; missing required arguments are not detected; decorators are
    #   ignored; assignments inside the function never reach the outer state (no global/nonlocal).
    source_code = ast.unparse(func_def)

    def new_func(*args: Any, **kwargs: Any) -> Any:
        # [FUNCTION DOC]
        # What it does: HELPER: The callable representing the interpreted function; see create_function.
        # Inputs: args, kwargs as passed by the caller.
        # Outputs: the returned value, the last statement's value if no return, or None for __init__.
        # Exceptions: errors from evaluating defaults or the body propagate.
        # Side effects: mutates only a copy of the state (and any shared objects).
        func_state = state.copy()
        arg_names = [arg.arg for arg in func_def.args.args]
        default_values = [
            evaluate_ast(d, state, static_tools, custom_tools, authorized_imports) for d in func_def.args.defaults
        ]

        # Apply default values
        defaults = dict(zip(arg_names[-len(default_values) :], default_values))

        # Set positional arguments
        for name, value in zip(arg_names, args):
            func_state[name] = value

        # Set keyword arguments
        for name, value in kwargs.items():
            func_state[name] = value

        # Handle variable arguments
        if func_def.args.vararg:
            vararg_name = func_def.args.vararg.arg
            func_state[vararg_name] = args

        if func_def.args.kwarg:
            kwarg_name = func_def.args.kwarg.arg
            func_state[kwarg_name] = kwargs

        # Set default values for arguments that were not provided
        for name, value in defaults.items():
            if name not in func_state:
                func_state[name] = value

        # Update function state with self and __class__
        if func_def.args.args and func_def.args.args[0].arg == "self":
            if args:
                func_state["self"] = args[0]
                func_state["__class__"] = args[0].__class__

        result = None
        try:
            for stmt in func_def.body:
                result = evaluate_ast(stmt, func_state, static_tools, custom_tools, authorized_imports)
        except ReturnException as e:
            result = e.value

        if func_def.name == "__init__":
            return None

        return result

    # Store original AST, source code, and name
    new_func.__ast__ = func_def
    new_func.__source__ = source_code
    new_func.__name__ = func_def.name

    return new_func


def evaluate_function_def(
    func_def: ast.FunctionDef,
    state: dict[str, Any],
    static_tools: dict[str, Callable],
    custom_tools: dict[str, Callable],
    authorized_imports: list[str],
) -> Callable:
    # [FUNCTION DOC]
    # What it does: Evaluates a def statement: creates the function and registers it in custom_tools under its
    #   name.
    # Inputs: expression/node (the AST node to evaluate), state (dict of interpreter variables, mutated by
    #   assignments), static_tools (dict of non-overwritable callables), custom_tools (dict of overwritable
    #   callables), authorized_imports (list of importable module names/patterns).
    # Outputs: the created callable.
    # Exceptions: none raised directly.
    # Side effects: mutates custom_tools (not state).
    custom_tools[func_def.name] = create_function(func_def, state, static_tools, custom_tools, authorized_imports)
    return custom_tools[func_def.name]


def evaluate_class_def(
    class_def: ast.ClassDef,
    state: dict[str, Any],
    static_tools: dict[str, Callable],
    custom_tools: dict[str, Callable],
    authorized_imports: list[str],
) -> type:
    # [FUNCTION DOC]
    # What it does: Evaluates a class definition: evaluates bases, picks the first non-type metaclass among
    #   bases (else type), prepares the namespace via __prepare__ when available, fills it from the body
    #   (methods, annotated and plain assignments, pass, a leading docstring), creates the class and stores it
    #   in state.
    # Inputs: expression/node (the AST node to evaluate), state (dict of interpreter variables, mutated by
    #   assignments), static_tools (dict of non-overwritable callables), custom_tools (dict of overwritable
    #   callables), authorized_imports (list of importable module names/patterns).
    # Outputs: the new class.
    # Exceptions: InterpreterError for unsupported body statements or annotated targets.
    # Side effects: mutates state; methods are also registered in custom_tools via evaluate_function_def.
    # KNOWN ISSUE: methods leak into custom_tools as global callables under their names; plain assignments to
    #   targets other than names/attributes (e.g. tuples) are silently ignored; annotated/attribute targets
    #   are evaluated against the class namespace instead of state; keyword arguments (metaclass=) and
    #   decorators are ignored.
    class_name = class_def.name
    bases = [evaluate_ast(base, state, static_tools, custom_tools, authorized_imports) for base in class_def.bases]

    # Determine the metaclass to use
    # If any base class has a custom metaclass, use it
    metaclass = type
    for base in bases:
        base_metaclass = type(base)
        if base_metaclass is not type:
            metaclass = base_metaclass
            break

    # Use __prepare__ if the metaclass provides it (e.g., Enum uses _EnumDict)
    if hasattr(metaclass, "__prepare__"):
        class_dict = metaclass.__prepare__(class_name, bases)
    else:
        class_dict = {}

    for stmt in class_def.body:
        if isinstance(stmt, ast.FunctionDef):
            class_dict[stmt.name] = evaluate_ast(stmt, state, static_tools, custom_tools, authorized_imports)
        elif isinstance(stmt, ast.AnnAssign):
            if stmt.value:
                value = evaluate_ast(stmt.value, state, static_tools, custom_tools, authorized_imports)
            target = stmt.target
            # Handle target types for annotation
            if isinstance(target, ast.Name):
                # Simple variable annotation like "x: int"
                annotation = evaluate_ast(stmt.annotation, state, static_tools, custom_tools, authorized_imports)
                class_dict.setdefault("__annotations__", {})[target.id] = annotation
                # Assign value if provided
                if stmt.value:
                    class_dict[target.id] = value
            elif isinstance(target, ast.Attribute):
                # Attribute annotation like "obj.attr: int"
                obj = evaluate_ast(target.value, class_dict, static_tools, custom_tools, authorized_imports)
                # If there's a value assignment, set the attribute
                if stmt.value:
                    setattr(obj, target.attr, value)
            elif isinstance(target, ast.Subscript):
                # Subscript annotation like "dict[key]: int"
                container = evaluate_ast(target.value, class_dict, static_tools, custom_tools, authorized_imports)
                index = evaluate_ast(target.slice, state, static_tools, custom_tools, authorized_imports)
                # If there's a value assignment, set the item
                if stmt.value:
                    container[index] = value
            else:
                raise InterpreterError(f"Unsupported AnnAssign target in class body: {type(target).__name__}")
        elif isinstance(stmt, ast.Assign):
            value = evaluate_ast(stmt.value, state, static_tools, custom_tools, authorized_imports)
            for target in stmt.targets:
                if isinstance(target, ast.Name):
                    class_dict[target.id] = value
                elif isinstance(target, ast.Attribute):
                    obj = evaluate_ast(target.value, class_dict, static_tools, custom_tools, authorized_imports)
                    setattr(obj, target.attr, value)
        elif isinstance(stmt, ast.Pass):
            pass
        elif (
            isinstance(stmt, ast.Expr)
            and stmt == class_def.body[0]
            and isinstance(stmt.value, ast.Constant)
            and isinstance(stmt.value.value, str)
        ):
            # Check if it is a docstring: first statement in class body which is a string literal expression
            class_dict["__doc__"] = stmt.value.value
        else:
            raise InterpreterError(f"Unsupported statement in class body: {stmt.__class__.__name__}")

    new_class = metaclass(class_name, tuple(bases), class_dict)
    state[class_name] = new_class
    return new_class


def evaluate_annassign(
    annassign: ast.AnnAssign,
    state: dict[str, Any],
    static_tools: dict[str, Callable],
    custom_tools: dict[str, Callable],
    authorized_imports: list[str],
) -> Any:
    # If there's a value to assign, evaluate it
    # [FUNCTION DOC]
    # What it does: Evaluates an annotated assignment: if a value is present, evaluates and assigns it via
    #   set_value; the annotation itself is not evaluated.
    # Inputs: expression/node (the AST node to evaluate), state (dict of interpreter variables, mutated by
    #   assignments), static_tools (dict of non-overwritable callables), custom_tools (dict of overwritable
    #   callables), authorized_imports (list of importable module names/patterns).
    # Outputs: the assigned value, or None for bare declarations.
    # Exceptions: errors from evaluation/assignment propagate.
    # Side effects: mutates state or targets.
    if annassign.value:
        value = evaluate_ast(annassign.value, state, static_tools, custom_tools, authorized_imports)
        # Set the value for the target
        set_value(annassign.target, value, state, static_tools, custom_tools, authorized_imports)
        return value
    # For declarations without values (x: int), just return None
    return None


def evaluate_augassign(
    expression: ast.AugAssign,
    state: dict[str, Any],
    static_tools: dict[str, Callable],
    custom_tools: dict[str, Callable],
    authorized_imports: list[str],
) -> Any:
    # [FUNCTION DOC]
    # What it does: Evaluates augmented assignment: reads the current target value (missing names default to
    #   0), evaluates the right side, applies the arithmetic/bitwise operator (lists may only be extended with
    #   lists) and writes the result back with set_value.
    # Inputs: expression/node (the AST node to evaluate), state (dict of interpreter variables, mutated by
    #   assignments), static_tools (dict of non-overwritable callables), custom_tools (dict of overwritable
    #   callables), authorized_imports (list of importable module names/patterns).
    # Outputs: the new value.
    # Exceptions: InterpreterError for unsupported operators/targets or adding a non-list to a list.
    # Side effects: mutates state, containers or attributes (lists mutate in place).
    # KNOWN ISSUE: an undefined name silently starts from 0 instead of raising NameError; the unsupported-
    #   target error message lacks the f-string prefix so the target type is not interpolated; the target
    #   subexpressions are evaluated twice (read then write); matrix multiplication is unsupported.
    def get_current_value(target: ast.AST) -> Any:
        # [FUNCTION DOC]
        # What it does: HELPER: Reads the current value of an augmented-assignment target (name, subscript,
        #   attribute, or tuple/list of those).
        # Inputs: target (AST node).
        # Outputs: the value, or 0 for undefined names.
        # Exceptions: InterpreterError for other target types; lookup errors propagate.
        # Side effects: none.
        if isinstance(target, ast.Name):
            return state.get(target.id, 0)
        elif isinstance(target, ast.Subscript):
            obj = evaluate_ast(target.value, state, static_tools, custom_tools, authorized_imports)
            key = evaluate_ast(target.slice, state, static_tools, custom_tools, authorized_imports)
            return obj[key]
        elif isinstance(target, ast.Attribute):
            obj = evaluate_ast(target.value, state, static_tools, custom_tools, authorized_imports)
            return getattr(obj, target.attr)
        elif isinstance(target, ast.Tuple):
            return tuple(get_current_value(elt) for elt in target.elts)
        elif isinstance(target, ast.List):
            return [get_current_value(elt) for elt in target.elts]
        else:
            raise InterpreterError("AugAssign not supported for {type(target)} targets.")

    current_value = get_current_value(expression.target)
    value_to_add = evaluate_ast(expression.value, state, static_tools, custom_tools, authorized_imports)

    if isinstance(expression.op, ast.Add):
        if isinstance(current_value, list):
            if not isinstance(value_to_add, list):
                raise InterpreterError(f"Cannot add non-list value {value_to_add} to a list.")
            current_value += value_to_add
        else:
            current_value += value_to_add
    elif isinstance(expression.op, ast.Sub):
        current_value -= value_to_add
    elif isinstance(expression.op, ast.Mult):
        current_value *= value_to_add
    elif isinstance(expression.op, ast.Div):
        current_value /= value_to_add
    elif isinstance(expression.op, ast.Mod):
        current_value %= value_to_add
    elif isinstance(expression.op, ast.Pow):
        current_value **= value_to_add
    elif isinstance(expression.op, ast.FloorDiv):
        current_value //= value_to_add
    elif isinstance(expression.op, ast.BitAnd):
        current_value &= value_to_add
    elif isinstance(expression.op, ast.BitOr):
        current_value |= value_to_add
    elif isinstance(expression.op, ast.BitXor):
        current_value ^= value_to_add
    elif isinstance(expression.op, ast.LShift):
        current_value <<= value_to_add
    elif isinstance(expression.op, ast.RShift):
        current_value >>= value_to_add
    else:
        raise InterpreterError(f"Operation {type(expression.op).__name__} is not supported.")

    # Update the state: current_value has been updated in-place
    set_value(
        expression.target,
        current_value,
        state,
        static_tools,
        custom_tools,
        authorized_imports,
    )

    return current_value


def evaluate_boolop(
    node: ast.BoolOp,
    state: dict[str, Any],
    static_tools: dict[str, Callable],
    custom_tools: dict[str, Callable],
    authorized_imports: list[str],
) -> Any:
    # Determine which value should trigger short-circuit based on operation type:
    # - 'and' returns the first falsy value encountered (or the last value if all are truthy)
    # - 'or' returns the first truthy value encountered (or the last value if all are falsy)
    # [FUNCTION DOC]
    # What it does: Evaluates and/or with Python short-circuit semantics, returning the deciding operand
    #   value.
    # Inputs: expression/node (the AST node to evaluate), state (dict of interpreter variables, mutated by
    #   assignments), static_tools (dict of non-overwritable callables), custom_tools (dict of overwritable
    #   callables), authorized_imports (list of importable module names/patterns).
    # Outputs: the first falsy (and) / truthy (or) value, else the last value.
    # Exceptions: none raised directly.
    # Side effects: only operands actually evaluated run.
    is_short_circuit_value = (lambda x: not x) if isinstance(node.op, ast.And) else (lambda x: bool(x))
    for value in node.values:
        result = evaluate_ast(value, state, static_tools, custom_tools, authorized_imports)
        # Short-circuit: return immediately if the condition is met
        if is_short_circuit_value(result):
            return result
    # If no short-circuit occurred, return the last evaluated value
    return result


def evaluate_binop(
    binop: ast.BinOp,
    state: dict[str, Any],
    static_tools: dict[str, Callable],
    custom_tools: dict[str, Callable],
    authorized_imports: list[str],
) -> Any:
    # Recursively evaluate the left and right operands
    # [FUNCTION DOC]
    # What it does: Evaluates binary arithmetic/bitwise operators after evaluating both operands.
    # Inputs: expression/node (the AST node to evaluate), state (dict of interpreter variables, mutated by
    #   assignments), static_tools (dict of non-overwritable callables), custom_tools (dict of overwritable
    #   callables), authorized_imports (list of importable module names/patterns).
    # Outputs: the operation result.
    # Exceptions: NotImplementedError for unsupported operators (e.g. matrix multiplication); operand errors
    #   propagate.
    # Side effects: none beyond operand evaluation.
    # KNOWN ISSUE: raises NotImplementedError instead of InterpreterError like the rest of the interpreter.
    left_val = evaluate_ast(binop.left, state, static_tools, custom_tools, authorized_imports)
    right_val = evaluate_ast(binop.right, state, static_tools, custom_tools, authorized_imports)

    # Determine the operation based on the type of the operator in the BinOp
    if isinstance(binop.op, ast.Add):
        return left_val + right_val
    elif isinstance(binop.op, ast.Sub):
        return left_val - right_val
    elif isinstance(binop.op, ast.Mult):
        return left_val * right_val
    elif isinstance(binop.op, ast.Div):
        return left_val / right_val
    elif isinstance(binop.op, ast.Mod):
        return left_val % right_val
    elif isinstance(binop.op, ast.Pow):
        return left_val**right_val
    elif isinstance(binop.op, ast.FloorDiv):
        return left_val // right_val
    elif isinstance(binop.op, ast.BitAnd):
        return left_val & right_val
    elif isinstance(binop.op, ast.BitOr):
        return left_val | right_val
    elif isinstance(binop.op, ast.BitXor):
        return left_val ^ right_val
    elif isinstance(binop.op, ast.LShift):
        return left_val << right_val
    elif isinstance(binop.op, ast.RShift):
        return left_val >> right_val
    else:
        raise NotImplementedError(f"Binary operation {type(binop.op).__name__} is not implemented.")


def evaluate_assign(
    assign: ast.Assign,
    state: dict[str, Any],
    static_tools: dict[str, Callable],
    custom_tools: dict[str, Callable],
    authorized_imports: list[str],
) -> Any:
    # [FUNCTION DOC]
    # What it does: Evaluates an assignment statement: evaluates the value once and assigns it to each target
    #   via set_value (starred targets get the value's items extended).
    # Inputs: expression/node (the AST node to evaluate), state (dict of interpreter variables, mutated by
    #   assignments), static_tools (dict of non-overwritable callables), custom_tools (dict of overwritable
    #   callables), authorized_imports (list of importable module names/patterns).
    # Outputs: the evaluated value.
    # Exceptions: errors from set_value propagate.
    # Side effects: mutates state/targets.
    result = evaluate_ast(assign.value, state, static_tools, custom_tools, authorized_imports)
    if len(assign.targets) == 1:
        target = assign.targets[0]
        set_value(target, result, state, static_tools, custom_tools, authorized_imports)
    else:
        expanded_values = []
        for tgt in assign.targets:
            if isinstance(tgt, ast.Starred):
                expanded_values.extend(result)
            else:
                expanded_values.append(result)

        for tgt, val in zip(assign.targets, expanded_values):
            set_value(tgt, val, state, static_tools, custom_tools, authorized_imports)
    return result


def set_value(
    target: ast.AST,
    value: Any,
    state: dict[str, Any],
    static_tools: dict[str, Callable],
    custom_tools: dict[str, Callable],
    authorized_imports: list[str],
) -> None:
    # [FUNCTION DOC]
    # What it does: Assigns a value to an assignment target: names (refusing static tool names), tuples
    #   (converting non-string iterables to tuples and requiring matching length, recursing per element),
    #   subscripts and attributes.
    # Inputs: target (AST node), value (any), state, static_tools, custom_tools, authorized_imports.
    # Outputs: None.
    # Exceptions: InterpreterError when overwriting a static tool, unpacking a non-iterable or wrong-sized
    #   value.
    # Side effects: mutates state, containers or object attributes.
    # KNOWN ISSUE: list targets and starred targets are silently ignored (no assignment happens), so starred
    #   unpacking fails with a size error and list-style unpacking does nothing.
    if isinstance(target, ast.Name):
        if target.id in static_tools:
            raise InterpreterError(f"Cannot assign to name '{target.id}': doing this would erase the existing tool!")
        state[target.id] = value
    elif isinstance(target, ast.Tuple):
        if not isinstance(value, tuple):
            if hasattr(value, "__iter__") and not isinstance(value, (str, bytes)):
                value = tuple(value)
            else:
                raise InterpreterError("Cannot unpack non-tuple value")
        if len(target.elts) != len(value):
            raise InterpreterError("Cannot unpack tuple of wrong size")
        for i, elem in enumerate(target.elts):
            set_value(elem, value[i], state, static_tools, custom_tools, authorized_imports)
    elif isinstance(target, ast.Subscript):
        obj = evaluate_ast(target.value, state, static_tools, custom_tools, authorized_imports)
        key = evaluate_ast(target.slice, state, static_tools, custom_tools, authorized_imports)
        obj[key] = value
    elif isinstance(target, ast.Attribute):
        obj = evaluate_ast(target.value, state, static_tools, custom_tools, authorized_imports)
        setattr(obj, target.attr, value)


def evaluate_call(
    call: ast.Call,
    state: dict[str, Any],
    static_tools: dict[str, Callable],
    custom_tools: dict[str, Callable],
    authorized_imports: list[str],
) -> Any:
    # [FUNCTION DOC]
    # What it does: Evaluates a function call: resolves the callee (from a call, lambda, attribute, name
    #   looked up in state, static_tools, custom_tools then builtin exceptions, or subscript), evaluates
    #   positional args (with star unpacking) and keyword args (with dict unpacking). Special cases: super
    #   (zero-arg form uses state's __class__ and self) and print (appends space-joined args plus newline to
    #   state['_print_outputs']). Refuses builtin functions not provided as static tools and dunder-named
    #   callables outside static tools and ALLOWED_DUNDER_METHODS.
    # Inputs: expression/node (the AST node to evaluate), state (dict of interpreter variables, mutated by
    #   assignments), static_tools (dict of non-overwritable callables), custom_tools (dict of overwritable
    #   callables), authorized_imports (list of importable module names/patterns).
    # Outputs: the call result (None for print).
    # Exceptions: InterpreterError for invalid callees, unknown names, missing attributes, bad unpacking,
    #   forbidden builtins/dunders, invalid super arguments; callee exceptions propagate.
    # Side effects: mutates state['_print_outputs']; whatever the callee does.
    # KNOWN ISSUE: print/super detection is by name only, so a method call like obj.print(...) is captured as
    #   print output instead of calling the method, and a user function named print is never called.
    if not isinstance(call.func, (ast.Call, ast.Lambda, ast.Attribute, ast.Name, ast.Subscript)):
        raise InterpreterError(f"This is not a correct function: {call.func}).")

    func, func_name = None, None

    if isinstance(call.func, ast.Call):
        func = evaluate_ast(call.func, state, static_tools, custom_tools, authorized_imports)
    elif isinstance(call.func, ast.Lambda):
        func = evaluate_ast(call.func, state, static_tools, custom_tools, authorized_imports)
    elif isinstance(call.func, ast.Attribute):
        obj = evaluate_ast(call.func.value, state, static_tools, custom_tools, authorized_imports)
        func_name = call.func.attr
        if not hasattr(obj, func_name):
            raise InterpreterError(f"Object {obj} has no attribute {func_name}")
        func = getattr(obj, func_name)
    elif isinstance(call.func, ast.Name):
        func_name = call.func.id
        if func_name in state:
            func = state[func_name]
        elif func_name in static_tools:
            func = static_tools[func_name]
        elif func_name in custom_tools:
            func = custom_tools[func_name]
        elif func_name in ERRORS:
            func = ERRORS[func_name]
        else:
            raise InterpreterError(
                f"Forbidden function evaluation: '{call.func.id}' is not among the explicitly allowed tools or defined/imported in the preceding code"
            )
    elif isinstance(call.func, ast.Subscript):
        func = evaluate_ast(call.func, state, static_tools, custom_tools, authorized_imports)
        if not callable(func):
            raise InterpreterError(f"This is not a correct function: {call.func}).")
        func_name = None

    args = []
    for arg in call.args:
        if isinstance(arg, ast.Starred):
            args.extend(evaluate_ast(arg.value, state, static_tools, custom_tools, authorized_imports))
        else:
            args.append(evaluate_ast(arg, state, static_tools, custom_tools, authorized_imports))

    kwargs = {}
    for keyword in call.keywords:
        if keyword.arg is None:
            # **kwargs unpacking
            starred_dict = evaluate_ast(keyword.value, state, static_tools, custom_tools, authorized_imports)
            if not isinstance(starred_dict, dict):
                raise InterpreterError(f"Cannot unpack non-dict value in **kwargs: {type(starred_dict).__name__}")
            kwargs.update(starred_dict)
        else:
            # Normal keyword argument
            kwargs[keyword.arg] = evaluate_ast(keyword.value, state, static_tools, custom_tools, authorized_imports)

    if func_name == "super":
        if not args:
            if "__class__" in state and "self" in state:
                return super(state["__class__"], state["self"])
            else:
                raise InterpreterError("super() needs at least one argument")
        cls = args[0]
        if not isinstance(cls, type):
            raise InterpreterError("super() argument 1 must be type")
        if len(args) == 1:
            return super(cls)
        elif len(args) == 2:
            instance = args[1]
            return super(cls, instance)
        else:
            raise InterpreterError("super() takes at most 2 arguments")
    elif func_name == "print":
        state["_print_outputs"] += " ".join(map(str, args)) + "\n"
        return None
    else:  # Assume it's a callable object
        if (inspect.getmodule(func) == builtins) and inspect.isbuiltin(func) and (func not in static_tools.values()):
            raise InterpreterError(
                f"Invoking a builtin function that has not been explicitly added as a tool is not allowed ({func_name})."
            )
        if (
            hasattr(func, "__name__")
            and func.__name__.startswith("__")
            and func.__name__.endswith("__")
            and (func.__name__ not in static_tools)
            and (func.__name__ not in ALLOWED_DUNDER_METHODS)
        ):
            raise InterpreterError(f"Forbidden call to dunder function: {func.__name__}")
        return func(*args, **kwargs)


def evaluate_subscript(
    subscript: ast.Subscript,
    state: dict[str, Any],
    static_tools: dict[str, Callable],
    custom_tools: dict[str, Callable],
    authorized_imports: list[str],
) -> Any:
    # [FUNCTION DOC]
    # What it does: Evaluates indexing: evaluates the index first, then the container, and returns the item;
    #   on failure builds an error message, adding close key matches for string keys of mappings.
    # Inputs: expression/node (the AST node to evaluate), state (dict of interpreter variables, mutated by
    #   assignments), static_tools (dict of non-overwritable callables), custom_tools (dict of overwritable
    #   callables), authorized_imports (list of importable module names/patterns).
    # Outputs: the indexed value.
    # Exceptions: InterpreterError (chained) for KeyError, IndexError or TypeError.
    # Side effects: none.
    # KNOWN ISSUE: index is evaluated before the container, the reverse of Python's evaluation order.
    index = evaluate_ast(subscript.slice, state, static_tools, custom_tools, authorized_imports)
    value = evaluate_ast(subscript.value, state, static_tools, custom_tools, authorized_imports)
    try:
        return value[index]
    except (KeyError, IndexError, TypeError) as e:
        error_message = f"Could not index {value} with '{index}': {type(e).__name__}: {e}"
        if isinstance(index, str) and isinstance(value, Mapping):
            close_matches = difflib.get_close_matches(index, list(value.keys()))
            if len(close_matches) > 0:
                error_message += f". Maybe you meant one of these indexes instead: {str(close_matches)}"
        raise InterpreterError(error_message) from e


def evaluate_name(
    name: ast.Name,
    state: dict[str, Any],
    static_tools: dict[str, Callable],
    custom_tools: dict[str, Callable],
    authorized_imports: list[str],
) -> Any:
    # [FUNCTION DOC]
    # What it does: Resolves a variable name: state first, then static tools (wrapped with safer_func), custom
    #   tools, builtin exception classes, and finally the closest fuzzy match among state names.
    # Inputs: expression/node (the AST node to evaluate), state (dict of interpreter variables, mutated by
    #   assignments), static_tools (dict of non-overwritable callables), custom_tools (dict of overwritable
    #   callables), authorized_imports (list of importable module names/patterns).
    # Outputs: the resolved value.
    # Exceptions: InterpreterError when nothing matches.
    # Side effects: none.
    # KNOWN ISSUE: the fuzzy fallback silently returns a different variable for a misspelled/undefined name
    #   instead of raising an error.
    if name.id in state:
        return state[name.id]
    elif name.id in static_tools:
        return safer_func(static_tools[name.id], static_tools=static_tools, authorized_imports=authorized_imports)
    elif name.id in custom_tools:
        return custom_tools[name.id]
    elif name.id in ERRORS:
        return ERRORS[name.id]
    close_matches = difflib.get_close_matches(name.id, list(state.keys()))
    if len(close_matches) > 0:
        return state[close_matches[0]]
    raise InterpreterError(f"The variable `{name.id}` is not defined.")


def evaluate_condition(
    condition: ast.Compare,
    state: dict[str, Any],
    static_tools: dict[str, Callable],
    custom_tools: dict[str, Callable],
    authorized_imports: list[str],
) -> bool | object:
    # [FUNCTION DOC]
    # What it does: Evaluates (possibly chained) comparisons pairwise, short-circuiting to False as soon as a
    #   comparison returns exactly False; non-bool results (e.g. arrays) are combined with and.
    # Inputs: expression/node (the AST node to evaluate), state (dict of interpreter variables, mutated by
    #   assignments), static_tools (dict of non-overwritable callables), custom_tools (dict of overwritable
    #   callables), authorized_imports (list of importable module names/patterns).
    # Outputs: bool or the combined comparison object.
    # Exceptions: InterpreterError for unknown operators; comparison errors propagate.
    # Side effects: none.
    result = True
    left = evaluate_ast(condition.left, state, static_tools, custom_tools, authorized_imports)
    for i, (op, comparator) in enumerate(zip(condition.ops, condition.comparators)):
        op = type(op)
        right = evaluate_ast(comparator, state, static_tools, custom_tools, authorized_imports)
        if op == ast.Eq:
            current_result = left == right
        elif op == ast.NotEq:
            current_result = left != right
        elif op == ast.Lt:
            current_result = left < right
        elif op == ast.LtE:
            current_result = left <= right
        elif op == ast.Gt:
            current_result = left > right
        elif op == ast.GtE:
            current_result = left >= right
        elif op == ast.Is:
            current_result = left is right
        elif op == ast.IsNot:
            current_result = left is not right
        elif op == ast.In:
            current_result = left in right
        elif op == ast.NotIn:
            current_result = left not in right
        else:
            raise InterpreterError(f"Unsupported comparison operator: {op}")

        if current_result is False:
            return False
        result = current_result if i == 0 else (result and current_result)
        left = right
    return result


def evaluate_if(
    if_statement: ast.If,
    state: dict[str, Any],
    static_tools: dict[str, Callable],
    custom_tools: dict[str, Callable],
    authorized_imports: list[str],
) -> Any:
    # [FUNCTION DOC]
    # What it does: Executes an if statement: evaluates the test and runs either the body or the else body.
    # Inputs: expression/node (the AST node to evaluate), state (dict of interpreter variables, mutated by
    #   assignments), static_tools (dict of non-overwritable callables), custom_tools (dict of overwritable
    #   callables), authorized_imports (list of importable module names/patterns).
    # Outputs: the last non-None statement result of the executed branch, else None.
    # Exceptions: none raised directly.
    # Side effects: mutates state via the branch.
    result = None
    test_result = evaluate_ast(if_statement.test, state, static_tools, custom_tools, authorized_imports)
    if test_result:
        for line in if_statement.body:
            line_result = evaluate_ast(line, state, static_tools, custom_tools, authorized_imports)
            if line_result is not None:
                result = line_result
    else:
        for line in if_statement.orelse:
            line_result = evaluate_ast(line, state, static_tools, custom_tools, authorized_imports)
            if line_result is not None:
                result = line_result
    return result


def evaluate_for(
    for_loop: ast.For,
    state: dict[str, Any],
    static_tools: dict[str, Callable],
    custom_tools: dict[str, Callable],
    authorized_imports: list[str],
) -> Any:
    # [FUNCTION DOC]
    # What it does: Executes a for loop: evaluates the iterable, assigns each item to the target via
    #   set_value, runs the body; break returns, continue skips the rest of the body.
    # Inputs: expression/node (the AST node to evaluate), state (dict of interpreter variables, mutated by
    #   assignments), static_tools (dict of non-overwritable callables), custom_tools (dict of overwritable
    #   callables), authorized_imports (list of importable module names/patterns).
    # Outputs: the last non-None body statement result, else None.
    # Exceptions: none raised directly.
    # Side effects: mutates state (loop variable persists).
    # KNOWN ISSUE: the else clause of a for loop is ignored.
    result = None
    iterator = evaluate_ast(for_loop.iter, state, static_tools, custom_tools, authorized_imports)
    for counter in iterator:
        set_value(
            for_loop.target,
            counter,
            state,
            static_tools,
            custom_tools,
            authorized_imports,
        )
        for node in for_loop.body:
            try:
                line_result = evaluate_ast(node, state, static_tools, custom_tools, authorized_imports)
                if line_result is not None:
                    result = line_result
            except BreakException:
                return result
            except ContinueException:
                break
    return result


def _evaluate_comprehensions(
    comprehensions: list[ast.comprehension],
    evaluate_element: Callable[[dict[str, Any]], Any],
    state: dict[str, Any],
    static_tools: dict[str, Callable],
    custom_tools: dict[str, Callable],
    authorized_imports: list[str],
) -> Generator[Any, None, None]:
    # [FUNCTION DOC]
    # What it does: HELPER: Recursive generator driving list/set/dict comprehensions: iterates the first
    #   comprehension clause, binds the target in a state copy, applies all if filters, then recurses on the
    #   remaining clauses; with no clauses left it yields evaluate_element applied to the state.
    # Inputs: comprehensions (list of comprehension nodes), evaluate_element (callable taking a state dict),
    #   state, static_tools, custom_tools, authorized_imports.
    # Outputs: yields produced elements.
    # Exceptions: none raised directly.
    # Side effects: none on the outer state (copies used).
    """
    Recursively evaluate nested comprehensions and yields elements.

    Args:
        comprehensions (`list[ast.comprehension]`): Comprehensions to evaluate.
        evaluate_element (`Callable`): Function that evaluates the final element when comprehensions are exhausted.
        state (`dict[str, Any]`): Current evaluation state.
        static_tools (`dict[str, Callable]`): Static tools.
        custom_tools (`dict[str, Callable]`): Custom tools.
        authorized_imports (`list[str]`): Authorized imports.

    Yields:
        `Any`: Individual elements produced by the comprehension
    """
    # Base case: no more comprehensions
    if not comprehensions:
        yield evaluate_element(state)
        return
    # Evaluate first comprehension
    comprehension = comprehensions[0]
    iter_value = evaluate_ast(comprehension.iter, state, static_tools, custom_tools, authorized_imports)
    for value in iter_value:
        new_state = state.copy()
        set_value(comprehension.target, value, new_state, static_tools, custom_tools, authorized_imports)
        # Check all filter conditions
        if all(
            evaluate_ast(if_clause, new_state, static_tools, custom_tools, authorized_imports)
            for if_clause in comprehension.ifs
        ):
            # Recurse with remaining comprehensions
            yield from _evaluate_comprehensions(
                comprehensions[1:], evaluate_element, new_state, static_tools, custom_tools, authorized_imports
            )


def evaluate_listcomp(
    listcomp: ast.ListComp,
    state: dict[str, Any],
    static_tools: dict[str, Callable],
    custom_tools: dict[str, Callable],
    authorized_imports: list[str],
) -> list[Any]:
    # [FUNCTION DOC]
    # What it does: Evaluates a list comprehension via _evaluate_comprehensions.
    # Inputs: expression/node (the AST node to evaluate), state (dict of interpreter variables, mutated by
    #   assignments), static_tools (dict of non-overwritable callables), custom_tools (dict of overwritable
    #   callables), authorized_imports (list of importable module names/patterns).
    # Outputs: list.
    # Exceptions: none raised directly.
    # Side effects: none on state.
    return list(
        _evaluate_comprehensions(
            listcomp.generators,
            lambda comp_state: evaluate_ast(listcomp.elt, comp_state, static_tools, custom_tools, authorized_imports),
            state,
            static_tools,
            custom_tools,
            authorized_imports,
        )
    )


def evaluate_setcomp(
    setcomp: ast.SetComp,
    state: dict[str, Any],
    static_tools: dict[str, Callable],
    custom_tools: dict[str, Callable],
    authorized_imports: list[str],
) -> set[Any]:
    # [FUNCTION DOC]
    # What it does: Evaluates a set comprehension via _evaluate_comprehensions.
    # Inputs: expression/node (the AST node to evaluate), state (dict of interpreter variables, mutated by
    #   assignments), static_tools (dict of non-overwritable callables), custom_tools (dict of overwritable
    #   callables), authorized_imports (list of importable module names/patterns).
    # Outputs: set.
    # Exceptions: none raised directly.
    # Side effects: none on state.
    return set(
        _evaluate_comprehensions(
            setcomp.generators,
            lambda comp_state: evaluate_ast(setcomp.elt, comp_state, static_tools, custom_tools, authorized_imports),
            state,
            static_tools,
            custom_tools,
            authorized_imports,
        )
    )


def evaluate_dictcomp(
    dictcomp: ast.DictComp,
    state: dict[str, Any],
    static_tools: dict[str, Callable],
    custom_tools: dict[str, Callable],
    authorized_imports: list[str],
) -> dict[Any, Any]:
    # [FUNCTION DOC]
    # What it does: Evaluates a dict comprehension via _evaluate_comprehensions, yielding key/value pairs.
    # Inputs: expression/node (the AST node to evaluate), state (dict of interpreter variables, mutated by
    #   assignments), static_tools (dict of non-overwritable callables), custom_tools (dict of overwritable
    #   callables), authorized_imports (list of importable module names/patterns).
    # Outputs: dict.
    # Exceptions: none raised directly.
    # Side effects: none on state.
    return dict(
        _evaluate_comprehensions(
            dictcomp.generators,
            lambda comp_state: (
                evaluate_ast(dictcomp.key, comp_state, static_tools, custom_tools, authorized_imports),
                evaluate_ast(dictcomp.value, comp_state, static_tools, custom_tools, authorized_imports),
            ),
            state,
            static_tools,
            custom_tools,
            authorized_imports,
        )
    )


def evaluate_try(
    try_node: ast.Try,
    state: dict[str, Any],
    static_tools: dict[str, Callable],
    custom_tools: dict[str, Callable],
    authorized_imports: list[str],
) -> None:
    # [FUNCTION DOC]
    # What it does: Executes try/except/else/finally: runs the body; on an Exception finds the first handler
    #   whose evaluated type matches (or a bare handler), binds the name in state and runs it, otherwise re-
    #   raises; runs else on success and finally always.
    # Inputs: expression/node (the AST node to evaluate), state (dict of interpreter variables, mutated by
    #   assignments), static_tools (dict of non-overwritable callables), custom_tools (dict of overwritable
    #   callables), authorized_imports (list of importable module names/patterns).
    # Outputs: None.
    # Exceptions: re-raises unmatched exceptions.
    # Side effects: mutates state.
    # KNOWN ISSUE: the interpreter's internal BreakException/ContinueException/ReturnException derive from
    #   Exception, so a bare or broad except in user code swallows break/continue/return inside the try body;
    #   the handler name binding is never removed afterwards.
    try:
        for stmt in try_node.body:
            evaluate_ast(stmt, state, static_tools, custom_tools, authorized_imports)
    except Exception as e:
        matched = False
        for handler in try_node.handlers:
            if handler.type is None or isinstance(
                e,
                evaluate_ast(handler.type, state, static_tools, custom_tools, authorized_imports),
            ):
                matched = True
                if handler.name:
                    state[handler.name] = e
                for stmt in handler.body:
                    evaluate_ast(stmt, state, static_tools, custom_tools, authorized_imports)
                break
        if not matched:
            raise e
    else:
        if try_node.orelse:
            for stmt in try_node.orelse:
                evaluate_ast(stmt, state, static_tools, custom_tools, authorized_imports)
    finally:
        if try_node.finalbody:
            for stmt in try_node.finalbody:
                evaluate_ast(stmt, state, static_tools, custom_tools, authorized_imports)


def evaluate_raise(
    raise_node: ast.Raise,
    state: dict[str, Any],
    static_tools: dict[str, Callable],
    custom_tools: dict[str, Callable],
    authorized_imports: list[str],
) -> None:
    # [FUNCTION DOC]
    # What it does: Executes a raise statement: evaluates the exception and optional cause and raises it.
    # Inputs: expression/node (the AST node to evaluate), state (dict of interpreter variables, mutated by
    #   assignments), static_tools (dict of non-overwritable callables), custom_tools (dict of overwritable
    #   callables), authorized_imports (list of importable module names/patterns).
    # Outputs: never returns normally.
    # Exceptions: the evaluated exception (with cause); InterpreterError for a bare re-raise.
    # Side effects: none.
    # KNOWN ISSUE: a bare raise inside an except handler is not supported.
    if raise_node.exc is not None:
        exc = evaluate_ast(raise_node.exc, state, static_tools, custom_tools, authorized_imports)
    else:
        exc = None
    if raise_node.cause is not None:
        cause = evaluate_ast(raise_node.cause, state, static_tools, custom_tools, authorized_imports)
    else:
        cause = None
    if exc is not None:
        if cause is not None:
            raise exc from cause
        else:
            raise exc
    else:
        raise InterpreterError("Re-raise is not supported without an active exception")


def evaluate_assert(
    assert_node: ast.Assert,
    state: dict[str, Any],
    static_tools: dict[str, Callable],
    custom_tools: dict[str, Callable],
    authorized_imports: list[str],
) -> None:
    # [FUNCTION DOC]
    # What it does: Executes an assert statement; on failure raises AssertionError with the evaluated message
    #   or with 'Assertion failed: ' plus the unparsed test.
    # Inputs: expression/node (the AST node to evaluate), state (dict of interpreter variables, mutated by
    #   assignments), static_tools (dict of non-overwritable callables), custom_tools (dict of overwritable
    #   callables), authorized_imports (list of importable module names/patterns).
    # Outputs: None.
    # Exceptions: AssertionError.
    # Side effects: none.
    test_result = evaluate_ast(assert_node.test, state, static_tools, custom_tools, authorized_imports)
    if not test_result:
        if assert_node.msg:
            msg = evaluate_ast(assert_node.msg, state, static_tools, custom_tools, authorized_imports)
            raise AssertionError(msg)
        else:
            # Include the failing condition in the assertion message
            test_code = ast.unparse(assert_node.test)
            raise AssertionError(f"Assertion failed: {test_code}")


def evaluate_with(
    with_node: ast.With,
    state: dict[str, Any],
    static_tools: dict[str, Callable],
    custom_tools: dict[str, Callable],
    authorized_imports: list[str],
) -> None:
    # [FUNCTION DOC]
    # What it does: Executes a with statement: enters each context manager in order (binding as-names in
    #   state), runs the body, then exits them in reverse order, propagating suppression and replacement of
    #   exceptions like Python does.
    # Inputs: expression/node (the AST node to evaluate), state (dict of interpreter variables, mutated by
    #   assignments), static_tools (dict of non-overwritable callables), custom_tools (dict of overwritable
    #   callables), authorized_imports (list of importable module names/patterns).
    # Outputs: None.
    # Exceptions: re-raises unsuppressed exceptions from the body or from __exit__.
    # Side effects: mutates state; calls __enter__/__exit__ on the managers.
    # KNOWN ISSUE: as-targets must be simple names (tuples fail); if a later __enter__ fails, earlier managers
    #   are not exited; BaseException subclasses such as FinalAnswerException skip __exit__ entirely.
    contexts = []
    for item in with_node.items:
        context_expr = evaluate_ast(item.context_expr, state, static_tools, custom_tools, authorized_imports)
        enter_result = context_expr.__enter__()
        contexts.append(context_expr)
        if item.optional_vars:
            state[item.optional_vars.id] = enter_result

    try:
        for stmt in with_node.body:
            evaluate_ast(stmt, state, static_tools, custom_tools, authorized_imports)
    except Exception as e:
        # exc_info tracks the active exception as we unwind (from innermost context manager)
        # Resetting it to (None, None, None) signals suppression to the remaining outer managers
        exc_info = (type(e), e, e.__traceback__)
        for context in reversed(contexts):
            try:
                if context.__exit__(*exc_info):
                    exc_info = (None, None, None)  # suppressed; outer CMs see no exception
            except Exception as exit_exc:
                exc_info = (type(exit_exc), exit_exc, exit_exc.__traceback__)  # new exc replaces active
        if exc_info[1] is not None:
            raise exc_info[1].with_traceback(exc_info[2])
    else:
        for context in reversed(contexts):
            context.__exit__(None, None, None)


def get_safe_module(raw_module, authorized_imports, visited=None):
    # [FUNCTION DOC]
    # What it does: Returns a non-module value unchanged; for a module, builds a new ModuleType with the same
    #   name and copies every attribute by reference, recursively processing nested modules and returning
    #   originals for already-visited (circular) modules; attributes raising ImportError/AttributeError are
    #   skipped.
    # Inputs: raw_module (any), authorized_imports (list, unused), visited (set of module ids, default None).
    # Outputs: the copy or the original value.
    # Exceptions: none raised directly.
    # Side effects: logs skipped attributes at INFO level; may trigger lazy imports.
    # KNOWN ISSUE: despite the name and docstring, no filtering or authorization check is applied;
    #   authorized_imports is unused, so submodules reachable as attributes are accessible.
    """Creates a safe copy of a module or returns the original if it's a function"""
    # If it's a function or non-module object, return it directly
    if not isinstance(raw_module, ModuleType):
        return raw_module

    # Handle circular references: Initialize visited set for the first call
    if visited is None:
        visited = set()

    module_id = id(raw_module)
    if module_id in visited:
        return raw_module  # Return original for circular refs

    visited.add(module_id)

    # Create new module for actual modules
    safe_module = ModuleType(raw_module.__name__)

    # Copy all attributes by reference, recursively checking modules
    for attr_name in dir(raw_module):
        try:
            attr_value = getattr(raw_module, attr_name)
        except (ImportError, AttributeError) as e:
            # lazy / dynamic loading module -> INFO log and skip
            logger.info(
                f"Skipping import error while copying {raw_module.__name__}.{attr_name}: {type(e).__name__} - {e}"
            )
            continue
        # Recursively process nested modules, passing visited set
        if isinstance(attr_value, ModuleType):
            attr_value = get_safe_module(attr_value, authorized_imports, visited=visited)

        setattr(safe_module, attr_name, attr_value)

    return safe_module


def evaluate_import(expression, state, authorized_imports):
    # [FUNCTION DOC]
    # What it does: Executes import and from-import statements after checking authorization: plain imports
    #   bind the (copied) module under its alias or full dotted name; from-imports bind requested names (or
    #   all of __all__ / all public names for star imports).
    # Inputs: expression (Import or ImportFrom node), state (dict), authorized_imports (list).
    # Outputs: None.
    # Exceptions: InterpreterError for unauthorized modules or missing names; ImportError if the import fails.
    # Side effects: imports modules into the process; mutates state.
    # KNOWN ISSUE: import a.b without alias binds the key 'a.b' in state (to the submodule) instead of binding
    #   'a', so it is not reachable by name; relative imports (module None) crash with AttributeError.
    if isinstance(expression, ast.Import):
        for alias in expression.names:
            if check_import_authorized(alias.name, authorized_imports):
                raw_module = import_module(alias.name)
                state[alias.asname or alias.name] = get_safe_module(raw_module, authorized_imports)
            else:
                raise InterpreterError(
                    f"Import of {alias.name} is not allowed. Authorized imports are: {str(authorized_imports)}"
                )
        return None
    elif isinstance(expression, ast.ImportFrom):
        if check_import_authorized(expression.module, authorized_imports):
            raw_module = __import__(expression.module, fromlist=[alias.name for alias in expression.names])
            module = get_safe_module(raw_module, authorized_imports)
            if expression.names[0].name == "*":  # Handle "from module import *"
                if hasattr(module, "__all__"):  # If module has __all__, import only those names
                    for name in module.__all__:
                        state[name] = getattr(module, name)
                else:  # If no __all__, import all public names (those not starting with '_')
                    for name in dir(module):
                        if not name.startswith("_"):
                            state[name] = getattr(module, name)
            else:  # regular from imports
                for alias in expression.names:
                    if hasattr(module, alias.name):
                        state[alias.asname or alias.name] = getattr(module, alias.name)
                    else:
                        raise InterpreterError(f"Module {expression.module} has no attribute {alias.name}")
        else:
            raise InterpreterError(
                f"Import from {expression.module} is not allowed. Authorized imports are: {str(authorized_imports)}"
            )
        return None


def evaluate_generatorexp(
    genexp: ast.GeneratorExp,
    state: dict[str, Any],
    static_tools: dict[str, Callable],
    custom_tools: dict[str, Callable],
    authorized_imports: list[str],
) -> Generator[Any]:
    # [FUNCTION DOC]
    # What it does: Evaluates a generator expression lazily, returning a generator that iterates each for-
    #   clause, binds targets in a state copy, applies filters and yields the element.
    # Inputs: expression/node (the AST node to evaluate), state (dict of interpreter variables, mutated by
    #   assignments), static_tools (dict of non-overwritable callables), custom_tools (dict of overwritable
    #   callables), authorized_imports (list of importable module names/patterns).
    # Outputs: a generator.
    # Exceptions: none raised directly.
    # Side effects: none on state.
    # KNOWN ISSUE: multiple for-clauses are iterated one after another rather than nested, so expressions with
    #   several clauses produce wrong results.
    def generator():
        # [FUNCTION DOC]
        # What it does: HELPER: The generator implementing evaluate_generatorexp.
        # Inputs: none.
        # Outputs: yields evaluated elements.
        # Exceptions: evaluation errors propagate on iteration.
        # Side effects: none on state.
        for gen in genexp.generators:
            iter_value = evaluate_ast(gen.iter, state, static_tools, custom_tools, authorized_imports)
            for value in iter_value:
                new_state = state.copy()
                set_value(
                    gen.target,
                    value,
                    new_state,
                    static_tools,
                    custom_tools,
                    authorized_imports,
                )
                if all(
                    evaluate_ast(if_clause, new_state, static_tools, custom_tools, authorized_imports)
                    for if_clause in gen.ifs
                ):
                    yield evaluate_ast(
                        genexp.elt,
                        new_state,
                        static_tools,
                        custom_tools,
                        authorized_imports,
                    )

    return generator()


def evaluate_delete(
    delete_node: ast.Delete,
    state: dict[str, Any],
    static_tools: dict[str, Callable],
    custom_tools: dict[str, Callable],
    authorized_imports: list[str],
) -> None:
    # [FUNCTION DOC]
    # What it does: Executes del statements for names (removed from state) and subscripts (item deletion).
    # Inputs: expression/node (the AST node to evaluate), state (dict of interpreter variables, mutated by
    #   assignments), static_tools (dict of non-overwritable callables), custom_tools (dict of overwritable
    #   callables), authorized_imports (list of importable module names/patterns).
    # Outputs: None.
    # Exceptions: InterpreterError for undefined names, failed item deletion or other target types such as
    #   attributes.
    # Side effects: mutates state or containers.
    """
    Evaluate a delete statement (del x, del x[y]).

    Args:
        delete_node: The AST Delete node to evaluate
        state: The current state dictionary
        static_tools: Dictionary of static tools
        custom_tools: Dictionary of custom tools
        authorized_imports: List of authorized imports
    """
    for target in delete_node.targets:
        if isinstance(target, ast.Name):
            # Handle simple variable deletion (del x)
            if target.id in state:
                del state[target.id]
            else:
                raise InterpreterError(f"Cannot delete name '{target.id}': name is not defined")
        elif isinstance(target, ast.Subscript):
            # Handle index/key deletion (del x[y])
            obj = evaluate_ast(target.value, state, static_tools, custom_tools, authorized_imports)
            index = evaluate_ast(target.slice, state, static_tools, custom_tools, authorized_imports)
            try:
                del obj[index]
            except (TypeError, KeyError, IndexError) as e:
                raise InterpreterError(f"Cannot delete index/key: {str(e)}")
        else:
            raise InterpreterError(f"Deletion of {type(target).__name__} targets is not supported")


@safer_eval
def evaluate_ast(
    expression: ast.AST,
    state: dict[str, Any],
    static_tools: dict[str, Callable],
    custom_tools: dict[str, Callable],
    authorized_imports: list[str] = BASE_BUILTIN_MODULES,
):
    # [FUNCTION DOC]
    # What it does: Central dispatcher: increments the operation counter stored in state, then evaluates the
    #   node by type (assignments, calls, constants, containers, comprehensions, operators, control flow,
    #   f-strings, slices, imports, classes, try/raise/assert/with, return, pass, del). Decorated with
    #   safer_eval so every result is safety-checked.
    # Inputs: expression/node (the AST node to evaluate), state (dict of interpreter variables, mutated by
    #   assignments), static_tools (dict of non-overwritable callables), custom_tools (dict of overwritable
    #   callables), authorized_imports (list of importable module names/patterns). authorized_imports defaults
    #   to BASE_BUILTIN_MODULES.
    # Outputs: the node's value (None for most statements).
    # Exceptions: InterpreterError for exceeding MAX_OPERATIONS or unsupported nodes; BreakException,
    #   ContinueException and ReturnException as control flow signals.
    # Side effects: mutates state['_operations_count'] and state via sub-evaluators.
    """
    Evaluate an abstract syntax tree using the content of the variables stored in a state and only evaluating a given
    set of functions.

    This function will recurse through the nodes of the tree provided.

    Args:
        expression (`ast.AST`):
            The code to evaluate, as an abstract syntax tree.
        state (`Dict[str, Any]`):
            A dictionary mapping variable names to values. The `state` is updated if need be when the evaluation
            encounters assignments.
        static_tools (`Dict[str, Callable]`):
            Functions that may be called during the evaluation. Trying to change one of these static_tools will raise an error.
        custom_tools (`Dict[str, Callable]`):
            Functions that may be called during the evaluation. These custom_tools can be overwritten.
        authorized_imports (`List[str]`):
            The list of modules that can be imported by the code. By default, only a few safe modules are allowed.
            If it contains "*", it will authorize any import. Use this at your own risk!
    """
    if state.setdefault("_operations_count", {"counter": 0})["counter"] >= MAX_OPERATIONS:
        raise InterpreterError(
            f"Reached the max number of operations of {MAX_OPERATIONS}. Maybe there is an infinite loop somewhere in the code, or you're just asking too many calculations."
        )
    state["_operations_count"]["counter"] += 1
    common_params = (state, static_tools, custom_tools, authorized_imports)
    if isinstance(expression, ast.Assign):
        # Assignment -> we evaluate the assignment which should update the state
        # We return the variable assigned as it may be used to determine the final result.
        return evaluate_assign(expression, *common_params)
    elif isinstance(expression, ast.AnnAssign):
        return evaluate_annassign(expression, *common_params)
    elif isinstance(expression, ast.AugAssign):
        return evaluate_augassign(expression, *common_params)
    elif isinstance(expression, ast.Call):
        # Function call -> we return the value of the function call
        return evaluate_call(expression, *common_params)
    elif isinstance(expression, ast.Constant):
        # Constant -> just return the value
        return expression.value
    elif isinstance(expression, ast.Tuple):
        return tuple((evaluate_ast(elt, *common_params) for elt in expression.elts))
    elif isinstance(expression, ast.GeneratorExp):
        return evaluate_generatorexp(expression, *common_params)
    elif isinstance(expression, ast.ListComp):
        return evaluate_listcomp(expression, *common_params)
    elif isinstance(expression, ast.DictComp):
        return evaluate_dictcomp(expression, *common_params)
    elif isinstance(expression, ast.SetComp):
        return evaluate_setcomp(expression, *common_params)
    elif isinstance(expression, ast.UnaryOp):
        return evaluate_unaryop(expression, *common_params)
    elif isinstance(expression, ast.Starred):
        return evaluate_ast(expression.value, *common_params)
    elif isinstance(expression, ast.BoolOp):
        # Boolean operation -> evaluate the operation
        return evaluate_boolop(expression, *common_params)
    elif isinstance(expression, ast.Break):
        raise BreakException()
    elif isinstance(expression, ast.Continue):
        raise ContinueException()
    elif isinstance(expression, ast.BinOp):
        # Binary operation -> execute operation
        return evaluate_binop(expression, *common_params)
    elif isinstance(expression, ast.Compare):
        # Comparison -> evaluate the comparison
        return evaluate_condition(expression, *common_params)
    elif isinstance(expression, ast.Lambda):
        return evaluate_lambda(expression, *common_params)
    elif isinstance(expression, ast.FunctionDef):
        return evaluate_function_def(expression, *common_params)
    elif isinstance(expression, ast.Dict):
        # Dict -> evaluate all keys and values
        keys = (evaluate_ast(k, *common_params) for k in expression.keys)
        values = (evaluate_ast(v, *common_params) for v in expression.values)
        return dict(zip(keys, values))
    elif isinstance(expression, ast.Expr):
        # Expression -> evaluate the content
        return evaluate_ast(expression.value, *common_params)
    elif isinstance(expression, ast.For):
        # For loop -> execute the loop
        return evaluate_for(expression, *common_params)
    elif isinstance(expression, ast.FormattedValue):
        # Formatted value (part of f-string) -> evaluate the content and format it
        value = evaluate_ast(expression.value, *common_params)
        # Early return if no format spec
        if not expression.format_spec:
            return value
        # Apply format specification
        format_spec = evaluate_ast(expression.format_spec, *common_params)
        return format(value, format_spec)
    elif isinstance(expression, ast.If):
        # If -> execute the right branch
        return evaluate_if(expression, *common_params)
    elif hasattr(ast, "Index") and isinstance(expression, ast.Index):
        return evaluate_ast(expression.value, *common_params)
    elif isinstance(expression, ast.JoinedStr):
        return "".join([str(evaluate_ast(v, *common_params)) for v in expression.values])
    elif isinstance(expression, ast.List):
        # List -> evaluate all elements
        return [evaluate_ast(elt, *common_params) for elt in expression.elts]
    elif isinstance(expression, ast.Name):
        # Name -> pick up the value in the state
        return evaluate_name(expression, *common_params)
    elif isinstance(expression, ast.Subscript):
        # Subscript -> return the value of the indexing
        return evaluate_subscript(expression, *common_params)
    elif isinstance(expression, ast.IfExp):
        test_val = evaluate_ast(expression.test, *common_params)
        if test_val:
            return evaluate_ast(expression.body, *common_params)
        else:
            return evaluate_ast(expression.orelse, *common_params)
    elif isinstance(expression, ast.Attribute):
        return evaluate_attribute(expression, *common_params)
    elif isinstance(expression, ast.Slice):
        return slice(
            evaluate_ast(expression.lower, *common_params) if expression.lower is not None else None,
            evaluate_ast(expression.upper, *common_params) if expression.upper is not None else None,
            evaluate_ast(expression.step, *common_params) if expression.step is not None else None,
        )
    elif isinstance(expression, ast.While):
        return evaluate_while(expression, *common_params)
    elif isinstance(expression, (ast.Import, ast.ImportFrom)):
        return evaluate_import(expression, state, authorized_imports)
    elif isinstance(expression, ast.ClassDef):
        return evaluate_class_def(expression, *common_params)
    elif isinstance(expression, ast.Try):
        return evaluate_try(expression, *common_params)
    elif isinstance(expression, ast.Raise):
        return evaluate_raise(expression, *common_params)
    elif isinstance(expression, ast.Assert):
        return evaluate_assert(expression, *common_params)
    elif isinstance(expression, ast.With):
        return evaluate_with(expression, *common_params)
    elif isinstance(expression, ast.Set):
        return set((evaluate_ast(elt, *common_params) for elt in expression.elts))
    elif isinstance(expression, ast.Return):
        raise ReturnException(evaluate_ast(expression.value, *common_params) if expression.value else None)
    elif isinstance(expression, ast.Pass):
        return None
    elif isinstance(expression, ast.Delete):
        return evaluate_delete(expression, *common_params)
    else:
        # For now we refuse anything else. Let's add things as we need them.
        raise InterpreterError(f"{expression.__class__.__name__} is not supported.")


class FinalAnswerException(BaseException):
    # [CLASS DOC]
    # What it does: Signal raised by the wrapped final_answer tool to stop execution and carry the answer;
    #   derives from BaseException so generic except clauses in agent code do not catch it.
    # Inputs: value (any) stored as attribute value.
    # Outputs: n/a.
    # Exceptions: none raised directly.
    # Side effects: none.
    """Exception raised when final_answer is called.

    Inherits from BaseException instead of Exception to prevent being caught
    by generic `except Exception` clauses in agent-generated code.
    """

    def __init__(self, value):
        # [FUNCTION DOC]
        # What it does: HELPER: Stores the final answer value.
        # Inputs: value (any).
        # Outputs: None.
        # Exceptions: none raised directly.
        # Side effects: sets self.value.
        self.value = value


def evaluate_python_code(
    code: str,
    static_tools: dict[str, Callable] | None = None,
    custom_tools: dict[str, Callable] | None = None,
    state: dict[str, Any] | None = None,
    authorized_imports: list[str] = BASE_BUILTIN_MODULES,
    max_print_outputs_length: int = DEFAULT_MAX_LEN_OUTPUT,
    timeout_seconds: int | None = MAX_EXECUTION_TIME_SECONDS,
):
    # [FUNCTION DOC]
    # What it does: Parses and executes a code string with the restricted interpreter. Resets
    #   state['_print_outputs'] and the operation counter, copies static_tools and wraps a final_answer tool
    #   so that it raises FinalAnswerException, evaluates each top-level statement, truncates print output,
    #   and optionally enforces a wall-clock timeout.
    # Inputs: code (str), static_tools (dict or None), custom_tools (dict or None, used and mutated directly),
    #   state (dict or None), authorized_imports (list, default BASE_BUILTIN_MODULES),
    #   max_print_outputs_length (int, default DEFAULT_MAX_LEN_OUTPUT), timeout_seconds (int or None, default
    #   MAX_EXECUTION_TIME_SECONDS; None disables).
    # Outputs: a tuple of (value of the last statement or the final answer, bool is_final_answer).
    # Exceptions: InterpreterError on syntax errors or wrapping any Exception during execution (with the
    #   offending source line); ExecutionTimeoutError on timeout.
    # Side effects: mutates state and custom_tools; may import modules and run arbitrary allowed callables.
    # KNOWN ISSUE: the docstring omits authorized_imports and max_print_outputs_length and the tuple return;
    #   wrapped errors are raised without exception chaining.
    """
    Evaluate a python expression using the content of the variables stored in a state and only evaluating a given set
    of functions.

    This function will recurse through the nodes of the tree provided.

    Args:
        code (`str`):
            The code to evaluate.
        static_tools (`Dict[str, Callable]`):
            The functions that may be called during the evaluation. These can also be agents in a multiagent setting.
            These tools cannot be overwritten in the code: any assignment to their name will raise an error.
        custom_tools (`Dict[str, Callable]`):
            The functions that may be called during the evaluation.
            These tools can be overwritten in the code: any assignment to their name will overwrite them.
        state (`Dict[str, Any]`):
            A dictionary mapping variable names to values. The `state` should contain the initial inputs but will be
            updated by this function to contain all variables as they are evaluated.
            The print outputs will be stored in the state under the key "_print_outputs".
        timeout_seconds (`int`, *optional*, defaults to `MAX_EXECUTION_TIME_SECONDS`):
            Maximum time in seconds allowed for code execution. Set to `None` to disable timeout.
    """
    try:
        expression = ast.parse(code)
    except SyntaxError as e:
        raise InterpreterError(
            f"Code parsing failed on line {e.lineno} due to: {type(e).__name__}: {str(e)}\n"
            f"{e.text}"
            f"{' ' * (e.offset or 0)}^"
        )

    if state is None:
        state = {}
    static_tools = static_tools.copy() if static_tools is not None else {}
    custom_tools = custom_tools if custom_tools is not None else {}
    state["_print_outputs"] = PrintContainer()
    state["_operations_count"] = {"counter": 0}

    if "final_answer" in static_tools:
        previous_final_answer = static_tools["final_answer"]

        def final_answer(*args, **kwargs):  # Allow arbitrary arguments to be passed
            # [FUNCTION DOC]
            # What it does: HELPER: Replacement final_answer tool that calls the original and raises its
            #   result inside a FinalAnswerException.
            # Inputs: arbitrary args/kwargs forwarded to the original tool.
            # Outputs: never returns normally.
            # Exceptions: FinalAnswerException; original tool errors propagate.
            # Side effects: those of the original tool.
            raise FinalAnswerException(previous_final_answer(*args, **kwargs))

        static_tools["final_answer"] = final_answer

    # Define the actual execution logic
    def _execute_code():
        # [FUNCTION DOC]
        # What it does: HELPER: Executes the parsed module body statement by statement, truncating print
        #   output in every outcome.
        # Inputs: none (closure over code, state, tools).
        # Outputs: tuple (result, is_final_answer).
        # Exceptions: InterpreterError wrapping any Exception.
        # Side effects: mutates state, including state['_print_outputs'].value.
        result = None
        try:
            for node in expression.body:
                result = evaluate_ast(node, state, static_tools, custom_tools, authorized_imports)
            state["_print_outputs"].value = truncate_content(
                str(state["_print_outputs"]), max_length=max_print_outputs_length
            )
            is_final_answer = False
            return result, is_final_answer
        except FinalAnswerException as e:
            state["_print_outputs"].value = truncate_content(
                str(state["_print_outputs"]), max_length=max_print_outputs_length
            )
            is_final_answer = True
            return e.value, is_final_answer
        except Exception as e:
            state["_print_outputs"].value = truncate_content(
                str(state["_print_outputs"]), max_length=max_print_outputs_length
            )
            raise InterpreterError(
                f"Code execution failed at line '{ast.get_source_segment(code, node)}' due to: {type(e).__name__}: {e}"
            )

    # Apply timeout if specified
    if timeout_seconds is not None:
        _execute_code = timeout(timeout_seconds)(_execute_code)

    return _execute_code()


@dataclass
class CodeOutput:
    # [CLASS DOC]
    # What it does: Data container for the result of one executor call.
    # Inputs: output (any final value), logs (str captured print output), is_final_answer (bool).
    # Outputs: n/a.
    # Exceptions: none raised directly.
    # Side effects: none.
    output: Any
    logs: str
    is_final_answer: bool


class PythonExecutor(ABC):
    @abstractmethod
    # [CLASS DOC]
    # What it does: Abstract interface for code executors (local or remote): tools and variables are sent in,
    #   then code strings are executed by calling the instance.
    # Inputs: none (abstract).
    # Outputs: n/a.
    # Exceptions: TypeError on instantiating without implementing all abstract methods.
    # Side effects: none.
    def send_tools(self, tools: dict[str, Tool]) -> None:
        # [FUNCTION DOC]
        # What it does: Abstract: make tools callable by subsequently executed code.
        # Inputs: tools (dict of name to Tool).
        # Outputs: None.
        # Exceptions: implementation-defined.
        # Side effects: implementation-defined.
        ...

    @abstractmethod
    def send_variables(self, variables: dict[str, Any]) -> None:
        # [FUNCTION DOC]
        # What it does: Abstract: inject variables into the execution namespace.
        # Inputs: variables (dict of name to value).
        # Outputs: None.
        # Exceptions: implementation-defined.
        # Side effects: implementation-defined.
        ...

    @abstractmethod
    def __call__(self, code_action: str) -> CodeOutput:
        # [FUNCTION DOC]
        # What it does: Abstract: execute a code string.
        # Inputs: code_action (str).
        # Outputs: CodeOutput.
        # Exceptions: implementation-defined.
        # Side effects: implementation-defined.
        ...


class LocalPythonExecutor(PythonExecutor):
    # [CLASS DOC]
    # What it does: In-process PythonExecutor built on evaluate_python_code, keeping state across calls.
    # Inputs: additional_authorized_imports (list of str), max_print_outputs_length (int or None ->
    #   DEFAULT_MAX_LEN_OUTPUT), additional_functions (dict of callables or None), timeout_seconds (int or
    #   None, default MAX_EXECUTION_TIME_SECONDS). Attributes: state (starts with __name__ set to __main__),
    #   custom_tools, static_tools (None until send_tools), authorized_imports (union with
    #   BASE_BUILTIN_MODULES).
    # Outputs: n/a.
    # Exceptions: InterpreterError at construction if an authorized module is not installed.
    # Side effects: none beyond held state.
    """
    Executor of Python code in a local environment.

    This executor evaluates Python code with restricted access to imports and built-in functions.
    It is not a security sandbox: for isolated execution of untrusted code, use a remote executor.
    It maintains state between executions, allows for custom tools and functions to be made available
    to the code, and captures print outputs separately from return values.

    Args:
        additional_authorized_imports (`list[str]`):
            Additional authorized imports for the executor.
        max_print_outputs_length (`int`, defaults to `DEFAULT_MAX_LEN_OUTPUT=50_000`):
            Maximum length of the print outputs.
        additional_functions (`dict[str, Callable]`, *optional*):
            Additional Python functions to be added to the executor.
        timeout_seconds (`int`, *optional*, defaults to `MAX_EXECUTION_TIME_SECONDS`):
            Maximum time in seconds allowed for code execution. Set to `None` to disable timeout.
    """

    def __init__(
        self,
        additional_authorized_imports: list[str],
        max_print_outputs_length: int | None = None,
        additional_functions: dict[str, Callable] | None = None,
        timeout_seconds: int | None = MAX_EXECUTION_TIME_SECONDS,
    ):
        # [FUNCTION DOC]
        # What it does: Initializes state, tool dicts, limits and the authorized import list, and verifies the
        #   authorized modules are installed.
        # Inputs: see class doc.
        # Outputs: None.
        # Exceptions: InterpreterError for non-installed authorized modules.
        # Side effects: calls importlib find_spec, which may import parent packages.
        self.custom_tools = {}
        self.state = {"__name__": "__main__"}
        self.max_print_outputs_length = max_print_outputs_length
        if max_print_outputs_length is None:
            self.max_print_outputs_length = DEFAULT_MAX_LEN_OUTPUT
        self.additional_authorized_imports = additional_authorized_imports
        self.authorized_imports = list(set(BASE_BUILTIN_MODULES) | set(self.additional_authorized_imports))
        self._check_authorized_imports_are_installed()
        self.static_tools = None
        self.additional_functions = additional_functions or {}
        self.timeout_seconds = timeout_seconds

    def _check_authorized_imports_are_installed(self):
        # [FUNCTION DOC]
        # What it does: HELPER: Checks that the top-level package of every authorized import (except '*') can
        #   be found by importlib.
        # Inputs: none (uses self.authorized_imports).
        # Outputs: None.
        # Exceptions: InterpreterError listing missing modules.
        # Side effects: find_spec may import packages.
        """
        Check that all authorized imports are installed on the system.

        Handles wildcard imports ("*") and partial star-pattern imports (e.g., "os.*").

        Raises:
            InterpreterError: If any of the authorized modules are not installed.
        """
        missing_modules = [
            base_module
            for imp in self.authorized_imports
            if imp != "*" and find_spec(base_module := imp.split(".")[0]) is None
        ]
        if missing_modules:
            raise InterpreterError(
                f"Non-installed authorized modules: {', '.join(missing_modules)}. "
                f"Please install these modules or remove them from the authorized imports list."
            )

    def __call__(self, code_action: str) -> CodeOutput:
        # [FUNCTION DOC]
        # What it does: Executes a code string with the executor's tools, state, imports and limits.
        # Inputs: code_action (str).
        # Outputs: CodeOutput with output, captured logs and is_final_answer.
        # Exceptions: InterpreterError and ExecutionTimeoutError from evaluate_python_code.
        # Side effects: mutates self.state and self.custom_tools.
        output, is_final_answer = evaluate_python_code(
            code_action,
            static_tools=self.static_tools,
            custom_tools=self.custom_tools,
            state=self.state,
            authorized_imports=self.authorized_imports,
            max_print_outputs_length=self.max_print_outputs_length,
            timeout_seconds=self.timeout_seconds,
        )
        logs = str(self.state["_print_outputs"])
        return CodeOutput(output=output, logs=logs, is_final_answer=is_final_answer)

    def send_variables(self, variables: dict[str, Any]):
        # [FUNCTION DOC]
        # What it does: Merges the given variables into the persistent state.
        # Inputs: variables (dict).
        # Outputs: None.
        # Exceptions: none raised directly.
        # Side effects: mutates self.state.
        self.state.update(variables)

    def send_tools(self, tools: dict[str, Tool]):
        # Combine agent tools, base Python tools, and additional Python functions
        # [FUNCTION DOC]
        # What it does: Sets static_tools to the merge of the agent tools, BASE_PYTHON_TOOLS and additional
        #   functions, later entries overriding earlier ones.
        # Inputs: tools (dict of name to Tool).
        # Outputs: None.
        # Exceptions: none raised directly.
        # Side effects: replaces self.static_tools.
        # KNOWN ISSUE: an agent tool whose name collides with a base Python tool (e.g. print, type) is
        #   silently overridden by the builtin.
        self.static_tools = {**tools, **BASE_PYTHON_TOOLS.copy(), **self.additional_functions}


__all__ = ["evaluate_python_code", "LocalPythonExecutor"]
