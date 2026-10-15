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
# Purpose: remote (sandboxed) Python code executors for code agents. Defines the abstract base
#   RemotePythonExecutor and four concrete backends: E2BExecutor (E2B sandbox SDK), DockerExecutor (local
#   Docker container running Jupyter Kernel Gateway), ModalExecutor (Modal sandbox running Jupyter Kernel
#   Gateway) and BlaxelExecutor (Blaxel sandbox running a Jupyter image). The Jupyter-based backends share
#   module-level helpers that create a kernel over HTTP and run code over the kernel websocket protocol.
# Exports: __all__ lists BlaxelExecutor, E2BExecutor, ModalExecutor, DockerExecutor (RemotePythonExecutor is
#   also importable). Depends on requests, PIL, smolagents FinalAnswerTool, CodeOutput/PythonExecutor,
#   LogLevel, SafeSerializer/SerializationError, Tool/get_tools_definition_code, AgentError; optional backends
#   import e2b_code_interpreter, docker, modal, blaxel and websocket lazily.
# Final answer protocol: the final_answer tool is patched so calling it raises a remote FinalAnswerException
#   whose value is a prefixed string ('safe:' + JSON or 'pickle:' + base64 pickle); executors detect that
#   exception name in the remote error and deserialize the value into the final answer.
# Exceptions: none raised at import time (missing python-dotenv is ignored).
# Side effects: on import, if python-dotenv is installed, load_dotenv() reads a .env file into os.environ.
import base64
import inspect
import json
import pickle
import re
import secrets
import time
import uuid
from contextlib import closing
from io import BytesIO
from textwrap import dedent
from typing import Any, Optional

import PIL.Image
import requests
from requests.exceptions import RequestException

from .default_tools import FinalAnswerTool
from .local_python_executor import CodeOutput, PythonExecutor
from .monitoring import LogLevel
from .serialization import SafeSerializer, SerializationError
from .tools import Tool, get_tools_definition_code
from .utils import AgentError


__all__ = ["BlaxelExecutor", "E2BExecutor", "ModalExecutor", "DockerExecutor"]


try:
    from dotenv import load_dotenv

    load_dotenv()
except ModuleNotFoundError:
    pass


class RemotePythonExecutor(PythonExecutor):
    # [CLASS DOC]
    # Abstract base for executors that run agent code in a remote Python kernel/sandbox. Subclasses must
    #   implement run_code_raise_errors; this class supplies tool/variable shipping, package installation,
    #   final-answer patching and final-answer deserialization.
    # Inputs (constructor): additional_imports (list of str, packages to install), logger (object with
    #   log/log_error), allow_pickle (bool, default False, permits pickle fallback when
    #   serializing/deserializing).
    # Key attributes: FINAL_ANSWER_EXCEPTION (class constant, the string 'FinalAnswerException' used to
    #   recognize the final answer in remote errors), additional_imports, logger, allow_pickle,
    #   installed_packages (list of str).
    # Outputs: n/a. Exceptions: see methods. Side effects: see methods.
    """
    Executor of Python code in a remote environment.

    Args:
        additional_imports (`list[str]`): Additional Python packages to install.
        logger (`Logger`): Logger to use for output and errors.
        allow_pickle (`bool`, default `False`): Whether to allow pickle serialization for objects that cannot be safely serialized to JSON.
            - `False` (default, recommended): Only safe JSON serialization is used. Raises error if object cannot be safely serialized.
            - `True` (legacy mode): Tries safe JSON serialization first, falls back to pickle with warning if needed.

            **Security Warning:** Pickle deserialization can execute arbitrary code. Only set `allow_pickle=True`
            if you fully trust the execution environment and need backward compatibility with custom types.
    """

    FINAL_ANSWER_EXCEPTION = "FinalAnswerException"

    def __init__(
        self,
        additional_imports: list[str],
        logger,
        allow_pickle: bool = False,
    ):
        # [FUNCTION DOC]
        # Stores constructor arguments and initializes installed_packages to an empty list. Does not install
        #   anything itself.
        # Inputs: additional_imports (list of str), logger, allow_pickle (bool, default False).
        # Outputs: none.
        # Exceptions: none raised directly (AttributeError if logger lacks log).
        # Side effects: logs 'Initializing executor, hold on...' via logger.log.
        self.additional_imports = additional_imports
        self.logger = logger
        self.allow_pickle = allow_pickle
        self.logger.log("Initializing executor, hold on...")
        self.installed_packages = []

    def run_code_raise_errors(self, code: str) -> CodeOutput:
        # [FUNCTION DOC]
        # Abstract hook: subclasses execute code remotely and return a CodeOutput (output, logs,
        #   is_final_answer), raising on remote errors other than the final-answer signal.
        # Inputs: code (str, Python source to run).
        # Outputs: CodeOutput (in subclasses).
        # Exceptions: always raises NotImplementedError in the base class.
        # Side effects: none.
        """
        Execute Python code in the remote environment and return the result.

        Args:
            code (`str`): Python code to execute.

        Returns:
            `CodeOutput`: Code output containing the result, logs, and whether it is the final answer.
        """
        raise NotImplementedError

    def send_tools(self, tools: dict[str, Tool]):
        # [FUNCTION DOC]
        # Ships tools to the remote environment. If a 'final_answer' tool is present, patches it in place so
        #   it raises FinalAnswerException. Collects every requirement from each tool's serialized dict,
        #   excluding already-installed packages and 'smolagents'; maps 'PIL' to 'pillow'; installs the
        #   remaining set via install_packages and appends the result to installed_packages. Then generates
        #   the tool definition code and, if non-empty, runs it remotely and logs its logs.
        # Inputs: tools (dict of tool name to Tool).
        # Outputs: none.
        # Exceptions: propagates errors from install_packages / run_code_raise_errors (e.g. AgentError) and
        #   tool.to_dict.
        # Side effects: mutates the final_answer tool instance's class, mutates installed_packages, remote
        #   installs and code execution, logging.
        # KNOWN ISSUE: package set ordering is arbitrary (set), so the pip command order is nondeterministic.
        if "final_answer" in tools:
            self._patch_final_answer_with_exception(tools["final_answer"])
        # Install tool packages
        packages_to_install = {
            pkg
            for tool in tools.values()
            for pkg in tool.to_dict()["requirements"]
            if pkg not in self.installed_packages + ["smolagents"]
        }
        if "PIL" in packages_to_install:
            packages_to_install.discard("PIL")
            packages_to_install.add("pillow")
        if packages_to_install:
            self.installed_packages += self.install_packages(list(packages_to_install))
        # Get tool definitions
        code = get_tools_definition_code(tools)
        if code:
            code_output = self.run_code_raise_errors(code)
            self.logger.log(code_output.logs)

    def send_variables(self, variables: dict[str, Any]):
        # [FUNCTION DOC]
        # Injects variables into the remote kernel namespace. No-op for an empty/falsy dict. Otherwise
        #   serializes the dict with SafeSerializer (pickle fallback only if allow_pickle), then runs remote
        #   code consisting of the deserializer helper source, a call that deserializes the repr of the
        #   payload into a dict, and an update of the remote locals with it (at kernel top level locals are
        #   the globals).
        # Inputs: variables (dict of name to value).
        # Outputs: none.
        # Exceptions: SerializationError if a value cannot be safely serialized with allow_pickle False;
        #   errors from run_code_raise_errors propagate.
        # Side effects: remote code execution that defines/overwrites names (including the helper names
        #   vars_dict and _deserialize) in the remote namespace.
        """Send variables to the kernel namespace using SafeSerializer.

        Uses prefix-based format ("safe:..." or "pickle:...").
        When allow_pickle=False, only safe JSON serialization is allowed.
        When allow_pickle=True, pickle fallback is enabled for complex types.
        """
        if not variables:
            return

        serialized = SafeSerializer.dumps(variables, allow_pickle=self.allow_pickle)
        code = f"""
{SafeSerializer.get_deserializer_code(self.allow_pickle)}
vars_dict = _deserialize({repr(serialized)})
locals().update(vars_dict)
"""
        self.run_code_raise_errors(code)

    def __call__(self, code_action: str) -> CodeOutput:
        # [FUNCTION DOC]
        # Makes the executor callable: delegates directly to run_code_raise_errors.
        # Inputs: code_action (str, code to run).
        # Outputs: the CodeOutput from run_code_raise_errors.
        # Exceptions: whatever run_code_raise_errors raises.
        # Side effects: remote code execution.
        """Run the code and determine if it is the final answer."""
        return self.run_code_raise_errors(code_action)

    def install_packages(self, additional_imports: list[str]):
        # [FUNCTION DOC]
        # Installs packages remotely by running a notebook-style shell command (pip install followed by space-
        #   joined names) through run_code_raise_errors and logging its logs; does nothing remotely if the
        #   list is empty.
        # Inputs: additional_imports (list of str).
        # Outputs: the same list passed in (assumed installed; success is not verified).
        # Exceptions: errors from run_code_raise_errors propagate.
        # Side effects: remote pip install, logging.
        # KNOWN ISSUE: returns the input list even if pip fails without raising, so installed_packages may be
        #   inaccurate.
        if additional_imports:
            code_output = self.run_code_raise_errors(f"!pip install {' '.join(additional_imports)}")
            self.logger.log(code_output.logs)
        return additional_imports

    def _patch_final_answer_with_exception(self, final_answer_tool: FinalAnswerTool):
        # [FUNCTION DOC]
        # HELPER: Replaces the class of the given final_answer tool instance with a dynamically created
        #   subclass whose forward wraps the original: it calls the original forward (renamed _forward),
        #   serializes the result into a prefixed string and raises FinalAnswerException carrying it. Because
        #   tool code is regenerated from source and sent remotely, it sets a source attribute on the new
        #   forward (with the allow_pickle value textually baked in instead of the closure variable) and on
        #   _forward (original source with the method renamed).
        # Inputs: final_answer_tool (FinalAnswerTool instance).
        # Outputs: none.
        # Exceptions: OSError/TypeError from inspect.getsource if the original forward source is unavailable.
        # Side effects: mutates final_answer_tool's class; sets source attributes on function objects (the
        #   original forward function object is shared, so its source attribute is overwritten globally).
        # KNOWN ISSUE: the class docstring says pickle fallback happens 'with warning', but no warning is
        #   emitted here.
        """Patch the FinalAnswerTool to raise an exception.

        This is necessary because the remote executors
        rely on the FinalAnswerTool to detect the final answer.
        It modifies the `forward` method of the FinalAnswerTool to raise
        a `FinalAnswerException` with the final answer as a serialized value.
        This allows the executor to catch this exception and return the final answer.

        Uses prefix-based format ("safe:" or "pickle:") for serialization.

        Args:
            final_answer_tool (`FinalAnswerTool`): FinalAnswerTool instance to patch.
        """

        # Create a new class that inherits from the original FinalAnswerTool
        class _FinalAnswerTool(final_answer_tool.__class__):
            # [CLASS DOC]
            # HELPER: Empty dynamic subclass of the tool's original class; gets forward and _forward
            #   attributes assigned after creation.
            # Inputs: none. Attributes: forward (patched), _forward (original). Outputs/Exceptions/Side
            #   effects: none.
            pass

        # Add a new forward method that raises the FinalAnswerException
        # NOTE: Serialization logic is inlined here because this method's source code
        # is extracted and sent to remote environments where external references don't exist
        # Capture settings via closure
        allow_pickle_setting = self.allow_pickle

        def forward(self, *args, **kwargs) -> Any:
            # [FUNCTION DOC]
            # HELPER: Patched tool entry point (also executed remotely from its source). Calls the original
            #   forward with all arguments, serializes the result, and raises a locally defined
            #   FinalAnswerException with the serialized string. All helpers are defined inline because it
            #   must be self-contained when sent as source.
            # Inputs: self (tool), any positional/keyword args forwarded to the original forward.
            # Outputs: never returns normally.
            # Exceptions: always raises FinalAnswerException (a BaseException subclass); raises the local
            #   SerializationError if the result cannot be safely serialized and pickle is disallowed.
            # Side effects: imports base64, json, io (and lazily PIL, datetime, decimal, pathlib, numpy,
            #   dataclasses, pickle).
            import base64
            import json
            from io import BytesIO

            # Baked in from closure at patch time
            ALLOW_PICKLE = allow_pickle_setting

            class SerializationError(Exception):
                # [CLASS DOC]
                # HELPER: Local exception type signaling an object cannot be converted to a JSON-safe form
                #   (redefined inline so the remote source has no external references).
                # Inputs: message. Outputs/Side effects: none.
                pass

            def _to_json_safe(obj):
                # [FUNCTION DOC]
                # HELPER: Recursively converts an object to JSON-compatible data. Primitives (str, int, float,
                #   bool, None) are returned as-is; dicts with all-string keys map values recursively, else
                #   become a tagged list of key/value pairs; lists map recursively; tuple, set, frozenset,
                #   bytes (base64), complex, PIL images (PNG base64), datetime, date, time, timedelta (total
                #   seconds), Decimal, Path, numpy arrays (list + dtype) and dataclass instances (class name,
                #   module and field data) become dicts tagged with a type marker key; numpy scalars become
                #   Python scalars.
                # Inputs: obj (any).
                # Outputs: JSON-safe structure.
                # Exceptions: SerializationError (local) for unsupported types.
                # Side effects: none (lazy imports only).
                if isinstance(obj, (str, int, float, bool, type(None))):
                    return obj
                elif isinstance(obj, dict):
                    # Check if all keys are strings (JSON-compatible)
                    if all(isinstance(k, str) for k in obj.keys()):
                        return {k: _to_json_safe(v) for k, v in obj.items()}
                    else:
                        return {
                            "__type__": "dict_with_complex_keys",
                            "data": [[_to_json_safe(k), _to_json_safe(v)] for k, v in obj.items()],
                        }
                elif isinstance(obj, list):
                    return [_to_json_safe(item) for item in obj]
                elif isinstance(obj, tuple):
                    return {"__type__": "tuple", "data": [_to_json_safe(item) for item in obj]}
                elif isinstance(obj, set):
                    return {"__type__": "set", "data": [_to_json_safe(item) for item in obj]}
                elif isinstance(obj, bytes):
                    return {"__type__": "bytes", "data": base64.b64encode(obj).decode()}
                elif isinstance(obj, complex):
                    return {"__type__": "complex", "real": obj.real, "imag": obj.imag}
                elif isinstance(obj, frozenset):
                    return {"__type__": "frozenset", "data": [_to_json_safe(item) for item in obj]}

                # Try PIL Image
                try:
                    import PIL.Image

                    if isinstance(obj, PIL.Image.Image):
                        buffer = BytesIO()
                        obj.save(buffer, format="PNG")
                        return {"__type__": "PIL.Image", "data": base64.b64encode(buffer.getvalue()).decode()}
                except ImportError:
                    pass

                # Lazy imports for less common types
                from datetime import date, datetime, time, timedelta
                from decimal import Decimal
                from pathlib import Path

                if isinstance(obj, datetime):
                    return {"__type__": "datetime", "data": obj.isoformat()}
                elif isinstance(obj, date):
                    return {"__type__": "date", "data": obj.isoformat()}
                elif isinstance(obj, time):
                    return {"__type__": "time", "data": obj.isoformat()}
                elif isinstance(obj, timedelta):
                    return {"__type__": "timedelta", "total_seconds": obj.total_seconds()}
                elif isinstance(obj, Decimal):
                    return {"__type__": "Decimal", "data": str(obj)}
                elif isinstance(obj, Path):
                    return {"__type__": "Path", "data": str(obj)}

                # Try numpy if available
                try:
                    import numpy as np

                    if isinstance(obj, np.ndarray):
                        return {"__type__": "ndarray", "data": obj.tolist(), "dtype": str(obj.dtype)}
                    elif isinstance(obj, (np.integer, np.floating)):
                        return obj.item()
                except ImportError:
                    pass

                # Try dataclass
                import dataclasses

                if dataclasses.is_dataclass(obj) and not isinstance(obj, type):
                    return {
                        "__type__": "dataclass",
                        "class_name": type(obj).__name__,
                        "module": type(obj).__module__,
                        "data": {f.name: _to_json_safe(getattr(obj, f.name)) for f in dataclasses.fields(obj)},
                    }

                # Cannot safely serialize - raise error for safe mode
                raise SerializationError(f"Cannot safely serialize object of type {type(obj).__name__}")

            def _serialize_with_fallback(obj):
                # [FUNCTION DOC]
                # HELPER: Serializes obj to the prefixed wire format. With pickle disallowed: 'safe:' plus
                #   JSON of the JSON-safe form. With pickle allowed: tries the same; on SerializationError
                #   falls back to 'pickle:' plus base64 pickle; if pickling fails (PicklingError, TypeError,
                #   AttributeError) falls back to 'safe:' plus JSON of the string form.
                # Inputs: obj (any).
                # Outputs: str with 'safe:' or 'pickle:' prefix.
                # Exceptions: SerializationError when pickle disallowed and object unsupported; json errors
                #   propagate.
                # Side effects: none.
                """Serialize with safe method, fallback to pickle if allowed."""
                import pickle

                if not ALLOW_PICKLE:
                    # Safe ONLY mode - NO pickle fallback, raise error if can't serialize
                    json_safe = _to_json_safe(obj)  # Will raise SerializationError if fails
                    return "safe:" + json.dumps(json_safe)
                else:
                    # Try safe first, fallback to pickle if allowed
                    try:
                        json_safe = _to_json_safe(obj)
                        return "safe:" + json.dumps(json_safe)
                    except SerializationError:
                        # Fallback to pickle
                        try:
                            return "pickle:" + base64.b64encode(pickle.dumps(obj)).decode()
                        except (pickle.PicklingError, TypeError, AttributeError):
                            # Last resort: string representation
                            return "safe:" + json.dumps(str(obj))

            class FinalAnswerException(BaseException):
                # [CLASS DOC]
                # HELPER: Local BaseException subclass used as a control-flow signal carrying the serialized
                #   final answer; its class name is matched by executors against FINAL_ANSWER_EXCEPTION.
                # Inputs (constructor): value (serialized str). Attributes: value. Side effects: none.
                def __init__(self, value):
                    # [FUNCTION DOC]
                    # HELPER: Stores the serialized value on the exception.
                    # Inputs: value (str). Outputs: none. Exceptions: none raised directly. Side effects: sets
                    #   self.value.
                    # Note: the base initializer is not called, but BaseException construction still records
                    #   the value as its argument, so the remote error value string equals the serialized
                    #   payload.
                    self.value = value

            raise FinalAnswerException(_serialize_with_fallback(self._forward(*args, **kwargs)))

        # - Set the new forward method function to the _FinalAnswerTool class
        _FinalAnswerTool.forward = forward

        # Set __source__ with the actual values baked in (closures don't survive source extraction)
        source = inspect.getsource(forward)
        source = source.replace("ALLOW_PICKLE = allow_pickle_setting", f"ALLOW_PICKLE = {allow_pickle_setting}")
        forward.__source__ = source

        # Rename the original forward method to _forward
        # - Get the original forward method function from the final_answer_tool instance
        original_forward_function = final_answer_tool.forward.__func__
        # - Set the new _forward method function to the _FinalAnswerTool class
        _FinalAnswerTool._forward = original_forward_function
        # - Update the source code of the new forward method to match the original but with the new name
        _FinalAnswerTool._forward.__source__ = inspect.getsource(original_forward_function).replace(
            "def forward(", "def _forward("
        )

        # Set the new class as the class of the final_answer_tool instance
        final_answer_tool.__class__ = _FinalAnswerTool

    @staticmethod
    def _deserialize_final_answer(encoded_value: str, allow_pickle: bool = False) -> Any:
        # [FUNCTION DOC]
        # HELPER: Static method decoding a final-answer string. 'safe:' prefix: parse the rest as JSON and
        #   rebuild objects via SafeSerializer.from_json_safe. 'pickle:' prefix: if allowed, base64-decode and
        #   unpickle the rest.
        # Inputs: encoded_value (str), allow_pickle (bool, default False).
        # Outputs: the deserialized Python object.
        # Exceptions: SerializationError if pickle payload is given with allow_pickle False or prefix is
        #   unknown; json/base64/pickle errors propagate.
        # Side effects: unpickling may execute arbitrary code when allowed.
        """Deserialize final answer with format detection.

        Accepts explicit prefix-based formats only:
        - "safe:" for JSON-safe payloads
        - "pickle:" for pickle payloads (only when allow_pickle=True)

        Args:
            encoded_value (`str`): Serialized string from FinalAnswerException.
            allow_pickle (`bool`, default `False`): Whether to allow pickle deserialization.

        Returns:
            `Any`: Deserialized Python object.

        Raises:
            SerializationError: If pickle data is rejected.
        """
        if encoded_value.startswith("safe:"):
            json_data = json.loads(encoded_value[5:])
            return SafeSerializer.from_json_safe(json_data)
        elif encoded_value.startswith("pickle:"):
            if not allow_pickle:
                raise SerializationError("Pickle data rejected: allow_pickle=False")
            return pickle.loads(base64.b64decode(encoded_value[7:]))
        else:
            raise SerializationError("Unknown final answer format: expected 'safe:' or 'pickle:' prefix")


class E2BExecutor(RemotePythonExecutor):
    # [CLASS DOC]
    # Remote executor backed by an E2B code-interpreter sandbox.
    # Inputs (constructor): additional_imports, logger, allow_pickle (as base), **kwargs passed to the E2B
    #   Sandbox constructor/factory.
    # Key attributes: sandbox (E2B Sandbox instance), installed_packages.
    # Exceptions/Side effects: see methods.
    """
    Remote Python code executor in an E2B sandbox.

    Args:
        additional_imports (`list[str]`): Additional Python packages to install.
        logger (`Logger`): Logger to use for output and errors.
        allow_pickle (`bool`, default `False`): Whether to allow pickle serialization for objects that cannot be safely serialized to JSON.
            - `False` (default, recommended): Only safe JSON serialization is used. Raises error if object cannot be safely serialized.
            - `True` (legacy mode): Tries safe JSON serialization first, falls back to pickle with warning if needed.

            **Security Warning:** Pickle deserialization can execute arbitrary code. Only set `allow_pickle=True`
            if you fully trust the execution environment and need backward compatibility with custom types.
        **kwargs: Additional keyword arguments to pass to the E2B Sandbox instantiation.
    """

    def __init__(
        self,
        additional_imports: list[str],
        logger,
        allow_pickle: bool = False,
        **kwargs,
    ):
        # [FUNCTION DOC]
        # Initializes base state, imports e2b_code_interpreter.Sandbox, creates a sandbox (via the create
        #   factory when available for e2b v2, else the constructor for v1) with kwargs, installs
        #   additional_imports, and logs readiness.
        # Inputs: additional_imports (list of str), logger, allow_pickle (bool, default False), **kwargs for
        #   Sandbox.
        # Outputs: none.
        # Exceptions: ModuleNotFoundError with install hint if e2b is missing; sandbox creation and install
        #   errors propagate.
        # Side effects: creates a remote sandbox (network), remote pip install, logging.
        super().__init__(additional_imports, logger, allow_pickle)
        try:
            from e2b_code_interpreter import Sandbox
        except ModuleNotFoundError:
            raise ModuleNotFoundError(
                """Please install 'e2b' extra to use E2BExecutor: `pip install 'smolagents[e2b]'`"""
            )
        # Support both e2b v1 and v2 constructors
        # v2 exposes Sandbox.create(...), while v1 uses Sandbox(...)
        if hasattr(Sandbox, "create"):
            self.sandbox = Sandbox.create(**kwargs)
        else:
            self.sandbox = Sandbox(**kwargs)
        self.installed_packages = self.install_packages(additional_imports)
        self.logger.log("E2B is running", level=LogLevel.INFO)

    def run_code_raise_errors(self, code: str) -> CodeOutput:
        # [FUNCTION DOC]
        # Runs code in the sandbox and joins stdout log entries with newlines as logs. On error: if the error
        #   name equals FINAL_ANSWER_EXCEPTION, deserializes its value and returns it as a final answer;
        #   otherwise raises AgentError with logs, error name, value and traceback. On success: returns None
        #   output if there are no results; else for the first main result returns a PIL image decoded from
        #   jpeg or png base64 data if present, otherwise the first non-None of chart, data, html, javascript,
        #   json, latex, markdown, pdf, svg, text; else None output.
        # Inputs: code (str, Python source to run).
        # Outputs: CodeOutput (output, logs, is_final_answer).
        # Exceptions: AgentError on non-final remote errors; SerializationError on bad final answer payloads.
        # Side effects: remote execution (network).
        """
        Execute Python code in the E2B sandbox and return the result.

        Args:
            code (`str`): Python code to execute.

        Returns:
            `CodeOutput`: Code output containing the result, logs, and whether it is the final answer.
        """
        execution = self.sandbox.run_code(code)
        execution_logs = "\n".join([str(log) for log in execution.logs.stdout])

        # Handle errors
        if execution.error:
            # Check if the error is a FinalAnswerException
            if execution.error.name == RemotePythonExecutor.FINAL_ANSWER_EXCEPTION:
                final_answer = self._deserialize_final_answer(execution.error.value, self.allow_pickle)
                return CodeOutput(output=final_answer, logs=execution_logs, is_final_answer=True)

            # Construct error message
            error_message = (
                f"{execution_logs}\n"
                f"Executing code yielded an error:\n"
                f"{execution.error.name}\n"
                f"{execution.error.value}\n"
                f"{execution.error.traceback}"
            )
            raise AgentError(error_message, self.logger)

        # Handle results
        if not execution.results:
            return CodeOutput(output=None, logs=execution_logs, is_final_answer=False)

        for result in execution.results:
            if not result.is_main_result:
                continue
            # Handle image outputs
            for attribute_name in ["jpeg", "png"]:
                img_data = getattr(result, attribute_name, None)
                if img_data is not None:
                    decoded_bytes = base64.b64decode(img_data.encode("utf-8"))
                    return CodeOutput(
                        output=PIL.Image.open(BytesIO(decoded_bytes)), logs=execution_logs, is_final_answer=False
                    )
            # Handle other data formats
            for attribute_name in [
                "chart",
                "data",
                "html",
                "javascript",
                "json",
                "latex",
                "markdown",
                "pdf",
                "svg",
                "text",
            ]:
                data = getattr(result, attribute_name, None)
                if data is not None:
                    return CodeOutput(output=data, logs=execution_logs, is_final_answer=False)
        # If no main result found, return None
        return CodeOutput(output=None, logs=execution_logs, is_final_answer=False)

    def cleanup(self):
        # [FUNCTION DOC]
        # Kills the sandbox if the sandbox attribute exists, logs start/completion and deletes the attribute.
        # Inputs: none. Outputs: none.
        # Exceptions: none; any exception is caught and logged with log_error.
        # Side effects: remote sandbox termination, logging, removes self.sandbox.
        # KNOWN ISSUE: unlike other executors, E2BExecutor defines no delete method, so callers using delete()
        #   fail.
        """Clean up the E2B sandbox and resources."""
        try:
            if hasattr(self, "sandbox"):
                self.logger.log("Shutting down sandbox...", level=LogLevel.INFO)
                self.sandbox.kill()
                self.logger.log("Sandbox cleanup completed", level=LogLevel.INFO)
                del self.sandbox
        except Exception as e:
            self.logger.log_error(f"Error during cleanup: {e}")


def _websocket_send_execute_request(code: str, ws) -> str:
    # [FUNCTION DOC]
    # HELPER: Builds a Jupyter messaging protocol (v5.0) execute_request with a fresh random msg_id and
    #   session id, anonymous username, non-silent, history stored, stdin disallowed, and sends it as JSON
    #   over the websocket.
    # Inputs: code (str), ws (websocket object with send).
    # Outputs: the msg_id string, used to match replies.
    # Exceptions: websocket send errors propagate.
    # Side effects: network write on ws.
    """Send code execution request to kernel."""
    import uuid

    # Generate a unique message ID
    msg_id = str(uuid.uuid4())

    # Create execute request
    execute_request = {
        "header": {
            "msg_id": msg_id,
            "username": "anonymous",
            "session": str(uuid.uuid4()),
            "msg_type": "execute_request",
            "version": "5.0",
        },
        "parent_header": {},
        "metadata": {},
        "content": {
            "code": code,
            "silent": False,
            "store_history": True,
            "user_expressions": {},
            "allow_stdin": False,
        },
    }

    ws.send(json.dumps(execute_request))
    return msg_id


def _websocket_run_code_raise_errors(code: str, ws, logger, allow_pickle: bool = False) -> CodeOutput:
    # [FUNCTION DOC]
    # HELPER: Executes code on a Jupyter kernel over a websocket. Sends an execute request, then reads
    #   messages, ignoring those whose parent msg_id differs. Accumulates stream text as logs; execute_result
    #   sets the result to the text/plain data; error messages named FINAL_ANSWER_EXCEPTION set the
    #   deserialized final answer and flag it, other errors raise AgentError with the joined traceback; a
    #   status message with idle state ends the loop.
    # Inputs: code (str), ws (websocket), logger, allow_pickle (bool, default False).
    # Outputs: CodeOutput(output=result or None, logs=concatenated stream text, is_final_answer flag).
    # Exceptions: any exception is logged via logger.log_error then re-raised (AgentError, SerializationError,
    #   websocket and JSON errors).
    # Side effects: network I/O on ws, logging.
    # KNOWN ISSUE: no timeout; blocks forever if the kernel never reports idle. execute_result outputs are
    #   only plain text (rich outputs such as images are lost unless returned via final_answer).
    """Run code over a websocket.

    Args:
        code (`str`): Python code to execute.
        ws (`websocket.WebSocket`): Websocket instance.
        logger (`Logger`): Logger instance.
        allow_pickle (`bool`, default `False`): Whether to allow pickle deserialization.

    Returns:
        `CodeOutput`
    """
    try:
        # Send execute request
        msg_id = _websocket_send_execute_request(code, ws)

        # Collect output and results
        outputs = []
        result = None
        is_final_answer = False

        while True:
            msg = json.loads(ws.recv())
            parent_msg_id = msg.get("parent_header", {}).get("msg_id")
            # Skip unrelated messages
            if parent_msg_id != msg_id:
                continue
            msg_type = msg.get("msg_type", "")
            msg_content = msg.get("content", {})
            if msg_type == "stream":
                outputs.append(msg_content["text"])
            elif msg_type == "execute_result":
                result = msg_content["data"].get("text/plain", None)
            elif msg_type == "error":
                if msg_content.get("ename", "") == RemotePythonExecutor.FINAL_ANSWER_EXCEPTION:
                    result = RemotePythonExecutor._deserialize_final_answer(
                        msg_content.get("evalue", ""), allow_pickle
                    )
                    is_final_answer = True
                else:
                    raise AgentError("\n".join(msg_content.get("traceback", [])), logger)
            elif msg_type == "status" and msg_content["execution_state"] == "idle":
                break

        return CodeOutput(output=result, logs="".join(outputs), is_final_answer=is_final_answer)

    except Exception as e:
        logger.log_error(f"Code execution failed: {e}")
        raise


def _create_kernel_http(crate_kernel_endpoint: str, logger, headers: Optional[dict] = None) -> str:
    # [FUNCTION DOC]
    # HELPER: POSTs to a Jupyter kernels endpoint to start a new kernel and returns its id from the JSON
    #   response.
    # Inputs: crate_kernel_endpoint (str URL, including token query), logger, headers (optional dict of HTTP
    #   headers).
    # Outputs: kernel id (str).
    # Exceptions: RuntimeError with status and body if the status is not 201; requests/JSON errors propagate.
    # Side effects: HTTP request; on failure logs request/response details (including request headers) via
    #   log_error.
    # KNOWN ISSUE: parameter name is misspelled ('crate' for 'create'); failure log may leak auth
    #   headers/tokens.
    """Create kernel using http."""

    r = requests.post(crate_kernel_endpoint, headers=headers)
    if r.status_code != 201:
        error_details = {
            "status_code": r.status_code,
            "headers": dict(r.headers),
            "url": r.url,
            "body": r.text,
            "request_method": r.request.method,
            "request_headers": dict(r.request.headers),
            "request_body": r.request.body,
        }
        logger.log_error(f"Failed to create kernel. Details: {json.dumps(error_details, indent=2)}")
        raise RuntimeError(f"Failed to create kernel: Status {r.status_code}\nResponse: {r.text}") from None
    return r.json()["id"]


class DockerExecutor(RemotePythonExecutor):
    # [CLASS DOC]
    # Remote executor that runs a Jupyter Kernel Gateway inside a local Docker container and talks to it over
    #   HTTP and websockets.
    # Inputs (constructor): additional_imports, logger, allow_pickle, host (str, default 127.0.0.1), port
    #   (int, default 8888), image_name (str, default 'jupyter-kernel'), build_new_image (bool, default True),
    #   container_run_kwargs (optional dict), dockerfile_content (optional str, default python:3.12 image with
    #   kernel gateway).
    # Key attributes: host, port, image_name, dockerfile_content, client (docker client), container, base_url,
    #   kernel_id, ws_url.
    """
    Remote Python code executor using Jupyter Kernel Gateway in a Docker container.

    Args:
        additional_imports (`list[str]`): Additional Python packages to install.
        logger (`Logger`): Logger to use for output and errors.
        allow_pickle (`bool`, default `False`): Whether to allow pickle serialization for objects that cannot be safely serialized to JSON.
            - `False` (default, recommended): Only safe JSON serialization is used. Raises error if object cannot be safely serialized.
            - `True` (legacy mode): Tries safe JSON serialization first, falls back to pickle with warning if needed.

            **Security Warning:** Pickle deserialization can execute arbitrary code. Only set `allow_pickle=True`
            if you fully trust the execution environment and need backward compatibility with custom types.
        host (`str`, default `"127.0.0.1"`): Host to bind to.
        port (`int`, default `8888`): Port to bind to.
        image_name (`str`, default `"jupyter-kernel"`): Name of the Docker image to use. If the image doesn't exist, it will be built.
        build_new_image (`bool`, default `True`): Whether to rebuild a new image even if it already exists.
        container_run_kwargs (`dict`, *optional*): Additional keyword arguments to pass to the Docker container run command.
        dockerfile_content (`str`, *optional*): Custom Dockerfile content. If `None`, uses default.
    """

    def __init__(
        self,
        additional_imports: list[str],
        logger,
        allow_pickle: bool = False,
        host: str = "127.0.0.1",
        port: int = 8888,
        image_name: str = "jupyter-kernel",
        build_new_image: bool = True,
        container_run_kwargs: dict[str, Any] | None = None,
        dockerfile_content: str | None = None,
    ):
        # [FUNCTION DOC]
        # Imports docker, stores settings and Dockerfile content, connects to the Docker daemon, uses an
        #   existing image if build_new_image is False and it exists (otherwise builds from the Dockerfile,
        #   logging build stream at DEBUG), then runs a detached container with user kwargs plus port mapping
        #   8888/tcp to (host, port) and a random KG_AUTH_TOKEN env var (list-form environment is converted to
        #   a dict). Waits up to 5 seconds for running status, waits for the gateway, creates a kernel, builds
        #   the ws URL with the token, and installs additional_imports.
        # Inputs: see class doc.
        # Outputs: none.
        # Exceptions: ModuleNotFoundError if docker is missing; RuntimeError if the daemon is unreachable; any
        #   failure during build/start/kernel creation triggers cleanup and is re-raised as RuntimeError.
        # Side effects: builds images, starts a container, network calls, sleeps, logging; mutates the user's
        #   container_run_kwargs ports dict if one was provided (shallow copy only).
        # KNOWN ISSUE: if the container never reaches running state, or the server never becomes ready,
        #   initialization continues anyway and only fails later at kernel creation.
        super().__init__(additional_imports, logger, allow_pickle)
        try:
            import docker
        except ModuleNotFoundError:
            raise ModuleNotFoundError(
                "Please install 'docker' extra to use DockerExecutor: `pip install 'smolagents[docker]'`"
            )
        self.host = host
        self.port = port
        self.image_name = image_name

        self.dockerfile_content = dockerfile_content or dedent(
            """\
            FROM python:3.12-bullseye

            RUN pip install jupyter_kernel_gateway jupyter_client ipykernel

            EXPOSE 8888
            CMD ["jupyter", "kernelgateway", "--KernelGatewayApp.ip=0.0.0.0", "--KernelGatewayApp.port=8888"]
            """
        )

        # Initialize Docker
        try:
            self.client = docker.from_env()
        except docker.errors.DockerException as e:
            raise RuntimeError("Could not connect to Docker daemon: make sure Docker is running.") from e

        # Build and start container
        try:
            # Check if image exists, unless forced to rebuild
            if not build_new_image:
                try:
                    self.client.images.get(self.image_name)
                    self.logger.log(f"Using existing Docker image: {self.image_name}", level=LogLevel.INFO)
                except docker.errors.ImageNotFound:
                    self.logger.log(f"Image {self.image_name} not found, building...", level=LogLevel.INFO)
                    build_new_image = True

            if build_new_image:
                self.logger.log(f"Building Docker image {self.image_name}...", level=LogLevel.INFO)
                dockerfile_obj = BytesIO(self.dockerfile_content.encode("utf-8"))
                _, build_logs = self.client.images.build(fileobj=dockerfile_obj, tag=self.image_name)
                for log_chunk in build_logs:
                    # Only log non-empty messages
                    if log_message := log_chunk.get("stream", "").rstrip():
                        self.logger.log(log_message, level=LogLevel.DEBUG)

            self.logger.log(f"Starting container on {host}:{port}...", level=LogLevel.INFO)
            # Create base container parameters
            container_kwargs = {}
            if container_run_kwargs:
                container_kwargs.update(container_run_kwargs)

            # Ensure required port mapping and background running
            if not isinstance(container_kwargs.get("ports"), dict):
                container_kwargs["ports"] = {}
            container_kwargs["ports"]["8888/tcp"] = (host, port)
            container_kwargs["detach"] = True

            # Generate auth token and pass it to the kernel gateway via the standard KG_AUTH_TOKEN env var
            token = secrets.token_urlsafe(16)
            env = container_kwargs.get("environment") or {}
            if isinstance(env, list):
                env = dict(kv.split("=", 1) for kv in env if "=" in kv)
            env["KG_AUTH_TOKEN"] = token
            container_kwargs["environment"] = env

            self.container = self.client.containers.run(self.image_name, **container_kwargs)

            retries = 0
            while self.container.status != "running" and retries < 5:
                self.logger.log(f"Container status: {self.container.status}, waiting...", level=LogLevel.INFO)
                time.sleep(1)
                self.container.reload()
                retries += 1

            self.base_url = f"http://{host}:{port}"

            # Wait for Jupyter to start
            self._wait_for_server(token)

            # Create new kernel via HTTP
            self.kernel_id = _create_kernel_http(f"{self.base_url}/api/kernels?token={token}", self.logger)
            self.ws_url = f"ws://{host}:{port}/api/kernels/{self.kernel_id}/channels?token={token}"

            self.installed_packages = self.install_packages(additional_imports)
            self.logger.log(
                f"Container {self.container.short_id} is running with kernel {self.kernel_id}", level=LogLevel.INFO
            )

        except Exception as e:
            self.cleanup()
            raise RuntimeError(f"Failed to initialize Jupyter kernel: {e}") from e

    def run_code_raise_errors(self, code: str) -> CodeOutput:
        # [FUNCTION DOC]
        # Opens a new websocket to ws_url per call (closed afterwards) and delegates to
        #   _websocket_run_code_raise_errors.
        # Inputs: code (str, Python source to run).
        # Outputs: CodeOutput.
        # Exceptions: AgentError, connection and serialization errors propagate.
        # Side effects: network I/O, logging.
        """
        Execute Python code in the Docker container and return the result.

        Args:
            code (`str`): Python code to execute.

        Returns:
            `CodeOutput`: Code output containing the result, logs, and whether it is the final answer.
        """
        from websocket import create_connection

        with closing(create_connection(self.ws_url)) as ws:
            return _websocket_run_code_raise_errors(code, ws, self.logger, self.allow_pickle)

    def cleanup(self):
        # [FUNCTION DOC]
        # If a container attribute exists, stops and removes the container, logs, and deletes the attribute.
        # Inputs: none. Outputs: none.
        # Exceptions: none; errors are caught and logged with log_error.
        # Side effects: Docker container stop/remove, logging.
        """Clean up the Docker container and resources."""
        try:
            if hasattr(self, "container"):
                self.logger.log(f"Stopping and removing container {self.container.short_id}...", level=LogLevel.INFO)
                self.container.stop()
                self.container.remove()
                self.logger.log("Container cleanup completed", level=LogLevel.INFO)
                del self.container
        except Exception as e:
            self.logger.log_error(f"Error during cleanup: {e}")

    def delete(self):
        # [FUNCTION DOC]
        # Alias that calls cleanup.
        # Inputs: none. Outputs: none. Exceptions: none raised directly. Side effects: those of cleanup.
        """Ensure cleanup on deletion."""
        self.cleanup()

    def _wait_for_server(self, token: str):
        # [FUNCTION DOC]
        # HELPER: Polls the gateway kernelspecs endpoint (2 second timeout) up to 10 times, sleeping 1 second
        #   between failed attempts, until it returns status 200.
        # Inputs: token (str auth token).
        # Outputs: none.
        # Exceptions: none raised directly (request exceptions are caught).
        # Side effects: HTTP requests, sleeping, logging.
        # KNOWN ISSUE: gives up silently after 10 tries without raising.
        retries = 0
        jupyter_ready = False
        while not jupyter_ready and retries < 10:
            try:
                if requests.get(f"{self.base_url}/api/kernelspecs?token={token}", timeout=2).status_code == 200:
                    jupyter_ready = True
                else:
                    self.logger.log("Jupyter not ready, waiting...", level=LogLevel.INFO)
            except requests.RequestException:
                self.logger.log("Jupyter not ready, waiting...", level=LogLevel.INFO)
            if not jupyter_ready:
                time.sleep(1)
                retries += 1


class ModalExecutor(RemotePythonExecutor):
    # [CLASS DOC]
    # Remote executor that runs a Jupyter Kernel Gateway inside a Modal sandbox reached via an encrypted
    #   tunnel.
    # Inputs (constructor): additional_imports, logger, allow_pickle, app_name (str, default 'smolagent-
    #   executor'), port (int, default 8888), create_kwargs (optional dict passed to modal.Sandbox.create,
    #   overriding defaults).
    # Key attributes: _ANSI_ESCAPE (compiled regex for ANSI escape sequences), port, sandbox, ws_url.
    """
    Remote Python code executor in a Modal sandbox.

    Args:
        additional_imports (`list[str]`): Additional Python packages to install.
        logger (`Logger`): Logger to use for output and errors.
        allow_pickle (`bool`, default `False`): Whether to allow pickle serialization for objects that cannot be safely serialized to JSON.
            - `False` (default, recommended): Only safe JSON serialization is used. Raises error if object cannot be safely serialized.
            - `True` (legacy mode): Tries safe JSON serialization first, falls back to pickle with warning if needed.

            **Security Warning:** Pickle deserialization can execute arbitrary code. Only set `allow_pickle=True`
            if you fully trust the execution environment and need backward compatibility with custom types.
        app_name (`str`, default `"smolagent-executor"`): App name.
        port (`int`, default `8888`): Port for jupyter to bind to.
        create_kwargs (`dict`, *optional*): Additional keyword arguments to pass to the Modal Sandbox create command. See
            `modal.Sandbox.create` [docs](https://modal.com/docs/reference/modal.Sandbox#create) for all the
            keyword arguments.
    """

    _ANSI_ESCAPE = re.compile(r"\x1B(?:[@-Z\\-_]|\[[0-?]*[ -/]*[@-~])")

    def __init__(
        self,
        additional_imports: list[str],
        logger,
        allow_pickle: bool = False,
        app_name: str = "smolagent-executor",
        port: int = 8888,
        create_kwargs: Optional[dict] = None,
    ):
        # [FUNCTION DOC]
        # Imports modal and builds sandbox create kwargs: default image (debian slim with kernel gateway and
        #   ipykernel) and 5 minute timeout overridden by user kwargs; looks up/creates the app by name unless
        #   one is given; appends port to encrypted_ports; appends a secret with a random KG_AUTH_TOKEN to
        #   secrets. Creates the sandbox running the gateway on the port, waits for the tunnel host, creates a
        #   kernel via HTTPS, sets a wss URL and installs packages.
        # Inputs: see class doc.
        # Outputs: none.
        # Exceptions: ModuleNotFoundError if modal missing; RuntimeError from waiting or kernel creation;
        #   modal errors propagate.
        # Side effects: creates a remote sandbox/app, network calls, logging.
        # KNOWN ISSUE: unlike Docker/Blaxel, no cleanup is attempted if initialization fails after sandbox
        #   creation.
        super().__init__(additional_imports, logger, allow_pickle)
        self.port = port
        try:
            import modal
        except ModuleNotFoundError:
            raise ModuleNotFoundError(
                """Please install 'modal' extra to use ModalExecutor: `pip install 'smolagents[modal]'`"""
            )

        if create_kwargs is None:
            create_kwargs = {}

        create_kwargs = {
            "image": modal.Image.debian_slim().uv_pip_install("jupyter_kernel_gateway", "ipykernel"),
            "timeout": 60 * 5,
            **create_kwargs,
        }

        if "app" not in create_kwargs:
            create_kwargs["app"] = modal.App.lookup(app_name, create_if_missing=True)

        if "encrypted_ports" not in create_kwargs:
            create_kwargs["encrypted_ports"] = [port]
        else:
            create_kwargs["encrypted_ports"] = create_kwargs["encrypted_ports"] + [port]

        token = secrets.token_urlsafe(16)
        default_secrets = [modal.Secret.from_dict({"KG_AUTH_TOKEN": token})]

        if "secrets" not in create_kwargs:
            create_kwargs["secrets"] = default_secrets
        else:
            create_kwargs["secrets"] = create_kwargs["secrets"] + default_secrets

        entrypoint = [
            "jupyter",
            "kernelgateway",
            "--KernelGatewayApp.ip=0.0.0.0",
            f"--KernelGatewayApp.port={port}",
        ]

        self.logger.log("Starting Modal sandbox", level=LogLevel.INFO)
        self.sandbox = modal.Sandbox.create(
            *entrypoint,
            **create_kwargs,
        )

        tunnel = self.sandbox.tunnels()[port]
        self.logger.log(f"Waiting for Modal sandbox on {tunnel.host}:{port}", level=LogLevel.INFO)
        self._wait_for_server(tunnel.host, token)

        self.logger.log("Starting Jupyter kernel", level=LogLevel.INFO)
        kernel_id = _create_kernel_http(f"https://{tunnel.host}/api/kernels?token={token}", logger)
        self.ws_url = f"wss://{tunnel.host}/api/kernels/{kernel_id}/channels?token={token}"
        self.installed_packages = self.install_packages(additional_imports)

    def run_code_raise_errors(self, code: str) -> CodeOutput:
        # [FUNCTION DOC]
        # Opens a websocket to ws_url per call and delegates to _websocket_run_code_raise_errors.
        # Inputs: code (str, Python source to run).
        # Outputs: CodeOutput.
        # Exceptions: AgentError, connection and serialization errors propagate.
        # Side effects: network I/O, logging.
        """
        Execute Python code in the Modal sandbox and return the result.

        Args:
            code (`str`): Python code to execute.

        Returns:
            `CodeOutput`: Code output containing the result, logs, and whether it is the final answer.
        """
        from websocket import create_connection

        with closing(create_connection(self.ws_url)) as ws:
            return _websocket_run_code_raise_errors(code, ws, self.logger, self.allow_pickle)

    def cleanup(self):
        # [FUNCTION DOC]
        # Terminates the sandbox if the sandbox attribute exists.
        # Inputs: none. Outputs: none. Exceptions: modal errors propagate. Side effects: remote sandbox
        #   termination.
        """Clean up the Modal sandbox by terminating it."""
        if hasattr(self, "sandbox"):
            self.sandbox.terminate()

    def delete(self):
        # [FUNCTION DOC]
        # Alias that calls cleanup.
        # Inputs: none. Outputs: none. Exceptions: those of cleanup. Side effects: those of cleanup.
        """Ensure cleanup on deletion."""
        self.cleanup()

    def _wait_for_server(self, host: str, token: str):
        # [FUNCTION DOC]
        # HELPER: Polls the HTTPS kernelspecs endpoint on the tunnel host until status 200. On request
        #   exceptions counts retries, logs every 10th, sleeps 1 second, and raises after more than 60.
        # Inputs: host (str), token (str).
        # Outputs: none.
        # Exceptions: RuntimeError when retries exceed 60.
        # Side effects: HTTP requests, sleeping, logging.
        # KNOWN ISSUE: a non-200 response without an exception neither sleeps nor counts, causing an unbounded
        #   busy loop; requests have no timeout.
        """Wait for server to start up."""
        n_retries = 0
        while True:
            try:
                resp = requests.get(f"https://{host}/api/kernelspecs?token={token}")
                if resp.status_code == 200:
                    break
            except RequestException:
                n_retries += 1
                if n_retries % 10 == 0:
                    self.logger.log("Waiting for server to startup, retrying...", level=LogLevel.INFO)
                if n_retries > 60:
                    raise RuntimeError("Unable to connect to sandbox")
                time.sleep(1.0)

    @classmethod
    def _strip_ansi_colors(cls, text: str) -> str:
        # [FUNCTION DOC]
        # HELPER: Class method removing ANSI escape sequences from text using _ANSI_ESCAPE.
        # Inputs: text (str). Outputs: cleaned str. Exceptions: none raised directly. Side effects: none.
        # KNOWN ISSUE: not used anywhere in this module.
        """Remove ansi colors from text."""
        return cls._ANSI_ESCAPE.sub("", text)


class BlaxelExecutor(RemotePythonExecutor):
    # [CLASS DOC]
    # Remote executor backed by a Blaxel sandbox running a Jupyter server, reached over HTTP/websockets with
    #   Blaxel auth headers.
    # Inputs (constructor): additional_imports, logger, allow_pickle, sandbox_name (optional str), image (str,
    #   default 'blaxel/jupyter-notebook'), memory (int MB, default 4096), ttl (optional str), region
    #   (optional str).
    # Key attributes: sandbox_name, image, memory, region, port (fixed 8888), _cleaned_up flag, sandbox,
    #   ws_url.
    """
    Remote Python code executor in a Blaxel sandbox.

    Blaxel provides fast-launching virtual machines that start from hibernation in under 25ms
    and scale back to zero after inactivity while maintaining memory state.

    Args:
        additional_imports (`list[str]`): Additional Python packages to install.
        logger (`Logger`): Logger to use for output and errors.
        allow_pickle (`bool`, default `False`): Whether to allow pickle serialization for objects that cannot be safely serialized to JSON.
            - `False` (default, recommended): Only safe JSON serialization is used. Raises error if object cannot be safely serialized.
            - `True` (legacy mode): Tries safe JSON serialization first, falls back to pickle with warning if needed.

            **Security Warning:** Pickle deserialization can execute arbitrary code. Only set `allow_pickle=True`
            if you fully trust the execution environment and need backward compatibility with custom types.
        sandbox_name (`str`, *optional*): Name for the sandbox. Defaults to "smolagent-executor".
        image (`str`, default `"blaxel/jupyter-notebook"`): Docker image to use.
        memory (`int`, default `4096`): Memory allocation in MB.
        ttl (`str`, *optional*): Time to live in seconds.
        region (`str`, *optional*): Deployment region. If not specified, Blaxel chooses default.
    """

    def __init__(
        self,
        additional_imports: list[str],
        logger,
        allow_pickle: bool = False,
        sandbox_name: str | None = None,
        image: str = "blaxel/jupyter-notebook",
        memory: int = 4096,
        ttl: str | None = None,
        region: Optional[str] = None,
    ):
        # [FUNCTION DOC]
        # Verifies blaxel is importable, sets sandbox_name (given or 'smolagent-executor-' plus 8 random hex
        #   chars), builds a sandbox config (name, image, memory, port 8888, optional region and ttl), creates
        #   the sandbox, creates a kernel via HTTP at the sandbox URL's port path using Blaxel headers,
        #   derives ws/wss URL from the sandbox URL and installs packages.
        # Inputs: see class doc.
        # Outputs: none.
        # Exceptions: ModuleNotFoundError if blaxel missing; any later failure triggers cleanup and is re-
        #   raised as RuntimeError.
        # Side effects: creates a remote sandbox, network calls, logging.
        # KNOWN ISSUE: the docstring says sandbox_name defaults to 'smolagent-executor' but a random suffix is
        #   added; ttl is documented as seconds but typed str; the token is put in the kernel URL but never
        #   set as KG_AUTH_TOKEN in the sandbox. If the blaxel import fails, _cleaned_up is never set so
        #   __del__ cleanup errors (silently).
        super().__init__(additional_imports, logger, allow_pickle=allow_pickle)

        try:
            import blaxel  # noqa: F401
        except ModuleNotFoundError:
            raise ModuleNotFoundError(
                "Please install 'blaxel' extra to use BlaxelExecutor: `pip install 'smolagents[blaxel]'`"
            )

        self.sandbox_name = sandbox_name or f"smolagent-executor-{uuid.uuid4().hex[:8]}"
        self.image = image
        self.memory = memory
        self.region = region
        self.port = 8888
        self._cleaned_up = False  # Flag to prevent double cleanup

        # Prepare sandbox creation parameters
        token = secrets.token_urlsafe(16)
        sandbox_config = {
            "metadata": {
                "name": self.sandbox_name,
            },
            "spec": {
                "runtime": {"image": image, "memory": memory, "ports": [{"target": self.port}]},
            },
        }

        if region:
            sandbox_config["spec"]["region"] = region

        if ttl:
            sandbox_config["spec"]["runtime"]["ttl"] = ttl

        # Create the sandbox
        try:
            # Create sandbox environment on Blaxel
            self.sandbox = BlaxelExecutor._create_sandbox(sandbox_config)

            # Create kernel via HTTP
            from blaxel.core import settings

            kernel_id = _create_kernel_http(
                f"{self.sandbox.metadata.url}/port/{self.port}/api/kernels?token={token}",
                self.logger,
                headers=settings.headers,
            )

            # Set up websocket URL
            # Convert http/https to ws/wss
            ws_scheme = "wss" if self.sandbox.metadata.url.startswith("https") else "ws"
            ws_base = self.sandbox.metadata.url.replace("https://", "").replace("http://", "")
            self.ws_url = f"{ws_scheme}://{ws_base}/port/{self.port}/api/kernels/{kernel_id}/channels?token={token}"

            # Install additional packages
            self.installed_packages = self.install_packages(additional_imports)
            self.logger.log("Blaxel is running", level=LogLevel.INFO)
        except Exception as e:
            self.cleanup()
            raise RuntimeError(f"Failed to initialize Blaxel sandbox: {e}") from e

    @staticmethod
    def _create_sandbox(config):
        # [FUNCTION DOC]
        # HELPER: Static method calling Blaxel's synchronous create_sandbox API with the config and wrapping
        #   the response in a SandboxInstance.
        # Inputs: config (dict sandbox spec).
        # Outputs: SandboxInstance.
        # Exceptions: Blaxel API errors propagate.
        # Side effects: network call creating a sandbox.
        # KNOWN ISSUE: docstring says asynchronous but the call is synchronous.
        """Helper method to create sandbox asynchronously."""
        from blaxel.core import SandboxInstance
        from blaxel.core.client import client
        from blaxel.core.client.api.compute import create_sandbox

        response = create_sandbox.sync(client=client, body=config)
        return SandboxInstance(response)

    def run_code_raise_errors(self, code: str) -> CodeOutput:
        # [FUNCTION DOC]
        # Builds 'Key: value' header strings from Blaxel settings headers, opens a websocket to ws_url with
        #   them and delegates to _websocket_run_code_raise_errors.
        # Inputs: code (str, Python source to run).
        # Outputs: CodeOutput.
        # Exceptions: AgentError, connection and serialization errors propagate.
        # Side effects: network I/O, logging.
        """
        Execute Python code in the Blaxel sandbox and return the result.

        Args:
            code (`str`): Python code to execute.

        Returns:
            `CodeOutput`: Code output containing the result, logs, and whether it is the final answer.
        """
        from blaxel.core import settings
        from websocket import create_connection

        headers = []
        for key, value in settings.headers.items():
            headers.append(f"{key}: {value}")
        with closing(create_connection(self.ws_url, header=headers)) as ws:
            return _websocket_run_code_raise_errors(code, ws, self.logger, self.allow_pickle)

    def install_packages(self, additional_imports: list[str]) -> list[str]:
        # [FUNCTION DOC]
        # Overrides base install: returns empty list for no packages; otherwise configures the Blaxel sandbox
        #   client with sandbox URL and headers, starts a process named 'install-packages' running pip
        #   install, then polls it every second (up to 600 seconds) until not running.
        # Inputs: additional_imports (list of str).
        # Outputs: the input list on exit code 0; empty list on non-zero exit or any error.
        # Exceptions: none; all exceptions (including timeout and error responses) are caught and logged.
        # Side effects: remote process execution, mutates the shared Blaxel sandbox client configuration,
        #   sleeping, logging.
        # KNOWN ISSUE: docstring says asynchronous but it is synchronous; a fixed process name may collide on
        #   repeated installs; exit_code None is treated as failure.
        """Helper method to install packages asynchronously."""
        if not additional_imports:
            return []

        from blaxel.core import settings
        from blaxel.core.sandbox.client import client
        from blaxel.core.sandbox.client.api.process import get_process_identifier, post_process
        from blaxel.core.sandbox.client.models import ErrorResponse, ProcessResponse

        try:
            client.with_base_url(self.sandbox.metadata.url)
            client.with_headers(settings.headers)

            # Install packages using pip via run_code
            self.logger.log(f"Installing packages: {', '.join(additional_imports)}", level=LogLevel.INFO)
            pip_install_code = f"pip install --root-user-action=ignore {' '.join(additional_imports)}"

            identifier = "install-packages"
            body = {
                "name": identifier,
                "command": pip_install_code,
            }
            post_process.sync(client=client, body=body)

            status = "running"
            interval = 1000
            max_wait = 600000
            start_time = time.time() * 1000
            logs = ""
            exit_code = 0

            while status == "running":
                if (time.time() * 1000) - start_time > max_wait:
                    raise Exception("Process did not finish in time")
                data = get_process_identifier.sync(identifier, client=client)
                if isinstance(data, ProcessResponse):
                    status = data.status or "running"
                    exit_code = data.exit_code
                    logs = data.logs
                elif isinstance(data, ErrorResponse):
                    raise Exception(f"Failed to install packages: {data.message}")
                else:
                    raise Exception(f"Unknown response: {data}")

                if status == "running":
                    time.sleep(interval / 1000)  # Convert to seconds

            if exit_code != 0:
                self.logger.log_error(f"Failed to install packages (exit code {exit_code}): {logs}")
                return []

            self.logger.log(f"Successfully installed packages: {', '.join(additional_imports)}", level=LogLevel.INFO)
            return additional_imports

        except Exception as e:
            self.logger.log_error(f"Error installing packages: {e}")
            return []

    def _delete_sandbox(self):
        # [FUNCTION DOC]
        # HELPER: Requests deletion of the sandbox by name via Blaxel's synchronous delete API.
        # Inputs: none (uses sandbox_name).
        # Outputs: none.
        # Exceptions: Blaxel API errors propagate.
        # Side effects: network call, logging.
        # KNOWN ISSUE: docstring says it waits for completion, but it only issues the request.
        """Delete sandbox using Blaxel's sync API and wait for completion."""
        from blaxel.core.client import client
        from blaxel.core.client.api.compute import delete_sandbox

        self.logger.log(f"Requesting sandbox {self.sandbox_name} deletion...", level=LogLevel.INFO)
        delete_sandbox.sync(client=client, sandbox_name=self.sandbox_name)

    def cleanup(self):
        # [FUNCTION DOC]
        # Idempotent cleanup: returns immediately if already cleaned up; otherwise marks cleaned up, tries to
        #   delete the sandbox (errors logged at INFO), and always removes the sandbox attribute and logs
        #   completion.
        # Inputs: none. Outputs: none.
        # Exceptions: AttributeError if _cleaned_up was never set (e.g. init failed early); otherwise none.
        # Side effects: remote deletion, logging, sets _cleaned_up, removes self.sandbox.
        # KNOWN ISSUE: error log message contains a doubled colon.
        """Sync wrapper to clean up sandbox and resources."""
        # Prevent double cleanup
        if self._cleaned_up:
            return
        self.logger.log("Shutting down sandbox...", level=LogLevel.INFO)
        self._cleaned_up = True
        try:
            self._delete_sandbox()
        except Exception as e:
            # Log cleanup errors but don't raise - cleanup should be best-effort
            self.logger.log(f"Error during cleanup: : {e}", level=LogLevel.INFO)
        finally:
            # Always clean up local references
            if hasattr(self, "sandbox"):
                del self.sandbox
            self.logger.log("Sandbox cleanup completed", level=LogLevel.INFO)

    def delete(self):
        # [FUNCTION DOC]
        # Alias that calls cleanup.
        # Inputs: none. Outputs: none. Exceptions: those of cleanup. Side effects: those of cleanup.
        """Ensure cleanup on deletion."""
        self.cleanup()

    def __del__(self):
        # [FUNCTION DOC]
        # Finalizer that calls cleanup and silently ignores any exception.
        # Inputs: none. Outputs: none. Exceptions: none. Side effects: may delete the remote sandbox at
        #   garbage collection.
        """Ensure cleanup on deletion."""
        try:
            self.cleanup()
        except Exception:
            pass  # Silently ignore errors during cleanup
