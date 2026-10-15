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
# Core tool abstractions for smolagents: the Tool base class that agents call, helpers to build tools from
#   plain functions (the tool decorator), from Hub Spaces, Gradio tools, LangChain tools, Hub repos, code
#   strings and dicts, plus ToolCollection (load many tools from a Hub collection or an MCP server),
#   PipelineTool (transformers-backed tools), Gradio demo launching, sandbox tool-definition code generation and
#   runtime argument validation against a tool's input schema.
# Inputs: depends on huggingface_hub (repo/commit/download/collection APIs), the internal type-hint-to-JSON-
#   schema utilities, agent_types (AgentImage/AgentAudio and input/output sanitizers), tool_validation
#   (MethodChecker, validate_tool_attributes) and utils (source extraction, name validation, builtin module
#   list). Optional lazy deps: gradio, gradio_client, PIL, mcpadapt, transformers, accelerate, torch.
# Outputs: public exports are AUTHORIZED_TYPES, Tool, tool, load_tool, launch_gradio_demo, ToolCollection;
#   also defines BaseTool, PipelineTool, CONVERSION_DICT, validate_after_init, add_description,
#   get_tools_definition_code and validate_tool_arguments at module level.
# Exceptions: none raised at import time beyond import errors of hard dependencies.
# Side effects: creates a module-level logger; no other import-time side effects.
from __future__ import annotations

import ast
import inspect
import json
import logging
import os
import sys
import tempfile
import textwrap
import types
import warnings
from abc import ABC, abstractmethod
from collections.abc import Callable
from contextlib import contextmanager
from functools import wraps
from pathlib import Path
from typing import TYPE_CHECKING, Any

from huggingface_hub import (
    CommitOperationAdd,
    create_commit,
    create_repo,
    get_collection,
    hf_hub_download,
    metadata_update,
)

from ._function_type_hints_utils import (
    TypeHintParsingException,
    _convert_type_hints_to_json_schema,
    _get_json_schema_type,
    get_imports,
    get_json_schema,
)
from .agent_types import AgentAudio, AgentImage, handle_agent_input_types, handle_agent_output_types
from .tool_validation import MethodChecker, validate_tool_attributes
from .utils import (
    BASE_BUILTIN_MODULES,
    _is_package_available,
    get_source,
    instance_to_source,
    is_valid_name,
)


if TYPE_CHECKING:
    import mcp


logger = logging.getLogger(__name__)


def validate_after_init(cls):
    # [FUNCTION DOC]
    # HELPER: Class decorator that wraps a class's constructor so argument validation runs after construction.
    # Inputs: cls (a class that defines a validate_arguments method).
    # Outputs: the same class object, whose constructor is replaced by a wrapper that calls the original
    #   constructor and then the instance's validate_arguments method.
    # Exceptions: none raised directly; anything raised by the original constructor or validate_arguments
    #   propagates.
    # Side effects: mutates cls by reassigning its __init__ attribute.
    original_init = cls.__init__

    @wraps(original_init)
    def new_init(self, *args, **kwargs):
        # [FUNCTION DOC]
        # HELPER: Replacement constructor installed by validate_after_init.
        # Inputs: self (the instance), any positional and keyword args (forwarded unchanged to the original
        #   constructor).
        # Outputs: none.
        # Exceptions: propagates errors from the original constructor and from self.validate_arguments.
        # Side effects: initializes the instance via the original constructor, then validates it.
        # KNOWN ISSUE: since Tool.__init_subclass__ wraps every subclass, a subclass constructor that calls
        #   the parent constructor triggers validation once per wrapped level of the hierarchy (redundant
        #   repeated validation).
        original_init(self, *args, **kwargs)
        self.validate_arguments()

    cls.__init__ = new_init
    return cls


AUTHORIZED_TYPES = [
    "string",
    "boolean",
    "integer",
    "number",
    "image",
    "audio",
    "array",
    "object",
    "any",
    "null",
]

CONVERSION_DICT = {"str": "string", "int": "integer", "float": "number"}


class BaseTool(ABC):
    # [CLASS DOC]
    # Minimal abstract interface for anything callable as a tool by an agent.
    # Inputs: no constructor args; subclasses must provide a name class attribute (str).
    # Key attributes: name (str), the identifier used by agents.
    # Outputs: n/a (abstract base; cannot be instantiated without implementing the call method).
    # Exceptions: TypeError on instantiation if the abstract call method is not implemented.
    # Side effects: none.
    name: str

    @abstractmethod
    def __call__(self, *args, **kwargs) -> Any:
        # [FUNCTION DOC]
        # Abstract call entry point that concrete tools must implement.
        # Inputs: arbitrary positional and keyword arguments.
        # Outputs: any value (the tool's result); the base implementation does nothing and returns None.
        # Exceptions: none raised directly.
        # Side effects: none.
        pass


class Tool(BaseTool):
    # [CLASS DOC]
    # Base class for agent tools. Subclasses set class attributes name (valid Python identifier), description
    #   (str), inputs (dict mapping input name to a dict with at least type and description keys; type is one of
    #   AUTHORIZED_TYPES or a list of them; optional nullable key), output_type (one of AUTHORIZED_TYPES) and
    #   optionally output_schema (dict JSON schema, informational only), and implement forward whose parameters
    #   (besides self) must exactly match the keys of inputs.
    # Inputs: constructor accepts and ignores any args; sets is_initialized to False. Every subclass's
    #   constructor is automatically wrapped (via __init_subclass__ and validate_after_init) so
    #   validate_arguments runs after init.
    # Key attributes: name, description, inputs, output_type, output_schema, is_initialized, and optionally
    #   skip_forward_signature_validation (True disables forward-signature checks).
    # Outputs: instances are callable; calling runs setup lazily then forward.
    # Exceptions: construction of subclasses raises TypeError/ValueError/AssertionError/Exception on invalid
    #   attributes (see validate_arguments).
    # Side effects: subclass creation mutates the subclass's constructor.
    """
    A base class for the functions used by the agent. Subclass this and implement the `forward` method as well as the
    following class attributes:

    - **description** (`str`) -- A short description of what your tool does, the inputs it expects and the output(s) it
      will return. For instance 'This is a tool that downloads a file from a `url`. It takes the `url` as input, and
      returns the text contained in the file'.
    - **name** (`str`) -- A performative name that will be used for your tool in the prompt to the agent. For instance
      `"text-classifier"` or `"image_generator"`.
    - **inputs** (`Dict[str, Dict[str, Union[str, type, bool]]]`) -- The dict of modalities expected for the inputs.
      It has one `type`key and a `description`key.
      This is used by `launch_gradio_demo` or to make a nice space from your tool, and also can be used in the generated
      description for your tool.
    - **output_type** (`type`) -- The type of the tool output. This is used by `launch_gradio_demo`
      or to make a nice space from your tool, and also can be used in the generated description for your tool.
    - **output_schema** (`Dict[str, Any]`, *optional*) -- The JSON schema defining the expected structure of the tool output.
      This can be included in system prompts to help agents understand the expected output format. Note: This is currently
      used for informational purposes only and does not perform actual output validation.

    You can also override the method [`~Tool.setup`] if your tool has an expensive operation to perform before being
    usable (such as loading a model). [`~Tool.setup`] will be called the first time you use your tool, but not at
    instantiation.
    """

    name: str
    description: str
    inputs: dict[str, dict[str, str | type | bool]]
    output_type: str
    output_schema: dict[str, Any] | None = None

    def __init__(self, *args, **kwargs):
        # [FUNCTION DOC]
        # Base constructor: marks the tool as not yet set up.
        # Inputs: any positional and keyword args (ignored).
        # Outputs: none.
        # Exceptions: none raised directly.
        # Side effects: sets self.is_initialized to False.
        self.is_initialized = False

    def __init_subclass__(cls, **kwargs):
        # [FUNCTION DOC]
        # Hook run whenever a subclass of Tool is defined; ensures validation runs after each instance is
        #   constructed.
        # Inputs: cls (the new subclass), kwargs (forwarded to the parent hook).
        # Outputs: none.
        # Exceptions: none raised directly.
        # Side effects: replaces the subclass's constructor with a validating wrapper (see
        #   validate_after_init).
        super().__init_subclass__(**kwargs)
        validate_after_init(cls)

    def validate_arguments(self):
        # [FUNCTION DOC]
        # Validates the tool's attributes and forward signature after construction. Checks, in order:
        #   description, name, inputs, output_type exist (non-None) and are str/str/dict/str; output_schema is
        #   None or a dict; name is a valid non-keyword identifier; each inputs entry is a dict with type and
        #   description keys, whose type is a string or list of strings, all within AUTHORIZED_TYPES;
        #   output_type is in AUTHORIZED_TYPES. Unless skip_forward_signature_validation is True: the set of
        #   forward parameter names (excluding self) must equal the set of inputs keys; every input must appear
        #   in the JSON schema derived from forward's type hints; and the nullable flag must be present in
        #   inputs iff the type-hint schema marks the parameter nullable (Optional type or default value).
        # Inputs: self.
        # Outputs: none.
        # Exceptions: TypeError (missing/wrong-typed attribute, bad type spec), ValueError (unauthorized
        #   type), generic Exception (invalid name, forward params mismatch), AssertionError (malformed inputs
        #   entry, bad output_type, missing schema key, nullable mismatch).
        # Side effects: none.
        # KNOWN ISSUE: several checks use assert statements, which are skipped when Python runs with
        #   optimizations (-O).
        required_attributes = {
            "description": str,
            "name": str,
            "inputs": dict,
            "output_type": str,
        }
        # Validate class attributes
        for attr, expected_type in required_attributes.items():
            attr_value = getattr(self, attr, None)
            if attr_value is None:
                raise TypeError(f"You must set an attribute {attr}.")
            if not isinstance(attr_value, expected_type):
                raise TypeError(
                    f"Attribute {attr} should have type {expected_type.__name__}, got {type(attr_value)} instead."
                )

        # Validate optional output_schema attribute
        output_schema = getattr(self, "output_schema", None)
        if output_schema is not None and not isinstance(output_schema, dict):
            raise TypeError(f"Attribute output_schema should have type dict, got {type(output_schema)} instead.")

        # - Validate name
        if not is_valid_name(self.name):
            raise Exception(
                f"Invalid Tool name '{self.name}': must be a valid Python identifier and not a reserved keyword"
            )
        # Validate inputs
        for input_name, input_content in self.inputs.items():
            assert isinstance(input_content, dict), f"Input '{input_name}' should be a dictionary."
            assert "type" in input_content and "description" in input_content, (
                f"Input '{input_name}' should have keys 'type' and 'description', has only {list(input_content.keys())}."
            )
            # Get input_types as a list, whether from a string or list
            if isinstance(input_content["type"], str):
                input_types = [input_content["type"]]
            elif isinstance(input_content["type"], list):
                input_types = input_content["type"]
                # Check if all elements are strings
                if not all(isinstance(t, str) for t in input_types):
                    raise TypeError(
                        f"Input '{input_name}': when type is a list, all elements must be strings, got {input_content['type']}"
                    )
            else:
                raise TypeError(
                    f"Input '{input_name}': type must be a string or list of strings, got {type(input_content['type']).__name__}"
                )
            # Check all types are authorized
            invalid_types = [t for t in input_types if t not in AUTHORIZED_TYPES]
            if invalid_types:
                raise ValueError(f"Input '{input_name}': types {invalid_types} must be one of {AUTHORIZED_TYPES}")
        # Validate output type
        assert getattr(self, "output_type", None) in AUTHORIZED_TYPES

        # Validate forward function signature, except for Tools that use a "generic" signature (PipelineTool, SpaceToolWrapper, LangChainToolWrapper)
        if not (
            hasattr(self, "skip_forward_signature_validation")
            and getattr(self, "skip_forward_signature_validation") is True
        ):
            signature = inspect.signature(self.forward)
            actual_keys = set(key for key in signature.parameters.keys() if key != "self")
            expected_keys = set(self.inputs.keys())
            if actual_keys != expected_keys:
                raise Exception(
                    f"In tool '{self.name}', 'forward' method parameters were {actual_keys}, but expected {expected_keys}. "
                    f"It should take 'self' as its first argument, then its next arguments should match the keys of tool attribute 'inputs'."
                )

            json_schema = _convert_type_hints_to_json_schema(self.forward, error_on_missing_type_hints=False)[
                "properties"
            ]  # This function will not raise an error on missing docstrings, contrary to get_json_schema
            for key, value in self.inputs.items():
                assert key in json_schema, (
                    f"Input '{key}' should be present in function signature, found only {json_schema.keys()}"
                )
                if "nullable" in value:
                    assert "nullable" in json_schema[key], (
                        f"Nullable argument '{key}' in inputs should have key 'nullable' set to True in function signature."
                    )
                if key in json_schema and "nullable" in json_schema[key]:
                    assert "nullable" in value, (
                        f"Nullable argument '{key}' in function signature should have key 'nullable' set to True in inputs."
                    )

    def forward(self, *args, **kwargs):
        # [FUNCTION DOC]
        # Default implementation of the tool's logic; subclasses must override it.
        # Inputs: any args (ignored).
        # Outputs: none (always raises).
        # Exceptions: NotImplementedError always.
        # Side effects: none.
        raise NotImplementedError("Write this method in your subclass of `Tool`.")

    def __call__(self, *args, sanitize_inputs_outputs: bool = False, **kwargs):
        # [FUNCTION DOC]
        # Invokes the tool: lazily runs setup on first use, normalizes arguments, optionally sanitizes
        #   inputs/outputs, and calls forward. If exactly one positional arg is given, no kwargs, the arg is a
        #   dict, and all its keys are input names, the dict is unpacked into keyword arguments.
        # Inputs: args/kwargs for forward; sanitize_inputs_outputs (bool, default False): when True, agent-
        #   typed inputs are converted to raw values before forward and the output is wrapped into the agent
        #   type matching output_type.
        # Outputs: the (optionally sanitized) result of forward.
        # Exceptions: none raised directly; errors from setup, forward or sanitizers propagate.
        # Side effects: may call setup (which sets is_initialized); whatever forward does.
        if not self.is_initialized:
            self.setup()

        # Handle the arguments might be passed as a single dictionary
        if len(args) == 1 and len(kwargs) == 0 and isinstance(args[0], dict):
            potential_kwargs = args[0]

            # If the dictionary keys match our input parameters, convert it to kwargs
            if all(key in self.inputs for key in potential_kwargs):
                args = ()
                kwargs = potential_kwargs

        if sanitize_inputs_outputs:
            args, kwargs = handle_agent_input_types(*args, **kwargs)
        outputs = self.forward(*args, **kwargs)
        if sanitize_inputs_outputs:
            outputs = handle_agent_output_types(outputs, self.output_type)
        return outputs

    def setup(self):
        # [FUNCTION DOC]
        # Hook for expensive one-time initialization run on first call; base version only marks the tool
        #   initialized.
        # Inputs: self.
        # Outputs: none.
        # Exceptions: none raised directly.
        # Side effects: sets self.is_initialized to True.
        """
        Overwrite this method here for any operation that is expensive and needs to be executed before you start using
        your tool. Such as loading a big model.
        """
        self.is_initialized = True

    def to_code_prompt(self) -> str:
        # [FUNCTION DOC]
        # Renders the tool as a Python-like function stub (signature plus docstring) for code-agent prompts.
        #   The signature lists each input as name: type (JSON type strings), and the return type is dict when
        #   an output_schema is set, otherwise output_type. The docstring holds the description, an extra note
        #   about structured output if a schema exists, an Args section (one line per input with its
        #   description) when there are inputs, and a Returns section with the JSON-dumped schema (indent 4)
        #   when a schema exists. Body is indented 4.
        # Inputs: self.
        # Outputs: str, the stub text starting with def followed by the tool name.
        # Exceptions: KeyError if an inputs entry lacks type or description.
        # Side effects: none.
        args_signature = ", ".join(f"{arg_name}: {arg_schema['type']}" for arg_name, arg_schema in self.inputs.items())

        # Use dict type for tools with output schema to indicate structured return
        has_schema = hasattr(self, "output_schema") and self.output_schema is not None
        output_type = "dict" if has_schema else self.output_type
        tool_signature = f"({args_signature}) -> {output_type}"
        tool_doc = self.description

        # Add an important note for smaller models (e.g. Mistral Small, Gemma 3, etc.) to properly handle structured output.
        if has_schema:
            tool_doc += "\n\nImportant: This tool returns structured output! Use the JSON schema below to directly access fields like result['field_name']. NO print() statements needed to inspect the output!"

        # Add arguments documentation
        if self.inputs:
            args_descriptions = "\n".join(
                f"{arg_name}: {arg_schema['description']}" for arg_name, arg_schema in self.inputs.items()
            )
            args_doc = f"Args:\n{textwrap.indent(args_descriptions, '    ')}"
            tool_doc += f"\n\n{args_doc}"

        # Add returns documentation with output schema if it exists
        if has_schema:
            formatted_schema = json.dumps(self.output_schema, indent=4)
            indented_schema = textwrap.indent(formatted_schema, "        ")
            returns_doc = f"\nReturns:\n    dict (structured output): This tool ALWAYS returns a dictionary that strictly adheres to the following JSON schema:\n{indented_schema}"
            tool_doc += f"\n{returns_doc}"

        tool_doc = f'"""{tool_doc}\n"""'
        return f"def {self.name}{tool_signature}:\n{textwrap.indent(tool_doc, '    ')}"

    def to_tool_calling_prompt(self) -> str:
        # [FUNCTION DOC]
        # Renders a short textual summary for tool-calling-agent prompts: name and description, then indented
        #   lines with the raw inputs dict representation and the output type.
        # Inputs: self.
        # Outputs: str.
        # Exceptions: none raised directly.
        # Side effects: none.
        return f"{self.name}: {self.description}\n    Takes inputs: {self.inputs}\n    Returns an output of type: {self.output_type}"

    def to_dict(self) -> dict:
        # [FUNCTION DOC]
        # Serializes the tool into a dict containing standalone source code that recreates it. For tools
        #   created with the tool decorator (class named SimpleTool): parses the forward source (with the
        #   decorator text removed) and validates it with MethodChecker for self-contained imports/names, then
        #   generates a Tool subclass source with name, description, inputs, output_type (and output_schema if
        #   set) class attributes, appending the original function source renamed to forward with self added as
        #   first parameter.
        # For other tools: refuses wrappers (SpaceToolWrapper, LangChainToolWrapper, GradioToolWrapper),
        #   validates class attributes with validate_tool_attributes, and uses instance_to_source to generate
        #   code, prefixed with a typing import line. Requirements are the non-stdlib top-level imports detected
        #   in the code, plus smolagents, sorted.
        # Inputs: self.
        # Outputs: dict with keys name, code, requirements (sorted list), and output_schema when set.
        # Exceptions: ValueError if SimpleTool validation fails or the tool is a non-serializable wrapper;
        #   errors from validate_tool_attributes/source extraction propagate.
        # Side effects: none.
        # KNOWN ISSUE: for SimpleTool it replaces every occurrence of the tool name in the forward source (not
        #   only the function name) with forward, which can corrupt code or strings that contain the name.
        # KNOWN ISSUE: the output_schema line appended for SimpleTool keeps a hard-coded 16-space indent after
        #   the class body was already dedented, so the generated class source is likely mis-indented when a
        #   schema is present.
        """Returns a dictionary representing the tool"""
        class_name = self.__class__.__name__
        if type(self).__name__ == "SimpleTool":
            # Check that imports are self-contained
            source_code = get_source(self.forward).replace("@tool", "")
            forward_node = ast.parse(source_code)
            # If tool was created using '@tool' decorator, it has only a forward pass, so it's simpler to just get its code
            method_checker = MethodChecker(set())
            method_checker.visit(forward_node)

            if len(method_checker.errors) > 0:
                errors = [f"- {error}" for error in method_checker.errors]
                raise (ValueError(f"SimpleTool validation failed for {self.name}:\n" + "\n".join(errors)))

            forward_source_code = get_source(self.forward)
            tool_code = textwrap.dedent(
                f"""
            from smolagents import Tool
            from typing import Any, Optional

            class {class_name}(Tool):
                name = "{self.name}"
                description = {json.dumps(textwrap.dedent(self.description).strip())}
                inputs = {repr(self.inputs)}
                output_type = "{self.output_type}"
            """
            ).strip()

            # Add output_schema if it exists
            if hasattr(self, "output_schema") and self.output_schema is not None:
                tool_code += f"\n                output_schema = {repr(self.output_schema)}"
            import re

            def add_self_argument(source_code: str) -> str:
                # [FUNCTION DOC]
                # HELPER: Inserts self as the first parameter of a forward function definition in source text.
                # Inputs: source_code (str).
                # Outputs: str, the source with forward's parameter list prefixed by self (unless it already
                #   starts with self).
                # Exceptions: none raised directly.
                # Side effects: none.
                # KNOWN ISSUE: the regex stops at the first closing parenthesis, so parameter lists containing
                #   parentheses (e.g. defaults with calls or tuple annotations) are not handled correctly.
                """Add 'self' as first argument to a function definition if not present."""
                pattern = r"def forward\(((?!self)[^)]*)\)"

                def replacement(match):
                    # [FUNCTION DOC]
                    # HELPER: Regex substitution callback producing the new forward definition header.
                    # Inputs: match (regex match whose first group is the original parameter text).
                    # Outputs: str, the forward header with self alone or self followed by the stripped
                    #   original parameters.
                    # Exceptions: none raised directly.
                    # Side effects: none.
                    args = match.group(1).strip()
                    if args:  # If there are other arguments
                        return f"def forward(self, {args})"
                    return "def forward(self)"

                return re.sub(pattern, replacement, source_code)

            forward_source_code = forward_source_code.replace(self.name, "forward")
            forward_source_code = add_self_argument(forward_source_code)
            forward_source_code = forward_source_code.replace("@tool", "").strip()
            tool_code += "\n\n" + textwrap.indent(forward_source_code, "    ")

        else:  # If the tool was not created by the @tool decorator, it was made by subclassing Tool
            if type(self).__name__ in [
                "SpaceToolWrapper",
                "LangChainToolWrapper",
                "GradioToolWrapper",
            ]:
                raise ValueError(
                    "Cannot save objects created with from_space, from_langchain or from_gradio, as this would create errors."
                )

            validate_tool_attributes(self.__class__)

            tool_code = "from typing import Any, Optional\n" + instance_to_source(self, base_cls=Tool)

        requirements = {el for el in get_imports(tool_code) if el not in sys.stdlib_module_names} | {"smolagents"}

        tool_dict = {"name": self.name, "code": tool_code, "requirements": sorted(requirements)}

        # Add output_schema if it exists
        if hasattr(self, "output_schema") and self.output_schema is not None:
            tool_dict["output_schema"] = self.output_schema

        return tool_dict

    @classmethod
    def from_dict(cls, tool_dict: dict[str, Any], **kwargs) -> "Tool":
        # [FUNCTION DOC]
        # Recreates a tool from the dict produced by to_dict.
        # Inputs: tool_dict (dict; must contain code, may contain output_schema); kwargs forwarded to the tool
        #   constructor.
        # Outputs: a Tool instance built via from_code, with output_schema overwritten from the dict when
        #   present.
        # Exceptions: ValueError if code is missing; errors from from_code propagate.
        # Side effects: executes arbitrary code from the dict (via from_code).
        """
        Create tool from a dictionary representation.

        Args:
            tool_dict (`dict[str, Any]`): Dictionary representation of the tool.
            **kwargs: Additional keyword arguments to pass to the tool's constructor.

        Returns:
            `Tool`: Tool object.
        """
        if "code" not in tool_dict:
            raise ValueError("Tool dictionary must contain 'code' key with the tool source code")

        tool = cls.from_code(tool_dict["code"], **kwargs)

        # Set output_schema if it exists in the dictionary
        if "output_schema" in tool_dict:
            tool.output_schema = tool_dict["output_schema"]

        return tool

    def save(self, output_dir: str | Path, tool_file_name: str = "tool", make_gradio_app: bool = True):
        # [FUNCTION DOC]
        # Writes the tool's files into a directory: the tool source file and, optionally, a Gradio app and
        #   requirements.
        # Inputs: output_dir (str or Path; created with parents if missing), tool_file_name (str, default
        #   tool; file is named with a .py suffix), make_gradio_app (bool, default True): also write app.py
        #   (importing the tool class from the tool module) and requirements.txt (newline-joined requirements).
        # Outputs: none.
        # Exceptions: errors from to_dict (e.g. ValueError for unsavable tools) and filesystem errors
        #   propagate.
        # Side effects: creates directories and writes UTF-8 files on disk.
        """
        Saves the relevant code files for your tool so it can be pushed to the Hub. This will copy the code of your
        tool in `output_dir` as well as autogenerate:

        - a `{tool_file_name}.py` file containing the logic for your tool.
        If you pass `make_gradio_app=True`, this will also write:
        - an `app.py` file providing a UI for your tool when it is exported to a Space with `tool.push_to_hub()`
        - a `requirements.txt` containing the names of the modules used by your tool (as detected when inspecting its
          code)

        Args:
            output_dir (`str` or `Path`): The folder in which you want to save your tool.
            tool_file_name (`str`, *optional*): The file name in which you want to save your tool.
            make_gradio_app (`bool`, *optional*, defaults to True): Whether to also export a `requirements.txt` file and Gradio UI.
        """
        # Ensure output directory exists
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        # Save tool file
        self._write_file(output_path / f"{tool_file_name}.py", self._get_tool_code())
        if make_gradio_app:
            #  Save app file
            self._write_file(output_path / "app.py", self._get_gradio_app_code(tool_module_name=tool_file_name))
            # Save requirements file
            self._write_file(output_path / "requirements.txt", self._get_requirements())

    def _write_file(self, file_path: Path, content: str) -> None:
        # [FUNCTION DOC]
        # HELPER: Writes text to a path using UTF-8 encoding (overwriting any existing file).
        # Inputs: file_path (Path), content (str).
        # Outputs: none.
        # Exceptions: OSError on filesystem failures.
        # Side effects: writes a file.
        """Writes content to a file with UTF-8 encoding."""
        file_path.write_text(content, encoding="utf-8")

    def push_to_hub(
        self,
        repo_id: str,
        commit_message: str = "Upload tool",
        private: bool | None = None,
        token: bool | str | None = None,
        create_pr: bool = False,
    ) -> str:
        # [FUNCTION DOC]
        # Uploads the tool as a Gradio Space on the Hugging Face Hub: ensures the Space repo exists (tagged
        #   smolagents and tool), then commits tool.py, app.py and requirements.txt in one commit.
        # Inputs: repo_id (str), commit_message (str, default Upload tool), private (bool or None; ignored if
        #   repo exists), token (bool, str or None; HF auth), create_pr (bool, default False: open a PR instead
        #   of committing directly).
        # Outputs: the result of huggingface_hub create_commit (a commit info object).
        # Exceptions: huggingface_hub HTTP/auth errors and to_dict errors propagate.
        # Side effects: network calls creating/updating a remote repo, its metadata and contents.
        # KNOWN ISSUE: annotated as returning str, but create_commit returns a CommitInfo object.
        """
        Upload the tool to the Hub.

        Parameters:
            repo_id (`str`):
                The name of the repository you want to push your tool to. It should contain your organization name when
                pushing to a given organization.
            commit_message (`str`, *optional*, defaults to `"Upload tool"`):
                Message to commit while pushing.
            private (`bool`, *optional*):
                Whether to make the repo private. If `None` (default), the repo will be public unless the organization's default is private. This value is ignored if the repo already exists.
            token (`bool` or `str`, *optional*):
                The token to use as HTTP bearer authorization for remote files. If unset, will use the token generated
                when running `huggingface-cli login` (stored in `~/.huggingface`).
            create_pr (`bool`, *optional*, defaults to `False`):
                Whether to create a PR with the uploaded files or directly commit.
        """
        # Initialize repository
        repo_id = self._initialize_hub_repo(repo_id, token, private)
        # Prepare files for commit
        additions = self._prepare_hub_files()
        # Create commit
        return create_commit(
            repo_id=repo_id,
            operations=additions,
            commit_message=commit_message,
            token=token,
            create_pr=create_pr,
            repo_type="space",
        )

    @staticmethod
    def _initialize_hub_repo(repo_id: str, token: bool | str | None, private: bool | None) -> str:
        # [FUNCTION DOC]
        # HELPER: Creates (or reuses) a Gradio Space repo on the Hub and tags it with smolagents and tool
        #   metadata.
        # Inputs: repo_id (str), token (bool, str or None), private (bool or None).
        # Outputs: str, the canonical repo id returned by the Hub.
        # Exceptions: huggingface_hub errors propagate.
        # Side effects: network calls creating the repo (exist_ok) and updating its metadata.
        """Initialize repository on Hugging Face Hub."""
        repo_url = create_repo(
            repo_id=repo_id,
            token=token,
            private=private,
            exist_ok=True,
            repo_type="space",
            space_sdk="gradio",
        )
        metadata_update(repo_url.repo_id, {"tags": ["smolagents", "tool"]}, repo_type="space", token=token)
        return repo_url.repo_id

    def _prepare_hub_files(self) -> list:
        # [FUNCTION DOC]
        # HELPER: Builds the list of commit operations for uploading the tool: tool.py (tool code), app.py
        #   (Gradio app importing from the tool module) and requirements.txt, each as UTF-8 bytes.
        # Inputs: self.
        # Outputs: list of three CommitOperationAdd objects.
        # Exceptions: errors from to_dict propagate.
        # Side effects: none.
        """Prepare files for Hub commit."""
        additions = [
            # Add tool code
            CommitOperationAdd(
                path_in_repo="tool.py",
                path_or_fileobj=self._get_tool_code().encode(),
            ),
            # Add Gradio app
            CommitOperationAdd(
                path_in_repo="app.py",
                path_or_fileobj=self._get_gradio_app_code().encode(),
            ),
            # Add requirements
            CommitOperationAdd(
                path_in_repo="requirements.txt",
                path_or_fileobj=self._get_requirements().encode(),
            ),
        ]
        return additions

    def _get_tool_code(self) -> str:
        # [FUNCTION DOC]
        # HELPER: Returns the serialized tool source code (the code entry of to_dict).
        # Inputs: self.
        # Outputs: str.
        # Exceptions: errors from to_dict propagate.
        # Side effects: none.
        """Get the tool's code."""
        return self.to_dict()["code"]

    def _get_gradio_app_code(self, tool_module_name: str = "tool") -> str:
        # [FUNCTION DOC]
        # HELPER: Generates the source of a small Gradio app that imports the tool class from a module,
        #   instantiates it with no arguments and launches the Gradio demo.
        # Inputs: tool_module_name (str, default tool).
        # Outputs: str, the app source.
        # Exceptions: none raised directly.
        # Side effects: none.
        """Get the Gradio app code."""
        class_name = self.__class__.__name__
        return textwrap.dedent(
            f"""\
            from smolagents import launch_gradio_demo
            from {tool_module_name} import {class_name}

            tool = {class_name}()
            launch_gradio_demo(tool)
            """
        )

    def _get_requirements(self) -> str:
        # [FUNCTION DOC]
        # HELPER: Returns the tool's requirements (from to_dict) joined by newlines.
        # Inputs: self.
        # Outputs: str.
        # Exceptions: errors from to_dict propagate.
        # Side effects: none.
        """Get the requirements."""
        return "\n".join(self.to_dict()["requirements"])

    @classmethod
    def from_hub(
        cls,
        repo_id: str,
        token: str | None = None,
        trust_remote_code: bool = False,
        **kwargs,
    ):
        # [FUNCTION DOC]
        # Downloads a tool's tool.py from a Hub Space and instantiates it.
        # Inputs: repo_id (str, Space id), token (str or None), trust_remote_code (bool, default False; must
        #   be True), kwargs: cache_dir, force_download, proxies, revision, subfolder, local_files_only are used
        #   for the download.
        # Outputs: a Tool instance built via Tool.from_code.
        # Exceptions: ValueError if trust_remote_code is False; download errors and from_code errors
        #   propagate.
        # Side effects: network download into the HF cache; reads the file; executes remote code.
        # KNOWN ISSUE: contrary to the docstring, kwargs are not split: all kwargs (including Hub download
        #   ones) are also passed to the tool constructor, which can fail for constructors not accepting them.
        # KNOWN ISSUE: always uses Tool.from_code instead of cls.from_code, ignoring the calling subclass.
        """
        Loads a tool defined on the Hub.

        <Tip warning={true}>

        Loading a tool from the Hub means that you'll download the tool and execute it locally.
        ALWAYS inspect the tool you're downloading before loading it within your runtime, as you would do when
        installing a package using pip/npm/apt.

        </Tip>

        Args:
            repo_id (`str`):
                The name of the Space repo on the Hub where your tool is defined.
            token (`str`, *optional*):
                The token to identify you on hf.co. If unset, will use the token generated when running
                `huggingface-cli login` (stored in `~/.huggingface`).
            trust_remote_code(`str`, *optional*, defaults to False):
                This flags marks that you understand the risk of running remote code and that you trust this tool.
                If not setting this to True, loading the tool from Hub will fail.
            kwargs (additional keyword arguments, *optional*):
                Additional keyword arguments that will be split in two: all arguments relevant to the Hub (such as
                `cache_dir`, `revision`, `subfolder`) will be used when downloading the files for your tool, and the
                others will be passed along to its init.
        """
        if not trust_remote_code:
            raise ValueError(
                "Loading a tool from Hub requires to acknowledge you trust its code: to do so, pass `trust_remote_code=True`."
            )

        # Get the tool's tool.py file.
        tool_file = hf_hub_download(
            repo_id,
            "tool.py",
            token=token,
            repo_type="space",
            cache_dir=kwargs.get("cache_dir"),
            force_download=kwargs.get("force_download"),
            proxies=kwargs.get("proxies"),
            revision=kwargs.get("revision"),
            subfolder=kwargs.get("subfolder"),
            local_files_only=kwargs.get("local_files_only"),
        )

        tool_code = Path(tool_file).read_text()
        return Tool.from_code(tool_code, **kwargs)

    @classmethod
    def from_code(cls, tool_code: str, **kwargs):
        # [FUNCTION DOC]
        # Executes a source string in a fresh module and instantiates the first Tool subclass found in it. If
        #   the found class's inputs (or output_schema) is a string, it is converted with a literal evaluation.
        # Inputs: tool_code (str, Python source defining a Tool subclass), kwargs forwarded to the class
        #   constructor.
        # Outputs: an instance of the discovered Tool subclass.
        # Exceptions: ValueError if no Tool subclass is found; any exception from executing the code or
        #   constructing.
        # Side effects: executes arbitrary code; may mutate class attributes inputs/output_schema of the found
        #   class.
        # KNOWN ISSUE: the first subclass is chosen by inspect.getmembers alphabetical order, so a module that
        #   also imports or defines other Tool subclasses may yield the wrong class.
        module = types.ModuleType("dynamic_tool")

        exec(tool_code, module.__dict__)

        # Find the Tool subclass
        tool_class = next(
            (
                obj
                for _, obj in inspect.getmembers(module, inspect.isclass)
                if issubclass(obj, Tool) and obj is not Tool
            ),
            None,
        )

        if tool_class is None:
            raise ValueError("No Tool subclass found in the code.")

        if not isinstance(tool_class.inputs, dict):
            tool_class.inputs = ast.literal_eval(tool_class.inputs)

        # Handle output_schema if it exists and is a string representation
        if hasattr(tool_class, "output_schema") and isinstance(tool_class.output_schema, str):
            tool_class.output_schema = ast.literal_eval(tool_class.output_schema)

        return tool_class(**kwargs)

    @staticmethod
    def from_space(
        space_id: str,
        name: str,
        description: str = "",
        api_name: str | None = None,
        token: str | None = None,
    ):
        # [FUNCTION DOC]
        # Creates a tool that calls a Gradio Space via gradio_client.
        # Inputs: space_id (str), name (str), description (str, default empty), api_name (str or None:
        #   endpoint to use, defaults to the first named endpoint with a warning), token (str or None: HF token
        #   for private Spaces/quota).
        # Outputs: a SpaceToolWrapper instance (Tool subclass) whose inputs and output_type are derived from
        #   the Space API.
        # Exceptions: ImportError if gradio_client is missing; KeyError if api_name is not found; client
        #   errors propagate.
        # Side effects: network calls to the Space to fetch its API description; emits a warning when api_name
        #   is None.
        """
        Creates a [`Tool`] from a Space given its id on the Hub.

        Args:
            space_id (`str`):
                The id of the Space on the Hub.
            name (`str`):
                The name of the tool.
            description (`str`):
                The description of the tool.
            api_name (`str`, *optional*):
                The specific api_name to use, if the space has several tabs. If not precised, will default to the first available api.
            token (`str`, *optional*):
                Add your token to access private spaces or increase your GPU quotas.
        Returns:
            [`Tool`]:
                The Space, as a tool.

        Examples:
        ```py
        >>> image_generator = Tool.from_space(
        ...     space_id="black-forest-labs/FLUX.1-schnell",
        ...     name="image-generator",
        ...     description="Generate an image from a prompt"
        ... )
        >>> image = image_generator("Generate an image of a cool surfer in Tahiti")
        ```
        ```py
        >>> face_swapper = Tool.from_space(
        ...     "tuan2308/face-swap",
        ...     "face_swapper",
        ...     "Tool that puts the face shown on the first image on the second image. You can give it paths to images.",
        ... )
        >>> image = face_swapper('./aymeric.jpeg', './ruth.jpg')
        ```
        """
        from gradio_client import Client, handle_file

        class SpaceToolWrapper(Tool):
            # [CLASS DOC]
            # HELPER: Tool subclass wrapping a gradio_client Client for a Space endpoint. Skips forward
            #   signature validation.
            # Inputs: constructor args space_id, name, description, api_name, token (see from_space).
            # Key attributes: client (gradio Client), api_name, inputs (built from endpoint parameters),
            #   output_type.
            # Outputs: instances call the Space and return its prediction.
            # Exceptions: see constructor and forward.
            # Side effects: network calls.
            skip_forward_signature_validation = True

            def __init__(
                self,
                space_id: str,
                name: str,
                description: str = "",
                api_name: str | None = None,
                token: str | None = None,
            ):
                # [FUNCTION DOC]
                # HELPER: Connects to the Space, reads its named endpoints, picks the endpoint and builds
                #   inputs/output_type. Each endpoint parameter becomes an input with type from the parameter's
                #   type (object mapped to any), description from its python type description, and nullable set
                #   to whether it has a default. output_type is image for an Image return component, audio for
                #   Audio, else any. Marks the tool initialized.
                # Inputs: space_id (str), name (str), description (str, default empty), api_name (str or
                #   None), token (str or None).
                # Outputs: none.
                # Exceptions: AssertionError if the API description is not a dict; KeyError if api_name is
                #   unknown; client errors.
                # Side effects: network calls; warning when api_name defaults; does not call the base
                #   constructor.
                # KNOWN ISSUE: parameter types from the Space (e.g. lists or unknown type names) are not
                #   normalized and can fail the AUTHORIZED_TYPES validation run after construction.
                self.name = name
                self.description = description
                self.client = Client(space_id, hf_token=token)
                space_api = self.client.view_api(return_format="dict", print_info=False)
                assert isinstance(space_api, dict)
                space_description = space_api["named_endpoints"]

                # If api_name is not defined, take the first of the available APIs for this space
                if api_name is None:
                    api_name = list(space_description.keys())[0]
                    warnings.warn(
                        f"Since `api_name` was not defined, it was automatically set to the first available API: `{api_name}`."
                    )
                self.api_name = api_name

                try:
                    space_description_api = space_description[api_name]
                except KeyError:
                    raise KeyError(f"Could not find specified {api_name=} among available api names.")
                self.inputs = {}
                for parameter in space_description_api["parameters"]:
                    parameter_type = parameter["type"]["type"]
                    if parameter_type == "object":
                        parameter_type = "any"
                    self.inputs[parameter["parameter_name"]] = {
                        "type": parameter_type,
                        "description": parameter["python_type"]["description"],
                        "nullable": parameter["parameter_has_default"],
                    }
                output_component = space_description_api["returns"][0]["component"]
                if output_component == "Image":
                    self.output_type = "image"
                elif output_component == "Audio":
                    self.output_type = "audio"
                else:
                    self.output_type = "any"
                self.is_initialized = True

            def sanitize_argument_for_prediction(self, arg):
                # [FUNCTION DOC]
                # HELPER: Converts an argument into a form gradio_client accepts: PIL images are saved to a
                #   temporary PNG and replaced by its path; existing file paths (str or Path) and HTTP URLs are
                #   wrapped with handle_file.
                # Inputs: arg (any).
                # Outputs: the possibly converted argument.
                # Exceptions: ImportError if gradio_client or PIL is missing; image save errors propagate.
                # Side effects: may create a temporary file that is never deleted.
                from gradio_client.utils import is_http_url_like
                from PIL.Image import Image

                if isinstance(arg, Image):
                    temp_file = tempfile.NamedTemporaryFile(suffix=".png", delete=False)
                    arg.save(temp_file.name)
                    arg = temp_file.name
                if (
                    (isinstance(arg, str) and os.path.isfile(arg))
                    or (isinstance(arg, Path) and arg.exists() and arg.is_file())
                    or is_http_url_like(arg)
                ):
                    arg = handle_file(arg)
                return arg

            def forward(self, *args, **kwargs):
                # [FUNCTION DOC]
                # HELPER: Sanitizes all args/kwargs, calls the Space endpoint, and post-processes the result.
                #   If the result is a tuple/list and its second element is a string, it is treated as an error
                #   message; otherwise the first element is kept. String results ending with an image or audio
                #   extension are wrapped as AgentImage or AgentAudio.
                # Inputs: any positional and keyword arguments for the endpoint.
                # Outputs: the prediction (raw value, AgentImage or AgentAudio).
                # Exceptions: ValueError when the Space returns an error message; client errors propagate.
                # Side effects: network call; mutates the kwargs dict in place; may create temporary files.
                # KNOWN ISSUE: a list/tuple result with a single element raises IndexError when checking the
                #   second element, and a legitimate string second element (not an error) is misreported as an
                #   error.
                # Preprocess args and kwargs:
                args = list(args)
                for i, arg in enumerate(args):
                    args[i] = self.sanitize_argument_for_prediction(arg)
                for arg_name, arg in kwargs.items():
                    kwargs[arg_name] = self.sanitize_argument_for_prediction(arg)

                output = self.client.predict(*args, api_name=self.api_name, **kwargs)
                if isinstance(output, tuple) or isinstance(output, list):
                    if isinstance(output[1], str):
                        raise ValueError("The space returned this message: " + output[1])
                    output = output[
                        0
                    ]  # Sometime the space also returns the generation seed, in which case the result is at index 0
                IMAGE_EXTENSIONS = [".png", ".jpg", ".jpeg", ".gif", ".webp"]
                AUDIO_EXTENSIONS = [".mp3", ".wav", ".ogg", ".m4a", ".flac"]
                if isinstance(output, str) and any([output.endswith(ext) for ext in IMAGE_EXTENSIONS]):
                    output = AgentImage(output)
                elif isinstance(output, str) and any([output.endswith(ext) for ext in AUDIO_EXTENSIONS]):
                    output = AgentAudio(output)
                return output

        return SpaceToolWrapper(
            space_id=space_id,
            name=name,
            description=description,
            api_name=api_name,
            token=token,
        )

    @staticmethod
    def from_gradio(gradio_tool):
        # [FUNCTION DOC]
        # Wraps a gradio_tools-style tool object as a smolagents Tool.
        # Inputs: gradio_tool (object with name, description and a run method whose parameters are annotated
        #   with str, int or float type names).
        # Outputs: a GradioToolWrapper instance with output_type string and forward bound to gradio_tool.run.
        # Exceptions: KeyError if a run parameter annotation is not str/int/float; validation errors
        #   propagate.
        # Side effects: none.
        """
        Creates a [`Tool`] from a gradio tool.
        """
        import inspect

        class GradioToolWrapper(Tool):
            # [CLASS DOC]
            # HELPER: Tool subclass adapting a gradio tool; forward is the gradio tool's run method.
            # Inputs: constructor arg _gradio_tool.
            # Key attributes: name, description, output_type (string), inputs (derived from run's signature,
            #   empty descriptions), forward.
            # Outputs: n/a.
            # Exceptions: see constructor.
            # Side effects: none.
            def __init__(self, _gradio_tool):
                # [FUNCTION DOC]
                # HELPER: Copies name/description from the gradio tool, sets output_type string, builds inputs
                #   from run's parameters (annotation mapped through CONVERSION_DICT, empty description) and
                #   assigns forward to run.
                # Inputs: _gradio_tool (gradio tool object).
                # Outputs: none.
                # Exceptions: KeyError for annotations not in CONVERSION_DICT.
                # Side effects: none.
                # KNOWN ISSUE: neither calls the base constructor nor sets is_initialized, so calling the
                #   wrapper raises AttributeError when Tool.__call__ reads is_initialized.
                # KNOWN ISSUE: annotations are looked up as given; real type objects (not strings) like the
                #   str class would not match the string keys of CONVERSION_DICT.
                self.name = _gradio_tool.name
                self.description = _gradio_tool.description
                self.output_type = "string"
                self._gradio_tool = _gradio_tool
                func_args = list(inspect.signature(_gradio_tool.run).parameters.items())
                self.inputs = {
                    key: {"type": CONVERSION_DICT[value.annotation], "description": ""} for key, value in func_args
                }
                self.forward = self._gradio_tool.run

        return GradioToolWrapper(gradio_tool)

    @staticmethod
    def from_langchain(langchain_tool):
        # [FUNCTION DOC]
        # Wraps a LangChain tool as a smolagents Tool.
        # Inputs: langchain_tool (object with name, description, args dict and run method).
        # Outputs: a LangChainToolWrapper instance (output_type string).
        # Exceptions: validation errors propagate (e.g. invalid name or unauthorized input types).
        # Side effects: mutates the per-input dicts of the LangChain tool's args (they are shallow-copied
        #   only).
        """
        Creates a [`Tool`] from a langchain tool.
        """

        class LangChainToolWrapper(Tool):
            # [CLASS DOC]
            # HELPER: Tool subclass adapting a LangChain tool; skips forward signature validation.
            # Inputs: constructor arg _langchain_tool.
            # Key attributes: name (lowercased), description, inputs, output_type (string), langchain_tool.
            # Outputs: n/a.
            # Exceptions: see constructor.
            # Side effects: see constructor.
            skip_forward_signature_validation = True

            def __init__(self, _langchain_tool):
                # [FUNCTION DOC]
                # HELPER: Builds inputs from the LangChain tool's args: removes title keys and sets empty
                #   descriptions; lowercases the name; sets output_type string; marks initialized.
                # Inputs: _langchain_tool.
                # Outputs: none.
                # Exceptions: none raised directly.
                # Side effects: mutates the nested dicts of the LangChain tool's args (shallow copy of the
                #   outer dict only).
                self.name = _langchain_tool.name.lower()
                self.description = _langchain_tool.description
                self.inputs = _langchain_tool.args.copy()
                for input_content in self.inputs.values():
                    if "title" in input_content:
                        input_content.pop("title")
                    input_content["description"] = ""
                self.output_type = "string"
                self.langchain_tool = _langchain_tool
                self.is_initialized = True

            def forward(self, *args, **kwargs):
                # [FUNCTION DOC]
                # HELPER: Builds a tool-input dict from kwargs plus positional args, then runs the LangChain
                #   tool.
                # Inputs: positional args (mapped to input names) and keyword args.
                # Outputs: the LangChain tool's run result.
                # Exceptions: errors from the LangChain tool propagate.
                # Side effects: whatever the LangChain tool does.
                # KNOWN ISSUE: every positional argument is assigned to the first input key, so with multiple
                #   positional args each overwrites the previous and other inputs are never filled.
                tool_input = kwargs.copy()
                for index, argument in enumerate(args):
                    if index < len(self.inputs):
                        input_key = next(iter(self.inputs))
                        tool_input[input_key] = argument
                return self.langchain_tool.run(tool_input)

        return LangChainToolWrapper(langchain_tool)


def launch_gradio_demo(tool: Tool):
    # [FUNCTION DOC]
    # Launches a Gradio Interface for a tool: one component per input (by type: boolean checkbox, image,
    #   audio, string textbox, integer/number number field) and one output component for output_type; the
    #   interface calls the tool with input/output sanitization.
    # Inputs: tool (Tool).
    # Outputs: none.
    # Exceptions: ImportError if gradio is not installed; KeyError for input or output types without a mapped
    #   component (e.g. array, object, any, null, or list-typed inputs).
    # Side effects: starts a Gradio web server (blocking launch).
    """
    Launches a gradio demo for a tool. The corresponding tool class needs to properly implement the class attributes
    `inputs` and `output_type`.

    Args:
        tool (`Tool`): The tool for which to launch the demo.
    """
    try:
        import gradio as gr
    except ImportError:
        raise ImportError("Gradio should be installed in order to launch a gradio demo.")

    TYPE_TO_COMPONENT_CLASS_MAPPING = {
        "boolean": gr.Checkbox,
        "image": gr.Image,
        "audio": gr.Audio,
        "string": gr.Textbox,
        "integer": gr.Number,
        "number": gr.Number,
    }

    def tool_forward(*args, **kwargs):
        # [FUNCTION DOC]
        # HELPER: Gradio callback that calls the tool with sanitize_inputs_outputs enabled; its signature is
        #   set to the tool's forward signature.
        # Inputs: any args/kwargs for the tool.
        # Outputs: the tool's sanitized output.
        # Exceptions: tool errors propagate.
        # Side effects: whatever the tool does.
        return tool(*args, sanitize_inputs_outputs=True, **kwargs)

    tool_forward.__signature__ = inspect.signature(tool.forward)

    gradio_inputs = []
    for input_name, input_details in tool.inputs.items():
        input_gradio_component_class = TYPE_TO_COMPONENT_CLASS_MAPPING[input_details["type"]]
        new_component = input_gradio_component_class(label=input_name)
        gradio_inputs.append(new_component)

    output_gradio_component_class = TYPE_TO_COMPONENT_CLASS_MAPPING[tool.output_type]
    gradio_output = output_gradio_component_class(label="Output")

    gr.Interface(
        fn=tool_forward,
        inputs=gradio_inputs,
        outputs=gradio_output,
        title=tool.name,
        description=tool.description,
        api_name=tool.name,
    ).launch()


def load_tool(
    repo_id,
    model_repo_id: str | None = None,
    token: str | None = None,
    trust_remote_code: bool = False,
    **kwargs,
):
    # [FUNCTION DOC]
    # Convenience wrapper that loads a tool from a Hub Space by delegating to Tool.from_hub.
    # Inputs: repo_id (str), model_repo_id (str or None), token (str or None), trust_remote_code (bool,
    #   default False), kwargs (download options and constructor args).
    # Outputs: a Tool instance.
    # Exceptions: ValueError if trust_remote_code is False; errors from from_hub propagate.
    # Side effects: network download and execution of remote code.
    # KNOWN ISSUE: model_repo_id is forwarded as a keyword argument into kwargs and thus to the tool's
    #   constructor; tools whose constructor does not accept it will fail.
    """
    Main function to quickly load a tool from the Hub.

    <Tip warning={true}>

    Loading a tool means that you'll download the tool and execute it locally.
    ALWAYS inspect the tool you're downloading before loading it within your runtime, as you would do when
    installing a package using pip/npm/apt.

    </Tip>

    Args:
        repo_id (`str`):
            Space repo ID of a tool on the Hub.
        model_repo_id (`str`, *optional*):
            Use this argument to use a different model than the default one for the tool you selected.
        token (`str`, *optional*):
            The token to identify you on hf.co. If unset, will use the token generated when running `huggingface-cli
            login` (stored in `~/.huggingface`).
        trust_remote_code (`bool`, *optional*, defaults to False):
            This needs to be accepted in order to load a tool from Hub.
        kwargs (additional keyword arguments, *optional*):
            Additional keyword arguments that will be split in two: all arguments relevant to the Hub (such as
            `cache_dir`, `revision`, `subfolder`) will be used when downloading the files for your tool, and the others
            will be passed along to its init.
    """
    return Tool.from_hub(
        repo_id,
        model_repo_id=model_repo_id,
        token=token,
        trust_remote_code=trust_remote_code,
        **kwargs,
    )


def add_description(description):
    # [FUNCTION DOC]
    # Decorator factory attaching a description and name to a function.
    # Inputs: description (any, normally str).
    # Outputs: a decorator that returns the same function with description and name (the function's own name)
    #   attributes set.
    # Exceptions: none raised directly.
    # Side effects: the returned decorator mutates the decorated function's attributes.
    """
    A decorator that adds a description to a function.
    """

    def inner(func):
        # [FUNCTION DOC]
        # HELPER: Sets description and name attributes on the given function and returns it.
        # Inputs: func (function).
        # Outputs: the same function.
        # Exceptions: AttributeError if func does not accept attribute assignment.
        # Side effects: mutates func.
        func.description = description
        func.name = func.__name__
        return func

    return inner


class ToolCollection:
    # [CLASS DOC]
    # Container holding a list of tools, with constructors loading them from a Hub collection or an MCP
    #   server.
    # Inputs: constructor arg tools (list of Tool).
    # Key attributes: tools (list of Tool).
    # Outputs: n/a.
    # Exceptions: none raised directly by the constructor.
    # Side effects: none.
    """
    Tool collections enable loading a collection of tools in the agent's toolbox.

    Collections can be loaded from a collection in the Hub or from an MCP server, see:
    - [`ToolCollection.from_hub`]
    - [`ToolCollection.from_mcp`]

    For example and usage, see: [`ToolCollection.from_hub`] and [`ToolCollection.from_mcp`]
    """

    def __init__(self, tools: list[Tool]):
        # [FUNCTION DOC]
        # Stores the given list of tools.
        # Inputs: tools (list of Tool).
        # Outputs: none.
        # Exceptions: none raised directly.
        # Side effects: sets self.tools (no copy).
        self.tools = tools

    @classmethod
    def from_hub(
        cls,
        collection_slug: str,
        token: str | None = None,
        trust_remote_code: bool = False,
    ) -> "ToolCollection":
        # [FUNCTION DOC]
        # Loads all Space items of a Hub collection as tools via Tool.from_hub.
        # Inputs: collection_slug (str), token (str or None), trust_remote_code (bool, default False).
        # Outputs: a ToolCollection (constructed via cls) with one tool per unique Space id (order
        #   unspecified, from a set).
        # Exceptions: ValueError if trust_remote_code is False (from Tool.from_hub, only if the collection has
        #   Spaces); Hub/network errors propagate.
        # Side effects: network calls; downloads and executes remote code.
        """Loads a tool collection from the Hub.

        it adds a collection of tools from all Spaces in the collection to the agent's toolbox

        > [!NOTE]
        > Only Spaces will be fetched, so you can feel free to add models and datasets to your collection if you'd
        > like for this collection to showcase them.

        Args:
            collection_slug (str): The collection slug referencing the collection.
            token (str, *optional*): The authentication token if the collection is private.
            trust_remote_code (bool, *optional*, defaults to False): Whether to trust the remote code.

        Returns:
            ToolCollection: A tool collection instance loaded with the tools.

        Example:
        ```py
        >>> from smolagents import ToolCollection, CodeAgent

        >>> image_tool_collection = ToolCollection.from_hub("huggingface-tools/diffusion-tools-6630bb19a942c2306a2cdb6f")
        >>> agent = CodeAgent(tools=[*image_tool_collection.tools], add_base_tools=True)

        >>> agent.run("Please draw me a picture of rivers and lakes.")
        ```
        """
        _collection = get_collection(collection_slug, token=token)
        _hub_repo_ids = {item.item_id for item in _collection.items if item.item_type == "space"}

        tools = [Tool.from_hub(repo_id, token, trust_remote_code) for repo_id in _hub_repo_ids]

        return cls(tools)

    @classmethod
    @contextmanager
    def from_mcp(
        cls,
        server_parameters: "mcp.StdioServerParameters" | dict,
        trust_remote_code: bool = False,
        structured_output: bool | None = None,
    ) -> "ToolCollection":
        # [FUNCTION DOC]
        # Context manager that connects to an MCP server via mcpadapt and yields a ToolCollection of its
        #   tools; the connection is closed on exit. For dict parameters, transport defaults to streamable-http
        #   (written into the dict) and must be sse or streamable-http. structured_output None emits a
        #   FutureWarning and is treated as False.
        # Inputs: server_parameters (mcp StdioServerParameters or dict with url and optional transport),
        #   trust_remote_code (bool, default False; must be True), structured_output (bool or None).
        # Outputs: yields a ToolCollection built from the adapter's tools.
        # Exceptions: ImportError if mcpadapt is missing; ValueError for unsupported transport or untrusted
        #   code.
        # Side effects: FutureWarning; mutates a dict server_parameters; starts a background thread /
        #   subprocess or network connection to the MCP server for the duration of the context.
        # KNOWN ISSUE: annotated as returning ToolCollection although it is a context manager yielding one;
        #   the docstring says structured_output defaults to False while the default is None (with a warning).
        # KNOWN ISSUE: the transport check runs before the trust check, so an untrusted call with a dict can
        #   still mutate the caller's dict before failing.
        """Automatically load a tool collection from an MCP server.

        This method supports Stdio, Streamable HTTP, and legacy HTTP+SSE MCP servers. Look at the `server_parameters`
        argument for more details on how to connect to each MCP server.

        Note: a separate thread will be spawned to run an asyncio event loop handling
        the MCP server.

        Args:
            server_parameters (`mcp.StdioServerParameters` or `dict`):
                Configuration parameters to connect to the MCP server. This can be:

                - An instance of `mcp.StdioServerParameters` for connecting a Stdio MCP server via standard input/output using a subprocess.

                - A `dict` with at least:
                  - "url": URL of the server.
                  - "transport": Transport protocol to use, one of:
                    - "streamable-http": Streamable HTTP transport (default).
                    - "sse": Legacy HTTP+SSE transport (deprecated).
            trust_remote_code (`bool`, *optional*, defaults to `False`):
                Whether to trust the execution of code from tools defined on the MCP server.
                This option should only be set to `True` if you trust the MCP server,
                and understand the risks associated with running remote code on your local machine.
                If set to `False`, loading tools from MCP will fail.
            structured_output (`bool`, *optional*, defaults to `False`):
                Whether to enable structured output features for MCP tools. If True, enables:
                - Support for outputSchema in MCP tools
                - Structured content handling (structuredContent from MCP responses)
                - JSON parsing fallback for structured data
                If False, uses the original simple text-only behavior for backwards compatibility.

        Returns:
            ToolCollection: A tool collection instance.

        Example with a Stdio MCP server:
        ```py
        >>> import os
        >>> from smolagents import ToolCollection, CodeAgent, InferenceClientModel
        >>> from mcp import StdioServerParameters

        >>> model = InferenceClientModel()

        >>> server_parameters = StdioServerParameters(
        >>>     command="uvx",
        >>>     args=["--quiet", "pubmedmcp@0.1.3"],
        >>>     env={"UV_PYTHON": "3.12", **os.environ},
        >>> )

        >>> with ToolCollection.from_mcp(server_parameters, trust_remote_code=True) as tool_collection:
        >>>     agent = CodeAgent(tools=[*tool_collection.tools], add_base_tools=True, model=model)
        >>>     agent.run("Please find a remedy for hangover.")
        ```

        Example with structured output enabled:
        ```py
        >>> with ToolCollection.from_mcp(server_parameters, trust_remote_code=True, structured_output=True) as tool_collection:
        >>>     agent = CodeAgent(tools=[*tool_collection.tools], add_base_tools=True, model=model)
        >>>     agent.run("Please find a remedy for hangover.")
        ```

        Example with a Streamable HTTP MCP server:
        ```py
        >>> with ToolCollection.from_mcp({"url": "http://127.0.0.1:8000/mcp", "transport": "streamable-http"}, trust_remote_code=True) as tool_collection:
        >>>     agent = CodeAgent(tools=[*tool_collection.tools], add_base_tools=True, model=model)
        >>>     agent.run("Please find a remedy for hangover.")
        ```
        """
        # Handle future warning for structured_output default value change
        if structured_output is None:
            warnings.warn(
                "Parameter 'structured_output' was not specified. "
                "Currently it defaults to False, but in version 1.25, the default will change to True. "
                "To suppress this warning, explicitly set structured_output=True (new behavior) or structured_output=False (legacy behavior). "
                "See documentation at https://huggingface.co/docs/smolagents/tutorials/tools#structured-output-and-output-schema-support for more details.",
                FutureWarning,
                stacklevel=2,
            )
            structured_output = False

        try:
            from mcpadapt.core import MCPAdapt
            from mcpadapt.smolagents_adapter import SmolAgentsAdapter
        except ImportError:
            raise ImportError(
                """Please install 'mcp' extra to use ToolCollection.from_mcp: `pip install 'smolagents[mcp]'`."""
            )
        if isinstance(server_parameters, dict):
            transport = server_parameters.get("transport")
            if transport is None:
                transport = "streamable-http"
                server_parameters["transport"] = transport
            if transport not in {"sse", "streamable-http"}:
                raise ValueError(
                    f"Unsupported transport: {transport}. Supported transports are 'streamable-http' and 'sse'."
                )
        if not trust_remote_code:
            raise ValueError(
                "Loading tools from MCP requires you to acknowledge you trust the MCP server, "
                "as it will execute code on your local machine: pass `trust_remote_code=True`."
            )
        with MCPAdapt(server_parameters, SmolAgentsAdapter(structured_output=structured_output)) as tools:
            yield cls(tools)


def tool(tool_function: Callable) -> Tool:
    # [FUNCTION DOC]
    # Converts a type-hinted, documented function into an instance of a dynamically created Tool subclass
    #   named SimpleTool. Uses get_json_schema on the function for name, description, inputs (parameters
    #   properties) and output type (return type); a function without a return hint is allowed only if it has no
    #   parameters (output type null). output_schema is taken from the schema's output_schema or return schema
    #   when present. forward becomes a static wrapper of the function with a signature having self prepended.
    #   It also builds source text for the class and the forward method (decorator lines other than the tool
    #   decorator kept, body taken from the first body statement onwards) and stores it as __source__ on the
    #   class and forward.
    # Inputs: tool_function (callable with type hints and an Args docstring section).
    # Outputs: a SimpleTool instance (already initialized).
    # Exceptions: TypeHintParsingException for missing return hint (with parameters) or bad hints/docstring;
    #   ValueError if no function definition is found or there are multiple tool decorators; validation errors.
    # Side effects: warns (UserWarning) when other decorators are present; reads the function's source file.
    # KNOWN ISSUE: since forward source starts at the first body statement, comments placed between the def
    #   line and the first statement are dropped from the stored source.
    # KNOWN ISSUE: decorator_lines skips only up to the tool decorator end line, so if other decorators come
    #   before the tool decorator they are omitted, and only the tool decorator by bare name (not attribute
    #   access) is detected.
    """
    Convert a function into an instance of a dynamically created Tool subclass.

    Args:
        tool_function (`Callable`): Function to convert into a Tool subclass.
            Should have type hints for each input and a type hint for the output.
            Should also have a docstring including the description of the function
            and an 'Args:' part where each argument is described.
    """
    tool_json_schema = get_json_schema(tool_function)["function"]
    if "return" not in tool_json_schema:
        if len(tool_json_schema["parameters"]["properties"]) == 0:
            tool_json_schema["return"] = {"type": "null"}
        else:
            raise TypeHintParsingException(
                "Tool return type not found: make sure your function has a return type hint!"
            )

    class SimpleTool(Tool):
        # [CLASS DOC]
        # HELPER: Dynamically created Tool subclass whose class attributes are filled from the function's JSON
        #   schema.
        # Inputs: no constructor args.
        # Key attributes: name, description, inputs, output_type, optional output_schema, forward (static
        #   wrapper).
        # Outputs: n/a.
        # Exceptions: validation errors at construction.
        # Side effects: none.
        def __init__(self):
            # [FUNCTION DOC]
            # HELPER: Marks the tool initialized (no setup needed).
            # Inputs: self.
            # Outputs: none.
            # Exceptions: none raised directly.
            # Side effects: sets is_initialized to True.
            self.is_initialized = True

    # Set the class attributes
    SimpleTool.name = tool_json_schema["name"]
    SimpleTool.description = tool_json_schema["description"]
    SimpleTool.inputs = tool_json_schema["parameters"]["properties"]
    SimpleTool.output_type = tool_json_schema["return"]["type"]

    # Set output_schema if it exists in the JSON schema
    if "output_schema" in tool_json_schema:
        SimpleTool.output_schema = tool_json_schema["output_schema"]
    elif "return" in tool_json_schema and "schema" in tool_json_schema["return"]:
        SimpleTool.output_schema = tool_json_schema["return"]["schema"]

    @wraps(tool_function)
    def wrapped_function(*args, **kwargs):
        # [FUNCTION DOC]
        # HELPER: Thin wrapper calling the original function, used as SimpleTool.forward (preserves metadata
        #   via wraps).
        # Inputs: any args/kwargs.
        # Outputs: the original function's return value.
        # Exceptions: propagates the function's errors.
        # Side effects: those of the original function.
        return tool_function(*args, **kwargs)

    # Bind the copied function to the forward method
    SimpleTool.forward = staticmethod(wrapped_function)

    # Get the signature parameters of the tool function
    sig = inspect.signature(tool_function)
    # - Add "self" as first parameter to tool_function signature
    new_sig = sig.replace(
        parameters=[inspect.Parameter("self", inspect.Parameter.POSITIONAL_OR_KEYWORD)] + list(sig.parameters.values())
    )
    # - Set the signature of the forward method
    SimpleTool.forward.__signature__ = new_sig

    # Create and attach the source code of the dynamically created tool class and forward method
    # - Get the source code of tool_function
    tool_source = textwrap.dedent(inspect.getsource(tool_function))
    # - Remove the tool decorator and function definition line
    lines = tool_source.splitlines()
    tree = ast.parse(tool_source)
    #   - Find function definition
    func_node = next((node for node in ast.walk(tree) if isinstance(node, ast.FunctionDef)), None)
    if not func_node:
        raise ValueError(
            f"No function definition found in the provided source of {tool_function.__name__}. "
            "Ensure the input is a standard function."
        )
    #   - Extract decorator lines
    decorator_lines = ""
    if func_node.decorator_list:
        tool_decorators = [d for d in func_node.decorator_list if isinstance(d, ast.Name) and d.id == "tool"]
        if len(tool_decorators) > 1:
            raise ValueError(
                f"Multiple @tool decorators found on function '{func_node.name}'. Only one @tool decorator is allowed."
            )
        if len(tool_decorators) < len(func_node.decorator_list):
            warnings.warn(
                f"Function '{func_node.name}' has decorators other than @tool. "
                "This may cause issues with serialization in the remote executor. See issue #1626."
            )
        decorator_start = tool_decorators[0].end_lineno if tool_decorators else 0
        decorator_end = func_node.decorator_list[-1].end_lineno
        decorator_lines = "\n".join(lines[decorator_start:decorator_end])
    #   - Extract tool source body
    body_start = func_node.body[0].lineno - 1  # AST lineno starts at 1
    tool_source_body = "\n".join(lines[body_start:])
    # - Create the forward method source, including def line and indentation
    forward_method_source = f"def forward{new_sig}:\n{tool_source_body}"
    # - Create the class source
    indent = " " * 4  # for class method
    class_source = (
        textwrap.dedent(f"""
        class SimpleTool(Tool):
            name: str = "{tool_json_schema["name"]}"
            description: str = {json.dumps(textwrap.dedent(tool_json_schema["description"]).strip())}
            inputs: dict[str, dict[str, str]] = {tool_json_schema["parameters"]["properties"]}
            output_type: str = "{tool_json_schema["return"]["type"]}"

            def __init__(self):
                self.is_initialized = True

        """)
        + textwrap.indent(decorator_lines, indent)
        + textwrap.indent(forward_method_source, indent)
    )
    # - Store the source code on both class and method for inspection
    SimpleTool.__source__ = class_source
    SimpleTool.forward.__source__ = forward_method_source

    simple_tool = SimpleTool()
    return simple_tool


class PipelineTool(Tool):
    # [CLASS DOC]
    # Tool base for transformers models: lazily loads a pre-processor, model and post-processor, then runs
    #   encode, forward (no-grad model call) and decode on each call. Skips forward signature validation.
    # Inputs (class attributes): model_class, default_checkpoint, pre_processor_class and post_processor_class
    #   (default AutoProcessor when None). Constructor args: model (checkpoint name or model instance; defaults
    #   to default_checkpoint), pre_processor (defaults to model), post_processor (defaults to pre_processor at
    #   setup), device, device_map, model_kwargs (dict), token, hub_kwargs (extra Hub loading kwargs).
    # Key attributes: model, pre_processor, post_processor, device, device_map, model_kwargs, hub_kwargs.
    # Outputs: calling returns the decoded outputs.
    # Exceptions: see constructor/setup/call.
    # Side effects: model downloads and device moves at setup.
    # KNOWN ISSUE: the default class attributes inputs (a dict mapping to the str type) and output_type (the
    #   str type) are not valid per validate_arguments, so instantiating PipelineTool or a subclass that does
    #   not override them fails validation.
    """
    A [`Tool`] tailored towards Transformer models. On top of the class attributes of the base class [`Tool`], you will
    need to specify:

    - **model_class** (`type`) -- The class to use to load the model in this tool.
    - **default_checkpoint** (`str`) -- The default checkpoint that should be used when the user doesn't specify one.
    - **pre_processor_class** (`type`, *optional*, defaults to [`transformers.AutoProcessor`]) -- The class to use to load the
      pre-processor
    - **post_processor_class** (`type`, *optional*, defaults to [`transformers.AutoProcessor`]) -- The class to use to load the
      post-processor (when different from the pre-processor).

    Args:
        model (`str` or [`transformers.PreTrainedModel`], *optional*):
            The name of the checkpoint to use for the model, or the instantiated model. If unset, will default to the
            value of the class attribute `default_checkpoint`.
        pre_processor (`str` or `Any`, *optional*):
            The name of the checkpoint to use for the pre-processor, or the instantiated pre-processor (can be a
            tokenizer, an image processor, a feature extractor or a processor). Will default to the value of `model` if
            unset.
        post_processor (`str` or `Any`, *optional*):
            The name of the checkpoint to use for the post-processor, or the instantiated pre-processor (can be a
            tokenizer, an image processor, a feature extractor or a processor). Will default to the `pre_processor` if
            unset.
        device (`int`, `str` or `torch.device`, *optional*):
            The device on which to execute the model. Will default to any accelerator available (GPU, MPS etc...), the
            CPU otherwise.
        device_map (`str` or `dict`, *optional*):
            If passed along, will be used to instantiate the model.
        model_kwargs (`dict`, *optional*):
            Any keyword argument to send to the model instantiation.
        token (`str`, *optional*):
            The token to use as HTTP bearer authorization for remote files. If unset, will use the token generated when
            running `huggingface-cli login` (stored in `~/.huggingface`).
        hub_kwargs (additional keyword arguments, *optional*):
            Any additional keyword argument to send to the methods that will load the data from the Hub.
    """

    pre_processor_class = None
    model_class = None
    post_processor_class = None
    default_checkpoint = None
    description = "This is a pipeline tool"
    name = "pipeline"
    inputs = {"prompt": str}
    output_type = str
    skip_forward_signature_validation = True

    def __init__(
        self,
        model=None,
        pre_processor=None,
        post_processor=None,
        device=None,
        device_map=None,
        model_kwargs=None,
        token=None,
        **hub_kwargs,
    ):
        # [FUNCTION DOC]
        # Stores model/processor configuration after checking dependencies.
        # Inputs: model, pre_processor, post_processor, device, device_map, model_kwargs, token (all default
        #   None), hub_kwargs (extra keyword args).
        # Outputs: none.
        # Exceptions: ModuleNotFoundError if accelerate or torch is unavailable; ValueError if no model and no
        #   default_checkpoint.
        # Side effects: sets attributes; adds device_map into model_kwargs when given; adds token into
        #   hub_kwargs.
        if not _is_package_available("accelerate") or not _is_package_available("torch"):
            raise ModuleNotFoundError(
                "Please install 'transformers' extra to use a PipelineTool: `pip install 'smolagents[transformers]'`"
            )

        if model is None:
            if self.default_checkpoint is None:
                raise ValueError("This tool does not implement a default checkpoint, you need to pass one.")
            model = self.default_checkpoint
        if pre_processor is None:
            pre_processor = model

        self.model = model
        self.pre_processor = pre_processor
        self.post_processor = post_processor
        self.device = device
        self.device_map = device_map
        self.model_kwargs = {} if model_kwargs is None else model_kwargs
        if device_map is not None:
            self.model_kwargs["device_map"] = device_map
        self.hub_kwargs = hub_kwargs
        self.hub_kwargs["token"] = token

        super().__init__()

    def setup(self):
        # [FUNCTION DOC]
        # Instantiates processors and model if given as strings (using the class attributes, defaulting
        #   processors to AutoProcessor), defaults post_processor to pre_processor, determines the device (first
        #   entry of the model's device map when device_map is set, else accelerate's default device), moves the
        #   model to the device when no device_map, then marks initialized.
        # Inputs: self.
        # Outputs: none.
        # Exceptions: ImportError for missing transformers/accelerate; loading errors;
        #   TypeError/AttributeError if model_class is None while model is a string.
        # Side effects: downloads/loads models, moves model to device, mutates instance and class-processor
        #   attributes (sets pre_processor_class/post_processor_class on the instance).
        """
        Instantiates the `pre_processor`, `model` and `post_processor` if necessary.
        """
        if isinstance(self.pre_processor, str):
            if self.pre_processor_class is None:
                from transformers import AutoProcessor

                self.pre_processor_class = AutoProcessor
            self.pre_processor = self.pre_processor_class.from_pretrained(self.pre_processor, **self.hub_kwargs)

        if isinstance(self.model, str):
            self.model = self.model_class.from_pretrained(self.model, **self.model_kwargs, **self.hub_kwargs)

        if self.post_processor is None:
            self.post_processor = self.pre_processor
        elif isinstance(self.post_processor, str):
            if self.post_processor_class is None:
                from transformers import AutoProcessor

                self.post_processor_class = AutoProcessor
            self.post_processor = self.post_processor_class.from_pretrained(self.post_processor, **self.hub_kwargs)

        if self.device is None:
            if self.device_map is not None:
                self.device = list(self.model.hf_device_map.values())[0]
            else:
                from accelerate import PartialState

                self.device = PartialState().default_device

        if self.device_map is None:
            self.model.to(self.device)

        super().setup()

    def encode(self, raw_inputs):
        # [FUNCTION DOC]
        # Applies the pre-processor to raw inputs.
        # Inputs: raw_inputs (any).
        # Outputs: the pre-processor output (expected to be a dict-like mapping).
        # Exceptions: pre-processor errors propagate.
        # Side effects: none.
        """
        Uses the `pre_processor` to prepare the inputs for the `model`.
        """
        return self.pre_processor(raw_inputs)

    def forward(self, inputs):
        # [FUNCTION DOC]
        # Runs the model on encoded inputs under torch no-grad.
        # Inputs: inputs (dict of keyword arguments for the model).
        # Outputs: the model output.
        # Exceptions: ImportError if torch is missing; model errors propagate.
        # Side effects: none beyond computation.
        """
        Sends the inputs through the `model`.
        """
        import torch

        with torch.no_grad():
            return self.model(**inputs)

    def decode(self, outputs):
        # [FUNCTION DOC]
        # Applies the post-processor to model outputs.
        # Inputs: outputs (model output).
        # Outputs: the decoded result.
        # Exceptions: post-processor errors propagate.
        # Side effects: none.
        """
        Uses the `post_processor` to decode the model output.
        """
        return self.post_processor(outputs)

    def __call__(self, *args, sanitize_inputs_outputs: bool = False, **kwargs):
        # [FUNCTION DOC]
        # Full pipeline call: setup if needed, optional input sanitization, encode, move tensor inputs to the
        #   device, forward with tensor and non-tensor inputs merged, move outputs to CPU, decode, optional
        #   output sanitization.
        # Inputs: args/kwargs for encode; sanitize_inputs_outputs (bool, default False).
        # Outputs: decoded (optionally sanitized) outputs.
        # Exceptions: ImportError for torch/accelerate; errors from any stage propagate.
        # Side effects: device transfers; may trigger setup.
        # KNOWN ISSUE: unlike Tool.__call__, it does not unpack a single dict argument into kwargs.
        import torch
        from accelerate.utils import send_to_device

        if not self.is_initialized:
            self.setup()

        if sanitize_inputs_outputs:
            args, kwargs = handle_agent_input_types(*args, **kwargs)
        encoded_inputs = self.encode(*args, **kwargs)

        tensor_inputs = {k: v for k, v in encoded_inputs.items() if isinstance(v, torch.Tensor)}
        non_tensor_inputs = {k: v for k, v in encoded_inputs.items() if not isinstance(v, torch.Tensor)}

        encoded_inputs = send_to_device(tensor_inputs, self.device)
        outputs = self.forward({**encoded_inputs, **non_tensor_inputs})
        outputs = send_to_device(outputs, "cpu")
        decoded_outputs = self.decode(outputs)
        if sanitize_inputs_outputs:
            decoded_outputs = handle_agent_output_types(decoded_outputs, self.output_type)
        return decoded_outputs


def get_tools_definition_code(tools: dict[str, Tool]) -> str:
    # [FUNCTION DOC]
    # Generates Python source defining the given tools for a remote/sandbox executor: imports for all
    #   BASE_BUILTIN_MODULES, a minimal Tool stub class (call delegates to forward), then each tool's class
    #   source (with its smolagents Tool import removed) followed by an instantiation assigned to the tool's
    #   name.
    # Inputs: tools (dict mapping name to Tool).
    # Outputs: str, the combined source.
    # Exceptions: errors from validate_tool_attributes (imports not checked) or instance_to_source propagate.
    # Side effects: none.
    tool_codes = []
    for tool in tools.values():
        validate_tool_attributes(tool.__class__, check_imports=False)
        tool_code = instance_to_source(tool, base_cls=Tool)
        tool_code = tool_code.replace("from smolagents.tools import Tool", "")
        tool_code += f"\n\n{tool.name} = {tool.__class__.__name__}()\n"
        tool_codes.append(tool_code)

    tool_definition_code = "\n".join([f"import {module}" for module in BASE_BUILTIN_MODULES])
    tool_definition_code += textwrap.dedent(
        """
    from typing import Any

    class Tool:
        def __call__(self, *args, **kwargs):
            return self.forward(*args, **kwargs)

        def forward(self, *args, **kwargs):
            pass # to be implemented in child class
    """
    )
    tool_definition_code += "\n\n".join(tool_codes)
    return tool_definition_code


def validate_tool_arguments(tool: Tool, arguments: Any) -> None:
    # [FUNCTION DOC]
    # Checks that call arguments match a tool's input schema.
    # For dict arguments: every key must be an input; each value's JSON type (derived from its Python type)
    #   must equal the expected type (or be contained in it when the expected type is a list), unless the
    #   expected type is any, the value is None and the input is nullable, or the value is an integer and the
    #   expected type is number; every non-nullable input must be present. For non-dict arguments: the value's
    #   JSON type must equal the first input's type unless that type is any.
    # Inputs: tool (Tool), arguments (dict or single value).
    # Outputs: None.
    # Exceptions: ValueError (unknown argument or missing required argument), TypeError (type mismatch).
    # Side effects: none.
    # KNOWN ISSUE: list-typed expected types containing any, or number with an integer value, are not accepted
    #   in the dict path (those exemptions compare against the whole list); the non-dict path ignores list
    #   types, nullable and integer-to-number coercion, and raises IndexError for tools with no inputs.
    # KNOWN ISSUE: nullable inputs may be omitted even when they are required parameters (nullable conflates
    #   optional and accepts-None).
    """Validate tool arguments against tool's input schema.

    Checks that all provided arguments match the tool's expected input types and that
    all required arguments are present. Supports both dictionary arguments and single
    value arguments for tools with one input parameter.

    Args:
        tool (`Tool`): Tool whose input schema will be used for validation.
        arguments (`Any`): Arguments to validate. Can be a dictionary mapping
            argument names to values, or a single value for tools with one input.


    Raises:
        ValueError: If an argument is not in the tool's input schema, if a required
            argument is missing, or if the argument value doesn't match the expected type.
        TypeError: If an argument has an incorrect type that cannot be converted
            (e.g., string instead of number, excluding integer to number conversion).

    Note:
        - Supports type coercion from integer to number
        - Handles nullable parameters when explicitly marked in the schema
        - Accepts "any" type as a wildcard that matches all types
    """
    if isinstance(arguments, dict):
        for key, value in arguments.items():
            if key not in tool.inputs:
                raise ValueError(f"Argument {key} is not in the tool's input schema")

            actual_type = _get_json_schema_type(type(value))["type"]
            expected_type = tool.inputs[key]["type"]
            expected_type_is_nullable = tool.inputs[key].get("nullable", False)

            # Type is valid if it matches, is "any", or is null for nullable parameters
            if (
                (actual_type != expected_type if isinstance(expected_type, str) else actual_type not in expected_type)
                and expected_type != "any"
                and not (actual_type == "null" and expected_type_is_nullable)
            ):
                if actual_type == "integer" and expected_type == "number":
                    continue
                raise TypeError(f"Argument {key} has type '{actual_type}' but should be '{tool.inputs[key]['type']}'")

        for key, schema in tool.inputs.items():
            key_is_nullable = schema.get("nullable", False)
            if key not in arguments and not key_is_nullable:
                raise ValueError(f"Argument {key} is required")
        return None
    else:
        expected_type = list(tool.inputs.values())[0]["type"]
        if _get_json_schema_type(type(arguments))["type"] != expected_type and not expected_type == "any":
            raise TypeError(f"Argument has type '{type(arguments).__name__}' but should be '{expected_type}'")


__all__ = [
    "AUTHORIZED_TYPES",
    "Tool",
    "tool",
    "load_tool",
    "launch_gradio_demo",
    "ToolCollection",
]
