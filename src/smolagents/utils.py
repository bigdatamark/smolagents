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
# Purpose: Grab-bag of shared helpers for smolagents: agent exception hierarchy, JSON/code-blob parsing of LLM
#   output, content truncation, source-code extraction and class-to-source serialization, image encoding helpers,
#   identifier validation, the Gradio app template used when saving agents, a rate limiter and a retry controller.
# Exports: __all__ only lists AgentError, but other modules import many names directly (e.g. BASE_BUILTIN_MODULES,
#   MAX_LENGTH_TRUNCATE_CONTENT, AGENT_GRADIO_APP_TEMPLATE, parse_code_blobs, truncate_content, get_source,
#   instance_to_source, RateLimiter, Retrying and the Agent*Error classes).
# Constants: BASE_BUILTIN_MODULES (stdlib modules always importable by the local Python executor),
#   MAX_LENGTH_TRUNCATE_CONTENT (20000), AGENT_GRADIO_APP_TEMPLATE (Jinja2 source of an app.py that rebuilds an
#   agent and launches GradioUI).
# Dependencies: stdlib plus jinja2; IPython is imported lazily in get_source; AgentLogger only for type checking.
# Side effects on import: none.
import ast
import base64
import importlib.util
import inspect
import json
import keyword
import os
import random
import re
import time
from functools import lru_cache
from io import BytesIO
from logging import Logger
from pathlib import Path
from textwrap import dedent
from typing import TYPE_CHECKING, Any, Callable

import jinja2


if TYPE_CHECKING:
    from smolagents.memory import AgentLogger


__all__ = ["AgentError"]


@lru_cache
def _is_package_available(package_name: str) -> bool:
    # [FUNCTION DOC]
    # HELPER: Reports whether a top-level package can be imported, without importing it. Results are cached forever
    #   per name (lru_cache).
    # Inputs: package_name (str).
    # Outputs: True if importlib can find a module spec for the name, else False.
    # Exceptions: ModuleNotFoundError / ValueError may propagate for dotted names whose parent is missing or invalid.
    # Side effects: populates the function cache.
    return importlib.util.find_spec(package_name) is not None


BASE_BUILTIN_MODULES = [
    "collections",
    "datetime",
    "itertools",
    "math",
    "queue",
    "random",
    "re",
    "stat",
    "statistics",
    "time",
    "unicodedata",
]


def sanitize_for_rich(value) -> str:
    # [FUNCTION DOC]
    # Converts any value to a string safe to print, replacing control characters with visible escapes.
    # Inputs: value (any). None becomes empty string; str is used as-is; bytes, bytearray and memoryview are decoded
    #   as UTF-8 with replacement characters; other objects use their str form.
    # Outputs: the string where newline, tab and carriage return are kept, and every other character with code below
    #   32 or equal to 127 is replaced by a backslash-x two-digit lowercase hex escape.
    # Exceptions: none raised directly (str of an object may raise).
    # Side effects: none.
    # KNOWN ISSUE: the docstring says bracket sequences that could be interpreted as Rich markup are escaped, but the
    #   code performs no bracket escaping at all.
    """
    Convert arbitrary values (including bytes / control characters) into a safe string for Rich.
    - Decodes bytes-like inputs using UTF-8 with replacement.
    - Escapes bracket sequences that could be interpreted as markup while preserving valid Rich tags.
    - Replaces ASCII control characters (except common whitespace) with visible escape sequences.
    """
    if value is None:
        s = ""
    elif isinstance(value, str):
        s = value
    elif isinstance(value, (bytes, bytearray, memoryview)):
        s = bytes(value).decode("utf-8", errors="replace")
    else:
        s = str(value)

    out: list[str] = []
    for ch in s:
        code = ord(ch)
        if ch in ("\n", "\t", "\r"):
            out.append(ch)
        elif code < 32 or code == 127:
            out.append(f"\\x{code:02x}")
        else:
            out.append(ch)
    return "".join(out)


class AgentError(Exception):
    # [CLASS DOC]
    # Base exception for agent-related errors; logs itself on creation.
    # Constructor args: message (any, usually str), logger (AgentLogger providing log_error).
    # Key attributes: message (the original message).
    """Base class for other agent-related exceptions"""

    def __init__(self, message, logger: "AgentLogger"):
        # [FUNCTION DOC]
        # Initializes the exception with the message, stores it and immediately logs it through the logger.
        # Inputs: message (any), logger (object with a log_error method).
        # Outputs: none.
        # Exceptions: AttributeError if the logger lacks log_error.
        # Side effects: emits an error log entry via the logger.
        super().__init__(message)
        self.message = message
        logger.log_error(message)

    def dict(self) -> dict[str, str]:
        # [FUNCTION DOC]
        # Serializes the error to a plain dict.
        # Inputs: none.
        # Outputs: dict with "type" (the concrete class name) and "message" (string form of the message).
        # Exceptions: none raised directly. Side effects: none.
        return {"type": self.__class__.__name__, "message": str(self.message)}


class AgentParsingError(AgentError):
    # [CLASS DOC]
    # Raised when the agent fails to parse model output (e.g. no code or JSON blob). Same constructor as AgentError.
    """Exception raised for errors in parsing in the agent"""

    pass


class AgentExecutionError(AgentError):
    # [CLASS DOC]
    # Raised when executing an agent action fails. Same constructor as AgentError. Parent of the tool error classes.
    """Exception raised for errors in execution in the agent"""

    pass


class AgentMaxStepsError(AgentError):
    # [CLASS DOC]
    # Raised when the agent reaches its maximum number of steps. Same constructor as AgentError.
    # KNOWN ISSUE: its docstring is copy-pasted from AgentExecutionError ("errors in execution") and does not describe
    #   the max-steps condition.
    """Exception raised for errors in execution in the agent"""

    pass


class AgentToolCallError(AgentExecutionError):
    # [CLASS DOC]
    # Raised when a tool is called with incorrect arguments. Subclass of AgentExecutionError; same constructor.
    """Exception raised for errors when incorrect arguments are passed to the tool"""

    pass


class AgentToolExecutionError(AgentExecutionError):
    # [CLASS DOC]
    # Raised when a tool fails during execution. Subclass of AgentExecutionError; same constructor.
    """Exception raised for errors when executing a tool"""

    pass


class AgentGenerationError(AgentError):
    # [CLASS DOC]
    # Raised when the model fails to generate output. Same constructor as AgentError.
    """Exception raised for errors in generation in the agent"""

    pass


def make_json_serializable(obj: Any) -> Any:
    # [FUNCTION DOC]
    # Recursively converts an arbitrary object into JSON-compatible data.
    # Inputs: obj (any).
    # Outputs: None stays None; int, float and bool are returned unchanged; a string that starts and ends with curly
    #   braces or square brackets is parsed as JSON and the result is converted recursively (falling back to the
    #   original string on decode errors), other strings unchanged; lists and tuples become lists of converted
    #   items; dicts get string keys and converted values; objects with a __dict__ become a dict with "_type" set to
    #   the class name plus their converted attributes; anything else becomes its str form.
    # Exceptions: RecursionError on cyclic structures. Side effects: none.
    """Recursive function to make objects JSON serializable"""
    if obj is None:
        return None
    elif isinstance(obj, (str, int, float, bool)):
        # Try to parse string as JSON if it looks like a JSON object/array
        if isinstance(obj, str):
            try:
                if (obj.startswith("{") and obj.endswith("}")) or (obj.startswith("[") and obj.endswith("]")):
                    parsed = json.loads(obj)
                    return make_json_serializable(parsed)
            except json.JSONDecodeError:
                pass
        return obj
    elif isinstance(obj, (list, tuple)):
        return [make_json_serializable(item) for item in obj]
    elif isinstance(obj, dict):
        return {str(k): make_json_serializable(v) for k, v in obj.items()}
    elif hasattr(obj, "__dict__"):
        # For custom objects, convert their __dict__ to a serializable format
        return {"_type": obj.__class__.__name__, **{k: make_json_serializable(v) for k, v in obj.__dict__.items()}}
    else:
        # For any other type, convert to string
        return str(obj)


def parse_json_blob(json_blob: str) -> tuple[dict[str, str], str]:
    # [FUNCTION DOC]
    # Extracts a JSON object embedded in text (typically an LLM tool call).
    # Inputs: json_blob (str).
    # Behavior: takes the substring from the first opening curly brace to the last closing curly brace and parses it
    #   with non-strict JSON decoding (control characters allowed inside strings).
    # Outputs: tuple of (parsed JSON data, text preceding the first opening brace).
    # Exceptions: ValueError if no closing brace exists; ValueError with a "multiple tool calls" hint when decoding
    #   fails right at a closing brace followed by comma and newline; otherwise ValueError describing the decode
    #   error and the surrounding characters.
    # Side effects: none.
    # KNOWN ISSUE: if there is no opening brace the find returns -1, so slicing starts at the last character and the
    #   prefix returned would drop it; that case usually fails to decode anyway. Text after the JSON is discarded.
    "Extracts the JSON blob from the input and returns the JSON data and the rest of the input."
    try:
        first_accolade_index = json_blob.find("{")
        last_accolade_index = [a.start() for a in list(re.finditer("}", json_blob))][-1]
        json_str = json_blob[first_accolade_index : last_accolade_index + 1]
        json_data = json.loads(json_str, strict=False)
        return json_data, json_blob[:first_accolade_index]
    except IndexError:
        raise ValueError("The model output does not contain any JSON blob.")
    except json.JSONDecodeError as e:
        place = e.pos
        if json_blob[place - 1 : place + 2] == "},\n":
            raise ValueError(
                "JSON is invalid: you probably tried to provide multiple tool calls in one action. PROVIDE ONLY ONE TOOL CALL."
            )
        raise ValueError(
            f"The JSON blob you used is invalid due to the following error: {e}.\n"
            f"JSON blob was: {json_blob}, decoding failed on that specific part of the blob:\n"
            f"'{json_blob[place - 4 : place + 5]}'."
        )


def extract_code_from_text(text: str, code_block_tags: tuple[str, str]) -> str | None:
    # [FUNCTION DOC]
    # HELPER: Finds every region between an opening and closing tag (treated as regex fragments, non-greedy, dot
    #   matches newline).
    # Inputs: text (str), code_block_tags (pair of regex strings: opening, closing).
    # Outputs: the stripped matches joined with blank lines, or None if there are no matches.
    # Exceptions: re.error if the tags are invalid regexes. Side effects: none.
    """Extract code from the LLM's output."""
    pattern = rf"{code_block_tags[0]}(.*?){code_block_tags[1]}"
    matches = re.findall(pattern, text, re.DOTALL)
    if matches:
        return "\n\n".join(match.strip() for match in matches)
    return None


def parse_code_blobs(text: str, code_block_tags: tuple[str, str]) -> str:
    # [FUNCTION DOC]
    # Extracts the code an LLM produced.
    # Inputs: text (str, model output), code_block_tags (pair of opening/closing regex strings).
    # Behavior: first tries the given tags; if nothing is found, falls back to markdown fences labelled python or py;
    #   if still nothing, returns the whole text if it parses as valid Python.
    # Outputs: the extracted code string.
    # Exceptions: ValueError with guidance when no code is found; the message suggests the final_answer pattern if
    #   the text contains both "final" and "answer".
    # Side effects: none.
    # KNOWN ISSUE: the docstring's Args section omits the code_block_tags parameter.
    """Extract code blocs from the LLM's output.

    If a valid code block is passed, it returns it directly.

    Args:
        text (`str`): LLM's output text to parse.

    Returns:
        `str`: Extracted code block.

    Raises:
        ValueError: If no valid code block is found in the text.
    """
    matches = extract_code_from_text(text, code_block_tags)
    if not matches:  # Fallback to markdown pattern
        matches = extract_code_from_text(text, ("```(?:python|py)", "\n```"))
    if matches:
        return matches
    # Maybe the LLM outputted a code blob directly
    try:
        ast.parse(text)
        return text
    except SyntaxError:
        pass

    if "final" in text and "answer" in text:
        raise ValueError(
            dedent(
                f"""
                Your code snippet is invalid, because the regex pattern {code_block_tags[0]}(.*?){code_block_tags[1]} was not found in it.
                Here is your code snippet:
                {text}
                It seems like you're trying to return the final answer, you can do it as follows:
                {code_block_tags[0]}
                final_answer("YOUR FINAL ANSWER HERE")
                {code_block_tags[1]}
                """
            ).strip()
        )
    raise ValueError(
        dedent(
            f"""
            Your code snippet is invalid, because the regex pattern {code_block_tags[0]}(.*?){code_block_tags[1]} was not found in it.
            Here is your code snippet:
            {text}
            Make sure to include code with the correct pattern, for instance:
            Thoughts: Your thoughts
            {code_block_tags[0]}
            # Your python code here
            {code_block_tags[1]}
            """
        ).strip()
    )


MAX_LENGTH_TRUNCATE_CONTENT = 20000


def truncate_content(content: str, max_length: int = MAX_LENGTH_TRUNCATE_CONTENT) -> str:
    # [FUNCTION DOC]
    # Shortens long text by keeping its head and tail.
    # Inputs: content (str), max_length (int, default MAX_LENGTH_TRUNCATE_CONTENT i.e. 20000).
    # Outputs: content unchanged if its length is at most max_length; otherwise the first half of max_length
    #   characters, a notice line saying content was truncated, and the last half.
    # Exceptions: none raised directly. Side effects: none.
    # KNOWN ISSUE: for odd max_length the tail slice uses negative floor division, so it keeps one more character
    #   than the head (total kept is max_length plus one).
    if len(content) <= max_length:
        return content
    else:
        return (
            content[: max_length // 2]
            + f"\n..._This content has been truncated to stay below {max_length} characters_...\n"
            + content[-max_length // 2 :]
        )


class ImportFinder(ast.NodeVisitor):
    # [CLASS DOC]
    # HELPER: AST visitor that collects the top-level package names imported by a piece of code.
    # Key attributes: packages (set of base package names, e.g. "os" for "os.path"). Relative imports without a
    #   module name are ignored.
    def __init__(self):
        # [FUNCTION DOC]
        # HELPER: Initializes an empty packages set.
        # Inputs: none. Outputs: none. Exceptions: none raised directly. Side effects: sets attribute.
        self.packages = set()

    def visit_Import(self, node):
        # [FUNCTION DOC]
        # HELPER: Adds the first dotted component of each imported name to packages.
        # Inputs: node (ast Import). Outputs: none. Exceptions: none raised directly. Side effects: mutates packages.
        # KNOWN ISSUE: does not call generic_visit, which is harmless for import nodes.
        for alias in node.names:
            # Get the base package name (before any dots)
            base_package = alias.name.split(".")[0]
            self.packages.add(base_package)

    def visit_ImportFrom(self, node):
        # [FUNCTION DOC]
        # HELPER: Adds the first dotted component of the source module of a from-import, if a module name is present.
        # Inputs: node (ast ImportFrom). Outputs: none. Exceptions: none raised directly. Side effects: mutates packages.
        if node.module:  # for "from x import y" statements
            # Get the base package name (before any dots)
            base_package = node.module.split(".")[0]
            self.packages.add(base_package)


def instance_to_source(instance, base_cls=None):
    # [FUNCTION DOC]
    # Generates standalone Python source for the class of an instance (used when saving tools).
    # Inputs: instance (any object); base_cls (class or None, default None) used as the declared base class and to
    #   filter out inherited members.
    # Behavior: emits a class header (with the base name if given); the class docstring if it differs from the
    #   base's; class attributes that are not dunder, not _abc_impl, not callable and not equal to the base's value
    #   (multiline strings in triple quotes, other strings via JSON quoting, everything else via repr); then the
    #   source of each callable defined on the class that is new or whose bytecode differs from the base
    #   (staticmethods/classmethods always), unwrapping decorated functions and re-indenting by four spaces.
    #   Imports used in the generated class are detected with ImportFinder.
    # Outputs: source string: optional "from <base module> import <base name>" line, one "import" line per detected
    #   package (set order), a blank line if any imports, then the class code.
    # Exceptions: errors from get_source (TypeError, OSError, ValueError); SyntaxError if the generated class does
    #   not parse; AttributeError when comparing bytecode of a base attribute that is not a function.
    # Side effects: none.
    """Convert an instance to its class source code representation."""
    cls = instance.__class__
    class_name = cls.__name__

    # Start building class lines
    class_lines = []
    if base_cls:
        class_lines.append(f"class {class_name}({base_cls.__name__}):")
    else:
        class_lines.append(f"class {class_name}:")

    # Add docstring if it exists and differs from base
    if cls.__doc__ and (not base_cls or cls.__doc__ != base_cls.__doc__):
        class_lines.append(f'    """{cls.__doc__}"""')

    # Add class-level attributes
    class_attrs = {
        name: value
        for name, value in cls.__dict__.items()
        if not name.startswith("__")
        and not name == "_abc_impl"
        and not callable(value)
        and not (base_cls and hasattr(base_cls, name) and getattr(base_cls, name) == value)
    }

    for name, value in class_attrs.items():
        if isinstance(value, str):
            # multiline value
            if "\n" in value:
                escaped_value = value.replace('"""', r"\"\"\"")  # Escape triple quotes
                class_lines.append(f'    {name} = """{escaped_value}"""')
            else:
                class_lines.append(f"    {name} = {json.dumps(value)}")
        else:
            class_lines.append(f"    {name} = {repr(value)}")

    if class_attrs:
        class_lines.append("")

    # Add methods
    methods = {
        name: func.__wrapped__ if hasattr(func, "__wrapped__") else func
        for name, func in cls.__dict__.items()
        if callable(func)
        and (
            not base_cls
            or not hasattr(base_cls, name)
            or (
                isinstance(func, (staticmethod, classmethod))
                or (getattr(base_cls, name).__code__.co_code != func.__code__.co_code)
            )
        )
    }

    for name, method in methods.items():
        method_source = get_source(method)
        # Clean up the indentation
        method_lines = method_source.split("\n")
        first_line = method_lines[0]
        indent = len(first_line) - len(first_line.lstrip())
        method_lines = [line[indent:] for line in method_lines]
        method_source = "\n".join(["    " + line if line.strip() else line for line in method_lines])
        class_lines.append(method_source)
        class_lines.append("")

    # Find required imports using ImportFinder
    import_finder = ImportFinder()
    import_finder.visit(ast.parse("\n".join(class_lines)))
    required_imports = import_finder.packages

    # Build final code with imports
    final_lines = []

    # Add base class import if needed
    if base_cls:
        final_lines.append(f"from {base_cls.__module__} import {base_cls.__name__}")

    # Add discovered imports
    for package in required_imports:
        final_lines.append(f"import {package}")

    if final_lines:  # Add empty line after imports
        final_lines.append("")

    # Add the class code
    final_lines.extend(class_lines)

    return "\n".join(final_lines)


def get_source(obj) -> str:
    # [FUNCTION DOC]
    # Returns the source code of a class or callable.
    # Inputs: obj (class or callable).
    # Behavior: uses the object's __source__ attribute if set, otherwise inspect.getsource; result is dedented and
    #   stripped. If inspect fails with OSError, falls back to the active IPython shell's input history ("In"),
    #   parsing all cells and returning the first class or function definition with the same name.
    # Outputs: the dedented, stripped source string.
    # Exceptions: TypeError if obj is neither class nor callable; the original OSError if IPython is unavailable or
    #   no shell is active; ValueError (chained from the OSError) if the history is empty or has no matching
    #   definition.
    # Side effects: reads the IPython shell history.
    # KNOWN ISSUE: the fallback only matches top-level-style function and class definitions (not async defs) and
    #   takes the first match, which may be an older redefinition; the docstring claims a ValueError is raised when
    #   there is no IPython shell, but an ImportError is raised internally and re-raised as the OSError.
    """Get the source code of a class or callable object (e.g.: function, method).
    First attempts to get the source code using `inspect.getsource`.
    In a dynamic environment (e.g.: Jupyter, IPython), if this fails,
    falls back to retrieving the source code from the current interactive shell session.

    Args:
        obj: A class or callable object (e.g.: function, method)

    Returns:
        str: The source code of the object, dedented and stripped

    Raises:
        TypeError: If object is not a class or callable
        OSError: If source code cannot be retrieved from any source
        ValueError: If source cannot be found in IPython history

    Note:
        TODO: handle Python standard REPL
    """
    if not (isinstance(obj, type) or callable(obj)):
        raise TypeError(f"Expected class or callable, got {type(obj)}")

    inspect_error = None
    try:
        # Handle dynamically created classes
        source = getattr(obj, "__source__", None) or inspect.getsource(obj)
        return dedent(source).strip()
    except OSError as e:
        # let's keep track of the exception to raise it if all further methods fail
        inspect_error = e
    try:
        import IPython

        shell = IPython.get_ipython()
        if not shell:
            raise ImportError("No active IPython shell found")
        all_cells = "\n".join(shell.user_ns.get("In", [])).strip()
        if not all_cells:
            raise ValueError("No code cells found in IPython session")

        tree = ast.parse(all_cells)
        for node in ast.walk(tree):
            if isinstance(node, (ast.ClassDef, ast.FunctionDef)) and node.name == obj.__name__:
                return dedent("\n".join(all_cells.split("\n")[node.lineno - 1 : node.end_lineno])).strip()
        raise ValueError(f"Could not find source code for {obj.__name__} in IPython history")
    except ImportError:
        # IPython is not available, let's just raise the original inspect error
        raise inspect_error
    except ValueError as e:
        # IPython is available but we couldn't find the source code, let's raise the error
        raise e from inspect_error


def encode_image_base64(image):
    # [FUNCTION DOC]
    # Encodes a PIL-like image as base64 PNG.
    # Inputs: image (object with a save method accepting a buffer and format).
    # Outputs: base64 string (UTF-8 decoded) of the PNG bytes.
    # Exceptions: propagates errors from image.save. Side effects: none.
    buffered = BytesIO()
    image.save(buffered, format="PNG")
    return base64.b64encode(buffered.getvalue()).decode("utf-8")


def make_image_url(base64_image):
    # [FUNCTION DOC]
    # Builds a PNG data URL.
    # Inputs: base64_image (str).
    # Outputs: "data:image/png;base64," followed by the input.
    # Exceptions: none raised directly. Side effects: none.
    return f"data:image/png;base64,{base64_image}"


def make_init_file(folder: str | Path):
    # [FUNCTION DOC]
    # Ensures a folder exists and creates an empty __init__.py in it.
    # Inputs: folder (str or Path).
    # Outputs: none.
    # Exceptions: OSError on filesystem failures.
    # Side effects: creates directories; creates or truncates the __init__.py file.
    os.makedirs(folder, exist_ok=True)
    # Create __init__
    with open(os.path.join(folder, "__init__.py"), "w"):
        pass


def is_valid_name(name: str) -> bool:
    # [FUNCTION DOC]
    # Checks whether a value is a valid Python identifier that is not a keyword.
    # Inputs: name (any).
    # Outputs: True for identifier strings that are not reserved keywords; False otherwise, including for non-strings.
    # Exceptions: none raised directly. Side effects: none.
    return name.isidentifier() and not keyword.iskeyword(name) if isinstance(name, str) else False


AGENT_GRADIO_APP_TEMPLATE = """import yaml
import os
from smolagents import GradioUI, {{ class_name }}, {{ agent_dict['model']['class'] }}

# Get current directory path
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))

{% for tool in tools.values() -%}
from {{managed_agent_relative_path}}tools.{{ tool.name }} import {{ tool.__class__.__name__ }} as {{ tool.name | camelcase }}
{% endfor %}
{% for managed_agent in managed_agents.values() -%}
from {{managed_agent_relative_path}}managed_agents.{{ managed_agent.name }}.app import agent_{{ managed_agent.name }}
{% endfor %}

model = {{ agent_dict['model']['class'] }}(
{% for key in agent_dict['model']['data'] if key != 'class' -%}
    {{ key }}={{ agent_dict['model']['data'][key]|repr }},
{% endfor %})

{% for tool in tools.values() -%}
{{ tool.name }} = {{ tool.name | camelcase }}()
{% endfor %}

with open(os.path.join(CURRENT_DIR, "prompts.yaml"), 'r') as stream:
    prompt_templates = yaml.safe_load(stream)

{{ agent_name }} = {{ class_name }}(
    model=model,
    tools=[{% for tool_name in tools.keys() if tool_name != "final_answer" %}{{ tool_name }}{% if not loop.last %}, {% endif %}{% endfor %}],
    managed_agents=[{% for subagent_name in managed_agents.keys() %}agent_{{ subagent_name }}{% if not loop.last %}, {% endif %}{% endfor %}],
    {% for attribute_name, value in agent_dict.items() if attribute_name not in ["class", "model", "tools", "prompt_templates", "authorized_imports", "managed_agents", "requirements"] -%}
    {{ attribute_name }}={{ value|repr }},
    {% endfor %}prompt_templates=prompt_templates
)
if __name__ == "__main__":
    GradioUI({{ agent_name }}).launch()
""".strip()


def create_agent_gradio_app_template():
    # [FUNCTION DOC]
    # Builds the Jinja2 template used to generate an agent's app.py.
    # Inputs: none.
    # Behavior: creates an environment with StrictUndefined (missing variables raise) and two filters: "repr" (Python
    #   repr) and "camelcase" (splits on underscores and capitalizes each word), then compiles
    #   AGENT_GRADIO_APP_TEMPLATE.
    # Outputs: a jinja2 Template expecting agent_name, class_name, agent_dict, tools, managed_agents and
    #   managed_agent_relative_path when rendered.
    # Exceptions: jinja2 TemplateSyntaxError if the template were malformed. Side effects: none.
    env = jinja2.Environment(loader=jinja2.BaseLoader(), undefined=jinja2.StrictUndefined)
    env.filters["repr"] = repr
    env.filters["camelcase"] = lambda value: "".join(word.capitalize() for word in value.split("_"))
    return env.from_string(AGENT_GRADIO_APP_TEMPLATE)


class RateLimiter:
    # [CLASS DOC]
    # Enforces a minimum delay between consecutive calls to throttle.
    # Constructor args: requests_per_minute (float or None, default None meaning disabled).
    # Key attributes: _enabled (bool), _interval (seconds between calls, 60 divided by the rate), _last_call (epoch
    #   seconds of the last throttled call, initially 0).
    """Simple rate limiter that enforces a minimum delay between consecutive requests.

    This class is useful for limiting the rate of operations such as API requests,
    by ensuring that calls to `throttle()` are spaced out by at least a given interval
    based on the desired requests per minute.

    If no rate is specified (i.e., `requests_per_minute` is None), rate limiting
    is disabled and `throttle()` becomes a no-op.

    Args:
        requests_per_minute (`float | None`): Maximum number of allowed requests per minute.
            Use `None` to disable rate limiting.
    """

    def __init__(self, requests_per_minute: float | None = None):
        # [FUNCTION DOC]
        # Computes whether limiting is enabled and the interval.
        # Inputs: requests_per_minute (float or None, default None).
        # Outputs: none.
        # Exceptions: ZeroDivisionError if requests_per_minute is 0.
        # Side effects: sets attributes.
        self._enabled = requests_per_minute is not None
        self._interval = 60.0 / requests_per_minute if self._enabled else 0.0
        self._last_call = 0.0

    def throttle(self):
        # [FUNCTION DOC]
        # Sleeps if needed so that at least the interval has passed since the previous call, then records the time.
        # Inputs: none. Outputs: none. Exceptions: none raised directly.
        # Side effects: may call time.sleep; updates _last_call. No-op when disabled.
        """Pause execution to respect the rate limit, if enabled."""
        if not self._enabled:
            return
        now = time.time()
        elapsed = now - self._last_call
        if elapsed < self._interval:
            time.sleep(self._interval - elapsed)
        self._last_call = time.time()


class Retrying:
    # [CLASS DOC]
    # Minimal retry controller inspired by tenacity: calls a function, retrying on exceptions accepted by a predicate
    #   with exponential backoff and optional jitter, and optional logging.
    # Constructor args: max_attempts (int, default 1), wait_seconds (float, default 0.0 initial delay base),
    #   exponential_base (float, default 2.0), jitter (bool, default True), retry_predicate (callable taking the
    #   exception and returning bool, or None meaning never retry), reraise (bool, default False),
    #   before_sleep_logger and after_logger (pair of logging.Logger and level, or None).
    # Key attributes: same names as the constructor args.
    """Simple retrying controller. Inspired from library [tenacity](https://github.com/jd/tenacity/)."""

    def __init__(
        self,
        max_attempts: int = 1,
        wait_seconds: float = 0.0,
        exponential_base: float = 2.0,
        jitter: bool = True,
        retry_predicate: Callable[[BaseException], bool] | None = None,
        reraise: bool = False,
        before_sleep_logger: tuple[Logger, int] | None = None,
        after_logger: tuple[Logger, int] | None = None,
    ):
        # [FUNCTION DOC]
        # Stores the retry configuration as attributes.
        # Inputs: see class doc. Outputs: none. Exceptions: none raised directly. Side effects: sets attributes.
        self.max_attempts = max_attempts
        self.wait_seconds = wait_seconds
        self.exponential_base = exponential_base
        self.jitter = jitter
        self.retry_predicate = retry_predicate
        self.reraise = reraise
        self.before_sleep_logger = before_sleep_logger
        self.after_logger = after_logger

    def __call__(self, fn, *args: Any, **kwargs: Any) -> Any:
        # [FUNCTION DOC]
        # Calls fn with the given arguments, retrying on failure.
        # Inputs: fn (callable), *args and **kwargs passed to fn.
        # Behavior: for attempt 1 to max_attempts, call fn; on success return its result (logging a "Finished call"
        #   message via after_logger if this was not the first attempt). On any BaseException, if the predicate is
        #   absent or rejects it, or this was the last attempt, re-raise it. Otherwise log via after_logger, multiply
        #   the delay by exponential_base times one plus (jitter times a random number in [0, 1)), log via
        #   before_sleep_logger, and sleep if the delay is positive.
        # Outputs: the return value of fn; None if max_attempts is less than 1 (fn is never called).
        # Exceptions: re-raises the last exception from fn.
        # Side effects: calls fn repeatedly; sleeps; writes log records.
        # KNOWN ISSUE: the reraise flag has no effect, since both branches simply re-raise; the delay is multiplied before
        #   the first sleep, so the first wait is wait_seconds times the base rather than wait_seconds; with the default
        #   wait_seconds of 0 no sleeping ever happens; BaseException catching includes KeyboardInterrupt and SystemExit.
        start_time = time.time()
        delay = self.wait_seconds

        for attempt_number in range(1, self.max_attempts + 1):
            try:
                result = fn(*args, **kwargs)

                # Log after successful call if we had retries
                if self.after_logger and attempt_number > 1:
                    logger, log_level = self.after_logger
                    seconds = time.time() - start_time
                    fn_name = getattr(fn, "__name__", repr(fn))
                    logger.log(
                        log_level,
                        f"Finished call to '{fn_name}' after {seconds:.3f}(s), this was attempt n°{attempt_number}/{self.max_attempts}.",
                    )

                return result

            except BaseException as e:
                # Check if we should retry
                should_retry = self.retry_predicate(e) if self.retry_predicate else False

                # If this is the last attempt or we shouldn't retry, raise
                if not should_retry or attempt_number >= self.max_attempts:
                    if self.reraise:
                        raise
                    raise

                # Log after failed attempt
                if self.after_logger:
                    logger, log_level = self.after_logger
                    seconds = time.time() - start_time
                    fn_name = getattr(fn, "__name__", repr(fn))
                    logger.log(
                        log_level,
                        f"Finished call to '{fn_name}' after {seconds:.3f}(s), this was attempt n°{attempt_number}/{self.max_attempts}.",
                    )

                # Exponential backoff with jitter
                # https://cookbook.openai.com/examples/how_to_handle_rate_limits#example-3-manual-backoff-implementation
                delay *= self.exponential_base * (1 + self.jitter * random.random())

                # Log before sleeping
                if self.before_sleep_logger:
                    logger, log_level = self.before_sleep_logger
                    fn_name = getattr(fn, "__name__", repr(fn))
                    logger.log(
                        log_level,
                        f"Retrying {fn_name} in {delay} seconds as it raised {e.__class__.__name__}: {e}.",
                    )

                # Sleep before next attempt
                if delay > 0:
                    time.sleep(delay)
