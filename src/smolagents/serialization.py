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
# What: safe (JSON-based) serialization layer used to pass values between the local agent process and remote
#   executors / sandboxes. Values are converted to a JSON-compatible tree in which non-JSON types are encoded
#   as dicts carrying a '__type__' marker; the JSON text is prefixed with 'safe:'. An opt-in insecure fallback
#   encodes values with pickle + base64 under a 'pickle:' prefix. Legacy unprefixed strings are treated as
#   base64 pickle. The module can also emit standalone Python source for a deserializer to be injected into a
#   remote interpreter.
# Inputs: depends on stdlib base64, json, pickle, io.BytesIO, typing.Any; lazily imports datetime, decimal,
#   pathlib, dataclasses, inspect, textwrap, warnings, and optionally numpy and PIL (when present).
# Outputs: exports SerializationError and SafeSerializer (listed in the module export list).
# Exceptions: none raised at import time.
# Side effects: none at import time (only class definitions).

"""
Safe serialization module for remote executor communication.

Provides JSON-based serialization with optional pickle fallback for types
that cannot be safely serialized.

**Security Note:** Pickle deserialization can execute arbitrary code. This module
defaults to safe JSON-only serialization. Only enable pickle fallback
(allow_insecure_serializer=True) if you fully trust the execution environment.
"""

import base64
import json
import pickle
from io import BytesIO
from typing import Any


__all__ = ["SerializationError", "SafeSerializer"]


class SerializationError(Exception):
    # [CLASS DOC]
    # What: exception type signalling that a value cannot be serialized (safe mode refusal, or pickle failure)
    #   or that pickle data was rejected during deserialization.
    # Inputs: standard Exception constructor arguments (typically a message string).
    # Outputs: n/a (exception class, no extra attributes).
    # Exceptions: none raised directly.
    # Side effects: none.
    """Raised when a type cannot be safely serialized."""

    pass


class SafeSerializer:
    # [CLASS DOC]
    # What: namespace class (all members static/class methods, never instantiated in practice) implementing
    #   safe serialization. Encoding: primitives (exact str/int/float/bool/None) pass through; list -> JSON
    #   list; dict with only exact-str keys -> JSON object; other dicts -> marker 'dict_with_complex_keys'
    #   with a list of [key, value] pairs; tuple/set/frozenset -> marker with 'data' list; bytes -> base64
    #   text; complex -> real/imag; datetime/date/time -> ISO text; timedelta -> total_seconds; Decimal ->
    #   string; pathlib paths -> string under marker 'Path'; PIL images -> PNG bytes in base64; numpy ndarray
    #   -> nested list plus dtype string; numpy integer/floating scalars -> plain Python number; dataclass
    #   instances -> class name, module and recursively encoded fields. Anything else is refused.
    # Inputs: no constructor arguments. Class attributes: SAFE_PREFIX (the text 'safe:'), and
    #   _optional_types_cache (class-level dict shared by all callers mapping 'module.attr' to the imported
    #   class or None when unavailable).
    # Outputs: n/a (class).
    # Exceptions: methods raise SerializationError as described on each method.
    # Side effects: _optional_types_cache is mutated lazily and shared process-wide.
    # KNOWN ISSUE: user dicts whose own keys include '__type__' with a recognised marker value (e.g. 'bytes')
    #   are indistinguishable from encoded values and are decoded as that type on the way back (no escaping).
    """JSON-based serializer with type markers for safe serialization.

    Supports:
    - Basic: str, int, float, bool, None, list, dict
    - Extended: tuple, set, frozenset, bytes, complex, datetime/date/time/timedelta
    - Optional: numpy.ndarray, PIL.Image, dataclasses, Decimal, Path

    The serializer uses a prefix system to distinguish between formats:
    - "safe:" prefix for JSON-serialized data
    - "pickle:" prefix for pickle-serialized data (when allowed)
    """

    SAFE_PREFIX = "safe:"

    # Cache for optional type classes (avoids repeated import attempts)
    _optional_types_cache: dict = {}

    @classmethod
    def _get_optional_type(cls, module: str, attr: str):
        # [FUNCTION DOC]
        # HELPER: lazily imports an optional class (e.g. PIL Image or numpy ndarray) and memoizes the result.
        # Inputs: module (str, dotted module name), attr (str, attribute name inside that module).
        # Outputs: the attribute (usually a class) or None if the module cannot be imported or lacks the
        #   attribute.
        # Exceptions: none raised directly (ImportError/AttributeError are swallowed and cached as None).
        # Side effects: imports the module on first lookup; writes the result into the class-level cache keyed
        #   by 'module.attr'. Not lock-protected, but races only cause duplicate imports.
        """Get optional type class with caching to avoid repeated imports."""
        key = f"{module}.{attr}"
        if key not in cls._optional_types_cache:
            try:
                mod = __import__(module, fromlist=[attr])
                cls._optional_types_cache[key] = getattr(mod, attr)
            except (ImportError, AttributeError):
                cls._optional_types_cache[key] = None
        return cls._optional_types_cache[key]

    @staticmethod
    def to_json_safe(obj: Any) -> Any:
        # [FUNCTION DOC]
        # What: recursively converts a Python value to a JSON-serializable tree with '__type__' markers (see
        #   the class doc for the per-type encodings). Type checks are exact type identity for builtins (so
        #   subclasses of str/int/list/dict etc. are not accepted by those branches);
        #   datetime/Decimal/pathlib/numpy are recognised by the type's module and name without importing; PIL
        #   is checked via isinstance on the cached class; dataclass instances (not dataclass types) are
        #   handled last.
        # Inputs: obj (any value).
        # Outputs: a JSON-compatible value: primitive, list, or dict (plain dict for str-keyed dicts, marker
        #   dict otherwise). Floats such as inf/nan are passed through (json module later writes them as
        #   non-standard tokens).
        # Exceptions: SerializationError ('Cannot safely serialize object of type <name>') for unsupported
        #   types anywhere in the tree; errors from PIL save propagate; RecursionError on self-referencing
        #   structures.
        # Side effects: may import PIL / numpy through the optional-type cache; imports dataclasses.
        # KNOWN ISSUE: numpy bool scalars, complex scalars and other numpy scalar kinds are not handled and
        #   raise; complex-dtype ndarrays produce complex numbers in the list output which json.dumps cannot
        #   encode.
        # KNOWN ISSUE: a set/frozenset whose elements encode to dicts (e.g. tuples) round-trips fine, but sets
        #   containing unhashable decoded values (dataclasses become dicts) would fail on decode.
        # KNOWN ISSUE: any pathlib type whose name contains 'Path' (including PurePosixPath/PureWindowsPath)
        #   is encoded as 'Path' and decodes to a concrete Path, so the original class is not preserved.
        """Convert Python objects to JSON-serializable format with type markers.

        Args:
            obj: Object to convert.

        Returns:
            JSON-serializable representation.

        Raises:
            SerializationError: If the object cannot be safely serialized.
        """
        # Fast path: use exact type check for primitives (most common case)
        obj_type = type(obj)
        if obj_type is str or obj_type is int or obj_type is float or obj_type is bool or obj is None:
            return obj

        # Fast path: list (very common for return values)
        if obj_type is list:
            return [SafeSerializer.to_json_safe(item) for item in obj]

        # Fast path: tuple (common for multiple return values)
        if obj_type is tuple:
            return {"__type__": "tuple", "data": [SafeSerializer.to_json_safe(item) for item in obj]}

        # Fast path: dict (common, check string keys)
        if obj_type is dict:
            if all(type(k) is str for k in obj):
                return {k: SafeSerializer.to_json_safe(v) for k, v in obj.items()}
            return {
                "__type__": "dict_with_complex_keys",
                "data": [[SafeSerializer.to_json_safe(k), SafeSerializer.to_json_safe(v)] for k, v in obj.items()],
            }

        # Other builtin types - exact type checks
        if obj_type is set:
            return {"__type__": "set", "data": [SafeSerializer.to_json_safe(item) for item in obj]}
        if obj_type is frozenset:
            return {"__type__": "frozenset", "data": [SafeSerializer.to_json_safe(item) for item in obj]}
        if obj_type is bytes:
            return {"__type__": "bytes", "data": base64.b64encode(obj).decode()}
        if obj_type is complex:
            return {"__type__": "complex", "real": obj.real, "imag": obj.imag}

        # Use type module/name for lazy-loaded types (avoids import until needed)
        type_module = getattr(obj_type, "__module__", "")
        type_name = obj_type.__name__

        # datetime module types (check module first to skip unrelated types quickly)
        if type_module == "datetime":
            if type_name == "datetime":
                return {"__type__": "datetime", "data": obj.isoformat()}
            if type_name == "date":
                return {"__type__": "date", "data": obj.isoformat()}
            if type_name == "time":
                return {"__type__": "time", "data": obj.isoformat()}
            if type_name == "timedelta":
                return {"__type__": "timedelta", "total_seconds": obj.total_seconds()}

        # decimal.Decimal
        if type_module == "decimal" and type_name == "Decimal":
            return {"__type__": "Decimal", "data": str(obj)}

        # pathlib.Path (and subclasses like PosixPath, WindowsPath)
        if type_module.startswith("pathlib") and "Path" in type_name:
            return {"__type__": "Path", "data": str(obj)}

        # PIL.Image - use cached import
        pil_image_cls = SafeSerializer._get_optional_type("PIL.Image", "Image")
        if pil_image_cls is not None and isinstance(obj, pil_image_cls):
            buffer = BytesIO()
            obj.save(buffer, format="PNG")
            return {"__type__": "PIL.Image", "data": base64.b64encode(buffer.getvalue()).decode()}

        # numpy types - use cached import
        if type_module == "numpy" or type_module.startswith("numpy."):
            np_ndarray = SafeSerializer._get_optional_type("numpy", "ndarray")
            if np_ndarray is not None and obj_type is np_ndarray:
                return {"__type__": "ndarray", "data": obj.tolist(), "dtype": str(obj.dtype)}
            np_integer = SafeSerializer._get_optional_type("numpy", "integer")
            np_floating = SafeSerializer._get_optional_type("numpy", "floating")
            if (np_integer and isinstance(obj, np_integer)) or (np_floating and isinstance(obj, np_floating)):
                return obj.item()

        # dataclass - check last as is_dataclass() has overhead
        import dataclasses

        if dataclasses.is_dataclass(obj) and not isinstance(obj, type):
            return {
                "__type__": "dataclass",
                "class_name": type_name,
                "module": type_module,
                "data": {f.name: SafeSerializer.to_json_safe(getattr(obj, f.name)) for f in dataclasses.fields(obj)},
            }

        raise SerializationError(f"Cannot safely serialize object of type {type_name}")

    @staticmethod
    def from_json_safe(obj: Any) -> Any:
        # [FUNCTION DOC]
        # What: inverse of to_json_safe. For a dict carrying a recognised '__type__' marker it rebuilds bytes,
        #   PIL image (opened from PNG bytes), set, tuple, complex, frozenset, dict with complex keys,
        #   datetime, date, time, timedelta, Decimal, Path, numpy ndarray (with dtype), or a dataclass
        #   stand-in; dicts without a recognised marker are decoded value-by-value as plain dicts (so unknown
        #   markers are kept as data); lists are decoded element-wise; anything else is returned unchanged.
        # Inputs: obj (JSON-decoded value).
        # Outputs: the reconstructed Python value. Dataclasses come back as a plain dict with keys
        #   '__dataclass__' (class name), '__module__' and the decoded fields. If PIL is not installed an
        #   image is returned as its marker dict; if numpy is not installed an ndarray is returned as a nested
        #   list.
        # Exceptions: KeyError for marker dicts missing required fields; binascii.Error / ValueError for bad
        #   base64, ISO strings or Decimal text; PIL errors for invalid image bytes; TypeError when set
        #   elements decode to unhashable values.
        # Side effects: lazily imports datetime, decimal, pathlib, PIL, numpy as needed.
        # Note: its source text is reused verbatim (via inspect) by get_safe_serializer_code and
        #   get_deserializer_code, so it must remain self-contained.
        """
        Convert JSON-safe format back to Python objects.

        Args:
            obj: JSON-safe representation

        Returns:
            Original Python object
        """
        if isinstance(obj, dict):
            if "__type__" in obj:
                obj_type = obj["__type__"]
                if obj_type == "bytes":
                    return base64.b64decode(obj["data"])
                elif obj_type == "PIL.Image":
                    try:
                        import PIL.Image

                        img_bytes = base64.b64decode(obj["data"])
                        return PIL.Image.open(BytesIO(img_bytes))
                    except ImportError:
                        return {"__type__": "PIL.Image", "data": obj["data"]}
                elif obj_type == "set":
                    return set(SafeSerializer.from_json_safe(item) for item in obj["data"])
                elif obj_type == "tuple":
                    return tuple(SafeSerializer.from_json_safe(item) for item in obj["data"])
                elif obj_type == "complex":
                    return complex(obj["real"], obj["imag"])
                elif obj_type == "frozenset":
                    return frozenset(SafeSerializer.from_json_safe(item) for item in obj["data"])
                elif obj_type == "dict_with_complex_keys":
                    return {SafeSerializer.from_json_safe(k): SafeSerializer.from_json_safe(v) for k, v in obj["data"]}
                elif obj_type == "datetime":
                    from datetime import datetime

                    return datetime.fromisoformat(obj["data"])
                elif obj_type == "date":
                    from datetime import date

                    return date.fromisoformat(obj["data"])
                elif obj_type == "time":
                    from datetime import time

                    return time.fromisoformat(obj["data"])
                elif obj_type == "timedelta":
                    from datetime import timedelta

                    return timedelta(seconds=obj["total_seconds"])
                elif obj_type == "Decimal":
                    from decimal import Decimal

                    return Decimal(obj["data"])
                elif obj_type == "Path":
                    from pathlib import Path

                    return Path(obj["data"])
                elif obj_type == "ndarray":
                    try:
                        import numpy as np

                        return np.array(obj["data"], dtype=obj["dtype"])
                    except ImportError:
                        return obj["data"]  # Return as list if numpy not available
                elif obj_type == "dataclass":
                    # For dataclasses, we return a dict representation
                    # since we can't reconstruct the actual class without access to it
                    return {
                        "__dataclass__": obj["class_name"],
                        "__module__": obj["module"],
                        **{k: SafeSerializer.from_json_safe(v) for k, v in obj["data"].items()},
                    }
            return {k: SafeSerializer.from_json_safe(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [SafeSerializer.from_json_safe(item) for item in obj]
        return obj

    @staticmethod
    def dumps(obj: Any, allow_pickle: bool = False) -> str:
        # [FUNCTION DOC]
        # What: serializes a value to a prefixed string. With pickle disallowed it always uses the safe JSON
        #   encoding. With pickle allowed it tries the safe encoding first and, only if that raises
        #   SerializationError, emits a FutureWarning and falls back to base64-encoded pickle.
        # Inputs: obj (any value); allow_pickle (bool, default False).
        # Outputs: str, either 'safe:' followed by JSON text, or 'pickle:' followed by base64 pickle bytes.
        # Exceptions: SerializationError when safe encoding fails and pickle is disallowed, or when pickling
        #   fails (PicklingError/TypeError/AttributeError are wrapped). json.dumps errors (e.g. TypeError for
        #   complex values inside ndarray lists) propagate unwrapped and do not trigger the pickle fallback.
        # Side effects: emits a FutureWarning (stacklevel 2) on pickle fallback.
        """
        Serialize object to string.

        Args:
            obj: Object to serialize
            allow_pickle: If False (default), use ONLY safe JSON serialization (error if fails).
                         If True, try safe first, fallback to pickle with warning.

        Returns:
            str: Serialized string ("safe:..." for JSON, "pickle:..." for pickle)

        Raises:
            SerializationError: If allow_pickle=False and object cannot be safely serialized
        """
        if not allow_pickle:
            # Safe ONLY mode - no pickle fallback
            json_safe = SafeSerializer.to_json_safe(obj)  # Raises SerializationError if fails
            return SafeSerializer.SAFE_PREFIX + json.dumps(json_safe)
        else:
            # Try safe first, fallback to pickle
            try:
                json_safe = SafeSerializer.to_json_safe(obj)
                return SafeSerializer.SAFE_PREFIX + json.dumps(json_safe)
            except SerializationError:
                # Warn about insecure pickle usage
                import warnings

                warnings.warn(
                    "Falling back to insecure pickle serialization. "
                    "This is a security risk and will be removed in a future version. "
                    "Consider using only safe serializable types (primitives, lists, dicts, "
                    "numpy arrays, PIL images, datetime objects, dataclasses).",
                    FutureWarning,
                    stacklevel=2,
                )
                # Fallback to pickle (with prefix)
                try:
                    return "pickle:" + base64.b64encode(pickle.dumps(obj)).decode()
                except (pickle.PicklingError, TypeError, AttributeError) as e:
                    raise SerializationError(f"Cannot serialize object: {e}") from e

    @staticmethod
    def loads(data: str, allow_pickle: bool = False) -> Any:
        # [FUNCTION DOC]
        # What: deserializes a string produced by dumps, dispatching on the prefix: 'safe:' -> JSON parse then
        #   from_json_safe; 'pickle:' -> base64-decode then unpickle (only if allowed); no recognised prefix
        #   -> treated as legacy base64 pickle (only if allowed).
        # Inputs: data (str); allow_pickle (bool, default False).
        # Outputs: the deserialized Python value.
        # Exceptions: SerializationError ('Pickle data rejected ...') for pickle or legacy data when pickle is
        #   disallowed; json.JSONDecodeError for malformed safe payloads; base64/pickle errors propagate;
        #   AttributeError if data is not a string.
        # Side effects: emits a FutureWarning (stacklevel 2) whenever pickle data is deserialized; unpickling
        #   can execute arbitrary code from the payload.
        """
        Deserialize string with format detection.

        Args:
            data: Serialized string (with "safe:" or "pickle:" prefix)
            allow_pickle: If False (default), reject pickle data (strict safe mode).
                         If True, accept both safe and pickle formats.

        Returns:
            Deserialized object

        Raises:
            SerializationError: If pickle data received but allow_pickle=False
        """
        if data.startswith(SafeSerializer.SAFE_PREFIX):
            json_data = json.loads(data[len(SafeSerializer.SAFE_PREFIX) :])
            return SafeSerializer.from_json_safe(json_data)
        elif data.startswith("pickle:"):
            # Explicit pickle prefix
            if not allow_pickle:
                raise SerializationError(
                    "Pickle data rejected: allow_pickle=False requires safe-only data. "
                    "This data is pickle-serialized. To deserialize it, set "
                    "allow_pickle=True (not recommended for untrusted data)."
                )
            # Warn about insecure pickle deserialization
            import warnings

            warnings.warn(
                "Deserializing pickle data. This is a security risk if the data is untrusted.",
                FutureWarning,
                stacklevel=2,
            )
            return pickle.loads(base64.b64decode(data[7:]))
        else:
            # No prefix - legacy format, assume pickle
            if not allow_pickle:
                raise SerializationError(
                    "Pickle data rejected: allow_pickle=False requires safe-only data. "
                    "This data appears to be pickle-serialized (legacy format). To deserialize it, set "
                    "allow_pickle=True (not recommended for untrusted data)."
                )
            # Warn about insecure pickle deserialization
            import warnings

            warnings.warn(
                "Deserializing pickle data. This is a security risk if the data is untrusted.",
                FutureWarning,
                stacklevel=2,
            )
            return pickle.loads(base64.b64decode(data))

    @staticmethod
    def _extract_method_body(method) -> str:
        # [FUNCTION DOC]
        # HELPER: returns the source of a function's body (without the def line and docstring), dedented.
        #   Scans lines after the first: if a line contains a triple double-quote, the body starts after the
        #   line closing that docstring; otherwise the first non-blank line that does not start with 'def ' or
        #   '@' is taken as the body start.
        # Inputs: method (function object whose source is retrievable).
        # Outputs: str, the dedented body source.
        # Exceptions: OSError/TypeError from inspect.getsource if source is unavailable.
        # Side effects: reads source files via inspect.
        # KNOWN ISSUE: unused anywhere in the code base. The heuristic is fragile: comment lines, multi-line
        #   signatures or single-quoted docstrings are treated as the start of the body, and triple quotes
        #   inside code are mistaken for a docstring. If no body line is found the whole source (including
        #   def) is returned.
        """Extract method body without the def line and dedent it."""
        import inspect
        import textwrap

        source = inspect.getsource(method)
        lines = source.split("\n")
        # Skip the def line and docstring
        body_start = 0
        for i, line in enumerate(lines):
            if '"""' in line and i > 0:
                # Find end of docstring
                if line.count('"""') == 2:
                    body_start = i + 1
                    break
                for j in range(i + 1, len(lines)):
                    if '"""' in lines[j]:
                        body_start = j + 1
                        break
                break
            elif line.strip() and not line.strip().startswith("def ") and not line.strip().startswith("@"):
                body_start = i
                break

        body = "\n".join(lines[body_start:])
        return textwrap.dedent(body)

    @staticmethod
    def get_safe_serializer_code() -> str:
        # [FUNCTION DOC]
        # What: intended to return standalone Python source defining SerializationError and a sandbox
        #   SafeSerializer class, built by splicing the source of to_json_safe and from_json_safe (with the
        #   staticmethod decorator and class-qualified recursive calls stripped) into a template that also
        #   defines simplified dumps/loads (no warnings, short rejection messages).
        # Inputs: none.
        # Outputs: str of Python source.
        # Exceptions: OSError/TypeError from inspect.getsource if source is unavailable.
        # Side effects: reads source via inspect.
        # KNOWN ISSUE: the generated code does not compile. The spliced method sources keep their original
        #   4-space indentation and are inserted after another 4 spaces, so each def line ends up at the same
        #   indentation as its body (IndentationError when executed). Even if re-indented, the methods lose
        #   the staticmethod decorator and call to_json_safe / from_json_safe as bare names, which are not
        #   defined at module level. The method is not used anywhere in the code base (remote executors use
        #   get_deserializer_code).
        """
        Returns the SafeSerializer class definition as string for injection into sandbox.

        This generates a standalone version from the actual implementation to avoid duplication.
        """
        import inspect

        # Generate to_json_safe from actual implementation
        to_json_safe_source = inspect.getsource(SafeSerializer.to_json_safe)
        # Make it standalone (remove @staticmethod, change self references)
        to_json_safe_source = to_json_safe_source.replace("@staticmethod\n    ", "")
        to_json_safe_source = to_json_safe_source.replace("SafeSerializer.to_json_safe", "to_json_safe")

        # Generate from_json_safe from actual implementation
        from_json_safe_source = inspect.getsource(SafeSerializer.from_json_safe)
        from_json_safe_source = from_json_safe_source.replace("@staticmethod\n    ", "")
        from_json_safe_source = from_json_safe_source.replace("SafeSerializer.from_json_safe", "from_json_safe")

        return f'''
class SerializationError(Exception):
    """Raised when a type cannot be safely serialized."""
    pass

class SafeSerializer:
    """Safe JSON-based serializer for sandbox use."""

    SAFE_PREFIX = "safe:"

    {to_json_safe_source}

    {from_json_safe_source}

    @staticmethod
    def dumps(obj, allow_pickle=False):
        import json
        import base64
        import pickle

        if not allow_pickle:
            # Safe ONLY - no pickle fallback
            json_safe = to_json_safe(obj)  # Raises SerializationError if fails
            return SafeSerializer.SAFE_PREFIX + json.dumps(json_safe)
        else:
            # Try safe first, fallback to pickle if allowed
            try:
                json_safe = to_json_safe(obj)
                return SafeSerializer.SAFE_PREFIX + json.dumps(json_safe)
            except SerializationError:
                try:
                    return "pickle:" + base64.b64encode(pickle.dumps(obj)).decode()
                except (pickle.PicklingError, TypeError, AttributeError) as e:
                    raise SerializationError(f"Cannot serialize object: {{e}}") from e

    @staticmethod
    def loads(data, allow_pickle=False):
        import json
        import base64
        import pickle

        if data.startswith(SafeSerializer.SAFE_PREFIX):
            json_data = json.loads(data[len(SafeSerializer.SAFE_PREFIX):])
            return from_json_safe(json_data)
        elif data.startswith("pickle:"):
            if not allow_pickle:
                raise SerializationError("Pickle data rejected: allow_pickle=False")
            return pickle.loads(base64.b64decode(data[7:]))
        else:
            # Legacy format (no prefix) - assume pickle
            if not allow_pickle:
                raise SerializationError("Pickle data rejected: allow_pickle=False")
            return pickle.loads(base64.b64decode(data))
'''

    @staticmethod
    def get_deserializer_code(allow_pickle: bool) -> str:
        # [FUNCTION DOC]
        # What: builds standalone Python source for a remote interpreter defining SerializationError, a
        #   module-level _from_json_safe function (the dedented source of from_json_safe with the decorator
        #   removed, renamed, and its recursive calls rewritten), and a _deserialize function. _deserialize
        #   decodes 'safe:' strings via JSON plus _from_json_safe; for 'pickle:' strings and for any other
        #   value (legacy) it either unpickles base64 data or raises SerializationError, depending on the flag
        #   baked in at generation time.
        # Inputs: allow_pickle (bool) - whether the generated code accepts pickle payloads.
        # Outputs: str of Python source (lines joined by newlines, ending with a blank line).
        # Exceptions: OSError/TypeError from inspect.getsource if source is unavailable.
        # Side effects: reads source via inspect. Used by remote_executors to inject the deserializer.
        # KNOWN ISSUE: the generated _deserialize emits no warning when unpickling, unlike loads, and the
        #   legacy branch also receives non-string values, which then fail inside base64 decoding.
        """
        Generate deserializer function for remote execution with setting baked in.

        This generates code from the actual implementation to avoid duplication.

        Args:
            allow_pickle: Whether to allow pickle deserialization

        Returns:
            Python code string with _deserialize function
        """
        import inspect
        import textwrap

        # Build a standalone _from_json_safe function from the source of from_json_safe.
        from_json_safe_source = inspect.getsource(SafeSerializer.from_json_safe)
        from_json_safe_source = textwrap.dedent(from_json_safe_source)
        if from_json_safe_source.startswith("@staticmethod\n"):
            from_json_safe_source = from_json_safe_source[len("@staticmethod\n") :]
        from_json_safe_source = from_json_safe_source.replace("def from_json_safe(", "def _from_json_safe(")
        from_json_safe_source = from_json_safe_source.replace("SafeSerializer.from_json_safe", "_from_json_safe")

        if allow_pickle:
            prefixed_pickle_branch = [
                "        import pickle",
                "        return pickle.loads(base64.b64decode(data[7:]))",
            ]
            legacy_pickle_branch = [
                "        import pickle",
                "        return pickle.loads(base64.b64decode(data))",
            ]
        else:
            prefixed_pickle_branch = [
                '        raise SerializationError("Pickle data rejected: allow_pickle=False")',
            ]
            legacy_pickle_branch = [
                '        raise SerializationError("Pickle data rejected: allow_pickle=False")',
            ]

        lines = [
            "import base64",
            "from io import BytesIO",
            "from typing import Any",
            "",
            "class SerializationError(Exception):",
            "    pass",
            "",
            from_json_safe_source.rstrip(),
            "",
            "def _deserialize(data):",
            "    import json",
            '    if isinstance(data, str) and data.startswith("safe:"):',
            "        json_data = json.loads(data[5:])",
            "        return _from_json_safe(json_data)",
            '    elif isinstance(data, str) and data.startswith("pickle:"):',
            *prefixed_pickle_branch,
            "    else:",
            "        # No safe prefix - legacy format, assume pickle",
            *legacy_pickle_branch,
            "",
        ]
        return "\n".join(lines)
