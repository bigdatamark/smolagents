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
# What: Model abstraction layer for smolagents. Defines chat message data structures (ChatMessage, tool-call and
#   streaming-delta dataclasses, MessageRole), helpers to normalize message lists / tool schemas / stop sequences,
#   the abstract Model base class, and concrete backends: VLLMModel, MLXModel, TransformersModel (local inference),
#   and ApiModel subclasses LiteLLMModel, LiteLLMRouterModel, InferenceClientModel, OpenAIModel (alias
#   OpenAIServerModel), AzureOpenAIModel (alias AzureOpenAIServerModel), AmazonBedrockModel (alias
#   AmazonBedrockServerModel).
# Inputs/deps: stdlib (json, logging, os, re, uuid, warnings, threading, dataclasses, enum, copy); smolagents
#   .monitoring.TokenUsage, .tools.Tool, .utils (RateLimiter, Retrying, package detection, image encoding, JSON blob
#   parsing). Optional third-party backends (vllm, mlx_lm, torch/transformers, litellm, huggingface_hub, openai,
#   boto3) are imported lazily inside constructors/methods.
# Outputs/exports: __all__ lists REMOVE_PARAMETER, MessageRole, tool_role_conversions, get_clean_message_list, the
#   model classes and aliases, and ChatMessage. Module constants: retry settings (RETRY_WAIT=60s,
#   RETRY_MAX_ATTEMPTS=3, RETRY_EXPONENTIAL_BASE=2, RETRY_JITTER=True), STRUCTURED_GENERATION_PROVIDERS (providers
#   allowed to do structured output on InferenceClientModel), CODEAGENT_RESPONSE_FORMAT (JSON-schema response format
#   requiring 'thought' and 'code' strings), tool_role_conversions (tool-call -> assistant, tool-response -> user),
#   and MODEL_REGISTRY (class-name to class map, the allow-list for safe deserialization).
# Exceptions: none raised at import time (optional backends are only imported on use).
# Side effects: creates a module-level logger; no I/O at import.
import json
import logging
import os
import re
import uuid
import warnings
from collections.abc import Generator
from copy import deepcopy
from dataclasses import asdict, dataclass
from enum import Enum
from threading import Thread
from typing import TYPE_CHECKING, Any

from .monitoring import TokenUsage
from .tools import Tool
from .utils import RateLimiter, Retrying, _is_package_available, encode_image_base64, make_image_url, parse_json_blob


if TYPE_CHECKING:
    from transformers import StoppingCriteriaList


logger = logging.getLogger(__name__)

RETRY_WAIT = 60
RETRY_MAX_ATTEMPTS = 3
RETRY_EXPONENTIAL_BASE = 2
RETRY_JITTER = True
STRUCTURED_GENERATION_PROVIDERS = ["cerebras", "fireworks-ai"]
CODEAGENT_RESPONSE_FORMAT = {
    "type": "json_schema",
    "json_schema": {
        "schema": {
            "additionalProperties": False,
            "properties": {
                "thought": {
                    "description": "A free form text description of the thought process.",
                    "title": "Thought",
                    "type": "string",
                },
                "code": {
                    "description": "Valid Python code snippet implementing the thought.",
                    "title": "Code",
                    "type": "string",
                },
            },
            "required": ["thought", "code"],
            "title": "ThoughtAndCodeAnswer",
            "type": "object",
        },
        "name": "ThoughtAndCodeAnswer",
        "strict": True,
    },
}


def get_dict_from_nested_dataclasses(obj, ignore_key=None):
    # [FUNCTION DOC]
    # HELPER: Recursively converts a dataclass instance into plain dicts, dropping one top-level key.
    # Inputs: obj (any; dataclass instances are converted, other values returned unchanged); ignore_key (str or None,
    #   default None): key name excluded from the top-level dict produced from the dataclass.
    # Outputs: a dict (if obj is a dataclass) or obj itself. Note dataclasses.asdict already recurses, so nested
    #   dataclasses become dicts and the ignore_key filter only effectively applies to the outermost level.
    # Exceptions: none raised directly (asdict may fail on non-deep-copyable field values).
    # Side effects: none.
    def convert(obj):
        # [FUNCTION DOC]
        # HELPER: Inner recursive converter used by get_dict_from_nested_dataclasses.
        # Inputs: obj (any). Uses the enclosing ignore_key.
        # Outputs: dict of the dataclass fields (minus ignore_key) if obj has dataclass fields, else obj unchanged.
        # Exceptions: none raised directly.
        # Side effects: none.
        if hasattr(obj, "__dataclass_fields__"):
            return {k: convert(v) for k, v in asdict(obj).items() if k != ignore_key}
        return obj

    return convert(obj)


def remove_content_after_stop_sequences(content: str | None, stop_sequences: list[str] | None) -> str | None:
    # [FUNCTION DOC]
    # Truncates text at the first occurrence of each stop sequence (applied in list order).
    # Inputs: content (str or None): generated text; stop_sequences (list of str or None).
    # Outputs: the content cut before each stop sequence in turn; content is returned unchanged if it is None or if
    #   stop_sequences is None/empty.
    # Exceptions: none raised directly (an empty-string stop sequence would make str.split raise ValueError).
    # Side effects: none.
    """Remove content after any stop sequence is encountered.

    Some providers may return ``None`` content (for example when responding purely with tool calls),
    so we skip processing in that case.
    """
    if content is None or not stop_sequences:
        return content

    for stop_seq in stop_sequences:
        split = content.split(stop_seq)
        content = split[0]
    return content


@dataclass
class ChatMessageToolCallFunction:
    # [CLASS DOC]
    # Dataclass describing the function part of a tool call.
    # Attributes: arguments (Any: JSON string or parsed dict of arguments), name (str: tool name),
    #   description (str or None, default None).
    # Outputs/Exceptions/Side effects: plain data container; none.
    arguments: Any
    name: str
    description: str | None = None


@dataclass
class ChatMessageToolCall:
    # [CLASS DOC]
    # Dataclass for one tool call emitted by a model.
    # Attributes: function (ChatMessageToolCallFunction), id (str: call identifier), type (str: usually 'function').
    # Exceptions/Side effects: none.
    function: ChatMessageToolCallFunction
    id: str
    type: str

    def __str__(self) -> str:
        # [FUNCTION DOC]
        # Human-readable representation of the tool call.
        # Inputs: self.
        # Outputs: string 'Call: <id>: Calling <function name> with arguments: <arguments>'.
        # Exceptions: none raised directly. Side effects: none.
        return f"Call: {self.id}: Calling {str(self.function.name)} with arguments: {str(self.function.arguments)}"


class MessageRole(str, Enum):
    # [CLASS DOC]
    # String enum of chat roles: USER 'user', ASSISTANT 'assistant', SYSTEM 'system', TOOL_CALL 'tool-call',
    #   TOOL_RESPONSE 'tool-response'. Members compare equal to their string values (str mixin).
    # Exceptions: constructing with an unknown value raises ValueError. Side effects: none.
    USER = "user"
    ASSISTANT = "assistant"
    SYSTEM = "system"
    TOOL_CALL = "tool-call"
    TOOL_RESPONSE = "tool-response"

    @classmethod
    def roles(cls):
        # [FUNCTION DOC]
        # Classmethod listing all role string values.
        # Inputs: cls.
        # Outputs: list of the string values of every member, in definition order.
        # Exceptions: none. Side effects: none.
        return [r.value for r in cls]


@dataclass
class ChatMessage:
    # [CLASS DOC]
    # Dataclass representing one chat message exchanged with a model.
    # Attributes: role (MessageRole or role string), content (str, list of content-part dicts such as
    #   {'type': 'text', 'text': ...} / {'type': 'image', 'image': ...}, or None), tool_calls (list of
    #   ChatMessageToolCall or None; dict/pydantic-like items are coerced in __post_init__), raw (Any: raw provider
    #   response, excluded from JSON dumps), token_usage (TokenUsage or None).
    # Exceptions: see __post_init__. Side effects: none.
    role: MessageRole
    content: str | list[dict[str, Any]] | None = None
    tool_calls: list[ChatMessageToolCall] | None = None
    raw: Any | None = None  # Stores the raw output from the API
    token_usage: TokenUsage | None = None

    def __post_init__(self) -> None:
        # [FUNCTION DOC]
        # Normalizes tool_calls after dataclass init by converting each entry to ChatMessageToolCall.
        # Inputs: self (tool_calls may be None or a list of ChatMessageToolCall / dict / objects with model_dump or dict).
        # Outputs: none (returns None).
        # Exceptions: propagates errors from _coerce_tool_call (e.g. KeyError on malformed dicts, UnboundLocalError for
        #   unsupported types).
        # Side effects: replaces self.tool_calls with a new list.
        if self.tool_calls is None:
            return
        self.tool_calls = [_coerce_tool_call(tool_call) for tool_call in self.tool_calls]

    def model_dump_json(self):
        # [FUNCTION DOC]
        # Serializes the message to a JSON string, omitting the 'raw' field.
        # Inputs: self.
        # Outputs: JSON string of the nested-dataclass dict.
        # Exceptions: TypeError from json.dumps if content/arguments/token_usage values are not JSON serializable.
        # Side effects: none.
        return json.dumps(get_dict_from_nested_dataclasses(self, ignore_key="raw"))

    @classmethod
    def from_dict(cls, data: dict, raw: Any | None = None, token_usage: TokenUsage | None = None) -> "ChatMessage":
        # [FUNCTION DOC]
        # Classmethod building a ChatMessage from a plain dict.
        # Inputs: data (dict with 'role', optional 'content', optional 'tool_calls' list of dicts having 'function'
        #   (kwargs for ChatMessageToolCallFunction), 'id', 'type'); raw (Any, default None); token_usage (TokenUsage or
        #   None, default None).
        # Outputs: a new ChatMessage with role converted to MessageRole.
        # Exceptions: ValueError for an unknown role; KeyError if 'role' or tool-call keys are missing; TypeError for
        #   unexpected function keys.
        # Side effects: mutates the input dict: replaces data['tool_calls'] with ChatMessageToolCall objects.
        # KNOWN ISSUE: the caller's dict is modified in place.
        if data.get("tool_calls"):
            tool_calls = [
                ChatMessageToolCall(
                    function=ChatMessageToolCallFunction(**tc["function"]), id=tc["id"], type=tc["type"]
                )
                for tc in data["tool_calls"]
            ]
            data["tool_calls"] = tool_calls
        return cls(
            role=MessageRole(data["role"]),
            content=data.get("content"),
            tool_calls=data.get("tool_calls"),
            raw=raw,
            token_usage=token_usage,
        )

    def dict(self):
        # [FUNCTION DOC]
        # Converts the message (including 'raw') into nested plain dicts.
        # Inputs: self. Outputs: dict of all fields. Exceptions: none raised directly. Side effects: none.
        return get_dict_from_nested_dataclasses(self)

    def render_as_markdown(self) -> str:
        # [FUNCTION DOC]
        # Renders the message as text: the stringified content followed by one JSON line per tool call
        #   ({'tool': name, 'arguments': arguments}), tool lines joined with newlines.
        # Inputs: self. Outputs: str.
        # Exceptions: TypeError if tool arguments are not JSON serializable. Side effects: none.
        # KNOWN ISSUE: str(None) is 'None', so the fallback to an empty string never triggers and None content renders as
        #   the text 'None'; also no separator is inserted between the content and the first tool-call line.
        rendered = str(self.content) or ""
        if self.tool_calls:
            rendered += "\n".join(
                [
                    json.dumps({"tool": tool.function.name, "arguments": tool.function.arguments})
                    for tool in self.tool_calls
                ]
            )
        return rendered


def _coerce_tool_call(tool_call: Any) -> ChatMessageToolCall:
    # [FUNCTION DOC]
    # HELPER: Converts a tool-call-like object into a ChatMessageToolCall.
    # Inputs: tool_call: a ChatMessageToolCall (returned as is), a dict, an object with model_dump() (pydantic v2), or
    #   an object with a callable dict() method. The dict form must contain 'function' (with 'arguments' and 'name'),
    #   'id' and 'type'.
    # Outputs: ChatMessageToolCall (function description is dropped).
    # Exceptions: KeyError for missing keys.
    # Side effects: none.
    # KNOWN ISSUE: if the input matches none of the supported shapes, the local dict variable is never bound and an
    #   UnboundLocalError is raised instead of a clear error.
    if isinstance(tool_call, ChatMessageToolCall):
        return tool_call

    if isinstance(tool_call, dict):
        tool_call_dict = tool_call
    elif hasattr(tool_call, "model_dump"):
        tool_call_dict = tool_call.model_dump()
    elif hasattr(tool_call, "dict") and callable(tool_call.dict):
        tool_call_dict = tool_call.dict()

    return ChatMessageToolCall(
        function=ChatMessageToolCallFunction(
            arguments=tool_call_dict["function"]["arguments"],
            name=tool_call_dict["function"]["name"],
        ),
        id=tool_call_dict["id"],
        type=tool_call_dict["type"],
    )


def parse_json_if_needed(arguments: str | dict) -> str | dict:
    # [FUNCTION DOC]
    # HELPER: Parses tool arguments from JSON if they are a string.
    # Inputs: arguments (str or dict).
    # Outputs: the dict unchanged; for a string, the json-decoded value (may be any JSON type) or the original string
    #   if decoding fails.
    # Exceptions: none (all decode errors are swallowed). Side effects: none.
    if isinstance(arguments, dict):
        return arguments
    else:
        try:
            return json.loads(arguments)
        except Exception:
            return arguments


@dataclass
class ChatMessageToolCallStreamDelta:
    # [CLASS DOC]
    # Dataclass for a partial tool call received during streaming.
    # Attributes (all default None): index (int: position of the tool call being built), id (str), type (str),
    #   function (ChatMessageToolCallFunction with partial name/arguments).
    # Exceptions/Side effects: none.
    """Represents a streaming delta for tool calls during generation."""

    index: int | None = None
    id: str | None = None
    type: str | None = None
    function: ChatMessageToolCallFunction | None = None


@dataclass
class ChatMessageStreamDelta:
    # [CLASS DOC]
    # Dataclass for one streaming chunk from a model.
    # Attributes (all default None): content (str text fragment), tool_calls (list of ChatMessageToolCallStreamDelta),
    #   token_usage (TokenUsage for this chunk).
    # Exceptions/Side effects: none.
    content: str | None = None
    tool_calls: list[ChatMessageToolCallStreamDelta] | None = None
    token_usage: TokenUsage | None = None


def agglomerate_stream_deltas(
    stream_deltas: list[ChatMessageStreamDelta], role: MessageRole = MessageRole.ASSISTANT
) -> ChatMessage:
    # [FUNCTION DOC]
    # Merges a sequence of streaming deltas into one complete ChatMessage.
    # Behavior: concatenates content fragments; sums input/output tokens across deltas; groups tool-call deltas by
    #   index, creating an entry (empty name/arguments) on first sight, overwriting id/type/name when provided and
    #   concatenating argument string fragments.
    # Inputs: stream_deltas (list of ChatMessageStreamDelta); role (MessageRole, default ASSISTANT).
    # Outputs: ChatMessage with the given role, accumulated content (empty string if none), tool calls of type
    #   'function' (id defaults to empty string), and a TokenUsage with the totals.
    # Exceptions: ValueError if a tool-call delta has no index.
    # Side effects: none on inputs (new objects are built).
    # KNOWN ISSUE: the docstring says it returns a single stream delta, but it returns a ChatMessage.
    """
    Agglomerate a list of stream deltas into a single stream delta.
    """
    accumulated_tool_calls: dict[int, ChatMessageToolCallStreamDelta] = {}
    accumulated_content = ""
    total_input_tokens = 0
    total_output_tokens = 0
    for stream_delta in stream_deltas:
        if stream_delta.token_usage:
            total_input_tokens += stream_delta.token_usage.input_tokens
            total_output_tokens += stream_delta.token_usage.output_tokens
        if stream_delta.content:
            accumulated_content += stream_delta.content
        if stream_delta.tool_calls:
            for tool_call_delta in stream_delta.tool_calls:  # ?ormally there should be only one call at a time
                # Extend accumulated_tool_calls list to accommodate the new tool call if needed
                if tool_call_delta.index is not None:
                    if tool_call_delta.index not in accumulated_tool_calls:
                        accumulated_tool_calls[tool_call_delta.index] = ChatMessageToolCallStreamDelta(
                            id=tool_call_delta.id,
                            type=tool_call_delta.type,
                            function=ChatMessageToolCallFunction(name="", arguments=""),
                        )
                    # Update the tool call at the specific index
                    tool_call = accumulated_tool_calls[tool_call_delta.index]
                    if tool_call_delta.id:
                        tool_call.id = tool_call_delta.id
                    if tool_call_delta.type:
                        tool_call.type = tool_call_delta.type
                    if tool_call_delta.function:
                        if tool_call_delta.function.name and len(tool_call_delta.function.name) > 0:
                            tool_call.function.name = tool_call_delta.function.name
                        if tool_call_delta.function.arguments:
                            tool_call.function.arguments += tool_call_delta.function.arguments
                else:
                    raise ValueError(f"Tool call index is not provided in tool delta: {tool_call_delta}")

    return ChatMessage(
        role=role,
        content=accumulated_content,
        tool_calls=[
            ChatMessageToolCall(
                function=ChatMessageToolCallFunction(
                    name=tool_call_stream_delta.function.name,
                    arguments=tool_call_stream_delta.function.arguments,
                ),
                id=tool_call_stream_delta.id or "",
                type="function",
            )
            for tool_call_stream_delta in accumulated_tool_calls.values()
            if tool_call_stream_delta.function
        ],
        token_usage=TokenUsage(
            input_tokens=total_input_tokens,
            output_tokens=total_output_tokens,
        ),
    )


tool_role_conversions = {
    MessageRole.TOOL_CALL: MessageRole.ASSISTANT,
    MessageRole.TOOL_RESPONSE: MessageRole.USER,
}


def get_tool_json_schema(tool: Tool) -> dict:
    # [FUNCTION DOC]
    # Builds an OpenAI-style function-calling schema for a smolagents Tool.
    # Behavior: deep-copies tool.inputs; maps type 'any' to 'string'; marks every input not flagged nullable as
    #   required; flattens 'anyOf' entries into a type (single type or list of types), mapping 'any' to 'string',
    #   setting nullable when a 'null' option exists and keeping the last enum found, then removes 'anyOf'.
    # Inputs: tool (Tool with name, description, inputs dict of JSON-schema-like property specs each having 'type').
    # Outputs: dict {'type': 'function', 'function': {'name', 'description', 'parameters': {'type': 'object',
    #   'properties', 'required'}}}.
    # Exceptions: KeyError if a property lacks 'type'; IndexError if an anyOf contains only 'null' options.
    # Side effects: none (works on a copy).
    # KNOWN ISSUE: 'required' is decided before anyOf processing, so a property made nullable via a 'null' anyOf
    #   option is still listed as required.
    properties = deepcopy(tool.inputs)
    required = []
    for key, value in properties.items():
        if value["type"] == "any":
            value["type"] = "string"
        if not ("nullable" in value and value["nullable"]):
            required.append(key)

        # parse anyOf
        if "anyOf" in value:
            types = []
            enum = None
            for t in value["anyOf"]:
                if t["type"] == "null":
                    value["nullable"] = True
                    continue
                if t["type"] == "any":
                    types.append("string")
                else:
                    types.append(t["type"])
                if "enum" in t:  # assuming there is only one enum in anyOf
                    enum = t["enum"]

            value["type"] = types if len(types) > 1 else types[0]
            if enum is not None:
                value["enum"] = enum

            value.pop("anyOf")

    return {
        "type": "function",
        "function": {
            "name": tool.name,
            "description": tool.description,
            "parameters": {
                "type": "object",
                "properties": properties,
                "required": required,
            },
        },
    }


def get_clean_message_list(
    message_list: list[ChatMessage | dict],
    role_conversions: dict[MessageRole, MessageRole] | dict[str, str] = {},
    convert_images_to_image_urls: bool = False,
    flatten_messages_as_text: bool = False,
) -> list[dict[str, Any]]:
    # [FUNCTION DOC]
    # Normalizes a list of messages into chat-template-ready dicts, merging consecutive messages with the same role.
    # Behavior: deep-copies the list; converts dicts to ChatMessage; validates roles; applies role_conversions; for
    #   list content, encodes 'image' parts either to base64 strings or (if convert_images_to_image_urls) to
    #   'image_url' parts with a data URL; when a message has the same role as the previous output message, its text
    #   is appended with a newline (flattened mode) or its parts are appended, merging adjacent text parts.
    # Inputs: message_list (list of ChatMessage or dict); role_conversions (mapping role->role, default empty);
    #   convert_images_to_image_urls (bool, default False); flatten_messages_as_text (bool, default False: if True,
    #   content becomes the text of the first content part only).
    # Outputs: list of dicts {'role': role, 'content': str or list of parts}.
    # Exceptions: ValueError for unsupported roles; AssertionError for non-dict content parts, images when
    #   flattening, or non-list content when merging; KeyError/TypeError if content is not a list of text parts
    #   when flattening.
    # Side effects: none on the input (deep copy).
    # KNOWN ISSUE: uses a mutable default argument for role_conversions (not mutated, so harmless); flattening keeps
    #   only the first content part's text and silently drops other parts; validation uses assert, which is
    #   disabled under python -O.
    """
    Creates a list of messages to give as input to the LLM. These messages are dictionaries and chat template compatible with transformers LLM chat template.
    Subsequent messages with the same role will be concatenated to a single message.

    Args:
        message_list (`list[ChatMessage | dict]`): List of chat messages. Mixed types are allowed.
        role_conversions (`dict[MessageRole, MessageRole]`, *optional* ): Mapping to convert roles.
        convert_images_to_image_urls (`bool`, default `False`): Whether to convert images to image URLs.
        flatten_messages_as_text (`bool`, default `False`): Whether to flatten messages as text.
    """
    output_message_list: list[dict[str, Any]] = []
    message_list = deepcopy(message_list)  # Avoid modifying the original list
    for message in message_list:
        if isinstance(message, dict):
            message = ChatMessage.from_dict(message)
        role = message.role
        if role not in MessageRole.roles():
            raise ValueError(f"Incorrect role {role}, only {MessageRole.roles()} are supported for now.")

        if role in role_conversions:
            message.role = role_conversions[role]  # type: ignore
        # encode images if needed
        if isinstance(message.content, list):
            for element in message.content:
                assert isinstance(element, dict), "Error: this element should be a dict:" + str(element)
                if element["type"] == "image":
                    assert not flatten_messages_as_text, f"Cannot use images with {flatten_messages_as_text=}"
                    if convert_images_to_image_urls:
                        element.update(
                            {
                                "type": "image_url",
                                "image_url": {"url": make_image_url(encode_image_base64(element.pop("image")))},
                            }
                        )
                    else:
                        element["image"] = encode_image_base64(element["image"])

        if len(output_message_list) > 0 and message.role == output_message_list[-1]["role"]:
            assert isinstance(message.content, list), "Error: wrong content:" + str(message.content)
            if flatten_messages_as_text:
                output_message_list[-1]["content"] += "\n" + message.content[0]["text"]
            else:
                for el in message.content:
                    if el["type"] == "text" and output_message_list[-1]["content"][-1]["type"] == "text":
                        # Merge consecutive text messages rather than creating new ones
                        output_message_list[-1]["content"][-1]["text"] += "\n" + el["text"]
                    else:
                        output_message_list[-1]["content"].append(el)
        else:
            if flatten_messages_as_text:
                content = message.content[0]["text"]
            else:
                content = message.content
            output_message_list.append(
                {
                    "role": message.role,
                    "content": content,
                }
            )
    return output_message_list


def get_tool_call_from_text(text: str, tool_name_key: str, tool_arguments_key: str) -> ChatMessageToolCall:
    # [FUNCTION DOC]
    # Extracts a tool call from free text containing a JSON blob.
    # Inputs: text (str); tool_name_key (str: key holding the tool name); tool_arguments_key (str: key holding the
    #   arguments).
    # Outputs: ChatMessageToolCall with a random uuid4 id, type 'function', the tool name, and arguments (string
    #   arguments are JSON-parsed when possible; missing arguments become None).
    # Exceptions: ValueError if the name key is absent; errors from parse_json_blob propagate (e.g. ValueError when
    #   no JSON is found).
    # Side effects: none.
    tool_call_dictionary, _ = parse_json_blob(text)
    try:
        tool_name = tool_call_dictionary[tool_name_key]
    except Exception as e:
        raise ValueError(
            f"Tool call needs to have a key '{tool_name_key}'. Got keys: {list(tool_call_dictionary.keys())} instead"
        ) from e
    tool_arguments = tool_call_dictionary.get(tool_arguments_key, None)
    if isinstance(tool_arguments, str):
        tool_arguments = parse_json_if_needed(tool_arguments)
    return ChatMessageToolCall(
        id=str(uuid.uuid4()),
        type="function",
        function=ChatMessageToolCallFunction(name=tool_name, arguments=tool_arguments),
    )


def supports_stop_parameter(model_id: str) -> bool:
    # [FUNCTION DOC]
    # Decides whether a model id accepts the 'stop' API parameter.
    # Behavior: takes the part after the last '/'; 'o3-mini' is supported; names matching o3/o3-* /o3.*, o4/o4-*,
    #   gpt-5*, or grok-* (optionally prefixed with an 'xxx.' namespace) are unsupported; everything else supported.
    # Inputs: model_id (str).
    # Outputs: bool.
    # Exceptions: none. Side effects: none.
    # KNOWN ISSUE: the docstring mentions only OpenAI reasoning models, but grok-* models are also excluded.
    """
    Check if the model supports the `stop` parameter.

    Not supported with reasoning models openai/o3, openai/o4-mini, and the openai/gpt-5 series (and their versioned variants).

    Args:
        model_id (`str`): Model identifier (e.g. "openai/o3", "o4-mini-2025-04-16")

    Returns:
        bool: True if the model supports the stop parameter, False otherwise
    """
    model_name = model_id.split("/")[-1]
    if model_name == "o3-mini":
        return True
    # o3* (except mini), o4*, all grok-* models, and the gpt-5* family (including versioned variants) don't support stop parameter
    openai_model_pattern = r"(o3(?:$|[-.].*)|o4(?:$|[-.].*)|gpt-5.*)"
    grok_model_pattern = r"([A-Za-z][A-Za-z0-9_-]*\.)?grok-[A-Za-z0-9][A-Za-z0-9_.-]*"
    pattern = rf"^({openai_model_pattern}|{grok_model_pattern})$"

    return not re.match(pattern, model_name)


class _ParameterRemove:
    # [CLASS DOC]
    # HELPER: Sentinel type; its singleton REMOVE_PARAMETER, when used as a value in a Model's kwargs, removes that
    #   key from the prepared completion kwargs.
    # Inputs/Attributes: none. Exceptions/Side effects: none.
    """Sentinel value to indicate a parameter should be removed."""

    def __repr__(self):
        # [FUNCTION DOC]
        # HELPER: Returns the string 'REMOVE_PARAMETER'. Inputs: self. Exceptions: none. Side effects: none.
        return "REMOVE_PARAMETER"


# Singleton instance for removing parameters
REMOVE_PARAMETER = _ParameterRemove()


class Model:
    # [CLASS DOC]
    # Abstract base class for all model backends used by agents.
    # Constructor args: flatten_messages_as_text (bool, default False), tool_name_key (str, default 'name'),
    #   tool_arguments_key (str, default 'arguments'), model_id (str or None), **kwargs (stored as self.kwargs and
    #   merged into every completion call with highest priority).
    # Key attributes: flatten_messages_as_text, tool_name_key, tool_arguments_key, kwargs, model_id.
    # Behavior: subclasses implement generate() (and optionally generate_stream()); calling the instance calls
    #   generate(); provides kwarg preparation, tool-call parsing and (de)serialization helpers.
    # Exceptions: generate raises NotImplementedError in the base class. Side effects: none at construction.
    """Base class for all language model implementations.

    This abstract class defines the core interface that all model implementations must follow
    to work with agents. It provides common functionality for message handling, tool integration,
    and model configuration while allowing subclasses to implement their specific generation logic.

    Parameters:
        flatten_messages_as_text (`bool`, default `False`):
            Whether to flatten complex message content into plain text format.
        tool_name_key (`str`, default `"name"`):
            The key used to extract tool names from model responses.
        tool_arguments_key (`str`, default `"arguments"`):
            The key used to extract tool arguments from model responses.
        model_id (`str`, *optional*):
            Identifier for the specific model being used.
        **kwargs:
            Additional keyword arguments to forward to the underlying model completion call.

    Note:
        This is an abstract base class. Subclasses must implement the `generate()` method
        to provide actual model inference capabilities.

    Example:
        ```python
        class CustomModel(Model):
            def generate(self, messages, **kwargs):
                # Implementation specific to your model
                pass
        ```
    """

    def __init__(
        self,
        flatten_messages_as_text: bool = False,
        tool_name_key: str = "name",
        tool_arguments_key: str = "arguments",
        model_id: str | None = None,
        **kwargs,
    ):
        # [FUNCTION DOC]
        # Stores configuration on the instance.
        # Inputs: see class doc. Outputs: none. Exceptions: none. Side effects: sets instance attributes.
        self.flatten_messages_as_text = flatten_messages_as_text
        self.tool_name_key = tool_name_key
        self.tool_arguments_key = tool_arguments_key
        self.kwargs = kwargs
        self.model_id: str | None = model_id

    @property
    def supports_stop_parameter(self) -> bool:
        # [FUNCTION DOC]
        # Property: whether this model id supports the 'stop' parameter (delegates to the module-level function, using an
        #   empty string when model_id is None, which yields True).
        # Inputs: self. Outputs: bool. Exceptions: none. Side effects: none.
        return supports_stop_parameter(self.model_id or "")

    def _prepare_completion_kwargs(
        self,
        messages: list[ChatMessage | dict],
        stop_sequences: list[str] | None = None,
        response_format: dict[str, str] | None = None,
        tools_to_call_from: list[Tool] | None = None,
        custom_role_conversions: dict[str, str] | None = None,
        convert_images_to_image_urls: bool = False,
        tool_choice: str | dict | None = "required",  # Configurable tool_choice parameter
        **kwargs,
    ) -> dict[str, Any]:
        # [FUNCTION DOC]
        # HELPER: Builds the keyword arguments for a backend completion call.
        # Behavior: pops 'flatten_messages_as_text' from kwargs (default self.flatten_messages_as_text); cleans messages
        #   with get_clean_message_list using custom_role_conversions or the default tool_role_conversions; then sets, in
        #   increasing priority: 'stop' (only if given and supported), 'response_format', 'tools' (JSON schemas) plus
        #   'tool_choice' (if not None and tools given); then the remaining kwargs; then self.kwargs, where a value of
        #   REMOVE_PARAMETER deletes the key instead.
        # Inputs: messages (list of ChatMessage or dict); stop_sequences (list of str or None); response_format (dict or
        #   None); tools_to_call_from (list of Tool or None); custom_role_conversions (dict or None);
        #   convert_images_to_image_urls (bool, default False); tool_choice (str, dict or None, default 'required');
        #   **kwargs.
        # Outputs: dict of completion kwargs containing at least 'messages'.
        # Exceptions: propagates errors from get_clean_message_list.
        # Side effects: none on self (the kwargs dict passed in is local).
        """
        Prepare parameters required for model invocation.

        Parameter priority (highest to lowest):
        1. self.kwargs (model defaults)
        2. Explicitly passed kwargs
        3. Specific parameters (stop_sequences, response_format, etc.)
        """
        # Clean and standardize the message list
        flatten_messages_as_text = kwargs.pop("flatten_messages_as_text", self.flatten_messages_as_text)
        messages_as_dicts = get_clean_message_list(
            messages,
            role_conversions=custom_role_conversions or tool_role_conversions,
            convert_images_to_image_urls=convert_images_to_image_urls,
            flatten_messages_as_text=flatten_messages_as_text,
        )
        # Start with messages
        completion_kwargs = {
            "messages": messages_as_dicts,
        }
        # Override with specific parameters
        if stop_sequences is not None and self.supports_stop_parameter:
            # Some models do not support stop parameter
            completion_kwargs["stop"] = stop_sequences
        if response_format is not None:
            completion_kwargs["response_format"] = response_format
        if tools_to_call_from:
            completion_kwargs["tools"] = [get_tool_json_schema(tool) for tool in tools_to_call_from]
            if tool_choice is not None:
                completion_kwargs["tool_choice"] = tool_choice
        # Override with passed-in kwargs
        completion_kwargs.update(kwargs)
        # Override with self.kwargs
        for kwarg_name, kwarg_value in self.kwargs.items():
            if kwarg_value is REMOVE_PARAMETER:
                completion_kwargs.pop(kwarg_name, None)  # Remove parameter if present
            else:
                completion_kwargs[kwarg_name] = kwarg_value  # Set/override parameter
        return completion_kwargs

    def generate(
        self,
        messages: list[ChatMessage],
        stop_sequences: list[str] | None = None,
        response_format: dict[str, str] | None = None,
        tools_to_call_from: list[Tool] | None = None,
        **kwargs,
    ) -> ChatMessage:
        # [FUNCTION DOC]
        # Abstract: produce a model response for the messages.
        # Inputs: messages (list of ChatMessage/dict), stop_sequences (list of str or None), response_format (dict or
        #   None), tools_to_call_from (list of Tool or None), **kwargs.
        # Outputs: ChatMessage (in subclasses).
        # Exceptions: always NotImplementedError here. Side effects: none.
        """Process the input messages and return the model's response.

        Parameters:
            messages (`list[dict[str, str | list[dict]]] | list[ChatMessage]`):
                A list of message dictionaries to be processed. Each dictionary should have the structure `{"role": "user/system", "content": "message content"}`.
            stop_sequences (`List[str]`, *optional*):
                A list of strings that will stop the generation if encountered in the model's output.
            response_format (`dict[str, str]`, *optional*):
                The response format to use in the model's response.
            tools_to_call_from (`List[Tool]`, *optional*):
                A list of tools that the model can use to generate responses.
            **kwargs:
                Additional keyword arguments to be passed to the underlying model.

        Returns:
            `ChatMessage`: A chat message object containing the model's response.
        """
        raise NotImplementedError("This method must be implemented in child classes")

    def __call__(self, *args, **kwargs):
        # [FUNCTION DOC]
        # Makes the model callable; forwards all positional and keyword arguments to generate().
        # Outputs: whatever generate returns. Exceptions: those of generate. Side effects: those of generate.
        return self.generate(*args, **kwargs)

    def parse_tool_calls(self, message: ChatMessage) -> ChatMessage:
        # [FUNCTION DOC]
        # Ensures a message has parsed tool calls.
        # Behavior: forces role to ASSISTANT; if there are no tool_calls, parses one from the text content using
        #   tool_name_key/tool_arguments_key; then JSON-parses string arguments of every tool call where possible.
        # Inputs: message (ChatMessage).
        # Outputs: the same message object.
        # Exceptions: AssertionError if there is neither content nor tool calls, or no tool call results; ValueError from
        #   get_tool_call_from_text.
        # Side effects: mutates the given message (role, tool_calls, arguments).
        """Sometimes APIs do not return the tool call as a specific object, so we need to parse it."""
        message.role = MessageRole.ASSISTANT  # Overwrite role if needed
        if not message.tool_calls:
            assert message.content is not None, "Message contains no content and no tool calls"
            message.tool_calls = [
                get_tool_call_from_text(message.content, self.tool_name_key, self.tool_arguments_key)
            ]
        assert len(message.tool_calls) > 0, "No tool call was found in the model output"
        for tool_call in message.tool_calls:
            tool_call.function.arguments = parse_json_if_needed(tool_call.function.arguments)
        return message

    def to_dict(self) -> dict:
        # [FUNCTION DOC]
        # Serializes the model configuration to a JSON-compatible dict.
        # Behavior: starts with self.kwargs plus model_id, then copies known optional attributes if present
        #   (custom_role_conversion, temperature, max_tokens, provider, timeout, api_base, torch_dtype, device_map,
        #   organization, project, azure_endpoint). Secrets ('token', 'api_key') are not exported.
        # Inputs: self. Outputs: dict.
        # Exceptions: none raised directly.
        # Side effects: prints a warning to stdout for each secret attribute present on the instance.
        # KNOWN ISSUE: looks up 'custom_role_conversion' (singular) while subclasses store 'custom_role_conversions', so
        #   role conversions are never exported; warnings use print rather than logging/warnings.
        """
        Converts the model into a JSON-compatible dictionary.
        """
        model_dictionary = {
            **self.kwargs,
            "model_id": self.model_id,
        }
        for attribute in [
            "custom_role_conversion",
            "temperature",
            "max_tokens",
            "provider",
            "timeout",
            "api_base",
            "torch_dtype",
            "device_map",
            "organization",
            "project",
            "azure_endpoint",
        ]:
            if hasattr(self, attribute):
                model_dictionary[attribute] = getattr(self, attribute)

        dangerous_attributes = ["token", "api_key"]
        for attribute_name in dangerous_attributes:
            if hasattr(self, attribute_name):
                print(
                    f"For security reasons, we do not export the `{attribute_name}` attribute of your model. Please export it manually."
                )
        return model_dictionary

    @classmethod
    def from_dict(cls, model_dictionary: dict[str, Any]) -> "Model":
        # [FUNCTION DOC]
        # Classmethod recreating a model by passing every dict entry as a constructor keyword.
        # Inputs: model_dictionary (dict). Outputs: new instance of cls.
        # Exceptions: TypeError if keys do not match constructor params; any constructor error.
        # Side effects: those of the constructor (may load weights or create clients).
        return cls(**{k: v for k, v in model_dictionary.items()})


class VLLMModel(Model):
    # [CLASS DOC]
    # Local inference backend using vLLM.
    # Constructor args: model_id (str: HF repo id or path), model_kwargs (dict or None: forwarded to vllm.LLM),
    #   apply_chat_template_kwargs (dict or None: forwarded to tokenizer.apply_chat_template), **kwargs (Model kwargs).
    # Key attributes: model (vllm.LLM), tokenizer, model_kwargs, apply_chat_template_kwargs, _is_vlm (always False).
    # Exceptions: ModuleNotFoundError if vllm is not installed.
    # Side effects: loads the model onto devices (heavy GPU/memory use).
    """Model to use [vLLM](https://docs.vllm.ai/) for fast LLM inference and serving.

    Parameters:
        model_id (`str`):
            The Hugging Face model ID to be used for inference.
            This can be a path or model identifier from the Hugging Face model hub.
        model_kwargs (`dict[str, Any]`, *optional*):
            Additional keyword arguments to forward to the vLLM LLM instantiation, such as `revision`, `max_model_len`, etc.
        apply_chat_template_kwargs (dict, *optional*):
            Additional keyword arguments to pass to the `apply_chat_template` method of the tokenizer.
        **kwargs:
            Additional keyword arguments to forward to the underlying vLLM model generate call.
    """

    def __init__(
        self,
        model_id,
        model_kwargs: dict[str, Any] | None = None,
        apply_chat_template_kwargs: dict[str, Any] | None = None,
        **kwargs,
    ):
        # [FUNCTION DOC]
        # Checks vllm availability, stores config, loads the vLLM engine and tokenizer.
        # Inputs: see class doc. Outputs: none.
        # Exceptions: ModuleNotFoundError when vllm is missing; AssertionError if LLM is None; vLLM loading errors.
        # Side effects: model weight download/loading.
        if not _is_package_available("vllm"):
            raise ModuleNotFoundError("Please install 'vllm' extra to use VLLMModel: `pip install 'smolagents[vllm]'`")

        from vllm import LLM  # type: ignore
        from vllm.transformers_utils.tokenizer import get_tokenizer  # type: ignore

        self.model_kwargs = model_kwargs or {}
        self.apply_chat_template_kwargs = apply_chat_template_kwargs or {}
        super().__init__(**kwargs)
        self.model_id = model_id
        self.model = LLM(model=model_id, **self.model_kwargs)
        assert self.model is not None
        self.tokenizer = get_tokenizer(model_id)
        self._is_vlm = False  # VLLMModel does not support vision models yet.

    def cleanup(self):
        # [FUNCTION DOC]
        # Releases vLLM distributed state and GPU memory.
        # Inputs: self. Outputs: none.
        # Exceptions: import errors if torch/vllm missing; AttributeError if the engine layout differs.
        # Side effects: destroys model-parallel and distributed groups, deletes the driver worker, runs garbage
        #   collection and empties the CUDA cache.
        import gc

        import torch
        from vllm.distributed.parallel_state import (  # type: ignore
            destroy_distributed_environment,
            destroy_model_parallel,
        )

        destroy_model_parallel()
        if self.model is not None:
            # taken from https://github.com/vllm-project/vllm/issues/1908#issuecomment-2076870351
            del self.model.llm_engine.model_executor.driver_worker
        gc.collect()
        destroy_distributed_environment()
        torch.cuda.empty_cache()

    def generate(
        self,
        messages: list[ChatMessage | dict],
        stop_sequences: list[str] | None = None,
        response_format: dict[str, str] | None = None,
        tools_to_call_from: list[Tool] | None = None,
        **kwargs,
    ) -> ChatMessage:
        # [FUNCTION DOC]
        # Generates a completion with vLLM.
        # Behavior: prepares kwargs (flattened text messages), builds structured-output params from
        #   response_format['json_schema']['schema'] if given, renders a prompt with the chat template (tools included),
        #   samples with n/temperature/max_tokens taken from kwargs (defaults 1, 0.0, 2048) and the stop sequences, then
        #   post-truncates on stop sequences when the model id does not support stop.
        # Inputs: messages, stop_sequences, response_format (OpenAI-style json_schema dict), tools_to_call_from, **kwargs.
        # Outputs: ChatMessage (assistant) with the first output text, raw dict {'out', 'completion_kwargs'} and token
        #   usage from prompt/output token ids.
        # Exceptions: KeyError for malformed response_format; vLLM errors.
        # Side effects: GPU inference.
        # KNOWN ISSUE: n/temperature/max_tokens are read from kwargs but also remain in completion_kwargs, which are
        #   forwarded to LLM.generate and may cause an unexpected-keyword error.
        from vllm import SamplingParams  # type: ignore
        from vllm.sampling_params import StructuredOutputsParams  # type: ignore

        completion_kwargs = self._prepare_completion_kwargs(
            messages=messages,
            flatten_messages_as_text=(not self._is_vlm),
            stop_sequences=stop_sequences,
            tools_to_call_from=tools_to_call_from,
            **kwargs,
        )
        # Override the OpenAI schema for VLLM compatibility
        structured_outputs = (
            StructuredOutputsParams(json=response_format["json_schema"]["schema"]) if response_format else None
        )

        messages = completion_kwargs.pop("messages")
        prepared_stop_sequences = completion_kwargs.pop("stop", [])
        tools = completion_kwargs.pop("tools", None)
        completion_kwargs.pop("tool_choice", None)

        prompt = self.tokenizer.apply_chat_template(
            messages,
            tools=tools,
            add_generation_prompt=True,
            tokenize=False,
            **self.apply_chat_template_kwargs,
        )

        sampling_params = SamplingParams(
            n=kwargs.get("n", 1),
            temperature=kwargs.get("temperature", 0.0),
            max_tokens=kwargs.get("max_tokens", 2048),
            stop=prepared_stop_sequences,
            structured_outputs=structured_outputs,
        )

        out = self.model.generate(
            prompt,
            sampling_params=sampling_params,
            **completion_kwargs,
        )

        output_text = out[0].outputs[0].text
        if stop_sequences is not None and not self.supports_stop_parameter:
            output_text = remove_content_after_stop_sequences(output_text, stop_sequences)
        return ChatMessage(
            role=MessageRole.ASSISTANT,
            content=output_text,
            raw={"out": output_text, "completion_kwargs": completion_kwargs},
            token_usage=TokenUsage(
                input_tokens=len(out[0].prompt_token_ids),
                output_tokens=len(out[0].outputs[0].token_ids),
            ),
        )


class MLXModel(Model):
    # [CLASS DOC]
    # Local inference backend for Apple silicon using mlx-lm (text only).
    # Constructor args: model_id (str), trust_remote_code (bool, default False), load_kwargs (dict or None: for
    #   mlx_lm.load; tokenizer_config.trust_remote_code is defaulted), apply_chat_template_kwargs (dict or None;
    #   add_generation_prompt defaults to True), **kwargs (forwarded to stream_generate, e.g. max_tokens).
    # Key attributes: model, tokenizer, stream_generate, load_kwargs, apply_chat_template_kwargs, is_vlm (False).
    # Exceptions: ModuleNotFoundError if mlx_lm is missing.
    # Side effects: loads model weights. Always uses flattened text messages.
    """A class to interact with models loaded using MLX on Apple silicon.

    > [!TIP]
    > You must have `mlx-lm` installed on your machine. Please run `pip install 'smolagents[mlx-lm]'` if it's not the case.

    Parameters:
        model_id (str):
            The Hugging Face model ID to be used for inference. This can be a path or model identifier from the Hugging Face model hub.
        tool_name_key (str):
            The key, which can usually be found in the model's chat template, for retrieving a tool name.
        tool_arguments_key (str):
            The key, which can usually be found in the model's chat template, for retrieving tool arguments.
        trust_remote_code (bool, default `False`):
            Some models on the Hub require running remote code: for this model, you would have to set this flag to True.
        load_kwargs (dict[str, Any], *optional*):
            Additional keyword arguments to pass to the `mlx.lm.load` method when loading the model and tokenizer.
        apply_chat_template_kwargs (dict, *optional*):
            Additional keyword arguments to pass to the `apply_chat_template` method of the tokenizer.
        **kwargs:
            Additional keyword arguments to forward to the underlying MLX model stream_generate call, for instance `max_tokens`.

    Example:
    ```python
    >>> engine = MLXModel(
    ...     model_id="mlx-community/Qwen2.5-Coder-32B-Instruct-4bit",
    ...     max_tokens=10000,
    ... )
    >>> messages = [
    ...     {
    ...         "role": "user",
    ...         "content": "Explain quantum mechanics in simple terms."
    ...     }
    ... ]
    >>> response = engine(messages, stop_sequences=["END"])
    >>> print(response)
    "Quantum mechanics is the branch of physics that studies..."
    ```
    """

    def __init__(
        self,
        model_id: str,
        trust_remote_code: bool = False,
        load_kwargs: dict[str, Any] | None = None,
        apply_chat_template_kwargs: dict[str, Any] | None = None,
        **kwargs,
    ):
        # [FUNCTION DOC]
        # Validates mlx_lm availability, sets defaults on load/template kwargs, and loads model and tokenizer.
        # Inputs: see class doc. Outputs: none.
        # Exceptions: ModuleNotFoundError; loading errors.
        # Side effects: mutates the passed load_kwargs/apply_chat_template_kwargs dicts (setdefault) when provided;
        #   downloads/loads weights.
        if not _is_package_available("mlx_lm"):
            raise ModuleNotFoundError(
                "Please install 'mlx-lm' extra to use 'MLXModel': `pip install 'smolagents[mlx-lm]'`"
            )
        import mlx_lm

        self.load_kwargs = load_kwargs or {}
        self.load_kwargs.setdefault("tokenizer_config", {}).setdefault("trust_remote_code", trust_remote_code)
        self.apply_chat_template_kwargs = apply_chat_template_kwargs or {}
        self.apply_chat_template_kwargs.setdefault("add_generation_prompt", True)
        # mlx-lm doesn't support vision models: flatten_messages_as_text=True
        super().__init__(model_id=model_id, flatten_messages_as_text=True, **kwargs)

        self.model, self.tokenizer = mlx_lm.load(self.model_id, **self.load_kwargs)
        self.stream_generate = mlx_lm.stream_generate
        self.is_vlm = False  # mlx-lm doesn't support vision models

    def generate(
        self,
        messages: list[ChatMessage | dict],
        stop_sequences: list[str] | None = None,
        response_format: dict[str, str] | None = None,
        tools_to_call_from: list[Tool] | None = None,
        **kwargs,
    ) -> ChatMessage:
        # [FUNCTION DOC]
        # Generates text by streaming tokens from mlx_lm and stopping at the first stop sequence found.
        # Inputs: messages, stop_sequences, response_format (must be None), tools_to_call_from, **kwargs.
        # Outputs: ChatMessage (assistant) with text cut before the stop string, raw dict, and token usage (prompt id
        #   count, number of streamed chunks).
        # Exceptions: ValueError if response_format is given; mlx errors.
        # Side effects: on-device inference.
        if response_format is not None:
            raise ValueError("MLX does not support structured outputs.")
        completion_kwargs = self._prepare_completion_kwargs(
            messages=messages,
            stop_sequences=stop_sequences,
            tools_to_call_from=tools_to_call_from,
            **kwargs,
        )
        messages = completion_kwargs.pop("messages")
        stops = completion_kwargs.pop("stop", [])
        tools = completion_kwargs.pop("tools", None)
        completion_kwargs.pop("tool_choice", None)

        prompt_ids = self.tokenizer.apply_chat_template(messages, tools=tools, **self.apply_chat_template_kwargs)

        output_tokens = 0
        text = ""
        for response in self.stream_generate(self.model, self.tokenizer, prompt=prompt_ids, **completion_kwargs):
            output_tokens += 1
            text += response.text
            if any((stop_index := text.rfind(stop)) != -1 for stop in stops):
                text = text[:stop_index]
                break
        if stop_sequences is not None and not self.supports_stop_parameter:
            text = remove_content_after_stop_sequences(text, stop_sequences)
        return ChatMessage(
            role=MessageRole.ASSISTANT,
            content=text,
            raw={"out": text, "completion_kwargs": completion_kwargs},
            token_usage=TokenUsage(
                input_tokens=len(prompt_ids),
                output_tokens=output_tokens,
            ),
        )


class TransformersModel(Model):
    # [CLASS DOC]
    # Local inference backend using Hugging Face transformers; tries an image-text-to-text (VLM) model first and falls
    #   back to a causal LM.
    # Constructor args: model_id (str or None; defaults with FutureWarning to 'HuggingFaceTB/SmolLM2-1.7B-Instruct'),
    #   device_map (str or None; defaults to 'cuda' if available else 'cpu'), torch_dtype (str or None),
    #   trust_remote_code (bool, default False), model_kwargs (dict or None, for from_pretrained), max_new_tokens (int,
    #   default 4096), max_tokens (int or None, overrides max_new_tokens), apply_chat_template_kwargs (dict or None),
    #   **kwargs (stored in self.kwargs, forwarded to model.generate).
    # Key attributes: model, processor (VLM) or tokenizer (LLM), streamer (TextIteratorStreamer), _is_vlm.
    # Exceptions: ModuleNotFoundError if torch/transformers are missing; ValueError if loading fails.
    # Side effects: loads weights, logs the chosen device.
    """A class that uses Hugging Face's Transformers library for language model interaction.

    This model allows you to load and use Hugging Face's models locally using the Transformers library. It supports features like stop sequences and grammar customization.

    > [!TIP]
    > You must have `transformers` and `torch` installed on your machine. Please run `pip install 'smolagents[transformers]'` if it's not the case.

    Parameters:
        model_id (`str`):
            The Hugging Face model ID to be used for inference. This can be a path or model identifier from the Hugging Face model hub.
            For example, `"Qwen/Qwen3-Next-80B-A3B-Thinking"`.
        device_map (`str`, *optional*):
            The device_map to initialize your model with.
        torch_dtype (`str`, *optional*):
            The torch_dtype to initialize your model with.
        trust_remote_code (bool, default `False`):
            Some models on the Hub require running remote code: for this model, you would have to set this flag to True.
        model_kwargs (`dict[str, Any]`, *optional*):
            Additional keyword arguments to pass to `AutoModel.from_pretrained` (like revision, model_args, config, etc.).
        max_new_tokens (`int`, default `4096`):
            Maximum number of new tokens to generate, ignoring the number of tokens in the prompt.
        max_tokens (`int`, *optional*):
            Alias for `max_new_tokens`. If provided, this value takes precedence.
        apply_chat_template_kwargs (dict, *optional*):
            Additional keyword arguments to pass to the `apply_chat_template` method of the tokenizer.
        **kwargs:
            Additional keyword arguments to forward to the underlying Transformers model generate call, such as `device`.
    Raises:
        ValueError:
            If the model name is not provided.

    Example:
    ```python
    >>> engine = TransformersModel(
    ...     model_id="Qwen/Qwen3-Next-80B-A3B-Thinking",
    ...     device="cuda",
    ...     max_new_tokens=5000,
    ... )
    >>> messages = [{"role": "user", "content": "Explain quantum mechanics in simple terms."}]
    >>> response = engine(messages, stop_sequences=["END"])
    >>> print(response)
    "Quantum mechanics is the branch of physics that studies..."
    ```
    """

    def __init__(
        self,
        model_id: str | None = None,
        device_map: str | None = None,
        torch_dtype: str | None = None,
        trust_remote_code: bool = False,
        model_kwargs: dict[str, Any] | None = None,
        max_new_tokens: int = 4096,
        max_tokens: int | None = None,
        apply_chat_template_kwargs: dict[str, Any] | None = None,
        **kwargs,
    ):
        # [FUNCTION DOC]
        # Imports dependencies, resolves defaults, loads model plus processor/tokenizer and creates a streamer.
        # Inputs: see class doc. Outputs: none.
        # Exceptions: ModuleNotFoundError; ValueError wrapping any non-'Unrecognized configuration class' load failure;
        #   a ValueError from the VLM loader that is not about an unrecognized config is re-raised.
        # Side effects: emits FutureWarning when model_id missing; logs device; downloads/loads weights.
        try:
            import torch
            from transformers import (
                AutoModelForCausalLM,
                AutoModelForImageTextToText,
                AutoProcessor,
                AutoTokenizer,
                TextIteratorStreamer,
            )
        except ModuleNotFoundError:
            raise ModuleNotFoundError(
                "Please install 'transformers' extra to use 'TransformersModel': `pip install 'smolagents[transformers]'`"
            )

        if not model_id:
            warnings.warn(
                "The 'model_id' parameter will be required in version 2.0.0. "
                "Please update your code to pass this parameter to avoid future errors. "
                "For now, it defaults to 'HuggingFaceTB/SmolLM2-1.7B-Instruct'.",
                FutureWarning,
            )
            model_id = "HuggingFaceTB/SmolLM2-1.7B-Instruct"

        max_new_tokens = max_tokens if max_tokens is not None else max_new_tokens

        if device_map is None:
            device_map = "cuda" if torch.cuda.is_available() else "cpu"
        logger.info(f"Using device: {device_map}")
        self._is_vlm = False
        self.model_kwargs = model_kwargs or {}
        self.apply_chat_template_kwargs = apply_chat_template_kwargs or {}
        try:
            self.model = AutoModelForImageTextToText.from_pretrained(
                model_id,
                device_map=device_map,
                torch_dtype=torch_dtype,
                trust_remote_code=trust_remote_code,
                **self.model_kwargs,
            )
            self.processor = AutoProcessor.from_pretrained(model_id, trust_remote_code=trust_remote_code)
            self._is_vlm = True
            self.streamer = TextIteratorStreamer(self.processor.tokenizer, skip_prompt=True, skip_special_tokens=True)  # type: ignore

        except ValueError as e:
            if "Unrecognized configuration class" in str(e):
                self.model = AutoModelForCausalLM.from_pretrained(
                    model_id,
                    device_map=device_map,
                    torch_dtype=torch_dtype,
                    trust_remote_code=trust_remote_code,
                    **self.model_kwargs,
                )
                self.tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=trust_remote_code)
                self.streamer = TextIteratorStreamer(self.tokenizer, skip_prompt=True, skip_special_tokens=True)  # type: ignore
            else:
                raise e
        except Exception as e:
            raise ValueError(f"Failed to load tokenizer and model for {model_id=}: {e}") from e
        super().__init__(
            flatten_messages_as_text=not self._is_vlm, model_id=model_id, max_new_tokens=max_new_tokens, **kwargs
        )

    def make_stopping_criteria(self, stop_sequences: list[str], tokenizer) -> "StoppingCriteriaList":
        # [FUNCTION DOC]
        # Creates a transformers StoppingCriteriaList that stops when decoded output ends with any stop string.
        # Inputs: stop_sequences (list of str); tokenizer (used to decode tokens).
        # Outputs: StoppingCriteriaList containing one StopOnStrings instance.
        # Exceptions: ImportError if transformers is missing. Side effects: none.
        from transformers import StoppingCriteria, StoppingCriteriaList

        class StopOnStrings(StoppingCriteria):
            # [CLASS DOC]
            # HELPER: Stopping criterion accumulating decoded tokens into a text stream and stopping on stop strings.
            # Constructor args: stop_strings (list of str), tokenizer. Attributes: stop_strings, tokenizer, stream (str).
            # Exceptions/Side effects: none beyond its own state.
            def __init__(self, stop_strings: list[str], tokenizer):
                # [FUNCTION DOC]
                # HELPER: Stores stop strings and tokenizer and starts with an empty stream.
                # Inputs: stop_strings (list of str), tokenizer. Outputs: none. Exceptions: none. Side effects: sets attributes.
                self.stop_strings = stop_strings
                self.tokenizer = tokenizer
                self.stream = ""

            def reset(self):
                # [FUNCTION DOC]
                # HELPER: Clears the accumulated stream. Inputs: self. Outputs: none. Exceptions: none.
                # Side effects: resets self.stream to an empty string.
                self.stream = ""

            def __call__(self, input_ids, scores, **kwargs):
                # [FUNCTION DOC]
                # HELPER: Called by transformers after each generation step.
                # Inputs: input_ids (tensor batch of token ids), scores (unused), **kwargs (unused).
                # Outputs: True if the accumulated stream ends with any stop string, else False.
                # Exceptions: none raised directly.
                # Side effects: appends the decoded last token of the first sequence to self.stream.
                # KNOWN ISSUE: only the first batch row is checked, and a single boolean is returned rather than a per-row tensor.
                generated = self.tokenizer.decode(input_ids[0][-1], skip_special_tokens=True)
                self.stream += generated
                if any([self.stream.endswith(stop_string) for stop_string in self.stop_strings]):
                    return True
                return False

        return StoppingCriteriaList([StopOnStrings(stop_sequences, tokenizer)])

    def _prepare_completion_args(
        self,
        messages: list[ChatMessage | dict],
        stop_sequences: list[str] | None = None,
        tools_to_call_from: list[Tool] | None = None,
        **kwargs,
    ) -> dict[str, Any]:
        # [FUNCTION DOC]
        # HELPER: Builds kwargs for model.generate.
        # Behavior: prepares completion kwargs with tool_choice None; pops messages, stop and tools; resolves
        #   max_new_tokens from kwargs or self.kwargs ('max_new_tokens' then 'max_tokens', default 1024); applies the
        #   chat template (processor if present, else tokenizer) to get tensors on the model device; extracts input_ids;
        #   builds stopping criteria if stop sequences exist.
        # Inputs: messages, stop_sequences (list or None), tools_to_call_from (list or None), **kwargs.
        # Outputs: dict with inputs (prompt id tensor), use_cache True, stopping_criteria (or None), max_new_tokens and the
        #   remaining completion kwargs.
        # Exceptions: tokenizer/template errors propagate.
        # Side effects: moves tensors to the model device.
        # KNOWN ISSUE: a 'max_tokens' kwarg is consumed for max_new_tokens but is also left in the completion kwargs
        #   passed to model.generate.
        completion_kwargs = self._prepare_completion_kwargs(
            messages=messages,
            stop_sequences=stop_sequences,
            tools_to_call_from=tools_to_call_from,
            tool_choice=None,
            **kwargs,
        )

        messages = completion_kwargs.pop("messages")
        stop_sequences = completion_kwargs.pop("stop", None)
        tools = completion_kwargs.pop("tools", None)

        max_new_tokens = (
            kwargs.get("max_new_tokens")
            or kwargs.get("max_tokens")
            or self.kwargs.get("max_new_tokens")
            or self.kwargs.get("max_tokens")
            or 1024
        )
        prompt_tensor = (self.processor if hasattr(self, "processor") else self.tokenizer).apply_chat_template(
            messages,
            tools=tools,
            return_tensors="pt",
            add_generation_prompt=True,
            tokenize=True,
            return_dict=True,
            **self.apply_chat_template_kwargs,
        )
        prompt_tensor = prompt_tensor.to(self.model.device)  # type: ignore
        if hasattr(prompt_tensor, "input_ids"):
            prompt_tensor = prompt_tensor["input_ids"]

        model_tokenizer = self.processor.tokenizer if hasattr(self, "processor") else self.tokenizer
        stopping_criteria = (
            self.make_stopping_criteria(stop_sequences, tokenizer=model_tokenizer) if stop_sequences else None
        )
        completion_kwargs["max_new_tokens"] = max_new_tokens
        return dict(
            inputs=prompt_tensor,
            use_cache=True,
            stopping_criteria=stopping_criteria,
            **completion_kwargs,
        )

    def generate(
        self,
        messages: list[ChatMessage | dict],
        stop_sequences: list[str] | None = None,
        response_format: dict[str, str] | None = None,
        tools_to_call_from: list[Tool] | None = None,
        **kwargs,
    ) -> ChatMessage:
        # [FUNCTION DOC]
        # Runs non-streaming generation and decodes only the new tokens.
        # Inputs: messages, stop_sequences, response_format (must be None), tools_to_call_from, **kwargs.
        # Outputs: ChatMessage (assistant) with decoded text truncated at stop sequences, raw dict (completion kwargs
        #   without inputs) and token usage (prompt length, generated length).
        # Exceptions: ValueError if response_format is given; generation errors.
        # Side effects: local inference on the model device.
        if response_format is not None:
            raise ValueError("Transformers does not support structured outputs, use VLLMModel for this.")
        generation_kwargs = self._prepare_completion_args(
            messages=messages,
            stop_sequences=stop_sequences,
            tools_to_call_from=tools_to_call_from,
            **kwargs,
        )
        count_prompt_tokens = generation_kwargs["inputs"].shape[1]  # type: ignore
        out = self.model.generate(
            **generation_kwargs,
        )
        generated_tokens = out[0, count_prompt_tokens:]
        if hasattr(self, "processor"):
            output_text = self.processor.decode(generated_tokens, skip_special_tokens=True)
        else:
            output_text = self.tokenizer.decode(generated_tokens, skip_special_tokens=True)

        if stop_sequences is not None:
            output_text = remove_content_after_stop_sequences(output_text, stop_sequences)
        return ChatMessage(
            role=MessageRole.ASSISTANT,
            content=output_text,
            raw={
                "out": output_text,
                "completion_kwargs": {key: value for key, value in generation_kwargs.items() if key != "inputs"},
            },
            token_usage=TokenUsage(
                input_tokens=count_prompt_tokens,
                output_tokens=len(generated_tokens),
            ),
        )

    def generate_stream(
        self,
        messages: list[ChatMessage | dict],
        stop_sequences: list[str] | None = None,
        response_format: dict[str, str] | None = None,
        tools_to_call_from: list[Tool] | None = None,
        **kwargs,
    ) -> Generator[ChatMessageStreamDelta]:
        # [FUNCTION DOC]
        # Streams generation by running model.generate in a background thread with the shared TextIteratorStreamer.
        # Inputs: messages, stop_sequences, response_format (must be None), tools_to_call_from, **kwargs.
        # Outputs: yields ChatMessageStreamDelta per text chunk, each with output_tokens 1 and input token count only on
        #   the first chunk.
        # Exceptions: ValueError if response_format is given.
        # Side effects: starts and joins a thread; sets self._last_output_token_count.
        # KNOWN ISSUE: the streamer is a single instance shared across calls (not safe for concurrent streams); stop
        #   sequences are only enforced by the stopping criteria, the stop text itself is not stripped from the stream;
        #   chunk count is used as the token count.
        if response_format is not None:
            raise ValueError("Transformers does not support structured outputs, use VLLMModel for this.")
        generation_kwargs = self._prepare_completion_args(
            messages=messages,
            stop_sequences=stop_sequences,
            response_format=response_format,
            tools_to_call_from=tools_to_call_from,
            **kwargs,
        )

        # Get prompt token count once
        count_prompt_tokens = generation_kwargs["inputs"].shape[1]  # type: ignore

        # Start generation in a separate thread
        thread = Thread(target=self.model.generate, kwargs={"streamer": self.streamer, **generation_kwargs})
        thread.start()

        # Process streaming output
        is_first_token = True
        count_generated_tokens = 0
        for new_text in self.streamer:
            count_generated_tokens += 1
            # Only include input tokens in the first yielded token
            input_tokens = count_prompt_tokens if is_first_token else 0
            is_first_token = False
            yield ChatMessageStreamDelta(
                content=new_text,
                tool_calls=None,
                token_usage=TokenUsage(input_tokens=input_tokens, output_tokens=1),
            )
            count_prompt_tokens = 0
        thread.join()

        # Update final output token count
        self._last_output_token_count = count_generated_tokens


class ApiModel(Model):
    # [CLASS DOC]
    # Base class for API-backed models: manages client creation, role conversions, rate limiting and retries.
    # Constructor args: model_id (str), custom_role_conversions (dict or None), client (Any or None; created via
    #   create_client otherwise), requests_per_minute (float or None), retry (bool, default True), **kwargs.
    # Key attributes: custom_role_conversions, client, rate_limiter (RateLimiter), retryer (Retrying: up to
    #   RETRY_MAX_ATTEMPTS attempts on rate-limit errors with exponential backoff and jitter, reraising at the end).
    # Exceptions: NotImplementedError from create_client if a subclass does not implement it.
    # Side effects: creates an API client; retry logging at INFO level.
    """
    Base class for API-based language models.

    This class serves as a foundation for implementing models that interact with
    external APIs. It handles the common functionality for managing model IDs,
    custom role mappings, and API client connections.

    Parameters:
        model_id (`str`):
            The identifier for the model to be used with the API.
        custom_role_conversions (`dict[str, str`], **optional**):
            Mapping to convert  between internal role names and API-specific role names. Defaults to None.
        client (`Any`, **optional**):
            Pre-configured API client instance. If not provided, a default client will be created. Defaults to None.
        requests_per_minute (`float`, **optional**):
            Rate limit in requests per minute.
        retry (`bool`, **optional**):
            Whether to retry on rate limit errors, up to RETRY_MAX_ATTEMPTS times. Defaults to True.
        **kwargs:
            Additional keyword arguments to forward to the underlying model completion call.
    """

    def __init__(
        self,
        model_id: str,
        custom_role_conversions: dict[str, str] | None = None,
        client: Any | None = None,
        requests_per_minute: float | None = None,
        retry: bool = True,
        **kwargs,
    ):
        # [FUNCTION DOC]
        # Initializes base config, client, rate limiter and retryer.
        # Inputs: see class doc. Outputs: none.
        # Exceptions: those of create_client. Side effects: client creation.
        super().__init__(model_id=model_id, **kwargs)
        self.custom_role_conversions = custom_role_conversions or {}
        self.client = client or self.create_client()
        self.rate_limiter = RateLimiter(requests_per_minute)
        self.retryer = Retrying(
            max_attempts=RETRY_MAX_ATTEMPTS if retry else 1,
            wait_seconds=RETRY_WAIT,
            exponential_base=RETRY_EXPONENTIAL_BASE,
            jitter=RETRY_JITTER,
            retry_predicate=is_rate_limit_error,
            reraise=True,
            before_sleep_logger=(logger, logging.INFO),
            after_logger=(logger, logging.INFO),
        )

    def create_client(self):
        # [FUNCTION DOC]
        # Abstract factory for the API client. Inputs: self. Outputs: client (in subclasses).
        # Exceptions: always NotImplementedError here. Side effects: none.
        """Create the API client for the specific service."""
        raise NotImplementedError("Subclasses must implement this method to create a client")

    def _apply_rate_limit(self):
        # [FUNCTION DOC]
        # HELPER: Blocks as needed to respect requests_per_minute. Inputs: self. Outputs: none.
        # Exceptions: none raised directly. Side effects: may sleep.
        """Apply rate limiting before making API calls."""
        self.rate_limiter.throttle()


def is_rate_limit_error(exception: BaseException) -> bool:
    # [FUNCTION DOC]
    # HELPER: Retry predicate detecting rate-limit errors by message text.
    # Inputs: exception (BaseException).
    # Outputs: True if the lowercased message contains '429', 'rate limit', 'too many requests' or 'rate_limit'.
    # Exceptions: none. Side effects: none.
    """Check if the exception is a rate limit error."""
    error_str = str(exception).lower()
    return (
        "429" in error_str
        or "rate limit" in error_str
        or "too many requests" in error_str
        or "rate_limit" in error_str
    )


class LiteLLMModel(ApiModel):
    # [CLASS DOC]
    # API backend using the LiteLLM SDK to reach many providers.
    # Constructor args: model_id (str or None; defaults with FutureWarning to 'anthropic/claude-3-5-sonnet-20240620'),
    #   api_base (str or None), api_key (str or None), custom_role_conversions (dict or None),
    #   flatten_messages_as_text (bool or None; defaults True for ids starting with ollama/groq/cerebras), **kwargs
    #   (ApiModel/Model kwargs).
    # Key attributes: api_base, api_key, client (the litellm module).
    # Exceptions: ModuleNotFoundError if litellm is missing. Side effects: FutureWarning on missing model_id.
    """Model to use [LiteLLM Python SDK](https://docs.litellm.ai/docs/#litellm-python-sdk) to access hundreds of LLMs.

    Parameters:
        model_id (`str`):
            The model identifier to use on the server (e.g. "gpt-3.5-turbo").
        api_base (`str`, *optional*):
            The base URL of the provider API to call the model.
        api_key (`str`, *optional*):
            The API key to use for authentication.
        custom_role_conversions (`dict[str, str]`, *optional*):
            Custom role conversion mapping to convert message roles in others.
            Useful for specific models that do not support specific message roles like "system".
        flatten_messages_as_text (`bool`, *optional*): Whether to flatten messages as text.
            Defaults to `True` for models that start with "ollama", "groq", "cerebras".
        **kwargs:
            Additional keyword arguments to forward to the underlying LiteLLM completion call.
    """

    def __init__(
        self,
        model_id: str | None = None,
        api_base: str | None = None,
        api_key: str | None = None,
        custom_role_conversions: dict[str, str] | None = None,
        flatten_messages_as_text: bool | None = None,
        **kwargs,
    ):
        # [FUNCTION DOC]
        # Resolves defaults and initializes the ApiModel.
        # Inputs: see class doc. Outputs: none. Exceptions: those of create_client.
        # Side effects: may emit FutureWarning.
        if not model_id:
            warnings.warn(
                "The 'model_id' parameter will be required in version 2.0.0. "
                "Please update your code to pass this parameter to avoid future errors. "
                "For now, it defaults to 'anthropic/claude-3-5-sonnet-20240620'.",
                FutureWarning,
            )
            model_id = "anthropic/claude-3-5-sonnet-20240620"
        self.api_base = api_base
        self.api_key = api_key
        flatten_messages_as_text = (
            flatten_messages_as_text
            if flatten_messages_as_text is not None
            else model_id.startswith(("ollama", "groq", "cerebras"))
        )
        super().__init__(
            model_id=model_id,
            custom_role_conversions=custom_role_conversions,
            flatten_messages_as_text=flatten_messages_as_text,
            **kwargs,
        )

    def create_client(self):
        # [FUNCTION DOC]
        # Imports and returns the litellm module as the client.
        # Inputs: self. Outputs: litellm module. Exceptions: ModuleNotFoundError with install hint. Side effects: import.
        """Create the LiteLLM client."""
        try:
            import litellm
        except ModuleNotFoundError as e:
            raise ModuleNotFoundError(
                "Please install 'litellm' extra to use LiteLLMModel: `pip install 'smolagents[litellm]'`"
            ) from e

        return litellm

    def generate(
        self,
        messages: list[ChatMessage | dict],
        stop_sequences: list[str] | None = None,
        response_format: dict[str, str] | None = None,
        tools_to_call_from: list[Tool] | None = None,
        **kwargs,
    ) -> ChatMessage:
        # [FUNCTION DOC]
        # Calls litellm completion (rate-limited, retried) and wraps the first choice.
        # Inputs: messages, stop_sequences, response_format, tools_to_call_from, **kwargs (images converted to URLs).
        # Outputs: ChatMessage with provider role, content (post-truncated on stop sequences if stop unsupported), tool
        #   calls, raw response and token usage.
        # Exceptions: RuntimeError if the response has no choices; provider/network errors after retries.
        # Side effects: network request; may sleep for rate limiting/retries.
        completion_kwargs = self._prepare_completion_kwargs(
            messages=messages,
            stop_sequences=stop_sequences,
            response_format=response_format,
            tools_to_call_from=tools_to_call_from,
            model=self.model_id,
            api_base=self.api_base,
            api_key=self.api_key,
            convert_images_to_image_urls=True,
            custom_role_conversions=self.custom_role_conversions,
            **kwargs,
        )
        self._apply_rate_limit()
        response = self.retryer(self.client.completion, **completion_kwargs)

        if not response.choices:
            raise RuntimeError(
                f"Unexpected API response: model '{self.model_id}' returned no choices. "
                " This may indicate a possible API or upstream issue. "
                f"Response details: {response.model_dump()}"
            )
        content = response.choices[0].message.content
        if stop_sequences is not None and not self.supports_stop_parameter:
            content = remove_content_after_stop_sequences(content, stop_sequences)
        return ChatMessage(
            role=response.choices[0].message.role,
            content=content,
            tool_calls=response.choices[0].message.tool_calls,
            raw=response,
            token_usage=TokenUsage(
                input_tokens=response.usage.prompt_tokens,
                output_tokens=response.usage.completion_tokens,
            ),
        )

    def generate_stream(
        self,
        messages: list[ChatMessage | dict],
        stop_sequences: list[str] | None = None,
        response_format: dict[str, str] | None = None,
        tools_to_call_from: list[Tool] | None = None,
        **kwargs,
    ) -> Generator[ChatMessageStreamDelta]:
        # [FUNCTION DOC]
        # Streams a litellm completion with usage included.
        # Inputs: same as generate.
        # Outputs: yields ChatMessageStreamDelta for usage events (empty content plus TokenUsage) and for each choice
        #   delta (content and tool-call deltas).
        # Exceptions: ValueError if a chunk has a choice without delta and without finish_reason; provider errors.
        # Side effects: network streaming request; rate limiting.
        # KNOWN ISSUE: retries only wrap stream creation, not iteration; stop sequences are not post-truncated in
        #   streaming mode for models that do not support stop.
        completion_kwargs = self._prepare_completion_kwargs(
            messages=messages,
            stop_sequences=stop_sequences,
            response_format=response_format,
            tools_to_call_from=tools_to_call_from,
            model=self.model_id,
            api_base=self.api_base,
            api_key=self.api_key,
            custom_role_conversions=self.custom_role_conversions,
            convert_images_to_image_urls=True,
            **kwargs,
        )
        self._apply_rate_limit()
        for event in self.retryer(
            self.client.completion, **completion_kwargs, stream=True, stream_options={"include_usage": True}
        ):
            if getattr(event, "usage", None):
                yield ChatMessageStreamDelta(
                    content="",
                    token_usage=TokenUsage(
                        input_tokens=event.usage.prompt_tokens,
                        output_tokens=event.usage.completion_tokens,
                    ),
                )
            if event.choices:
                choice = event.choices[0]
                if choice.delta:
                    yield ChatMessageStreamDelta(
                        content=choice.delta.content,
                        tool_calls=[
                            ChatMessageToolCallStreamDelta(
                                index=delta.index,
                                id=delta.id,
                                type=delta.type,
                                function=delta.function,
                            )
                            for delta in choice.delta.tool_calls
                        ]
                        if choice.delta.tool_calls
                        else None,
                    )
                else:
                    if not getattr(choice, "finish_reason", None):
                        raise ValueError(f"No content or tool calls in event: {event}")


class LiteLLMRouterModel(LiteLLMModel):
    # [CLASS DOC]
    # LiteLLM backend routing requests across a model list via litellm.router.Router.
    # Constructor args: model_id (str: model group name), model_list (list of dicts: router deployments),
    #   client_kwargs (dict or None: extra Router config), custom_role_conversions (dict or None),
    #   flatten_messages_as_text (bool or None), **kwargs.
    # Key attributes: client_kwargs (model_list merged with client_kwargs), client (Router).
    # Exceptions: ModuleNotFoundError if litellm is missing. Side effects: creates the Router.
    """Router‑based client for interacting with the [LiteLLM Python SDK Router](https://docs.litellm.ai/docs/routing).

    This class provides a high-level interface for distributing requests among multiple language models using
    the LiteLLM SDK's routing capabilities. It is responsible for initializing and configuring the router client,
    applying custom role conversions, and managing message formatting to ensure seamless integration with various LLMs.

    Parameters:
        model_id (`str`):
            Identifier for the model group to use from the model list (e.g., "model-group-1").
        model_list (`list[dict[str, Any]]`):
            Model configurations to be used for routing.
            Each configuration should include the model group name and any necessary parameters.
            For more details, refer to the [LiteLLM Routing](https://docs.litellm.ai/docs/routing#quick-start) documentation.
        client_kwargs (`dict[str, Any]`, *optional*):
            Additional configuration parameters for the Router client. For more details, see the
            [LiteLLM Routing Configurations](https://docs.litellm.ai/docs/routing).
        custom_role_conversions (`dict[str, str]`, *optional*):
            Custom role conversion mapping to convert message roles in others.
            Useful for specific models that do not support specific message roles like "system".
        flatten_messages_as_text (`bool`, *optional*): Whether to flatten messages as text.
            Defaults to `True` for models that start with "ollama", "groq", "cerebras".
        **kwargs:
            Additional keyword arguments to forward to the underlying LiteLLM Router completion call.

    Example:
    ```python
    >>> import os
    >>> from smolagents import CodeAgent, WebSearchTool, LiteLLMRouterModel
    >>> os.environ["OPENAI_API_KEY"] = ""
    >>> os.environ["AWS_ACCESS_KEY_ID"] = ""
    >>> os.environ["AWS_SECRET_ACCESS_KEY"] = ""
    >>> os.environ["AWS_REGION"] = ""
    >>> llm_loadbalancer_model_list = [
    ...     {
    ...         "model_name": "model-group-1",
    ...         "litellm_params": {
    ...             "model": "gpt-4o-mini",
    ...             "api_key": os.getenv("OPENAI_API_KEY"),
    ...         },
    ...     },
    ...     {
    ...         "model_name": "model-group-1",
    ...         "litellm_params": {
    ...             "model": "bedrock/anthropic.claude-3-sonnet-20240229-v1:0",
    ...             "aws_access_key_id": os.getenv("AWS_ACCESS_KEY_ID"),
    ...             "aws_secret_access_key": os.getenv("AWS_SECRET_ACCESS_KEY"),
    ...             "aws_region_name": os.getenv("AWS_REGION"),
    ...         },
    ...     },
    >>> ]
    >>> model = LiteLLMRouterModel(
    ...    model_id="model-group-1",
    ...    model_list=llm_loadbalancer_model_list,
    ...    client_kwargs={
    ...        "routing_strategy":"simple-shuffle"
    ...    }
    >>> )
    >>> agent = CodeAgent(tools=[WebSearchTool()], model=model)
    >>> agent.run("How many seconds would it take for a leopard at full speed to run through Pont des Arts?")
    ```
    """

    def __init__(
        self,
        model_id: str,
        model_list: list[dict[str, Any]],
        client_kwargs: dict[str, Any] | None = None,
        custom_role_conversions: dict[str, str] | None = None,
        flatten_messages_as_text: bool | None = None,
        **kwargs,
    ):
        # [FUNCTION DOC]
        # Stores router kwargs, then delegates to LiteLLMModel init (which creates the Router).
        # Inputs: see class doc. Outputs: none. Exceptions: those of create_client. Side effects: client creation.
        self.client_kwargs = {
            "model_list": model_list,
            **(client_kwargs or {}),
        }
        super().__init__(
            model_id=model_id,
            custom_role_conversions=custom_role_conversions,
            flatten_messages_as_text=flatten_messages_as_text,
            **kwargs,
        )

    def create_client(self):
        # [FUNCTION DOC]
        # Builds a litellm Router from self.client_kwargs.
        # Inputs: self. Outputs: Router. Exceptions: ModuleNotFoundError; Router config errors. Side effects: none beyond
        #   Router initialization.
        try:
            from litellm.router import Router
        except ModuleNotFoundError as e:
            raise ModuleNotFoundError(
                "Please install 'litellm' extra to use LiteLLMRouterModel: `pip install 'smolagents[litellm]'`"
            ) from e
        return Router(**self.client_kwargs)


class InferenceClientModel(ApiModel):
    # [CLASS DOC]
    # API backend for Hugging Face Inference Providers via huggingface_hub.InferenceClient.
    # Constructor args: model_id (str, default 'Qwen/Qwen3-Next-80B-A3B-Thinking'), provider (str or None), token (str
    #   or None; falls back to api_key then HF_TOKEN env var), timeout (int, default 120), client_kwargs (dict or None),
    #   custom_role_conversions (dict or None), api_key (alias of token), bill_to (str or None), base_url (str or
    #   None), **kwargs.
    # Key attributes: client_kwargs (model/provider/token/timeout/bill_to/base_url override user client_kwargs),
    #   client (InferenceClient).
    # Exceptions: ValueError if both token and api_key are given. Side effects: reads HF_TOKEN env var.
    """A class to interact with Hugging Face's Inference Providers for language model interaction.

    This model allows you to communicate with Hugging Face's models using Inference Providers. It can be used in both serverless mode, with a dedicated endpoint, or even with a local URL, supporting features like stop sequences and grammar customization.

    Providers include Cerebras, Cohere, Fal, Fireworks, HF-Inference, Hyperbolic, Nebius, Novita, Replicate, SambaNova, Together, and more.

    Parameters:
        model_id (`str`, *optional*, default `"Qwen/Qwen3-Next-80B-A3B-Thinking"`):
            The Hugging Face model ID to be used for inference.
            This can be a model identifier from the Hugging Face model hub or a URL to a deployed Inference Endpoint.
            Currently, it defaults to `"Qwen/Qwen3-Next-80B-A3B-Thinking"`, but this may change in the future.
        provider (`str`, *optional*):
            Name of the provider to use for inference. A list of supported providers can be found in the [Inference Providers documentation](https://huggingface.co/docs/inference-providers/index#partners).
            Defaults to "auto" i.e. the first of the providers available for the model, sorted by the user's order [here](https://hf.co/settings/inference-providers).
            If `base_url` is passed, then `provider` is not used.
        token (`str`, *optional*):
            Token used by the Hugging Face API for authentication. This token need to be authorized 'Make calls to the serverless Inference Providers'.
            If the model is gated (like Llama-3 models), the token also needs 'Read access to contents of all public gated repos you can access'.
            If not provided, the class will try to use environment variable 'HF_TOKEN', else use the token stored in the Hugging Face CLI configuration.
        timeout (`int`, *optional*, defaults to 120):
            Timeout for the API request, in seconds.
        client_kwargs (`dict[str, Any]`, *optional*):
            Additional keyword arguments to pass to the Hugging Face InferenceClient.
        custom_role_conversions (`dict[str, str]`, *optional*):
            Custom role conversion mapping to convert message roles in others.
            Useful for specific models that do not support specific message roles like "system".
        api_key (`str`, *optional*):
            Token to use for authentication. This is a duplicated argument from `token` to make [`InferenceClientModel`]
            follow the same pattern as `openai.OpenAI` client. Cannot be used if `token` is set. Defaults to None.
        bill_to (`str`, *optional*):
            The billing account to use for the requests. By default the requests are billed on the user's account. Requests can only be billed to
            an organization the user is a member of, and which has subscribed to Enterprise Hub.
        base_url (`str`, `optional`):
            Base URL to run inference. This is a duplicated argument from `model` to make [`InferenceClientModel`]
            follow the same pattern as `openai.OpenAI` client. Cannot be used if `model` is set. Defaults to None.
        **kwargs:
            Additional keyword arguments to forward to the underlying Hugging Face InferenceClient completion call.

    Raises:
        ValueError:
            If the model name is not provided.

    Example:
    ```python
    >>> engine = InferenceClientModel(
    ...     model_id="Qwen/Qwen3-Next-80B-A3B-Thinking",
    ...     provider="hyperbolic",
    ...     token="your_hf_token_here",
    ...     max_tokens=5000,
    ... )
    >>> messages = [{"role": "user", "content": "Explain quantum mechanics in simple terms."}]
    >>> response = engine(messages, stop_sequences=["END"])
    >>> print(response)
    "Quantum mechanics is the branch of physics that studies..."
    ```
    """

    def __init__(
        self,
        model_id: str = "Qwen/Qwen3-Next-80B-A3B-Thinking",
        provider: str | None = None,
        token: str | None = None,
        timeout: int = 120,
        client_kwargs: dict[str, Any] | None = None,
        custom_role_conversions: dict[str, str] | None = None,
        api_key: str | None = None,
        bill_to: str | None = None,
        base_url: str | None = None,
        **kwargs,
    ):
        # [FUNCTION DOC]
        # Resolves the token, builds client kwargs and initializes the ApiModel.
        # Inputs: see class doc. Outputs: none. Exceptions: ValueError for token+api_key.
        # Side effects: reads environment variable HF_TOKEN; creates the client.
        if token is not None and api_key is not None:
            raise ValueError(
                "Received both `token` and `api_key` arguments. Please provide only one of them."
                " `api_key` is an alias for `token` to make the API compatible with OpenAI's client."
                " It has the exact same behavior as `token`."
            )
        token = token if token is not None else api_key
        if token is None:
            token = os.getenv("HF_TOKEN")
        self.client_kwargs = {
            **(client_kwargs or {}),
            "model": model_id,
            "provider": provider,
            "token": token,
            "timeout": timeout,
            "bill_to": bill_to,
            "base_url": base_url,
        }
        super().__init__(model_id=model_id, custom_role_conversions=custom_role_conversions, **kwargs)

    def create_client(self):
        # [FUNCTION DOC]
        # Creates an InferenceClient with self.client_kwargs.
        # Inputs: self. Outputs: InferenceClient. Exceptions: ImportError if huggingface_hub is missing.
        # Side effects: none.
        """Create the Hugging Face client."""
        from huggingface_hub import InferenceClient

        return InferenceClient(**self.client_kwargs)

    def generate(
        self,
        messages: list[ChatMessage | dict],
        stop_sequences: list[str] | None = None,
        response_format: dict[str, str] | None = None,
        tools_to_call_from: list[Tool] | None = None,
        **kwargs,
    ) -> ChatMessage:
        # [FUNCTION DOC]
        # Calls chat_completion (rate-limited, retried) and wraps the first choice.
        # Inputs: messages, stop_sequences, response_format (only allowed for STRUCTURED_GENERATION_PROVIDERS),
        #   tools_to_call_from, **kwargs.
        # Outputs: ChatMessage with role, content (post-truncated if stop unsupported), tool calls, raw response, usage.
        # Exceptions: ValueError if response_format is used with another provider; API errors after retries.
        # Side effects: network request.
        # KNOWN ISSUE: response_format is validated but never passed to the API (the argument is commented out), so
        #   structured output is silently ignored even for allowed providers.
        if response_format is not None and self.client_kwargs["provider"] not in STRUCTURED_GENERATION_PROVIDERS:
            raise ValueError(
                "InferenceClientModel only supports structured outputs with these providers:"
                + ", ".join(STRUCTURED_GENERATION_PROVIDERS)
            )
        completion_kwargs = self._prepare_completion_kwargs(
            messages=messages,
            stop_sequences=stop_sequences,
            tools_to_call_from=tools_to_call_from,
            # response_format=response_format,
            convert_images_to_image_urls=True,
            custom_role_conversions=self.custom_role_conversions,
            **kwargs,
        )
        self._apply_rate_limit()
        response = self.retryer(self.client.chat_completion, **completion_kwargs)
        content = response.choices[0].message.content
        if stop_sequences is not None and not self.supports_stop_parameter:
            content = remove_content_after_stop_sequences(content, stop_sequences)
        return ChatMessage(
            role=response.choices[0].message.role,
            content=content,
            tool_calls=response.choices[0].message.tool_calls,
            raw=response,
            token_usage=TokenUsage(
                input_tokens=response.usage.prompt_tokens,
                output_tokens=response.usage.completion_tokens,
            ),
        )

    def generate_stream(
        self,
        messages: list[ChatMessage | dict],
        stop_sequences: list[str] | None = None,
        response_format: dict[str, str] | None = None,
        tools_to_call_from: list[Tool] | None = None,
        **kwargs,
    ) -> Generator[ChatMessageStreamDelta]:
        # [FUNCTION DOC]
        # Streams via client.chat.completions.create with usage included.
        # Inputs: messages, stop_sequences, response_format, tools_to_call_from, **kwargs (model id passed explicitly).
        # Outputs: yields ChatMessageStreamDelta for usage events and choice deltas.
        # Exceptions: ValueError for a choice with no delta and no finish_reason; API errors.
        # Side effects: network streaming request.
        # KNOWN ISSUE: unlike generate, it forwards response_format without the provider check and passes model
        #   explicitly; behavior differs between the two paths.
        completion_kwargs = self._prepare_completion_kwargs(
            messages=messages,
            stop_sequences=stop_sequences,
            response_format=response_format,
            tools_to_call_from=tools_to_call_from,
            model=self.model_id,
            custom_role_conversions=self.custom_role_conversions,
            convert_images_to_image_urls=True,
            **kwargs,
        )
        self._apply_rate_limit()
        for event in self.retryer(
            self.client.chat.completions.create,
            **completion_kwargs,
            stream=True,
            stream_options={"include_usage": True},
        ):
            if getattr(event, "usage", None):
                yield ChatMessageStreamDelta(
                    content="",
                    token_usage=TokenUsage(
                        input_tokens=event.usage.prompt_tokens,
                        output_tokens=event.usage.completion_tokens,
                    ),
                )
            if event.choices:
                choice = event.choices[0]
                if choice.delta:
                    yield ChatMessageStreamDelta(
                        content=choice.delta.content,
                        tool_calls=[
                            ChatMessageToolCallStreamDelta(
                                index=delta.index,
                                id=delta.id,
                                type=delta.type,
                                function=delta.function,
                            )
                            for delta in choice.delta.tool_calls
                        ]
                        if choice.delta.tool_calls
                        else None,
                    )
                else:
                    if not getattr(choice, "finish_reason", None):
                        raise ValueError(f"No content or tool calls in event: {event}")


class OpenAIModel(ApiModel):
    # [CLASS DOC]
    # API backend for any OpenAI-compatible server using the openai SDK.
    # Constructor args: model_id (str), api_base (str or None, used as base_url), api_key, organization, project (str
    #   or None), client_kwargs (dict or None; explicit args override it), custom_role_conversions (dict or None),
    #   flatten_messages_as_text (bool, default False), **kwargs (e.g. temperature).
    # Key attributes: client_kwargs, client (openai.OpenAI).
    # Exceptions: ModuleNotFoundError if openai is missing. Side effects: client creation.
    """This model connects to an OpenAI-compatible API server.

    Parameters:
        model_id (`str`):
            The model identifier to use on the server (e.g. "gpt-5").
        api_base (`str`, *optional*):
            The base URL of the OpenAI-compatible API server.
        api_key (`str`, *optional*):
            The API key to use for authentication.
        organization (`str`, *optional*):
            The organization to use for the API request.
        project (`str`, *optional*):
            The project to use for the API request.
        client_kwargs (`dict[str, Any]`, *optional*):
            Additional keyword arguments to pass to the OpenAI client (like organization, project, max_retries etc.).
        custom_role_conversions (`dict[str, str]`, *optional*):
            Custom role conversion mapping to convert message roles in others.
            Useful for specific models that do not support specific message roles like "system".
        flatten_messages_as_text (`bool`, default `False`):
            Whether to flatten messages as text.
        **kwargs:
            Additional keyword arguments to forward to the underlying OpenAI API completion call, for instance `temperature`.
    """

    def __init__(
        self,
        model_id: str,
        api_base: str | None = None,
        api_key: str | None = None,
        organization: str | None = None,
        project: str | None = None,
        client_kwargs: dict[str, Any] | None = None,
        custom_role_conversions: dict[str, str] | None = None,
        flatten_messages_as_text: bool = False,
        **kwargs,
    ):
        # [FUNCTION DOC]
        # Builds client kwargs and initializes the ApiModel.
        # Inputs: see class doc. Outputs: none. Exceptions: those of create_client. Side effects: client creation.
        self.client_kwargs = {
            **(client_kwargs or {}),
            "api_key": api_key,
            "base_url": api_base,
            "organization": organization,
            "project": project,
        }
        super().__init__(
            model_id=model_id,
            custom_role_conversions=custom_role_conversions,
            flatten_messages_as_text=flatten_messages_as_text,
            **kwargs,
        )

    def create_client(self):
        # [FUNCTION DOC]
        # Creates openai.OpenAI(**client_kwargs).
        # Inputs: self. Outputs: OpenAI client. Exceptions: ModuleNotFoundError with install hint. Side effects: none.
        try:
            import openai
        except ModuleNotFoundError as e:
            raise ModuleNotFoundError(
                "Please install 'openai' extra to use OpenAIModel: `pip install 'smolagents[openai]'`"
            ) from e

        return openai.OpenAI(**self.client_kwargs)

    def generate_stream(
        self,
        messages: list[ChatMessage | dict],
        stop_sequences: list[str] | None = None,
        response_format: dict[str, str] | None = None,
        tools_to_call_from: list[Tool] | None = None,
        **kwargs,
    ) -> Generator[ChatMessageStreamDelta]:
        # [FUNCTION DOC]
        # Streams chat completions with usage included.
        # Inputs: messages, stop_sequences, response_format, tools_to_call_from, **kwargs.
        # Outputs: yields ChatMessageStreamDelta for usage events and for each choice delta.
        # Exceptions: ValueError for a choice with no delta and no finish_reason; API errors.
        # Side effects: network streaming request; rate limiting.
        completion_kwargs = self._prepare_completion_kwargs(
            messages=messages,
            stop_sequences=stop_sequences,
            response_format=response_format,
            tools_to_call_from=tools_to_call_from,
            model=self.model_id,
            custom_role_conversions=self.custom_role_conversions,
            convert_images_to_image_urls=True,
            **kwargs,
        )
        self._apply_rate_limit()
        for event in self.retryer(
            self.client.chat.completions.create,
            **completion_kwargs,
            stream=True,
            stream_options={"include_usage": True},
        ):
            if event.usage:
                yield ChatMessageStreamDelta(
                    content="",
                    token_usage=TokenUsage(
                        input_tokens=event.usage.prompt_tokens,
                        output_tokens=event.usage.completion_tokens,
                    ),
                )
            if event.choices:
                choice = event.choices[0]
                if choice.delta:
                    yield ChatMessageStreamDelta(
                        content=choice.delta.content,
                        tool_calls=[
                            ChatMessageToolCallStreamDelta(
                                index=delta.index,
                                id=delta.id,
                                type=delta.type,
                                function=delta.function,
                            )
                            for delta in choice.delta.tool_calls
                        ]
                        if choice.delta.tool_calls
                        else None,
                    )
                else:
                    if not getattr(choice, "finish_reason", None):
                        raise ValueError(f"No content or tool calls in event: {event}")

    def generate(
        self,
        messages: list[ChatMessage | dict],
        stop_sequences: list[str] | None = None,
        response_format: dict[str, str] | None = None,
        tools_to_call_from: list[Tool] | None = None,
        **kwargs,
    ) -> ChatMessage:
        # [FUNCTION DOC]
        # Calls chat.completions.create (rate-limited, retried) and wraps the first choice.
        # Inputs: messages, stop_sequences, response_format, tools_to_call_from, **kwargs.
        # Outputs: ChatMessage with role, content (post-truncated if stop unsupported), tool calls, raw, token usage.
        # Exceptions: API/network errors after retries.
        # Side effects: network request.
        completion_kwargs = self._prepare_completion_kwargs(
            messages=messages,
            stop_sequences=stop_sequences,
            response_format=response_format,
            tools_to_call_from=tools_to_call_from,
            model=self.model_id,
            custom_role_conversions=self.custom_role_conversions,
            convert_images_to_image_urls=True,
            **kwargs,
        )
        self._apply_rate_limit()
        response = self.retryer(self.client.chat.completions.create, **completion_kwargs)
        content = response.choices[0].message.content
        if stop_sequences is not None and not self.supports_stop_parameter:
            content = remove_content_after_stop_sequences(content, stop_sequences)
        return ChatMessage(
            role=response.choices[0].message.role,
            content=content,
            tool_calls=response.choices[0].message.tool_calls,
            raw=response,
            token_usage=TokenUsage(
                input_tokens=response.usage.prompt_tokens,
                output_tokens=response.usage.completion_tokens,
            ),
        )


OpenAIServerModel = OpenAIModel


class AzureOpenAIModel(OpenAIModel):
    # [CLASS DOC]
    # OpenAIModel variant for Azure OpenAI deployments.
    # Constructor args: model_id (str: deployment name), azure_endpoint, api_key, api_version (str or None; openai SDK
    #   falls back to AZURE_OPENAI_ENDPOINT / AZURE_OPENAI_API_KEY / OPENAI_API_VERSION env vars), client_kwargs (dict
    #   or None), custom_role_conversions (dict or None), **kwargs.
    # Key attributes: client (openai.AzureOpenAI).
    # Exceptions: ModuleNotFoundError if openai is missing. Side effects: client creation.
    """This model connects to an Azure OpenAI deployment.

    Parameters:
        model_id (`str`):
            The model deployment name to use when connecting (e.g. "gpt-4o-mini").
        azure_endpoint (`str`, *optional*):
            The Azure endpoint, including the resource, e.g. `https://example-resource.azure.openai.com/`. If not provided, it will be inferred from the `AZURE_OPENAI_ENDPOINT` environment variable.
        api_key (`str`, *optional*):
            The API key to use for authentication. If not provided, it will be inferred from the `AZURE_OPENAI_API_KEY` environment variable.
        api_version (`str`, *optional*):
            The API version to use. If not provided, it will be inferred from the `OPENAI_API_VERSION` environment variable.
        client_kwargs (`dict[str, Any]`, *optional*):
            Additional keyword arguments to pass to the AzureOpenAI client (like organization, project, max_retries etc.).
        custom_role_conversions (`dict[str, str]`, *optional*):
            Custom role conversion mapping to convert message roles in others.
            Useful for specific models that do not support specific message roles like "system".
        **kwargs:
            Additional keyword arguments to forward to the underlying Azure OpenAI API completion call.
    """

    def __init__(
        self,
        model_id: str,
        azure_endpoint: str | None = None,
        api_key: str | None = None,
        api_version: str | None = None,
        client_kwargs: dict[str, Any] | None = None,
        custom_role_conversions: dict[str, str] | None = None,
        **kwargs,
    ):
        # [FUNCTION DOC]
        # Adds api_version and azure_endpoint to client_kwargs and delegates to OpenAIModel.
        # Inputs: see class doc. Outputs: none. Exceptions: those of create_client.
        # Side effects: mutates the caller-supplied client_kwargs dict.
        # KNOWN ISSUE: the caller's client_kwargs dict is updated in place.
        client_kwargs = client_kwargs or {}
        client_kwargs.update(
            {
                "api_version": api_version,
                "azure_endpoint": azure_endpoint,
            }
        )
        super().__init__(
            model_id=model_id,
            api_key=api_key,
            client_kwargs=client_kwargs,
            custom_role_conversions=custom_role_conversions,
            **kwargs,
        )

    def create_client(self):
        # [FUNCTION DOC]
        # Creates openai.AzureOpenAI(**client_kwargs).
        # Inputs: self. Outputs: AzureOpenAI client. Exceptions: ModuleNotFoundError. Side effects: may read env vars
        #   inside the SDK.
        try:
            import openai
        except ModuleNotFoundError as e:
            raise ModuleNotFoundError(
                "Please install 'openai' extra to use AzureOpenAIModel: `pip install 'smolagents[openai]'`"
            ) from e

        return openai.AzureOpenAI(**self.client_kwargs)


AzureOpenAIServerModel = AzureOpenAIModel


class AmazonBedrockModel(ApiModel):
    # [CLASS DOC]
    # API backend for Amazon Bedrock's Converse API via boto3.
    # Constructor args: model_id (str), client (boto3 client or None), client_kwargs (dict or None: for
    #   boto3.client('bedrock-runtime')), custom_role_conversions (dict or None; default maps system, assistant,
    #   tool-call and tool-response all to user), **kwargs (e.g. inferenceConfig, guardrailConfig).
    # Key attributes: client_kwargs, client. Messages are never flattened.
    # Exceptions: ModuleNotFoundError if boto3 is missing. Side effects: client creation (AWS credentials lookup).
    """
    A model class for interacting with Amazon Bedrock Server models through the Bedrock API.

    This class provides an interface to interact with various Bedrock language models,
    allowing for customized model inference, guardrail configuration, message handling,
    and other parameters allowed by boto3 API.

    Authentication:

    Amazon Bedrock supports multiple authentication methods:
    - Default AWS credentials:
       Use the default AWS credential chain (e.g., IAM roles, IAM users).
    - API Key Authentication (requires `boto3 >= 1.39.0`):
       Set the API key using the `AWS_BEARER_TOKEN_BEDROCK` environment variable.

    > [!TIP]
    > API key support requires `boto3 >= 1.39.0`.
    > For users not relying on API key authentication, the minimum supported version is `boto3 >= 1.36.18`.

    Parameters:
        model_id (`str`):
            The model identifier to use on Bedrock (e.g. "us.amazon.nova-pro-v1:0").
        client (`boto3.client`, *optional*):
            A custom boto3 client for AWS interactions. If not provided, a default client will be created.
        client_kwargs (dict[str, Any], *optional*):
            Keyword arguments used to configure the boto3 client if it needs to be created internally.
            Examples include `region_name`, `config`, or `endpoint_url`.
        custom_role_conversions (`dict[str, str]`, *optional*):
            Custom role conversion mapping to convert message roles in others.
            Useful for specific models that do not support specific message roles like "system".
            Defaults to converting all roles to "user" role to enable using all the Bedrock models.
        flatten_messages_as_text (`bool`, default `False`):
            Whether to flatten messages as text.
        **kwargs:
            Additional keyword arguments to forward to the underlying Amazon Bedrock model converse call.

    Examples:
        Creating a model instance with default settings:
        ```python
        >>> bedrock_model = AmazonBedrockModel(
        ...     model_id='us.amazon.nova-pro-v1:0'
        ... )
        ```

        Creating a model instance with a custom boto3 client:
        ```python
        >>> import boto3
        >>> client = boto3.client('bedrock-runtime', region_name='us-west-2')
        >>> bedrock_model = AmazonBedrockModel(
        ...     model_id='us.amazon.nova-pro-v1:0',
        ...     client=client
        ... )
        ```

        Creating a model instance with client_kwargs for internal client creation:
        ```python
        >>> bedrock_model = AmazonBedrockModel(
        ...     model_id='us.amazon.nova-pro-v1:0',
        ...     client_kwargs={'region_name': 'us-west-2', 'endpoint_url': 'https://custom-endpoint.com'}
        ... )
        ```

        Creating a model instance with inference and guardrail configurations:
        ```python
        >>> additional_api_config = {
        ...     "inferenceConfig": {
        ...         "maxTokens": 3000
        ...     },
        ...     "guardrailConfig": {
        ...         "guardrailIdentifier": "identify1",
        ...         "guardrailVersion": 'v1'
        ...     },
        ... }
        >>> bedrock_model = AmazonBedrockModel(
        ...     model_id='anthropic.claude-3-haiku-20240307-v1:0',
        ...     **additional_api_config
        ... )
        ```
    """

    def __init__(
        self,
        model_id: str,
        client=None,
        client_kwargs: dict[str, Any] | None = None,
        custom_role_conversions: dict[str, str] | None = None,
        **kwargs,
    ):
        # [FUNCTION DOC]
        # Sets client kwargs and default role conversions, then initializes the ApiModel with flattening disabled.
        # Inputs: see class doc. Outputs: none. Exceptions: those of create_client. Side effects: client creation.
        self.client_kwargs = client_kwargs or {}

        # Bedrock only supports `assistant` and `user` roles.
        # Many Bedrock models do not allow conversations to start with the `assistant` role, so the default is set to `user/user`.
        # This parameter is retained for future model implementations and extended support.
        custom_role_conversions = custom_role_conversions or {
            MessageRole.SYSTEM: MessageRole.USER,
            MessageRole.ASSISTANT: MessageRole.USER,
            MessageRole.TOOL_CALL: MessageRole.USER,
            MessageRole.TOOL_RESPONSE: MessageRole.USER,
        }

        super().__init__(
            model_id=model_id,
            custom_role_conversions=custom_role_conversions,
            flatten_messages_as_text=False,  # Bedrock API doesn't support flatten messages, must be a list of messages
            client=client,
            **kwargs,
        )

    def _prepare_completion_kwargs(
        self,
        messages: list[ChatMessage | dict],
        stop_sequences: list[str] | None = None,
        response_format: dict[str, str] | None = None,
        tools_to_call_from: list[Tool] | None = None,
        custom_role_conversions: dict[str, str] | None = None,
        convert_images_to_image_urls: bool = False,
        tool_choice: str | dict[Any, Any] | None = None,
        **kwargs,
    ) -> dict:
        # [FUNCTION DOC]
        # HELPER: Bedrock-specific kwargs: calls the base preparation without stop sequences or response_format, removes
        #   'toolConfig', strips the 'type' key from every message content part, and adds 'modelId'.
        # Inputs: messages, stop_sequences (ignored), response_format (ignored), tools_to_call_from,
        #   custom_role_conversions, convert_images_to_image_urls, tool_choice (ignored), **kwargs.
        # Outputs: dict for client.converse including modelId and messages.
        # Exceptions: those of the base method; TypeError if content is a string rather than a list of parts.
        # Side effects: mutates the freshly built message dicts only.
        # KNOWN ISSUE: stop_sequences and tool_choice are accepted but silently dropped; when tools are given the base
        #   method still adds 'tools' and 'tool_choice' (default 'required') since only 'toolConfig' is removed.
        """
        Overrides the base method to handle Bedrock-specific configurations.

        This implementation adapts the completion keyword arguments to align with
        Bedrock's requirements, ensuring compatibility with its unique setup and
        constraints.
        """
        completion_kwargs = super()._prepare_completion_kwargs(
            messages=messages,
            stop_sequences=None,  # Bedrock support stop_sequence using Inference Config
            tools_to_call_from=tools_to_call_from,
            custom_role_conversions=custom_role_conversions,
            convert_images_to_image_urls=convert_images_to_image_urls,
            **kwargs,
        )
        # Not all models in Bedrock support `toolConfig`. Also, smolagents already include the tool call in the prompt,
        # so adding `toolConfig` could cause conflicts. We remove it to avoid issues.
        completion_kwargs.pop("toolConfig", None)

        # The Bedrock API does not support the `type` key in requests.
        # This block of code modifies the object to meet Bedrock's requirements.
        for message in completion_kwargs.get("messages", []):
            for content in message.get("content", []):
                if "type" in content:
                    del content["type"]

        return {
            "modelId": self.model_id,
            **completion_kwargs,
        }

    def create_client(self):
        # [FUNCTION DOC]
        # Creates boto3.client('bedrock-runtime', **client_kwargs).
        # Inputs: self. Outputs: boto3 client. Exceptions: ModuleNotFoundError with install hint.
        # Side effects: AWS credential resolution.
        # KNOWN ISSUE: error message refers to AmazonBedrockServerModel (the alias) rather than AmazonBedrockModel.
        try:
            import boto3  # type: ignore
        except ModuleNotFoundError as e:
            raise ModuleNotFoundError(
                "Please install 'bedrock' extra to use AmazonBedrockServerModel: `pip install 'smolagents[bedrock]'`"
            ) from e

        return boto3.client("bedrock-runtime", **self.client_kwargs)

    def generate(
        self,
        messages: list[ChatMessage | dict],
        stop_sequences: list[str] | None = None,
        response_format: dict[str, str] | None = None,
        tools_to_call_from: list[Tool] | None = None,
        **kwargs,
    ) -> ChatMessage:
        # [FUNCTION DOC]
        # Calls client.converse (rate-limited, retried) and returns the last text content block.
        # Inputs: messages, stop_sequences (only used for post-truncation), response_format (must be None),
        #   tools_to_call_from, **kwargs.
        # Outputs: ChatMessage with role, content, tool_calls, raw response and token usage (inputTokens/outputTokens).
        # Exceptions: ValueError if response_format given; KeyError if no text block exists or expected keys are missing.
        # Side effects: network request to AWS.
        # KNOWN ISSUE: stop_sequences are not sent to the API; and the response is indexed with a 'tool_calls' key that the
        #   Bedrock Converse output message does not normally contain, so this likely raises KeyError.
        if response_format is not None:
            raise ValueError("Amazon Bedrock does not support response_format")
        completion_kwargs: dict = self._prepare_completion_kwargs(
            messages=messages,
            tools_to_call_from=tools_to_call_from,
            custom_role_conversions=self.custom_role_conversions,
            convert_images_to_image_urls=True,
            **kwargs,
        )
        self._apply_rate_limit()
        # self.client is created in ApiModel class
        response = self.retryer(self.client.converse, **completion_kwargs)

        # Get content blocks with "text" key: in case thinking blocks are present, discard them
        message_content_blocks_with_text = [
            block for block in response["output"]["message"]["content"] if "text" in block
        ]
        if not message_content_blocks_with_text:
            raise KeyError("No message content blocks with 'text' key found in response")
        # Keep the last one
        content = message_content_blocks_with_text[-1]["text"]
        if stop_sequences is not None and not self.supports_stop_parameter:
            content = remove_content_after_stop_sequences(content, stop_sequences)
        return ChatMessage(
            role=response["output"]["message"]["role"],
            content=content,
            tool_calls=response["output"]["message"]["tool_calls"],
            raw=response,
            token_usage=TokenUsage(
                input_tokens=response["usage"]["inputTokens"],
                output_tokens=response["usage"]["outputTokens"],
            ),
        )


AmazonBedrockServerModel = AmazonBedrockModel


# Model Registry for secure deserialization
# This registry maps model class names to their actual classes.
# Only classes listed here can be instantiated during deserialization (from_dict).
# This prevents arbitrary code execution via importlib-based dynamic loading.
MODEL_REGISTRY = {
    "VLLMModel": VLLMModel,
    "MLXModel": MLXModel,
    "TransformersModel": TransformersModel,
    "LiteLLMModel": LiteLLMModel,
    "LiteLLMRouterModel": LiteLLMRouterModel,
    "InferenceClientModel": InferenceClientModel,
    "OpenAIModel": OpenAIModel,
    "AzureOpenAIModel": AzureOpenAIModel,
    "AmazonBedrockModel": AmazonBedrockModel,
}

__all__ = [
    "REMOVE_PARAMETER",
    "MessageRole",
    "tool_role_conversions",
    "get_clean_message_list",
    "Model",
    "MLXModel",
    "TransformersModel",
    "ApiModel",
    "InferenceClientModel",
    "LiteLLMModel",
    "LiteLLMRouterModel",
    "OpenAIServerModel",
    "OpenAIModel",
    "VLLMModel",
    "AzureOpenAIServerModel",
    "AzureOpenAIModel",
    "AmazonBedrockServerModel",
    "AmazonBedrockModel",
    "ChatMessage",
]
