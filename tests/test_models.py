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
# Module: pytest suite for smolagents.models: Model base-class kwarg preparation, stream delta agglomeration,
#   ChatMessage helpers, message cleaning, stop-sequence handling, tool-call parsing and per-backend model
#   classes (InferenceClient, LiteLLM, LiteLLM router, OpenAI, Azure OpenAI, Bedrock, Transformers, MLX).
# Depends on: pytest, unittest.mock, huggingface_hub, smolagents models/tools/default_tools, the
#   require_run_all marker, conftest fixtures (example_tool, shared_datadir); some tests need openai, litellm,
#   transformers, mlx.
# Exports: test classes and functions only.
# Side effects: most tests are mocked; some download models or call real APIs (MLX on macOS, transformers, and
#   require_run_all tests).
import json
import sys
from contextlib import ExitStack
from unittest.mock import MagicMock, patch

import pytest
from huggingface_hub import ChatCompletionOutputMessage

from smolagents.default_tools import FinalAnswerTool
from smolagents.models import (
    AmazonBedrockModel,
    AzureOpenAIModel,
    ChatMessage,
    ChatMessageToolCall,
    InferenceClientModel,
    LiteLLMModel,
    LiteLLMRouterModel,
    MessageRole,
    MLXModel,
    Model,
    OpenAIModel,
    TransformersModel,
    get_clean_message_list,
    get_tool_call_from_text,
    get_tool_json_schema,
    parse_json_if_needed,
    remove_content_after_stop_sequences,
    supports_stop_parameter,
)
from smolagents.tools import tool

from .utils.markers import require_run_all


class TestModel:
    # [CLASS DOC]
    # Description: Tests for the generic Model base class and shared helpers (kwarg preparation, stream
    #   deltas, tool schemas, ChatMessage, local MLX/Transformers models, parse_json_if_needed).
    # Inputs: no constructor arguments (pytest test class).
    # Outputs: n/a.
    # Exceptions: n/a.
    # Side effects: none
    def test_prepare_completion_kwargs_parameter_precedence(self):
        # [FUNCTION DOC]
        # Description: Checks that constructor kwargs override call kwargs in _prepare_completion_kwargs
        #   (max_tokens 100 beats 50, temperature kept, extra top_p kept) and that REMOVE_PARAMETER in
        #   constructor kwargs removes max_tokens entirely.
        # Inputs: self (test instance); no other inputs.
        # Outputs: None (pytest test).
        # Exceptions: none raised directly; failed assertions raise AssertionError.
        # Side effects: none
        """Test that self.kwargs have highest precedence and REMOVE_PARAMETER works correctly"""
        from smolagents.models import REMOVE_PARAMETER

        # Test with self.kwargs having highest precedence
        model = Model(max_tokens=100, temperature=0.5)
        completion_kwargs = model._prepare_completion_kwargs(
            messages=[ChatMessage(role=MessageRole.USER, content=[{"type": "text", "text": "Hello"}])],
            max_tokens=50,  # This should be overridden by self.kwargs
            top_p=0.9,  # This should remain from kwargs
        )

        # self.kwargs should have highest precedence
        assert completion_kwargs["max_tokens"] == 100
        assert completion_kwargs["temperature"] == 0.5
        assert completion_kwargs["top_p"] == 0.9

        # Test REMOVE_PARAMETER functionality
        model_with_removal = Model(max_tokens=REMOVE_PARAMETER, temperature=0.7)
        completion_kwargs = model_with_removal._prepare_completion_kwargs(
            messages=[ChatMessage(role=MessageRole.USER, content=[{"type": "text", "text": "Hello"}])],
            max_tokens=200,  # This should be removed by REMOVE_PARAMETER
            top_p=0.8,
        )

        # max_tokens should be removed, temperature should be set
        assert "max_tokens" not in completion_kwargs
        assert completion_kwargs["temperature"] == 0.7
        assert completion_kwargs["top_p"] == 0.8

    def test_agglomerate_stream_deltas(self):
        # [FUNCTION DOC]
        # Description: Feeds four ChatMessageStreamDelta objects (text pieces, partial tool-call argument
        #   fragments, a final token usage) to agglomerate_stream_deltas and checks concatenated content,
        #   concatenated tool-call arguments and total tokens 1372.
        # Inputs: self (test instance); no other inputs.
        # Outputs: None (pytest test).
        # Exceptions: none raised directly; failed assertions raise AssertionError.
        # Side effects: none
        from smolagents.models import (
            ChatMessageStreamDelta,
            ChatMessageToolCallFunction,
            ChatMessageToolCallStreamDelta,
            agglomerate_stream_deltas,
        )
        from smolagents.monitoring import TokenUsage

        stream_deltas = [
            ChatMessageStreamDelta(
                content="Hi",
                tool_calls=[
                    ChatMessageToolCallStreamDelta(
                        index=0,
                        type="function",
                        function=ChatMessageToolCallFunction(arguments="", name="web_search", description=None),
                    )
                ],
                token_usage=None,
            ),
            ChatMessageStreamDelta(
                content=" everyone",
                tool_calls=[
                    ChatMessageToolCallStreamDelta(
                        index=0,
                        type="function",
                        function=ChatMessageToolCallFunction(arguments=' {"', name="web_search", description=None),
                    )
                ],
                token_usage=None,
            ),
            ChatMessageStreamDelta(
                content=", it's",
                tool_calls=[
                    ChatMessageToolCallStreamDelta(
                        index=0,
                        type="function",
                        function=ChatMessageToolCallFunction(
                            arguments='query": "current pope name and date of birth"}',
                            name="web_search",
                            description=None,
                        ),
                    )
                ],
                token_usage=None,
            ),
            ChatMessageStreamDelta(
                content="",
                tool_calls=None,
                token_usage=TokenUsage(input_tokens=1348, output_tokens=24),
            ),
        ]
        agglomerated_stream_delta = agglomerate_stream_deltas(stream_deltas)
        assert agglomerated_stream_delta.content == "Hi everyone, it's"
        assert (
            agglomerated_stream_delta.tool_calls[0].function.arguments
            == ' {"query": "current pope name and date of birth"}'
        )
        assert agglomerated_stream_delta.token_usage.total_tokens == 1372

    @pytest.mark.parametrize(
        "model_id, stop_sequences, should_contain_stop",
        [
            ("regular-model", ["stop1", "stop2"], True),  # Regular model should include stop
            ("openai/o3", ["stop1", "stop2"], False),  # o3 model should not include stop
            ("openai/o4-mini", ["stop1", "stop2"], False),  # o4-mini model should not include stop
            ("something/else/o3", ["stop1", "stop2"], False),  # Path ending with o3 should not include stop
            ("something/else/o4-mini", ["stop1", "stop2"], False),  # Path ending with o4-mini should not include stop
            ("o3", ["stop1", "stop2"], False),  # Exact o3 model should not include stop
            ("o4-mini", ["stop1", "stop2"], False),  # Exact o4-mini model should not include stop
            ("regular-model", None, False),  # None stop_sequences should not add stop parameter
        ],
    )
    def test_prepare_completion_kwargs_stop_sequences(self, model_id, stop_sequences, should_contain_stop):
        # [FUNCTION DOC]
        # Description: Parametrized: verifies the stop parameter is included only for models supporting it
        #   (excluded for o3/o4-mini, with or without path prefixes) and only when stop_sequences is not None.
        # Inputs: self; model_id (str), stop_sequences (list of str or None), should_contain_stop (bool).
        # Outputs: None (pytest test).
        # Exceptions: none raised directly; failed assertions raise AssertionError.
        # Side effects: none
        model = Model()
        model.model_id = model_id
        completion_kwargs = model._prepare_completion_kwargs(
            messages=[
                ChatMessage(role=MessageRole.USER, content=[{"type": "text", "text": "Hello"}]),
            ],
            stop_sequences=stop_sequences,
        )
        # Verify that the stop parameter is only included when appropriate
        if should_contain_stop:
            assert "stop" in completion_kwargs
            assert completion_kwargs["stop"] == stop_sequences
        else:
            assert "stop" not in completion_kwargs

    @pytest.mark.parametrize(
        "with_tools, tool_choice, expected_result",
        [
            # Default behavior: With tools but no explicit tool_choice, should default to "required"
            (True, ..., {"has_tool_choice": True, "value": "required"}),
            # Custom value: With tools and explicit tool_choice="auto"
            (True, "auto", {"has_tool_choice": True, "value": "auto"}),
            # Tool name as string
            (True, "valid_tool_function", {"has_tool_choice": True, "value": "valid_tool_function"}),
            # Tool choice as dictionary
            (
                True,
                {"type": "function", "function": {"name": "valid_tool_function"}},
                {"has_tool_choice": True, "value": {"type": "function", "function": {"name": "valid_tool_function"}}},
            ),
            # With tools but explicit None tool_choice: should exclude tool_choice
            (True, None, {"has_tool_choice": False, "value": None}),
            # Without tools: tool_choice should never be included
            (False, "required", {"has_tool_choice": False, "value": None}),
            (False, "auto", {"has_tool_choice": False, "value": None}),
            (False, None, {"has_tool_choice": False, "value": None}),
            (False, ..., {"has_tool_choice": False, "value": None}),
        ],
    )
    def test_prepare_completion_kwargs_tool_choice(self, with_tools, tool_choice, expected_result, example_tool):
        # [FUNCTION DOC]
        # Description: Parametrized: verifies tool_choice handling: defaults to 'required' when tools are
        #   given and tool_choice is omitted (Ellipsis sentinel), passes through explicit strings/dicts, omits
        #   it when None or when there are no tools.
        # Inputs: self; with_tools (bool), tool_choice (str, dict, None or Ellipsis meaning 'not passed'),
        #   expected_result (dict with has_tool_choice and value), example_tool (conftest fixture).
        # Outputs: None (pytest test).
        # Exceptions: none raised directly; failed assertions raise AssertionError.
        # Side effects: none
        model = Model()
        kwargs = {"messages": [ChatMessage(role=MessageRole.USER, content=[{"type": "text", "text": "Hello"}])]}
        if with_tools:
            kwargs["tools_to_call_from"] = [example_tool]
        if tool_choice is not ...:
            kwargs["tool_choice"] = tool_choice

        completion_kwargs = model._prepare_completion_kwargs(**kwargs)

        if expected_result["has_tool_choice"]:
            assert "tool_choice" in completion_kwargs
            assert completion_kwargs["tool_choice"] == expected_result["value"]
        else:
            assert "tool_choice" not in completion_kwargs

    def test_get_json_schema_has_nullable_args(self):
        # [FUNCTION DOC]
        # Description: Defines a @tool with an optional celsius argument and checks get_tool_json_schema marks
        #   it nullable.
        # Inputs: self (test instance); no other inputs.
        # Outputs: None (pytest test).
        # Exceptions: none raised directly; failed assertions raise AssertionError.
        # Side effects: none
        @tool
        def get_weather(location: str, celsius: bool | None = False) -> str:
            # [FUNCTION DOC]
            # Description: HELPER: dummy tool body used only for schema generation; always returns a fixed
            #   weather string.
            # Inputs: location (str), celsius (bool or None, default False); both ignored.
            # Outputs: constant string.
            # Exceptions: none raised directly.
            # Side effects: none
            """
            Get weather in the next days at given location.
            Secretly this tool does not care about the location, it hates the weather everywhere.

            Args:
                location: the location
                celsius: the temperature type
            """
            return "The weather is UNGODLY with torrential rains and temperatures below -10°C"

        assert "nullable" in get_tool_json_schema(get_weather)["function"]["parameters"]["properties"]["celsius"]

    def test_chatmessage_has_model_dumps_json(self):
        # [FUNCTION DOC]
        # Description: Checks ChatMessage.model_dump_json produces JSON whose content equals the original
        #   content list.
        # Inputs: self (test instance); no other inputs.
        # Outputs: None (pytest test).
        # Exceptions: none raised directly; failed assertions raise AssertionError.
        # Side effects: none
        message = ChatMessage("user", [{"type": "text", "text": "Hello!"}])
        data = json.loads(message.model_dump_json())
        assert data["content"] == [{"type": "text", "text": "Hello!"}]

    def test_chatmessage_from_dict_role_conversion(self):
        # [FUNCTION DOC]
        # Description: Checks ChatMessage.from_dict converts a role string, or an existing MessageRole, into a
        #   MessageRole enum.
        # Inputs: self (test instance); no other inputs.
        # Outputs: None (pytest test).
        # Exceptions: none raised directly; failed assertions raise AssertionError.
        # Side effects: none
        message_data = {
            "role": "user",
            "content": [{"type": "text", "text": "Hello!"}],
        }
        message = ChatMessage.from_dict(message_data)
        assert isinstance(message.role, MessageRole)
        assert message.role == MessageRole.USER
        assert message.role.value == "user"
        assert message.content == [{"type": "text", "text": "Hello!"}]

        message_data["role"] = MessageRole.ASSISTANT
        message2 = ChatMessage.from_dict(message_data)
        assert isinstance(message2.role, MessageRole)
        assert message2.role == MessageRole.ASSISTANT

    @pytest.mark.skipif(not sys.platform.startswith("darwin"), reason="requires macOS")
    def test_get_mlx_message_no_tool(self):
        # [FUNCTION DOC]
        # Description: macOS only: runs a real MLXModel (SmolLM2-135M) and checks the output starts with
        #   'Hello'.
        # Inputs: self (test instance); no other inputs.
        # Outputs: None (pytest test).
        # Exceptions: none raised directly; failed assertions raise AssertionError.
        # Side effects: downloads model weights from the Hugging Face Hub (network, disk cache).
        model = MLXModel(model_id="HuggingFaceTB/SmolLM2-135M-Instruct", max_tokens=10)
        messages = [ChatMessage(role=MessageRole.USER, content=[{"type": "text", "text": "Hello!"}])]
        output = model(messages, stop_sequences=["great"]).content
        assert output.startswith("Hello")

    @pytest.mark.skipif(not sys.platform.startswith("darwin"), reason="requires macOS")
    def test_get_mlx_message_tricky_stop_sequence(self):
        # [FUNCTION DOC]
        # Description: macOS only: checks MLXModel trims stop sequences even when the stop text is followed by
        #   extra characters within the same token; asserts exact model outputs with and without the stop
        #   sequence.
        # Inputs: self (test instance); no other inputs.
        # Outputs: None (pytest test).
        # Exceptions: none raised directly; failed assertions raise AssertionError.
        # Side effects: downloads model weights from the Hugging Face Hub (network, disk cache).
        # In this test HuggingFaceTB/SmolLM2-135M-Instruct generates the token ">'"
        # which is required to test capturing stop_sequences that have extra chars at the end.
        model = MLXModel(model_id="HuggingFaceTB/SmolLM2-135M-Instruct", max_tokens=100)
        stop_sequence = " print '>"
        messages = [
            ChatMessage(role=MessageRole.USER, content=[{"type": "text", "text": f"Please{stop_sequence}'"}]),
        ]
        # check our assumption that that ">" is followed by "'"
        assert model.tokenizer.vocab[">'"]
        assert model(messages, stop_sequences=[]).content == f"I'm ready to help you{stop_sequence}'"
        # check stop_sequence capture when output has trailing chars
        assert model(messages, stop_sequences=[stop_sequence]).content == "I'm ready to help you"

    def test_transformers_message_no_tool(self, monkeypatch):
        # [FUNCTION DOC]
        # Description: Runs a real TransformersModel (SmolLM2-135M, CPU, greedy, 5 tokens) and checks generate
        #   and generate_stream both yield exactly "Hello! I'm here".
        # Inputs: self; monkeypatch (pytest fixture, used to raise HF download timeout to 30s).
        # Outputs: None (pytest test).
        # Exceptions: none raised directly; failed assertions raise AssertionError.
        # Side effects: downloads model weights from the Hugging Face Hub (network, disk cache). Patches the
        #   HF download timeout constant for the test.
        # KNOWN ISSUE: not guarded by require_run_all despite requiring network and model download.
        monkeypatch.setattr("huggingface_hub.constants.HF_HUB_DOWNLOAD_TIMEOUT", 30)  # instead of 10
        model = TransformersModel(
            model_id="HuggingFaceTB/SmolLM2-135M-Instruct",
            max_new_tokens=5,
            device_map="cpu",
            do_sample=False,
        )
        messages = [ChatMessage(role=MessageRole.USER, content=[{"type": "text", "text": "Hello!"}])]
        output = model.generate(messages).content
        assert output == "Hello! I'm here"

        output = model.generate_stream(messages, stop_sequences=["great"])
        output_str = ""
        for el in output:
            output_str += el.content
        assert output_str == "Hello! I'm here"

    def test_transformers_message_vl_no_tool(self, shared_datadir, monkeypatch):
        # [FUNCTION DOC]
        # Description: Runs a real vision-language TransformersModel (llava-interleave-qwen-0.5b) on a test
        #   image and checks generate and generate_stream outputs equal 'I am a very'.
        # Inputs: self; shared_datadir (fixture path to test data), monkeypatch (fixture).
        # Outputs: None (pytest test).
        # Exceptions: none raised directly; failed assertions raise AssertionError.
        # Side effects: downloads model weights from the Hugging Face Hub (network, disk cache). Reads a PNG
        #   from the test data dir.
        # KNOWN ISSUE: the expected output is flagged in the code as a possible regression (#1416); test needs
        #   network but is not guarded by require_run_all.
        monkeypatch.setattr("huggingface_hub.constants.HF_HUB_DOWNLOAD_TIMEOUT", 30)  # instead of 10
        import PIL.Image

        img = PIL.Image.open(shared_datadir / "000000039769.png")
        model = TransformersModel(
            model_id="llava-hf/llava-interleave-qwen-0.5b-hf",
            max_new_tokens=4,
            device_map="cpu",
            do_sample=False,
        )
        messages = [
            ChatMessage(
                role=MessageRole.USER,
                content=[{"type": "text", "text": "What is this?"}, {"type": "image", "image": img}],
            )
        ]
        output = model.generate(messages).content
        assert output == "I am a very"  # TODO: Investigate possible regression; see #1416

        output = model.generate_stream(messages, stop_sequences=["great"])
        output_str = ""
        for el in output:
            output_str += el.content
        assert output_str == "I am a very"  # TODO: Investigate possible regression; see #1416

    def test_parse_json_if_needed(self):
        # [FUNCTION DOC]
        # Description: Checks parse_json_if_needed returns non-JSON strings unchanged, parses JSON strings
        #   (object, number), and returns non-strings unchanged.
        # Inputs: self (test instance); no other inputs.
        # Outputs: None (pytest test).
        # Exceptions: none raised directly; failed assertions raise AssertionError.
        # Side effects: none
        args = "abc"
        parsed_args = parse_json_if_needed(args)
        assert parsed_args == "abc"

        args = '{"a": 3}'
        parsed_args = parse_json_if_needed(args)
        assert parsed_args == {"a": 3}

        args = "3"
        parsed_args = parse_json_if_needed(args)
        assert parsed_args == 3

        args = 3
        parsed_args = parse_json_if_needed(args)
        assert parsed_args == 3


class TestInferenceClientModel:
    # [CLASS DOC]
    # Description: Tests for InferenceClientModel construction and generation (mostly mocked).
    # Inputs: no constructor arguments (pytest test class).
    # Outputs: n/a.
    # Exceptions: n/a.
    # Side effects: none
    def test_call_with_custom_role_conversions(self):
        # [FUNCTION DOC]
        # Description: With a mocked client, checks that custom_role_conversions (user to system) are applied
        #   to messages sent to chat_completion.
        # Inputs: self (test instance); no other inputs.
        # Outputs: None (pytest test).
        # Exceptions: none raised directly; failed assertions raise AssertionError.
        # Side effects: replaces the model's client attribute with a MagicMock.
        custom_role_conversions = {MessageRole.USER: MessageRole.SYSTEM}
        model = InferenceClientModel(model_id="test-model", custom_role_conversions=custom_role_conversions)
        model.client = MagicMock()
        mock_response = model.client.chat_completion.return_value
        mock_response.choices[0].message = ChatCompletionOutputMessage(role=MessageRole.ASSISTANT)
        messages = [ChatMessage(role=MessageRole.USER, content="Test message")]
        _ = model(messages)
        # Verify that the role conversion was applied
        assert model.client.chat_completion.call_args.kwargs["messages"][0]["role"] == "system", (
            "role conversion should be applied"
        )

    def test_init_model_with_tokens(self):
        # [FUNCTION DOC]
        # Description: Checks token and api_key both set the client token and that passing both raises
        #   ValueError.
        # Inputs: self (test instance); no other inputs.
        # Outputs: None (pytest test).
        # Exceptions: expects ValueError internally via pytest.raises; otherwise none raised directly; failed
        #   assertions raise AssertionError.
        # Side effects: none
        model = InferenceClientModel(model_id="test-model", token="abc")
        assert model.client.token == "abc"

        model = InferenceClientModel(model_id="test-model", api_key="abc")
        assert model.client.token == "abc"

        with pytest.raises(ValueError, match="Received both `token` and `api_key` arguments."):
            InferenceClientModel(model_id="test-model", token="abc", api_key="def")

    def test_structured_outputs_with_unsupported_provider(self):
        # [FUNCTION DOC]
        # Description: Checks generate with response_format raises ValueError for a provider not supporting
        #   structured outputs.
        # Inputs: self (test instance); no other inputs.
        # Outputs: None (pytest test).
        # Exceptions: expects ValueError via pytest.raises; none raised directly; failed assertions raise
        #   AssertionError.
        # Side effects: none
        with pytest.raises(
            ValueError, match="InferenceClientModel only supports structured outputs with these providers:"
        ):
            model = InferenceClientModel(model_id="test-model", token="abc", provider="some_provider")
            model.generate(
                messages=[ChatMessage(role=MessageRole.USER, content="Hello!")],
                response_format={"type": "json_object"},
            )

    @require_run_all
    def test_get_hfapi_message_no_tool(self):
        # [FUNCTION DOC]
        # Description: Integration (require_run_all): real call to Qwen2.5-Coder-32B via HF inference; only
        #   checks no exception.
        # Inputs: self (test instance); no other inputs.
        # Outputs: None (pytest test).
        # Exceptions: none raised directly; failed assertions raise AssertionError.
        # Side effects: network call to Hugging Face inference API.
        model = InferenceClientModel(model_id="Qwen/Qwen2.5-Coder-32B-Instruct", max_tokens=10)
        messages = [ChatMessage(role=MessageRole.USER, content=[{"type": "text", "text": "Hello!"}])]
        model(messages, stop_sequences=["great"])

    @require_run_all
    def test_get_hfapi_message_no_tool_external_provider(self):
        # [FUNCTION DOC]
        # Description: Integration (require_run_all): same as above via the 'together' provider.
        # Inputs: self (test instance); no other inputs.
        # Outputs: None (pytest test).
        # Exceptions: none raised directly; failed assertions raise AssertionError.
        # Side effects: network call.
        model = InferenceClientModel(model_id="Qwen/Qwen2.5-Coder-32B-Instruct", provider="together", max_tokens=10)
        messages = [ChatMessage(role=MessageRole.USER, content=[{"type": "text", "text": "Hello!"}])]
        model(messages, stop_sequences=["great"])

    @require_run_all
    def test_get_hfapi_message_stream_no_tool(self):
        # [FUNCTION DOC]
        # Description: Integration (require_run_all): streams from the HF API and checks every delta has non-
        #   None content.
        # Inputs: self (test instance); no other inputs.
        # Outputs: None (pytest test).
        # Exceptions: none raised directly; failed assertions raise AssertionError.
        # Side effects: network call.
        model = InferenceClientModel(model_id="Qwen/Qwen2.5-Coder-32B-Instruct", max_tokens=10)
        messages = [ChatMessage(role=MessageRole.USER, content=[{"type": "text", "text": "Hello!"}])]
        for el in model.generate_stream(messages, stop_sequences=["great"]):
            assert el.content is not None

    @require_run_all
    def test_get_hfapi_message_stream_no_tool_external_provider(self):
        # [FUNCTION DOC]
        # Description: Integration (require_run_all): streaming via the 'together' provider; checks deltas
        #   have content.
        # Inputs: self (test instance); no other inputs.
        # Outputs: None (pytest test).
        # Exceptions: none raised directly; failed assertions raise AssertionError.
        # Side effects: network call.
        model = InferenceClientModel(model_id="Qwen/Qwen2.5-Coder-32B-Instruct", provider="together", max_tokens=10)
        messages = [ChatMessage(role=MessageRole.USER, content=[{"type": "text", "text": "Hello!"}])]
        for el in model.generate_stream(messages, stop_sequences=["great"]):
            assert el.content is not None


class TestLiteLLMModel:
    # [CLASS DOC]
    # Description: Tests for LiteLLMModel error handling, retries and flatten option.
    # Inputs: no constructor arguments (pytest test class).
    # Outputs: n/a.
    # Exceptions: n/a.
    # Side effects: none
    @pytest.mark.parametrize(
        "model_id",
        [
            "groq/llama-3.3-70b",
            "cerebras/llama-3.3-70b",
            "mistral/mistral-tiny",
        ],
    )
    def test_call_different_providers_without_key(self, model_id):
        # [FUNCTION DOC]
        # Description: Parametrized over providers: without API keys, generate and generate_stream must raise
        #   an exception whose text contains one of several authentication-related phrases.
        # Inputs: self; model_id (str, provider-prefixed model).
        # Outputs: None (pytest test).
        # Exceptions: expects exceptions via pytest.raises; none raised directly; failed assertions raise
        #   AssertionError.
        # Side effects: may attempt network calls to provider endpoints; depends on env vars not containing
        #   keys.
        # KNOWN ISSUE: the message is taken from str() of the ExceptionInfo rather than of the exception
        #   value, which works only because its repr includes the message.
        # Different litellm versions produce different error messages for missing API keys
        # This test checks for the presence of any common authentication-related error phrases
        possible_error_messages = [
            "Missing API Key",
            "Wrong API Key",
            "Invalid API Key",
            "The api_key client option must be set",
            "AuthenticationError",
            "Unauthorized",
            "Missing credentials",
        ]
        model = LiteLLMModel(model_id=model_id)
        messages = [ChatMessage(role=MessageRole.USER, content=[{"type": "text", "text": "Test message"}])]
        # Test generate method
        with pytest.raises(Exception) as e:
            model.generate(messages)
        error_message = str(e)
        assert any(possible_error_message in error_message for possible_error_message in possible_error_messages), (
            f"Error message '{error_message}' does not contain any expected phrases"
        )
        # Test generate_stream method
        with pytest.raises(Exception) as e:
            for el in model.generate_stream(messages):
                assert el.content is not None
        error_message = str(e)
        assert any(possible_error_message in error_message for possible_error_message in possible_error_messages), (
            f"Error message '{error_message}' does not contain any expected phrases"
        )

    def test_retry_on_rate_limit_error(self):
        # [FUNCTION DOC]
        # Description: With RETRY_WAIT patched to 0.1, jitter random patched to 0.1 and a mocked litellm
        #   client failing twice with a 429 error then succeeding, checks three completion calls, correct
        #   content and token usage, and total elapsed time between 0.67 and 0.73 seconds.
        # Inputs: self (test instance); no other inputs.
        # Outputs: None (pytest test).
        # Exceptions: none raised directly; failed assertions raise AssertionError.
        # Side effects: patches module globals and sleeps for real (~0.7s).
        # KNOWN ISSUE: timing-based assertion can be flaky on slow machines; the comment says RETRY_WAIT is
        #   patched to 1 second but the value is 0.1.
        """Test that the retry mechanism does trigger on 429 rate limit errors"""
        import time

        # Patch RETRY_WAIT to 1 second for faster testing
        mock_litellm = MagicMock()

        with (
            patch("smolagents.models.RETRY_WAIT", 0.1),
            patch("smolagents.utils.random.random", side_effect=[0.1, 0.1]),
            patch("smolagents.models.LiteLLMModel.create_client", return_value=mock_litellm),
        ):
            model = LiteLLMModel(model_id="test-model")
            messages = [ChatMessage(role=MessageRole.USER, content=[{"type": "text", "text": "Test message"}])]

            # Create a mock response for successful call
            mock_success_response = MagicMock()
            mock_success_response.choices = [MagicMock()]
            # Set content directly (not through model_dump)
            mock_success_response.choices[0].message.content = "Success response"
            mock_success_response.choices[0].message.role = "assistant"
            mock_success_response.choices[0].message.tool_calls = None
            mock_success_response.usage.prompt_tokens = 10
            mock_success_response.usage.completion_tokens = 20

            # Create a 429 rate limit error
            rate_limit_error = Exception("Error code: 429 - Rate limit exceeded")

            # Mock the litellm client to raise an error twice, and then succeed
            model.client.completion.side_effect = [rate_limit_error, rate_limit_error, mock_success_response]

            # Measure time to verify retry wait time
            start_time = time.time()
            result = model.generate(messages)
            elapsed_time = time.time() - start_time

            # Verify that completion was called thrice (twice failed, once succeeded)
            assert model.client.completion.call_count == 3
            assert result.content == "Success response"
            assert result.token_usage.input_tokens == 10
            assert result.token_usage.output_tokens == 20

            # Verify that the wait time was around
            # 0.22s (1st retry) [0.1 * 2.0 * (1 + 1 * 0.1)]
            # + 0.48s (2nd retry) [0.22 * 2.0 * (1 + 1 * 0.1)]
            # = 0.704s (allow some tolerance)
            assert 0.67 <= elapsed_time <= 0.73

    def test_passing_flatten_messages(self):
        # [FUNCTION DOC]
        # Description: Checks an explicit flatten_messages_as_text argument overrides provider-based defaults
        #   (False for groq, True for fal).
        # Inputs: self (test instance); no other inputs.
        # Outputs: None (pytest test).
        # Exceptions: none raised directly; failed assertions raise AssertionError.
        # Side effects: none
        model = LiteLLMModel(model_id="groq/llama-3.3-70b", flatten_messages_as_text=False)
        assert not model.flatten_messages_as_text

        model = LiteLLMModel(model_id="fal/llama-3.3-70b", flatten_messages_as_text=True)
        assert model.flatten_messages_as_text


class TestLiteLLMRouterModel:
    # [CLASS DOC]
    # Description: Tests for LiteLLMRouterModel.
    # Inputs: no constructor arguments (pytest test class).
    # Outputs: n/a.
    # Exceptions: n/a.
    # Side effects: none
    @pytest.mark.parametrize(
        "model_id, expected",
        [
            ("llama-3.3-70b", False),
            ("llama-3.3-70b", True),
            ("mistral-tiny", True),
        ],
    )
    def test_flatten_messages_as_text(self, model_id, expected):
        # [FUNCTION DOC]
        # Description: Parametrized: checks the explicit flatten_messages_as_text value is stored as-is.
        # Inputs: self; model_id (str, router group name), expected (bool, passed in and expected back).
        # Outputs: None (pytest test).
        # Exceptions: none raised directly; failed assertions raise AssertionError.
        # Side effects: none
        model_list = [
            {"model_name": "llama-3.3-70b", "litellm_params": {"model": "groq/llama-3.3-70b"}},
            {"model_name": "llama-3.3-70b", "litellm_params": {"model": "cerebras/llama-3.3-70b"}},
            {"model_name": "mistral-tiny", "litellm_params": {"model": "mistral/mistral-tiny"}},
        ]
        model = LiteLLMRouterModel(model_id=model_id, model_list=model_list, flatten_messages_as_text=expected)
        assert model.flatten_messages_as_text is expected

    def test_create_client(self):
        # [FUNCTION DOC]
        # Description: Patches litellm Router and checks it is constructed once with the model_list and
        #   client_kwargs (routing_strategy) and becomes model.client.
        # Inputs: self (test instance); no other inputs.
        # Outputs: None (pytest test).
        # Exceptions: none raised directly; failed assertions raise AssertionError.
        # Side effects: patches litellm.router.Router during the test.
        model_list = [
            {"model_name": "llama-3.3-70b", "litellm_params": {"model": "groq/llama-3.3-70b"}},
            {"model_name": "llama-3.3-70b", "litellm_params": {"model": "cerebras/llama-3.3-70b"}},
        ]
        with patch("litellm.router.Router") as mock_router:
            router_model = LiteLLMRouterModel(
                model_id="model-group-1", model_list=model_list, client_kwargs={"routing_strategy": "simple-shuffle"}
            )
            # Ensure that the Router constructor was called with the expected keyword arguments
            mock_router.assert_called_once()
            assert mock_router.call_count == 1
            assert mock_router.call_args.kwargs["model_list"] == model_list
            assert mock_router.call_args.kwargs["routing_strategy"] == "simple-shuffle"
            assert router_model.client == mock_router.return_value


class TestOpenAIModel:
    # [CLASS DOC]
    # Description: Tests for OpenAIModel client creation, streaming tool calls and stop-sequence trimming.
    # Inputs: no constructor arguments (pytest test class).
    # Outputs: n/a.
    # Exceptions: n/a.
    # Side effects: none
    def test_client_kwargs_passed_correctly(self):
        # [FUNCTION DOC]
        # Description: Patches openai.OpenAI and checks it is called with base_url, api_key, organization,
        #   project and merged client_kwargs, and the instance becomes model.client.
        # Inputs: self (test instance); no other inputs.
        # Outputs: None (pytest test).
        # Exceptions: none raised directly; failed assertions raise AssertionError.
        # Side effects: patches openai.OpenAI.
        model_id = "gpt-3.5-turbo"
        api_base = "https://api.openai.com/v1"
        api_key = "test_api_key"
        organization = "test_org"
        project = "test_project"
        client_kwargs = {"max_retries": 5}

        with patch("openai.OpenAI") as MockOpenAI:
            model = OpenAIModel(
                model_id=model_id,
                api_base=api_base,
                api_key=api_key,
                organization=organization,
                project=project,
                client_kwargs=client_kwargs,
            )
        MockOpenAI.assert_called_once_with(
            base_url=api_base, api_key=api_key, organization=organization, project=project, max_retries=5
        )
        assert model.client == MockOpenAI.return_value

    @require_run_all
    def test_streaming_tool_calls(self):
        # [FUNCTION DOC]
        # Description: Integration (require_run_all): streams from gpt-4o-mini asking for two parallel
        #   final_answer calls and checks the accumulated arguments of both calls.
        # Inputs: self (test instance); no other inputs.
        # Outputs: None (pytest test).
        # Exceptions: none raised directly; failed assertions raise AssertionError.
        # Side effects: real OpenAI API call (needs OPENAI_API_KEY).
        # KNOWN ISSUE: args/args2 are unbound (NameError) if the model never emits tool calls or only one.
        model = OpenAIModel(model_id="gpt-4o-mini")
        messages = [
            ChatMessage(
                role=MessageRole.USER,
                content=[
                    {
                        "type": "text",
                        "text": "Hello! Please return the final answer 'blob' and the final answer 'blob2' in two parallel tool calls",
                    }
                ],
            ),
        ]
        for el in model.generate_stream(messages, tools_to_call_from=[FinalAnswerTool()]):
            if el.tool_calls:
                assert el.tool_calls[0].function.name == "final_answer"
                args = el.tool_calls[0].function.arguments
                if len(el.tool_calls) > 1:
                    assert el.tool_calls[1].function.name == "final_answer"
                    args2 = el.tool_calls[1].function.arguments
        assert args == '{"answer": "blob"}'
        assert args2 == '{"answer": "blob2"}'

    def test_stop_sequence_cutting_for_o4_mini(self):
        # [FUNCTION DOC]
        # Description: With a mocked OpenAI client returning text containing '<STOP>', checks that for o4-mini
        #   (no stop parameter support) the content is truncated client-side at the stop sequence.
        # Inputs: self (test instance); no other inputs.
        # Outputs: None (pytest test).
        # Exceptions: none raised directly; failed assertions raise AssertionError.
        # Side effects: patches openai.OpenAI.
        """Test that stop sequences are cut a posteriori for models that don't support stop parameter"""
        # Create a mock response that contains a stop sequence in the middle
        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].message.role = "assistant"
        mock_response.choices[0].message.content = "This is some text<STOP>and this should be removed"
        mock_response.choices[0].message.tool_calls = None
        mock_response.usage.prompt_tokens = 10
        mock_response.usage.completion_tokens = 20

        with patch("openai.OpenAI") as MockOpenAI:
            mock_client = MagicMock()
            MockOpenAI.return_value = mock_client
            mock_client.chat.completions.create.return_value = mock_response

            model = OpenAIModel(model_id="o4-mini")
            messages = [ChatMessage(role=MessageRole.USER, content=[{"type": "text", "text": "Hello"}])]
            result = model.generate(messages, stop_sequences=["<STOP>"])

            # Verify the stop sequence was removed
            assert result.content == "This is some text"
            assert "<STOP>" not in result.content
            assert "and this should be removed" not in result.content


class TestAmazonBedrockModel:
    # [CLASS DOC]
    # Description: Tests for AmazonBedrockModel client creation.
    # Inputs: no constructor arguments (pytest test class).
    # Outputs: n/a.
    # Exceptions: n/a.
    # Side effects: none
    def test_client_for_bedrock(self):
        # [FUNCTION DOC]
        # Description: Patches boto3.client and checks the model's client is the patched return value.
        # Inputs: self (test instance); no other inputs.
        # Outputs: None (pytest test).
        # Exceptions: none raised directly; failed assertions raise AssertionError.
        # Side effects: patches boto3.client.
        model_id = "us.amazon.nova-pro-v1:0"

        with patch("boto3.client") as MockBoto3:
            model = AmazonBedrockModel(
                model_id=model_id,
            )

        assert model.client == MockBoto3.return_value


class TestAzureOpenAIModel:
    # [CLASS DOC]
    # Description: Tests for AzureOpenAIModel client creation.
    # Inputs: no constructor arguments (pytest test class).
    # Outputs: n/a.
    # Exceptions: n/a.
    # Side effects: none
    def test_client_kwargs_passed_correctly(self):
        # [FUNCTION DOC]
        # Description: Patches openai.OpenAI and openai.AzureOpenAI; checks only AzureOpenAI is constructed,
        #   with base_url None plus api_key, api_version, azure_endpoint, organization, project and
        #   client_kwargs, and it becomes model.client.
        # Inputs: self (test instance); no other inputs.
        # Outputs: None (pytest test).
        # Exceptions: none raised directly; failed assertions raise AssertionError.
        # Side effects: patches openai classes.
        model_id = "gpt-3.5-turbo"
        api_key = "test_api_key"
        api_version = "2023-12-01-preview"
        azure_endpoint = "https://example-resource.azure.openai.com/"
        organization = "test_org"
        project = "test_project"
        client_kwargs = {"max_retries": 5}

        with patch("openai.OpenAI") as MockOpenAI, patch("openai.AzureOpenAI") as MockAzureOpenAI:
            model = AzureOpenAIModel(
                model_id=model_id,
                api_key=api_key,
                api_version=api_version,
                azure_endpoint=azure_endpoint,
                organization=organization,
                project=project,
                client_kwargs=client_kwargs,
            )
        assert MockOpenAI.call_count == 0
        MockAzureOpenAI.assert_called_once_with(
            base_url=None,
            api_key=api_key,
            api_version=api_version,
            azure_endpoint=azure_endpoint,
            organization=organization,
            project=project,
            max_retries=5,
        )
        assert model.client == MockAzureOpenAI.return_value


class TestTransformersModel:
    # [CLASS DOC]
    # Description: Tests for TransformersModel initialization paths.
    # Inputs: no constructor arguments (pytest test class).
    # Outputs: n/a.
    # Exceptions: n/a.
    # Side effects: none
    @pytest.mark.parametrize(
        "patching",
        [
            [
                (
                    "transformers.AutoModelForImageTextToText.from_pretrained",
                    {"side_effect": ValueError("Unrecognized configuration class")},
                ),
                ("transformers.AutoModelForCausalLM.from_pretrained", {}),
                ("transformers.AutoTokenizer.from_pretrained", {}),
            ],
            [
                ("transformers.AutoModelForImageTextToText.from_pretrained", {}),
                ("transformers.AutoProcessor.from_pretrained", {}),
            ],
        ],
    )
    def test_init(self, patching):
        # [FUNCTION DOC]
        # Description: Parametrized over two patch sets: when the image-text-to-text loader fails with
        #   'Unrecognized configuration class' the model falls back to AutoModelForCausalLM + AutoTokenizer;
        #   otherwise uses AutoModelForImageTextToText + AutoProcessor. Checks the loaders receive device_map,
        #   torch_dtype and trust_remote_code correctly.
        # Inputs: self; patching (list of (target, patch kwargs) tuples).
        # Outputs: None (pytest test).
        # Exceptions: none raised directly; failed assertions raise AssertionError.
        # Side effects: patches transformers loaders.
        with ExitStack() as stack:
            mocks = {target: stack.enter_context(patch(target, **kwargs)) for target, kwargs in patching}
            model = TransformersModel(
                model_id="test-model", device_map="cpu", torch_dtype="float16", trust_remote_code=True
            )
        assert model.model_id == "test-model"
        if "transformers.AutoTokenizer.from_pretrained" in mocks:
            assert model.model == mocks["transformers.AutoModelForCausalLM.from_pretrained"].return_value
            assert mocks["transformers.AutoModelForCausalLM.from_pretrained"].call_args.kwargs == {
                "device_map": "cpu",
                "torch_dtype": "float16",
                "trust_remote_code": True,
            }
            assert model.tokenizer == mocks["transformers.AutoTokenizer.from_pretrained"].return_value
            assert mocks["transformers.AutoTokenizer.from_pretrained"].call_args.args == ("test-model",)
            assert mocks["transformers.AutoTokenizer.from_pretrained"].call_args.kwargs == {"trust_remote_code": True}
        elif "transformers.AutoProcessor.from_pretrained" in mocks:
            assert model.model == mocks["transformers.AutoModelForImageTextToText.from_pretrained"].return_value
            assert mocks["transformers.AutoModelForImageTextToText.from_pretrained"].call_args.kwargs == {
                "device_map": "cpu",
                "torch_dtype": "float16",
                "trust_remote_code": True,
            }
            assert model.processor == mocks["transformers.AutoProcessor.from_pretrained"].return_value
            assert mocks["transformers.AutoProcessor.from_pretrained"].call_args.args == ("test-model",)
            assert mocks["transformers.AutoProcessor.from_pretrained"].call_args.kwargs == {"trust_remote_code": True}


def test_get_clean_message_list_basic():
    # [FUNCTION DOC]
    # Description: Checks get_clean_message_list converts ChatMessages to dicts with string roles and
    #   preserved content.
    # Inputs: none.
    # Outputs: None (pytest test).
    # Exceptions: none raised directly; failed assertions raise AssertionError.
    # Side effects: none
    messages = [
        ChatMessage(role=MessageRole.USER, content=[{"type": "text", "text": "Hello!"}]),
        ChatMessage(role=MessageRole.ASSISTANT, content=[{"type": "text", "text": "Hi there!"}]),
    ]
    result = get_clean_message_list(messages)
    assert len(result) == 2
    assert result[0]["role"] == "user"
    assert result[0]["content"][0]["text"] == "Hello!"
    assert result[1]["role"] == "assistant"
    assert result[1]["content"][0]["text"] == "Hi there!"


@pytest.mark.parametrize(
    "messages,expected_roles,expected_texts",
    [
        (
            [
                {"role": "user", "content": [{"type": "text", "text": "Hello!"}]},
                {"role": "assistant", "content": [{"type": "text", "text": "Hi there!"}]},
            ],
            ["user", "assistant"],
            ["Hello!", "Hi there!"],
        ),
        (
            [
                {"role": "user", "content": [{"type": "text", "text": "How are you?"}]},
            ],
            ["user"],
            ["How are you?"],
        ),
    ],
)
def test_get_clean_message_list_with_dicts(messages, expected_roles, expected_texts):
    # [FUNCTION DOC]
    # Description: Parametrized: checks plain dict messages are accepted and roles/texts preserved.
    # Inputs: messages (list of dicts), expected_roles (list of str), expected_texts (list of str).
    # Outputs: None (pytest test).
    # Exceptions: none raised directly; failed assertions raise AssertionError.
    # Side effects: none
    result = get_clean_message_list(messages)
    assert len(result) == len(messages)
    for i, msg in enumerate(result):
        assert msg["role"] == expected_roles[i]
        assert msg["content"][0]["text"] == expected_texts[i]


def test_get_clean_message_list_role_conversions():
    # [FUNCTION DOC]
    # Description: Checks role_conversions map tool-call to assistant and tool-response to user.
    # Inputs: none.
    # Outputs: None (pytest test).
    # Exceptions: none raised directly; failed assertions raise AssertionError.
    # Side effects: none
    messages = [
        ChatMessage(role=MessageRole.TOOL_CALL, content=[{"type": "text", "text": "Calling tool..."}]),
        ChatMessage(role=MessageRole.TOOL_RESPONSE, content=[{"type": "text", "text": "Tool response"}]),
    ]
    result = get_clean_message_list(messages, role_conversions={"tool-call": "assistant", "tool-response": "user"})
    assert len(result) == 2
    assert result[0]["role"] == "assistant"
    assert result[0]["content"][0]["text"] == "Calling tool..."
    assert result[1]["role"] == "user"
    assert result[1]["content"][0]["text"] == "Tool response"


def test_remove_content_after_stop_sequences():
    # [FUNCTION DOC]
    # Description: Checks content is cut at the first stop sequence.
    # Inputs: none.
    # Outputs: None (pytest test).
    # Exceptions: none raised directly; failed assertions raise AssertionError.
    # Side effects: none
    content = "Hello<code>world!"
    stop_sequences = ["<code>"]
    removed_content = remove_content_after_stop_sequences(content, stop_sequences)
    assert removed_content == "Hello"


def test_remove_content_after_stop_sequences_handles_none():
    # [FUNCTION DOC]
    # Description: Checks None stop sequences leave content unchanged and None content returns None.
    # Inputs: none.
    # Outputs: None (pytest test).
    # Exceptions: none raised directly; failed assertions raise AssertionError.
    # Side effects: none
    # Test with None stop sequence
    content = "Hello world!"
    removed_content = remove_content_after_stop_sequences(content, None)
    assert removed_content == content

    # Test with None content
    removed_content = remove_content_after_stop_sequences(None, ["<code>"])
    assert removed_content is None


@pytest.mark.parametrize(
    "convert_images_to_image_urls, expected_clean_message",
    [
        (
            False,
            dict(
                role=MessageRole.USER,
                content=[
                    {"type": "image", "image": "encoded_image"},
                    {"type": "image", "image": "second_encoded_image"},
                ],
            ),
        ),
        (
            True,
            dict(
                role=MessageRole.USER,
                content=[
                    {"type": "image_url", "image_url": {"url": "data:image/png;base64,encoded_image"}},
                    {"type": "image_url", "image_url": {"url": "data:image/png;base64,second_encoded_image"}},
                ],
            ),
        ),
    ],
)
def test_get_clean_message_list_image_encoding(convert_images_to_image_urls, expected_clean_message):
    # [FUNCTION DOC]
    # Description: Parametrized: with encode_image_base64 patched, checks image bytes are encoded and either
    #   kept as image parts or converted to image_url data URIs (PNG) depending on
    #   convert_images_to_image_urls.
    # Inputs: convert_images_to_image_urls (bool), expected_clean_message (dict).
    # Outputs: None (pytest test).
    # Exceptions: none raised directly; failed assertions raise AssertionError.
    # Side effects: patches smolagents.models.encode_image_base64.
    message = ChatMessage(
        role=MessageRole.USER,
        content=[{"type": "image", "image": b"image_data"}, {"type": "image", "image": b"second_image_data"}],
    )
    with patch("smolagents.models.encode_image_base64") as mock_encode:
        mock_encode.side_effect = ["encoded_image", "second_encoded_image"]
        result = get_clean_message_list([message], convert_images_to_image_urls=convert_images_to_image_urls)
        mock_encode.assert_any_call(b"image_data")
        mock_encode.assert_any_call(b"second_image_data")
        assert len(result) == 1
        assert result[0] == expected_clean_message


def test_get_clean_message_list_flatten_messages_as_text():
    # [FUNCTION DOC]
    # Description: Checks consecutive same-role messages are merged and, with flatten_messages_as_text,
    #   content becomes a single newline-joined string.
    # Inputs: none.
    # Outputs: None (pytest test).
    # Exceptions: none raised directly; failed assertions raise AssertionError.
    # Side effects: none
    messages = [
        ChatMessage(role=MessageRole.USER, content=[{"type": "text", "text": "Hello!"}]),
        ChatMessage(role=MessageRole.USER, content=[{"type": "text", "text": "How are you?"}]),
    ]
    result = get_clean_message_list(messages, flatten_messages_as_text=True)
    assert len(result) == 1
    assert result[0]["role"] == "user"
    assert result[0]["content"] == "Hello!\nHow are you?"


@pytest.mark.parametrize(
    "model_class, model_kwargs, patching, expected_flatten_messages_as_text",
    [
        (AzureOpenAIModel, {}, ("openai.AzureOpenAI", {}), False),
        (InferenceClientModel, {}, ("huggingface_hub.InferenceClient", {}), False),
        (LiteLLMModel, {}, None, False),
        (LiteLLMModel, {"model_id": "ollama"}, None, True),
        (LiteLLMModel, {"model_id": "groq"}, None, True),
        (LiteLLMModel, {"model_id": "cerebras"}, None, True),
        (MLXModel, {}, ("mlx_lm.load", {"return_value": (MagicMock(), MagicMock())}), True),
        (OpenAIModel, {}, ("openai.OpenAI", {}), False),
        (OpenAIModel, {"flatten_messages_as_text": True}, ("openai.OpenAI", {}), True),
        (
            TransformersModel,
            {},
            [
                (
                    "transformers.AutoModelForImageTextToText.from_pretrained",
                    {"side_effect": ValueError("Unrecognized configuration class")},
                ),
                ("transformers.AutoModelForCausalLM.from_pretrained", {}),
                ("transformers.AutoTokenizer.from_pretrained", {}),
            ],
            True,
        ),
        (
            TransformersModel,
            {},
            [
                ("transformers.AutoModelForImageTextToText.from_pretrained", {}),
                ("transformers.AutoProcessor.from_pretrained", {}),
            ],
            False,
        ),
    ],
)
def test_flatten_messages_as_text_for_all_models(
    model_class, model_kwargs, patching, expected_flatten_messages_as_text
):
    # [FUNCTION DOC]
    # Description: Parametrized over model classes with their constructors' dependencies patched: checks the
    #   default flatten_messages_as_text value per backend/model id (e.g. True for ollama/groq/cerebras
    #   LiteLLM, MLX and text-only Transformers; False for OpenAI/Azure/InferenceClient/VLM Transformers).
    # Inputs: model_class (Model subclass), model_kwargs (dict), patching (None, one (target, kwargs) tuple,
    #   or a list of them), expected_flatten_messages_as_text (bool).
    # Outputs: None (pytest test).
    # Exceptions: none raised directly; failed assertions raise AssertionError.
    # Side effects: patches client/loader constructors during model creation.
    with ExitStack() as stack:
        if isinstance(patching, list):
            for target, kwargs in patching:
                stack.enter_context(patch(target, **kwargs))
        elif patching:
            target, kwargs = patching
            stack.enter_context(patch(target, **kwargs))

        model = model_class(**{"model_id": "test-model", **model_kwargs})
    assert model.flatten_messages_as_text is expected_flatten_messages_as_text, f"{model_class.__name__} failed"


@pytest.mark.parametrize(
    "model_id,expected",
    [
        # Unsupported base models
        ("o3", False),
        ("o4-mini", False),
        ("gpt-5.1", False),
        ("gpt-5.2", False),
        ("gpt-5", False),
        ("gpt-5-mini", False),
        ("gpt-5-nano", False),
        ("gpt-5-turbo", False),
        ("gpt-5.2-mini", False),
        ("grok-4", False),
        ("grok-4-latest", False),
        ("grok-4.1", False),
        ("grok-3", False),
        ("grok-3-mini", False),
        ("grok-code-fast-1", False),
        # Unsupported versioned models
        ("o4-mini-2025-04-16", False),
        ("gpt-5-2025-01-01", False),
        # Unsupported models with path prefixes
        ("openai/o3", False),
        ("openai/o4-mini", False),
        ("openai/o3-2025-04-16", False),
        ("openai/o4-mini-2025-04-16", False),
        ("openai/gpt-5.2", False),
        ("openai/gpt-5.2-mini", False),
        ("openai/gpt-5.2-2025-01-01", False),
        ("oci/xai.grok-4", False),
        ("oci/xai.grok-3-mini", False),
        # Supported models
        ("o3-mini", True),
        ("gpt-4", True),
        ("claude-3-5-sonnet", True),
        ("mistral-large", True),
        # Supported models with path prefixes
        ("openai/gpt-4", True),
        ("anthropic/claude-3-5-sonnet", True),
        ("anthropic/claude-opus-4-5", True),
        ("mistralai/mistral-large", True),
        # Edge cases
        ("", True),  # Empty string doesn't match pattern
        ("o3x", True),  # Not exactly o3
        ("o4x", True),  # Not exactly o4
        ("gpt-5x", False),
        ("gpt-50", False),
        ("o3_mini", True),  # Not o3-mini format
        ("prefix-o3", True),  # o3 not at start
    ],
)
def test_supports_stop_parameter(model_id, expected):
    # [FUNCTION DOC]
    # Description: Parametrized: checks supports_stop_parameter returns False for o3, o4-mini, gpt-5*,
    #   grok-3/4 families (with optional provider prefixes and date suffixes) and True for others including
    #   o3-mini and near-miss names.
    # Inputs: model_id (str), expected (bool).
    # Outputs: None (pytest test).
    # Exceptions: none raised directly; failed assertions raise AssertionError.
    # Side effects: none
    """Test the supports_stop_parameter function with various model IDs"""
    assert supports_stop_parameter(model_id) == expected, f"Failed for model_id: {model_id}"


class TestGetToolCallFromText:
    # [CLASS DOC]
    # Description: Tests for get_tool_call_from_text parsing tool calls out of JSON text.
    # Inputs: no constructor arguments (pytest test class).
    # Outputs: n/a.
    # Exceptions: n/a.
    # Side effects: autouse fixture patches uuid.uuid4 for each test.
    @pytest.fixture(autouse=True)
    def mock_uuid4(self):
        # [FUNCTION DOC]
        # Description: HELPER: autouse pytest fixture that patches uuid.uuid4 to return 'test-uuid' for the
        #   duration of each test in the class.
        # Inputs: self.
        # Outputs: yields None once (generator fixture).
        # Exceptions: none raised directly.
        # Side effects: globally patches uuid.uuid4 while active.
        with patch("uuid.uuid4", return_value="test-uuid"):
            yield

    def test_get_tool_call_from_text_basic(self):
        # [FUNCTION DOC]
        # Description: Checks a JSON tool call yields a ChatMessageToolCall with the mocked id, type
        #   'function', name and string arguments.
        # Inputs: self (test instance); no other inputs.
        # Outputs: None (pytest test).
        # Exceptions: none raised directly; failed assertions raise AssertionError.
        # Side effects: none
        text = '{"name": "weather_tool", "arguments": "New York"}'
        result = get_tool_call_from_text(text, "name", "arguments")
        assert isinstance(result, ChatMessageToolCall)
        assert result.id == "test-uuid"
        assert result.type == "function"
        assert result.function.name == "weather_tool"
        assert result.function.arguments == "New York"

    def test_get_tool_call_from_text_name_key_missing(self):
        # [FUNCTION DOC]
        # Description: Checks a missing name key raises ValueError listing the available keys.
        # Inputs: self (test instance); no other inputs.
        # Outputs: None (pytest test).
        # Exceptions: expects ValueError via pytest.raises; none raised directly; failed assertions raise
        #   AssertionError.
        # Side effects: none
        text = '{"action": "weather_tool", "arguments": "New York"}'
        with pytest.raises(ValueError) as exc_info:
            get_tool_call_from_text(text, "name", "arguments")
        error_msg = str(exc_info.value)
        assert "Tool call needs to have a key 'name'" in error_msg
        assert "'action', 'arguments'" in error_msg

    def test_get_tool_call_from_text_json_object_args(self):
        # [FUNCTION DOC]
        # Description: Checks object-valued arguments are returned as a dict.
        # Inputs: self (test instance); no other inputs.
        # Outputs: None (pytest test).
        # Exceptions: none raised directly; failed assertions raise AssertionError.
        # Side effects: none
        text = '{"name": "weather_tool", "arguments": {"city": "New York"}}'
        result = get_tool_call_from_text(text, "name", "arguments")
        assert result.function.arguments == {"city": "New York"}

    def test_get_tool_call_from_text_json_string_args(self):
        # [FUNCTION DOC]
        # Description: Checks arguments given as a JSON-encoded string are parsed into a dict.
        # Inputs: self (test instance); no other inputs.
        # Outputs: None (pytest test).
        # Exceptions: none raised directly; failed assertions raise AssertionError.
        # Side effects: none
        text = '{"name": "weather_tool", "arguments": "{\\"city\\": \\"New York\\"}"}'
        result = get_tool_call_from_text(text, "name", "arguments")
        assert result.function.arguments == {"city": "New York"}

    def test_get_tool_call_from_text_missing_args(self):
        # [FUNCTION DOC]
        # Description: Checks a missing arguments key yields None arguments.
        # Inputs: self (test instance); no other inputs.
        # Outputs: None (pytest test).
        # Exceptions: none raised directly; failed assertions raise AssertionError.
        # Side effects: none
        text = '{"name": "weather_tool"}'
        result = get_tool_call_from_text(text, "name", "arguments")
        assert result.function.arguments is None

    def test_get_tool_call_from_text_custom_keys(self):
        # [FUNCTION DOC]
        # Description: Checks custom name/arguments key names are honored.
        # Inputs: self (test instance); no other inputs.
        # Outputs: None (pytest test).
        # Exceptions: none raised directly; failed assertions raise AssertionError.
        # Side effects: none
        text = '{"tool": "weather_tool", "params": "New York"}'
        result = get_tool_call_from_text(text, "tool", "params")
        assert result.function.name == "weather_tool"
        assert result.function.arguments == "New York"

    def test_get_tool_call_from_text_numeric_args(self):
        # [FUNCTION DOC]
        # Description: Checks numeric arguments are returned unchanged.
        # Inputs: self (test instance); no other inputs.
        # Outputs: None (pytest test).
        # Exceptions: none raised directly; failed assertions raise AssertionError.
        # Side effects: none
        text = '{"name": "calculator", "arguments": 42}'
        result = get_tool_call_from_text(text, "name", "arguments")
        assert result.function.name == "calculator"
        assert result.function.arguments == 42


@pytest.mark.parametrize(
    "model_class,model_id",
    [
        (LiteLLMModel, "gpt-4o-mini"),
        (OpenAIModel, "gpt-4o-mini"),
    ],
)
def test_tool_calls_json_serialization(model_class, model_id):
    # [FUNCTION DOC]
    # Description: Parametrized over OpenAIModel and LiteLLMModel: builds a realistic provider SDK response
    #   object (OpenAI ChatCompletion or litellm ModelResponse) containing a final_answer tool call, patches
    #   create_client to return a mock client returning it, and checks generate produces ChatMessageToolCall
    #   dataclasses that survive model_dump_json with correct name and arguments.
    # Inputs: model_class (OpenAIModel or LiteLLMModel), model_id (str).
    # Outputs: None (pytest test).
    # Exceptions: raises ValueError for an unexpected model_class; otherwise none raised directly; failed
    #   assertions raise AssertionError.
    # Side effects: patches the model class's create_client.
    """Test that tool_calls from various API models (Pydantic, dataclass, dict) are properly converted to dataclasses and can be JSON serialized.
    This tests the horizontal fix that ensures all models (LiteLLM, OpenAI, InferenceClient, AmazonBedrock)
    properly convert tool_calls to dataclasses regardless of the source format (Pydantic models, dataclasses, or dicts).
    """
    tool_arguments = "test_result"
    messages = [
        ChatMessage(
            role=MessageRole.USER,
            content=[
                {
                    "type": "text",
                    "text": "Hello! Please return the final answer 'hi there' in a tool call",
                }
            ],
        ),
    ]

    if model_class == OpenAIModel:
        from openai.types.chat.chat_completion import ChatCompletion, Choice
        from openai.types.chat.chat_completion_message import ChatCompletionMessage
        from openai.types.chat.chat_completion_message_tool_call import ChatCompletionMessageToolCall, Function
        from openai.types.completion_usage import CompletionUsage

        response = ChatCompletion(
            id="chatcmpl-test",
            created=0,
            model="gpt-4o-mini-2024-07-18",
            object="chat.completion",
            choices=[
                Choice(
                    finish_reason="tool_calls",
                    index=0,
                    logprobs=None,
                    message=ChatCompletionMessage(
                        role="assistant",
                        content=None,
                        tool_calls=[
                            ChatCompletionMessageToolCall(
                                id="call_test",
                                type="function",
                                function=Function(name="final_answer", arguments=tool_arguments),
                            )
                        ],
                    ),
                )
            ],
            usage=CompletionUsage(prompt_tokens=69, completion_tokens=15, total_tokens=84),
        )
        client = MagicMock()
        client.chat.completions.create.return_value = response
        create_call = client.chat.completions.create
        patch_target = "smolagents.models.OpenAIModel.create_client"
    elif model_class == LiteLLMModel:
        from litellm.types.utils import ChatCompletionMessageToolCall, Choices, Function, Message, ModelResponse, Usage

        response = ModelResponse(
            id="chatcmpl-test",
            created=0,
            object="chat.completion",
            choices=[
                Choices(
                    finish_reason="tool_calls",
                    index=0,
                    message=Message(
                        role="assistant",
                        content=None,
                        tool_calls=[
                            ChatCompletionMessageToolCall(
                                id="call_test",
                                type="function",
                                function=Function(name="final_answer", arguments=tool_arguments),
                            )
                        ],
                        function_call=None,
                        provider_specific_fields={"refusal": None, "annotations": []},
                    ),
                )
            ],
            usage=Usage(prompt_tokens=69, completion_tokens=15, total_tokens=84),
            model="gpt-4o-mini-2024-07-18",
        )
        client = MagicMock()
        client.completion.return_value = response
        create_call = client.completion
        patch_target = "smolagents.models.LiteLLMModel.create_client"
    else:
        raise ValueError(f"Unexpected model class: {model_class}")

    with patch(patch_target, return_value=client):
        model = model_class(model_id=model_id)
        result = model.generate(messages, tools_to_call_from=[FinalAnswerTool()])

    assert create_call.call_count == 1

    # Verify tool_calls are converted to dataclasses
    assert result.tool_calls is not None
    assert len(result.tool_calls) > 0
    assert isinstance(result.tool_calls[0], ChatMessageToolCall)

    # The critical test: verify JSON serialization works
    json_str = result.model_dump_json()
    data = json.loads(json_str)
    assert "tool_calls" in data
    assert len(data["tool_calls"]) > 0
    assert data["tool_calls"][0]["function"]["name"] == "final_answer"
    assert data["tool_calls"][0]["function"]["arguments"] == "test_result"
