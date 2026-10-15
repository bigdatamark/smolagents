# [MODULE DOC]
# Purpose: Tests for smolagents.cli load_model and run_smolagent, plus run_webagent from vision_web_browser.
# Dependencies: pytest, unittest.mock, smolagents cli, models, local_python_executor, vision_web_browser.
# Side effects: patches environment variables and external clients.
from unittest.mock import patch

import pytest

from smolagents.cli import load_model
from smolagents.local_python_executor import CodeOutput, LocalPythonExecutor
from smolagents.models import InferenceClientModel, LiteLLMModel, OpenAIModel, TransformersModel


@pytest.fixture
def set_env_vars(monkeypatch):
    # [FUNCTION DOC]
    # HELPER: Fixture setting FIREWORKS_API_KEY and HF_TOKEN environment variables to test values.
    # Inputs: monkeypatch (pytest fixture).
    # Outputs: none.
    # Exceptions: none raised directly.
    # Side effects: sets env vars for the test duration.
    # KNOWN ISSUE: sets HF_TOKEN while load_model reads HF_API_KEY; the HF test passes only because the model
    #   falls back to HF_TOKEN elsewhere.
    monkeypatch.setenv("FIREWORKS_API_KEY", "test_fireworks_api_key")
    monkeypatch.setenv("HF_TOKEN", "test_hf_api_key")


def test_load_model_openai_model(set_env_vars):
    # [FUNCTION DOC]
    # Checks OpenAIModel is built with the Fireworks base URL and API key from the environment.
    # Inputs: set_env_vars (fixture).
    # Outputs: none.
    # Exceptions: AssertionError on failure.
    # Side effects: none (client patched).
    with patch("openai.OpenAI") as MockOpenAI:
        model = load_model("OpenAIModel", "test_model_id")
    assert isinstance(model, OpenAIModel)
    assert model.model_id == "test_model_id"
    assert MockOpenAI.call_count == 1
    assert MockOpenAI.call_args.kwargs["base_url"] == "https://api.fireworks.ai/inference/v1"
    assert MockOpenAI.call_args.kwargs["api_key"] == "test_fireworks_api_key"


def test_load_model_litellm_model():
    # [FUNCTION DOC]
    # Checks LiteLLMModel receives the given model id, API key and base.
    # Inputs: none.
    # Outputs: none.
    # Exceptions: AssertionError on failure.
    # Side effects: none.
    model = load_model("LiteLLMModel", "test_model_id", api_key="test_api_key", api_base="https://api.test.com")
    assert isinstance(model, LiteLLMModel)
    assert model.api_key == "test_api_key"
    assert model.api_base == "https://api.test.com"
    assert model.model_id == "test_model_id"


def test_load_model_transformers_model():
    # [FUNCTION DOC]
    # Checks TransformersModel loads with patched transformers loaders (vision loader failing to force the
    #   causal LM path).
    # Inputs: none.
    # Outputs: none.
    # Exceptions: AssertionError on failure.
    # Side effects: none.
    with (
        patch(
            "transformers.AutoModelForImageTextToText.from_pretrained",
            side_effect=ValueError("Unrecognized configuration class"),
        ),
        patch("transformers.AutoModelForCausalLM.from_pretrained"),
        patch("transformers.AutoTokenizer.from_pretrained"),
    ):
        model = load_model("TransformersModel", "test_model_id")
    assert isinstance(model, TransformersModel)
    assert model.model_id == "test_model_id"


def test_load_model_hf_api_model(set_env_vars):
    # [FUNCTION DOC]
    # Checks InferenceClientModel creates one InferenceClient with the token from the environment.
    # Inputs: set_env_vars (fixture).
    # Outputs: none.
    # Exceptions: AssertionError on failure.
    # Side effects: none (client patched).
    with patch("huggingface_hub.InferenceClient") as huggingface_hub_InferenceClient:
        model = load_model("InferenceClientModel", "test_model_id")
    assert isinstance(model, InferenceClientModel)
    assert model.model_id == "test_model_id"
    assert huggingface_hub_InferenceClient.call_count == 1
    assert huggingface_hub_InferenceClient.call_args.kwargs["token"] == "test_hf_api_key"


def test_load_model_invalid_model_type():
    # [FUNCTION DOC]
    # Checks an unknown model type raises ValueError.
    # Inputs: none.
    # Outputs: none.
    # Exceptions: AssertionError on failure.
    # Side effects: none.
    with pytest.raises(ValueError, match="Unsupported model type: InvalidModel"):
        load_model("InvalidModel", "test_model_id")


def test_cli_main(capsys):
    # [FUNCTION DOC]
    # Checks run_smolagent calls load_model with the given arguments, builds a streaming CodeAgent with no
    #   imports and runs the prompt.
    # Inputs: capsys (unused fixture).
    # Outputs: none.
    # Exceptions: AssertionError on failure.
    # Side effects: loads .env.
    # KNOWN ISSUE: named after main but tests run_smolagent; capsys is unused.
    with patch("smolagents.cli.load_model") as mock_load_model:
        mock_load_model.return_value = "mock_model"
        with patch("smolagents.cli.CodeAgent") as mock_code_agent:
            from smolagents.cli import run_smolagent

            run_smolagent("test_prompt", [], "InferenceClientModel", "test_model_id", provider="hf-inference")
    # load_model
    assert len(mock_load_model.call_args_list) == 1
    assert mock_load_model.call_args.args == ("InferenceClientModel", "test_model_id")
    assert mock_load_model.call_args.kwargs == {"api_base": None, "api_key": None, "provider": "hf-inference"}
    # CodeAgent
    assert len(mock_code_agent.call_args_list) == 1
    assert mock_code_agent.call_args.args == ()
    assert mock_code_agent.call_args.kwargs == {
        "tools": [],
        "model": "mock_model",
        "additional_authorized_imports": None,
        "stream_outputs": True,
    }
    # agent.run
    assert len(mock_code_agent.return_value.run.call_args_list) == 1
    assert mock_code_agent.return_value.run.call_args.args == ("test_prompt",)


def test_vision_web_browser_main():
    # [FUNCTION DOC]
    # Checks run_webagent loads the model, builds a CodeAgent with 4 tools and helium imports, runs the helium
    #   import in its executor and runs the prompt with helium instructions appended.
    # Inputs: none.
    # Outputs: none.
    # Exceptions: AssertionError on failure.
    # Side effects: executes a helium import in a real LocalPythonExecutor.
    # KNOWN ISSUE: tests vision_web_browser, not cli, despite living in this file.
    with patch("smolagents.vision_web_browser.helium"):
        with patch("smolagents.vision_web_browser.load_model") as mock_load_model:
            mock_load_model.return_value = "mock_model"
            with patch("smolagents.vision_web_browser.CodeAgent") as mock_code_agent:
                from smolagents.vision_web_browser import helium_instructions, run_webagent

                run_webagent("test_prompt", "InferenceClientModel", "test_model_id", provider="hf-inference")
    # load_model
    assert len(mock_load_model.call_args_list) == 1
    assert mock_load_model.call_args.args == ("InferenceClientModel", "test_model_id")
    # CodeAgent
    assert len(mock_code_agent.call_args_list) == 1
    assert mock_code_agent.call_args.args == ()
    assert len(mock_code_agent.call_args.kwargs["tools"]) == 4
    assert mock_code_agent.call_args.kwargs["model"] == "mock_model"
    assert mock_code_agent.call_args.kwargs["additional_authorized_imports"] == ["helium"]
    # agent.python_executor
    assert len(mock_code_agent.return_value.python_executor.call_args_list) == 1
    assert mock_code_agent.return_value.python_executor.call_args.args == ("from helium import *",)
    assert LocalPythonExecutor(["helium"])("from helium import *") == CodeOutput(
        output=None, logs="", is_final_answer=False
    )
    # agent.run
    assert len(mock_code_agent.return_value.run.call_args_list) == 1
    assert mock_code_agent.return_value.run.call_args.args == ("test_prompt" + helium_instructions,)
