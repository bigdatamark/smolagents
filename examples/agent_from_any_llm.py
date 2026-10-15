# [MODULE DOC]
# Example script: shows that the same tool works with any smolagents model backend. A hard-coded string
#   selects one of five backends (InferenceClientModel on nebius with Llama-3.3-70B, local TransformersModel
#   SmolLM2-1.7B, LiteLLMModel against a local Ollama server, LiteLLMModel gpt-4o, OpenAIModel gpt-4o), then
#   runs a ToolCallingAgent and a streaming CodeAgent on a Paris-weather question using the dummy get_weather
#   tool and prints both answers.
# Inputs: none (edit chosen_inference in the source); depends on smolagents and credentials/servers of the
#   chosen backend.
# Outputs: none (script); prints the chosen model name and both agent answers.
# Exceptions: any backend/network/auth error propagates and aborts the script.
# Side effects: runs at import time: network calls to LLM providers (or local model load), console output.
# KNOWN ISSUE: available_inferences is defined but never used to validate chosen_inference; an unknown value
#   leaves the model name undefined and causes a NameError when building the agents.
from smolagents import (
    CodeAgent,
    InferenceClientModel,
    LiteLLMModel,
    OpenAIModel,
    ToolCallingAgent,
    TransformersModel,
    tool,
)


# Choose which inference type to use!

available_inferences = ["inference_client", "transformers", "ollama", "litellm", "openai"]
chosen_inference = "inference_client"

print(f"Chose model: '{chosen_inference}'")

if chosen_inference == "inference_client":
    model = InferenceClientModel(model_id="meta-llama/Llama-3.3-70B-Instruct", provider="nebius")

elif chosen_inference == "transformers":
    model = TransformersModel(model_id="HuggingFaceTB/SmolLM2-1.7B-Instruct", device_map="auto", max_new_tokens=1000)

elif chosen_inference == "ollama":
    model = LiteLLMModel(
        model_id="ollama_chat/llama3.2",
        api_base="http://localhost:11434",  # replace with remote open-ai compatible server if necessary
        api_key="your-api-key",  # replace with API key if necessary
        num_ctx=8192,  # ollama default is 2048 which will often fail horribly. 8192 works for easy tasks, more is better. Check https://huggingface.co/spaces/NyxKrage/LLM-Model-VRAM-Calculator to calculate how much VRAM this will need for the selected model.
    )

elif chosen_inference == "litellm":
    # For anthropic: change model_id below to 'anthropic/claude-3-5-sonnet-latest'
    model = LiteLLMModel(model_id="gpt-4o")

elif chosen_inference == "openai":
    # For anthropic: change model_id below to 'anthropic/claude-3-5-sonnet-latest'
    model = OpenAIModel(model_id="gpt-4o")


@tool
def get_weather(location: str, celsius: bool | None = False) -> str:
    # [FUNCTION DOC]
    # Tool (wrapped by the tool decorator) returning a fixed, fake weather report regardless of input.
    # Inputs: location (str, ignored), celsius (bool or None, default False, ignored).
    # Outputs: constant string describing torrential rain and temperatures below -10 degrees C.
    # Exceptions: none raised directly.
    # Side effects: none.
    """
    Get weather in the next days at given location.
    Secretly this tool does not care about the location, it hates the weather everywhere.

    Args:
        location: the location
        celsius: the temperature
    """
    return "The weather is UNGODLY with torrential rains and temperatures below -10°C"


agent = ToolCallingAgent(tools=[get_weather], model=model, verbosity_level=2)

print("ToolCallingAgent:", agent.run("What's the weather like in Paris?"))

agent = CodeAgent(tools=[get_weather], model=model, verbosity_level=2, stream_outputs=True)

print("CodeAgent:", agent.run("What's the weather like in Paris?"))
