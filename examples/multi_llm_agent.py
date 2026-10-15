# [MODULE DOC]
# Example script: load-balances a CodeAgent across several LLM deployments with LiteLLMRouterModel. Two
#   deployments (OpenAI gpt-4o-mini and Bedrock Claude 3 Sonnet) share the group name "model-group-1" and are
#   picked with the "simple-shuffle" routing strategy; the agent (WebSearchTool, streaming, full result)
#   answers a leopard/Pont des Arts question and the full result is printed.
# Inputs: environment variables OPENAI_API_KEY, AWS_ACCESS_KEY_ID, AWS_SECRET_ACCESS_KEY, AWS_REGION; depends
#   on smolagents with litellm.
# Outputs: none (script); module globals llm_loadbalancer_model_list, model, agent, full_result.
# Exceptions: auth/network errors propagate.
# Side effects: reads env vars, network calls, console output at import.
import os

from smolagents import CodeAgent, LiteLLMRouterModel, WebSearchTool


# Make sure to setup the necessary environment variables!

llm_loadbalancer_model_list = [
    {
        "model_name": "model-group-1",
        "litellm_params": {
            "model": "gpt-4o-mini",
            "api_key": os.getenv("OPENAI_API_KEY"),
        },
    },
    {
        "model_name": "model-group-1",
        "litellm_params": {
            "model": "bedrock/anthropic.claude-3-sonnet-20240229-v1:0",
            "aws_access_key_id": os.getenv("AWS_ACCESS_KEY_ID"),
            "aws_secret_access_key": os.getenv("AWS_SECRET_ACCESS_KEY"),
            "aws_region_name": os.getenv("AWS_REGION"),
        },
    },
    # {
    #     "model_name": "model-group-2",
    #     "litellm_params": {
    #         "model": "bedrock/anthropic.claude-3-sonnet-20240229-v1:0",
    #         "aws_access_key_id": os.getenv("AWS_ACCESS_KEY_ID"),
    #         "aws_secret_access_key": os.getenv("AWS_SECRET_ACCESS_KEY"),
    #         "aws_region_name": os.getenv("AWS_REGION"),
    #     },
    # },
]


model = LiteLLMRouterModel(
    model_id="model-group-1",
    model_list=llm_loadbalancer_model_list,
    client_kwargs={"routing_strategy": "simple-shuffle"},
)
agent = CodeAgent(tools=[WebSearchTool()], model=model, stream_outputs=True, return_full_result=True)

full_result = agent.run("How many seconds would it take for a leopard at full speed to run through Pont des Arts?")

print(full_result)
