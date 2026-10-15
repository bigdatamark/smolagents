# [MODULE DOC]
# Description: HELPER: Fixture module holding AGENT_DICTS, sample serialized agent dictionaries for format
#   versions v1.9, v1.10 (adds executor_type, executor_kwargs, max_print_outputs_length) and v1.20 (removes
#   grammar and token counts), used to test loading of older agent serializations.
# Inputs: depends on pytest.
# Outputs: AGENT_DICTS and the get_agent_dict fixture.
# Exceptions: none raised directly.
# Side effects: none.
import pytest


AGENT_DICTS = {
    "v1.9": {
        "tools": [],
        "model": {
            "class": "InferenceClientModel",
            "data": {
                "last_input_token_count": None,
                "last_output_token_count": None,
                "model_id": "Qwen/Qwen2.5-Coder-32B-Instruct",
                "provider": None,
            },
        },
        "managed_agents": {},
        "prompt_templates": {
            "system_prompt": "dummy system prompt",
            "planning": {
                "initial_facts": "dummy planning initial facts",
                "initial_plan": "dummy planning initial plan",
                "update_facts_pre_messages": "dummy planning update facts pre messages",
                "update_facts_post_messages": "dummy planning update facts post messages",
                "update_plan_pre_messages": "dummy planning update plan pre messages",
                "update_plan_post_messages": "dummy planning update plan post messages",
            },
            "managed_agent": {
                "task": "dummy managed agent task",
                "report": "dummy managed agent report",
            },
            "final_answer": {
                "pre_messages": "dummy final answer pre messages",
                "post_messages": "dummy final answer post messages",
            },
        },
        "max_steps": 10,
        "verbosity_level": 2,
        "grammar": None,
        "planning_interval": 2,
        "name": "test_agent",
        "description": "dummy description",
        "requirements": ["smolagents"],
        "authorized_imports": ["pandas"],
    },
    # Added: executor_type, executor_kwargs, max_print_outputs_length
    "v1.10": {
        "tools": [],
        "model": {
            "class": "InferenceClientModel",
            "data": {
                "last_input_token_count": None,
                "last_output_token_count": None,
                "model_id": "Qwen/Qwen2.5-Coder-32B-Instruct",
                "provider": None,
            },
        },
        "managed_agents": {},
        "prompt_templates": {
            "system_prompt": "dummy system prompt",
            "planning": {
                "initial_facts": "dummy planning initial facts",
                "initial_plan": "dummy planning initial plan",
                "update_facts_pre_messages": "dummy planning update facts pre messages",
                "update_facts_post_messages": "dummy planning update facts post messages",
                "update_plan_pre_messages": "dummy planning update plan pre messages",
                "update_plan_post_messages": "dummy planning update plan post messages",
            },
            "managed_agent": {
                "task": "dummy managed agent task",
                "report": "dummy managed agent report",
            },
            "final_answer": {
                "pre_messages": "dummy final answer pre messages",
                "post_messages": "dummy final answer post messages",
            },
        },
        "max_steps": 10,
        "verbosity_level": 2,
        "grammar": None,
        "planning_interval": 2,
        "name": "test_agent",
        "description": "dummy description",
        "requirements": ["smolagents"],
        "authorized_imports": ["pandas"],
        "executor_type": "local",
        "executor_kwargs": {},
        "max_print_outputs_length": None,
    },
    # Removed: grammar, last_input_token_count, last_output_token_count
    "v1.20": {
        "tools": [],
        "model": {
            "class": "InferenceClientModel",
            "data": {
                "model_id": "Qwen/Qwen2.5-Coder-32B-Instruct",
                "provider": None,
            },
        },
        "managed_agents": {},
        "prompt_templates": {
            "system_prompt": "dummy system prompt",
            "planning": {
                "initial_facts": "dummy planning initial facts",
                "initial_plan": "dummy planning initial plan",
                "update_facts_pre_messages": "dummy planning update facts pre messages",
                "update_facts_post_messages": "dummy planning update facts post messages",
                "update_plan_pre_messages": "dummy planning update plan pre messages",
                "update_plan_post_messages": "dummy planning update plan post messages",
            },
            "managed_agent": {
                "task": "dummy managed agent task",
                "report": "dummy managed agent report",
            },
            "final_answer": {
                "pre_messages": "dummy final answer pre messages",
                "post_messages": "dummy final answer post messages",
            },
        },
        "max_steps": 10,
        "verbosity_level": 2,
        "planning_interval": 2,
        "name": "test_agent",
        "description": "dummy description",
        "requirements": ["smolagents"],
        "authorized_imports": ["pandas"],
        "executor_type": "local",
        "executor_kwargs": {},
        "max_print_outputs_length": None,
    },
}


@pytest.fixture
def get_agent_dict():
    # [FUNCTION DOC]
    # Description: HELPER: Fixture returning a lookup function for AGENT_DICTS.
    # Inputs: none.
    # Outputs: a function taking a version key and returning the matching dict.
    # Exceptions: none raised directly.
    # Side effects: none.
    def _get_agent_dict(agent_dict_key):
        # [FUNCTION DOC]
        # Description: HELPER: Returns the agent dict for the given version key.
        # Inputs: agent_dict_key (str, e.g. v1.9).
        # Outputs: the shared dict object (not a copy).
        # Exceptions: KeyError for an unknown key.
        # Side effects: none.
        # KNOWN ISSUE: Returns the shared module-level dict, so a test that mutates it affects later tests.
        return AGENT_DICTS[agent_dict_key]

    return _get_agent_dict
