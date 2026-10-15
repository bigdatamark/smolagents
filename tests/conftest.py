# [MODULE DOC]
# Description: Pytest configuration shared by all tests: registers the fixture modules tests.fixtures.agents
#   and tests.fixtures.tools as plugins and installs an autouse fixture that silences agent logging by
#   default.
# Inputs: depends on pytest, unittest.mock.patch, MultiStepAgent and LogLevel.
# Outputs: pytest_plugins list, original_multi_step_agent_init (saved original constructor) and the autouse
#   fixture.
# Exceptions: none raised directly.
# Side effects: pytest loads the listed plugin modules.
from unittest.mock import patch

import pytest

from smolagents.agents import MultiStepAgent
from smolagents.monitoring import LogLevel


# Import fixture modules as plugins
pytest_plugins = ["tests.fixtures.agents", "tests.fixtures.tools"]

original_multi_step_agent_init = MultiStepAgent.__init__


@pytest.fixture(autouse=True)
def patch_multi_step_agent_with_suppressed_logging():
    # [FUNCTION DOC]
    # Description: Autouse fixture that, for every test, patches MultiStepAgent's constructor (autospec) so
    #   that verbosity_level defaults to LogLevel.OFF unless given explicitly, delegating to the saved
    #   original constructor.
    # Inputs: none.
    # Outputs: yields None while the patch is active.
    # Exceptions: none raised directly.
    # Side effects: temporarily replaces MultiStepAgent.__init__ for the duration of each test.
    # KNOWN ISSUE: Because the patch is a mock, MultiStepAgent.__init__ calls are recorded and subclass
    #   constructors that call it via super go through the mock; tests relying on the real function object
    #   identity may be affected.
    with patch.object(MultiStepAgent, "__init__", autospec=True) as mock_init:

        def init_with_suppressed_logging(self, *args, verbosity_level=LogLevel.OFF, **kwargs):
            # [FUNCTION DOC]
            # Description: HELPER: Replacement constructor that forwards all arguments to the original
            #   MultiStepAgent constructor with verbosity_level defaulting to LogLevel.OFF.
            # Inputs: self (agent), positional args, verbosity_level (default LogLevel.OFF), keyword args.
            # Outputs: None.
            # Exceptions: whatever the original constructor raises.
            # Side effects: initializes the agent.
            original_multi_step_agent_init(self, *args, verbosity_level=verbosity_level, **kwargs)

        mock_init.side_effect = init_with_suppressed_logging
        yield
