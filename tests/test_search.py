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
# Description: Tests for DuckDuckGoSearchTool using ToolTesterMixin; network tests are gated by
#   require_run_all.
# Inputs: depends on DuckDuckGoSearchTool, ToolTesterMixin, require_run_all.
# Outputs: TestDuckDuckGoSearchTool.
# Exceptions: none raised directly.
# Side effects: none.


from smolagents import DuckDuckGoSearchTool

from .test_tools import ToolTesterMixin
from .utils.markers import require_run_all


class TestDuckDuckGoSearchTool(ToolTesterMixin):
    # [CLASS DOC]
    # Description: Test class for the DuckDuckGo search tool.
    # Inputs: attribute tool set in setup_method.
    # Outputs: test outcomes.
    # Exceptions: AssertionError on failure.
    # Side effects: none.
    def setup_method(self):
        # [FUNCTION DOC]
        # Description: Instantiates the tool and calls its setup before each test.
        # Inputs: self.
        # Outputs: None.
        # Exceptions: ImportError if the ddgs dependency is missing.
        # Side effects: sets self.tool.
        self.tool = DuckDuckGoSearchTool()
        self.tool.setup()

    @require_run_all
    def test_exact_match_arg(self):
        # [FUNCTION DOC]
        # Description: Runs a real search for Agents and checks a string is returned.
        # Inputs: self.
        # Outputs: none (passes or fails via assertions).
        # Exceptions: AssertionError on failure.
        # Side effects: network request.
        result = self.tool("Agents")
        assert isinstance(result, str)

    @require_run_all
    def test_agent_type_output(self):
        # [FUNCTION DOC]
        # Description: Runs the mixin's agent-type output check, gated by RUN_ALL.
        # Inputs: self.
        # Outputs: none (passes or fails via assertions).
        # Exceptions: AssertionError on failure.
        # Side effects: network request via the mixin test.
        super().test_agent_type_output()
