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
# Description: Tests for FinalAnswerTool, reusing the generic ToolTesterMixin checks.
# Inputs: depends on numpy, PIL, pytest, _AGENT_TYPE_MAPPING, FinalAnswerTool, ToolTesterMixin and
#   require_torch.
# Outputs: TestFinalAnswerTool.
# Exceptions: none raised directly.
# Side effects: none.


import numpy as np
import PIL.Image
import pytest

from smolagents.agent_types import _AGENT_TYPE_MAPPING
from smolagents.default_tools import FinalAnswerTool

from .test_tools import ToolTesterMixin
from .utils.markers import require_torch


class TestFinalAnswerTool(ToolTesterMixin):
    # [CLASS DOC]
    # Description: Test class for FinalAnswerTool (inherits mixin tests that use self.tool and self.inputs).
    # Inputs: attributes tool and inputs set in setup_method.
    # Outputs: test outcomes.
    # Exceptions: AssertionError on failure.
    # Side effects: none.
    def setup_method(self):
        # [FUNCTION DOC]
        # Description: Creates the tool and a sample inputs dict before each test.
        # Inputs: self.
        # Outputs: None.
        # Exceptions: none raised directly.
        # Side effects: sets self.inputs and self.tool.
        self.inputs = {"answer": "Final answer"}
        self.tool = FinalAnswerTool()

    def test_exact_match_arg(self):
        # [FUNCTION DOC]
        # Description: Checks a positional answer is returned unchanged.
        # Inputs: self.
        # Outputs: none (passes or fails via assertions).
        # Exceptions: AssertionError on failure.
        # Side effects: none.
        result = self.tool("Final answer")
        assert result == "Final answer"

    def test_exact_match_kwarg(self):
        # [FUNCTION DOC]
        # Description: Checks a keyword answer is returned unchanged.
        # Inputs: self.
        # Outputs: none (passes or fails via assertions).
        # Exceptions: AssertionError on failure.
        # Side effects: none.
        result = self.tool(answer=self.inputs["answer"])
        assert result == "Final answer"

    @require_torch
    def test_agent_type_output(self, inputs):
        # [FUNCTION DOC]
        # Description: With sanitization enabled, checks string/image/audio answers are wrapped in the agent
        #   type mapped for each kind. Requires torch.
        # Inputs: self; inputs (fixture).
        # Outputs: none (passes or fails via assertions).
        # Exceptions: AssertionError on failure.
        # Side effects: none.
        for input_type, input in inputs.items():
            output = self.tool(**input, sanitize_inputs_outputs=True)
            agent_type = _AGENT_TYPE_MAPPING[input_type]
            assert isinstance(output, agent_type)

    @pytest.fixture
    def inputs(self, shared_datadir):
        # [FUNCTION DOC]
        # Description: HELPER: Fixture building sample answers: a string, a 512x512 image loaded from the
        #   shared data dir, and a torch tensor of 3000 ones as audio.
        # Inputs: self; shared_datadir (pytest-datadir fixture).
        # Outputs: dict mapping type name to kwargs dict.
        # Exceptions: ImportError if torch is missing; file errors.
        # Side effects: reads an image file.
        import torch

        return {
            "string": {"answer": "Text input"},
            "image": {"answer": PIL.Image.open(shared_datadir / "000000039769.png").resize((512, 512))},
            "audio": {"answer": torch.Tensor(np.ones(3000))},
        }
