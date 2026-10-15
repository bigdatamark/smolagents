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
# Purpose: Unit tests for smolagents.agent_types: AgentAudio from tensors and files, AgentImage from tensors,
#   paths and PIL images, AgentText from strings.
# Dependencies: unittest, PIL, torch and soundfile (optional, tests gated by markers), shared_datadir fixture
#   for image data.
# Side effects: writes temp audio/image files.
import os
import tempfile
import unittest
import uuid

import PIL.Image

from smolagents.agent_types import AgentAudio, AgentImage, AgentText

from .utils.markers import require_soundfile, require_torch


def get_new_path(suffix="") -> str:
    # [FUNCTION DOC]
    # HELPER: Returns a path to a new unique filename (UUID plus suffix) inside a freshly created temp
    #   directory.
    # Inputs: suffix (str, default empty).
    # Outputs: str path (file not created).
    # Exceptions: OSError if the temp dir cannot be created.
    # Side effects: creates a temp directory.
    directory = tempfile.mkdtemp()
    return os.path.join(directory, str(uuid.uuid4()) + suffix)


@require_soundfile
@require_torch
class AgentAudioTests(unittest.TestCase):
    # [CLASS DOC]
    # Tests for AgentAudio; skipped unless soundfile and torch are installed.
    # Exceptions: assertion failures on test failure.
    # Side effects: writes temp WAV files.
    def test_from_tensor(self):
        # [FUNCTION DOC]
        # Checks a tensor-based AgentAudio round-trips through to_raw and through the WAV file written by
        #   to_string, which persists after deletion of the object.
        # Inputs: none.
        # Outputs: none.
        # Exceptions: AssertionError on failure.
        # Side effects: writes a temp WAV file.
        import soundfile as sf
        import torch

        tensor = torch.rand(12, dtype=torch.float64) - 0.5
        agent_type = AgentAudio(tensor)
        path = str(agent_type.to_string())

        # Ensure that the tensor and the agent_type's tensor are the same
        self.assertTrue(torch.allclose(tensor, agent_type.to_raw(), atol=1e-4))

        del agent_type

        # Ensure the path remains even after the object deletion
        self.assertTrue(os.path.exists(path))

        # Ensure that the file contains the same value as the original tensor
        new_tensor, _ = sf.read(path)
        self.assertTrue(torch.allclose(tensor, torch.tensor(new_tensor), atol=1e-4))

    def test_from_string(self):
        # [FUNCTION DOC]
        # Checks a path-based AgentAudio reads the same samples and returns the original path.
        # Inputs: none.
        # Outputs: none.
        # Exceptions: AssertionError on failure.
        # Side effects: writes a temp WAV file.
        import soundfile as sf
        import torch

        tensor = torch.rand(12, dtype=torch.float64) - 0.5
        path = get_new_path(suffix=".wav")
        sf.write(path, tensor, 16000)

        agent_type = AgentAudio(path)

        self.assertTrue(torch.allclose(tensor, agent_type.to_raw(), atol=1e-4))
        self.assertEqual(agent_type.to_string(), path)


@require_torch
class TestAgentImage:
    # [CLASS DOC]
    # Tests for AgentImage; skipped unless torch is installed.
    # Exceptions: assertion failures on test failure.
    # Side effects: writes temp PNG files.
    def test_from_tensor(self):
        # [FUNCTION DOC]
        # Checks a tensor-based AgentImage keeps the tensor, converts to a PIL image and its saved file
        #   persists after deletion.
        # Inputs: none.
        # Outputs: none.
        # Exceptions: AssertionError on failure.
        # Side effects: writes a temp PNG.
        # KNOWN ISSUE: does not check pixel values, so the inversion in the tensor-to-image conversion goes
        #   undetected.
        import torch

        tensor = torch.randint(0, 256, (64, 64, 3))
        agent_type = AgentImage(tensor)
        path = str(agent_type.to_string())

        # Ensure that the tensor and the agent_type's tensor are the same
        assert torch.allclose(tensor, agent_type._tensor, atol=1e-4)

        assert isinstance(agent_type.to_raw(), PIL.Image.Image)

        # Ensure the path remains even after the object deletion
        del agent_type
        assert os.path.exists(path)

    def test_from_string(self, shared_datadir):
        # [FUNCTION DOC]
        # Checks a path-based AgentImage returns the same path and an image equal to opening the file.
        # Inputs: shared_datadir (pytest-datadir fixture path).
        # Outputs: none.
        # Exceptions: AssertionError on failure.
        # Side effects: none.
        path = shared_datadir / "000000039769.png"
        image = PIL.Image.open(path)
        agent_type = AgentImage(path)

        assert path.samefile(agent_type.to_string())
        assert image == agent_type.to_raw()

        # Ensure the path remains even after the object deletion
        del agent_type
        assert os.path.exists(path)

    def test_from_image(self, shared_datadir):
        # [FUNCTION DOC]
        # Checks a PIL-based AgentImage returns the same image and saves to a new path different from the
        #   source.
        # Inputs: shared_datadir (fixture path).
        # Outputs: none.
        # Exceptions: AssertionError on failure.
        # Side effects: writes a temp PNG.
        path = shared_datadir / "000000039769.png"
        image = PIL.Image.open(path)
        agent_type = AgentImage(image)

        assert not path.samefile(agent_type.to_string())
        assert image == agent_type.to_raw()

        # Ensure the path remains even after the object deletion
        del agent_type
        assert os.path.exists(path)


class AgentTextTests(unittest.TestCase):
    # [CLASS DOC]
    # Tests for AgentText.
    # Exceptions: assertion failures on test failure.
    # Side effects: none.
    def test_from_string(self):
        # [FUNCTION DOC]
        # Checks to_string and to_raw both return the original string.
        # Inputs: none.
        # Outputs: none.
        # Exceptions: AssertionError on failure.
        # Side effects: none.
        string = "Hey!"
        agent_type = AgentText(string)

        self.assertEqual(string, agent_type.to_string())
        self.assertEqual(string, agent_type.to_raw())
