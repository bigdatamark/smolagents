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
# Purpose: Defines the wrapper types that agents return for non-trivial outputs (text, image, audio) plus
#   helpers that convert agent inputs/outputs to and from these wrappers.
# Exports: AgentType, AgentText, AgentImage, AgentAudio (public); also module-level handle_agent_input_types,
#   handle_agent_output_types and the private mapping from type names ("string", "image", "audio") to wrapper
#   classes.
# Dependencies: PIL (images), requests (remote audio download), smolagents.utils for optional package
#   detection; torch, numpy, soundfile and IPython are imported lazily only when needed.
# Side effects: creates a module logger; several methods write temporary files to fresh temp directories.
import logging
import os
import pathlib
import tempfile
import uuid
from io import BytesIO
from typing import Any

import PIL.Image
import requests

from .utils import _is_package_available


logger = logging.getLogger(__name__)


class AgentType:
    # [CLASS DOC]
    # Base wrapper for values returned by agents; subclasses also inherit from the native type they emulate
    #   (str, PIL image) so they behave like that type while adding to_raw/to_string conversions.
    # Inputs: value (any) - the wrapped value, stored as the private attribute _value.
    # Key methods: to_raw returns the underlying value; to_string returns a string form; str() delegates to
    #   to_string.
    # Exceptions: none raised directly.
    # Side effects: the base to_raw and to_string log an error warning that the type is unknown.
    """
    Abstract class to be reimplemented to define types that can be returned by agents.

    These objects serve three purposes:

    - They behave as they were the type they're meant to be, e.g., a string for text, a PIL.Image.Image for images
    - They can be stringified: str(object) in order to return a string defining the object
    - They should be displayed correctly in ipython notebooks/colab/jupyter
    """

    def __init__(self, value):
        # [FUNCTION DOC]
        # Stores the given value as the wrapped value.
        # Inputs: value (any).
        # Outputs: none.
        # Exceptions: none raised directly.
        # Side effects: sets the _value attribute on the instance.
        self._value = value

    def __str__(self):
        # [FUNCTION DOC]
        # Returns the string form of the wrapper by delegating to to_string.
        # Inputs: none.
        # Outputs: str from to_string (subclasses may return a path object rather than str, which would make
        #   str() fail).
        # Exceptions: whatever to_string raises; TypeError if to_string returns a non-str.
        # Side effects: those of to_string (may log or write temp files in subclasses).
        return self.to_string()

    def to_raw(self):
        # [FUNCTION DOC]
        # Returns the wrapped value unchanged, after logging that the wrapper is of unknown type.
        # Inputs: none.
        # Outputs: the stored value.
        # Exceptions: none raised directly.
        # Side effects: logs an error-level message via the module logger.
        logger.error(
            "This is a raw AgentType of unknown type. Display in notebooks and string conversion will be unreliable"
        )
        return self._value

    def to_string(self) -> str:
        # [FUNCTION DOC]
        # Returns the string conversion of the wrapped value, after logging that the wrapper is of unknown
        #   type.
        # Inputs: none.
        # Outputs: str of the stored value.
        # Exceptions: none raised directly.
        # Side effects: logs an error-level message via the module logger.
        logger.error(
            "This is a raw AgentType of unknown type. Display in notebooks and string conversion will be unreliable"
        )
        return str(self._value)


class AgentText(AgentType, str):
    # [CLASS DOC]
    # Text wrapper that is also a real str (multiple inheritance from AgentType and str), so it can be used
    #   anywhere a string is expected.
    # Inputs: value (str-like) - passed to both str construction and AgentType initialisation.
    # Outputs/attributes: _value holds the original value.
    # Exceptions: none raised directly.
    # Side effects: none.
    """
    Text type returned by the agent. Behaves as a string.
    """

    def to_raw(self):
        # [FUNCTION DOC]
        # Returns the original wrapped value.
        # Inputs: none.
        # Outputs: the stored value (unchanged).
        # Exceptions: none raised directly.
        # Side effects: none.
        return self._value

    def to_string(self):
        # [FUNCTION DOC]
        # Returns the wrapped value converted to str.
        # Inputs: none.
        # Outputs: str.
        # Exceptions: none raised directly.
        # Side effects: none.
        return str(self._value)


class AgentImage(AgentType, PIL.Image.Image):
    # [CLASS DOC]
    # Image wrapper that also subclasses PIL.Image.Image. Internally keeps up to three representations: a file
    #   path, a raw PIL image, and a tensor; conversions are computed lazily and cached.
    # Inputs: value - another AgentImage (representations copied), a PIL image, raw bytes (decoded via PIL), a
    #   str or pathlib path, or a torch tensor / numpy array (numpy converted to a torch tensor).
    # Attributes: _path, _raw, _tensor (any may be None, at least one is set).
    # Exceptions: TypeError if no representation could be derived from the value.
    # Side effects: to_string writes PNG files to new temp directories.
    """
    Image type returned by the agent. Behaves as a PIL.Image.Image.
    """

    def __init__(self, value):
        # [FUNCTION DOC]
        # Initialises both parent classes and records the value in the matching internal slot based on its
        #   type (AgentImage copy, PIL image, bytes decoded to PIL, path, torch tensor, or numpy array
        #   converted to a tensor).
        # Inputs: value (see class doc).
        # Outputs: none.
        # Exceptions: TypeError when the value is of an unsupported type; ModuleNotFoundError from torch/numpy
        #   import is swallowed (leading to the TypeError).
        # Side effects: may open/decode image bytes with PIL; imports torch and numpy lazily.
        # KNOWN ISSUE: a numpy array is only recognised if torch is importable, since the numpy branch sits
        #   inside the torch import attempt.
        AgentType.__init__(self, value)
        PIL.Image.Image.__init__(self)

        self._path = None
        self._raw = None
        self._tensor = None

        if isinstance(value, AgentImage):
            self._raw, self._path, self._tensor = value._raw, value._path, value._tensor
        elif isinstance(value, PIL.Image.Image):
            self._raw = value
        elif isinstance(value, bytes):
            self._raw = PIL.Image.open(BytesIO(value))
        elif isinstance(value, (str, pathlib.Path)):
            self._path = value
        else:
            try:
                import torch

                if isinstance(value, torch.Tensor):
                    self._tensor = value
                import numpy as np

                if isinstance(value, np.ndarray):
                    self._tensor = torch.from_numpy(value)
            except ModuleNotFoundError:
                pass

        if self._path is None and self._raw is None and self._tensor is None:
            raise TypeError(f"Unsupported type for {self.__class__.__name__}: {type(value)}")

    def _ipython_display_(self, include=None, exclude=None):
        # [FUNCTION DOC]
        # HELPER: IPython display hook; renders the image in a notebook by displaying the file produced by
        #   to_string.
        # Inputs: include, exclude (ignored; part of the IPython protocol).
        # Outputs: none.
        # Exceptions: ImportError if IPython is not installed; errors from to_string.
        # Side effects: may write a temp PNG (via to_string) and emits notebook display output.
        """
        Displays correctly this type in an ipython notebook (ipython, colab, jupyter, ...)
        """
        from IPython.display import Image, display

        display(Image(self.to_string()))

    def to_raw(self):
        # [FUNCTION DOC]
        # Returns the image as a PIL image: the cached raw image if present, otherwise opens and caches the
        #   file at the path, otherwise builds an image from the tensor.
        # Inputs: none.
        # Outputs: PIL.Image.Image (or None if no representation is set, which the constructor prevents).
        # Exceptions: errors from PIL opening the file; numpy import errors.
        # Side effects: caches the opened image in _raw when loading from a path (the tensor-derived image is
        #   not cached).
        # KNOWN ISSUE: the tensor path computes 255 minus value times 255, i.e. it inverts and rescales the
        #   values, which is wrong for tensors already in 0-255 range (as in the tests) and inverts 0-1
        #   images.
        """
        Returns the "raw" version of that object. In the case of an AgentImage, it is a PIL.Image.Image.
        """
        if self._raw is not None:
            return self._raw

        if self._path is not None:
            self._raw = PIL.Image.open(self._path)
            return self._raw

        if self._tensor is not None:
            import numpy as np

            array = self._tensor.cpu().detach().numpy()
            return PIL.Image.fromarray((255 - array * 255).astype(np.uint8))

    def to_string(self):
        # [FUNCTION DOC]
        # Returns a filesystem path for the image: the existing path if known, otherwise saves the raw PIL
        #   image or the tensor-derived image as PNG to a new temp directory with a UUID filename, caches and
        #   returns that path.
        # Inputs: none.
        # Outputs: the path (str, or the original pathlib path if one was given), or None if nothing is set.
        # Exceptions: errors from PIL saving or numpy conversion.
        # Side effects: creates a temp directory and writes a PNG file; sets _path.
        # KNOWN ISSUE: can return a pathlib.Path (not a str) when constructed from a Path; same value
        #   inversion as to_raw for tensors.
        """
        Returns the stringified version of that object. In the case of an AgentImage, it is a path to the serialized
        version of the image.
        """
        if self._path is not None:
            return self._path

        if self._raw is not None:
            directory = tempfile.mkdtemp()
            self._path = os.path.join(directory, str(uuid.uuid4()) + ".png")
            self._raw.save(self._path, format="png")
            return self._path

        if self._tensor is not None:
            import numpy as np

            array = self._tensor.cpu().detach().numpy()

            # There is likely simpler than load into image into save
            img = PIL.Image.fromarray((255 - array * 255).astype(np.uint8))

            directory = tempfile.mkdtemp()
            self._path = os.path.join(directory, str(uuid.uuid4()) + ".png")
            img.save(self._path, format="png")

            return self._path

    def save(self, output_bytes, format: str = None, **params):
        # [FUNCTION DOC]
        # Saves the image (obtained via to_raw) to the given destination using PIL.
        # Inputs: output_bytes (file path or writable binary file object), format (str or None, PIL format
        #   name), **params (extra PIL save options).
        # Outputs: none.
        # Exceptions: errors from PIL save; AttributeError if to_raw returns None.
        # Side effects: writes to the given destination.
        """
        Saves the image to a file.
        Args:
            output_bytes (bytes): The output bytes to save the image to.
            format (str): The format to use for the output image. The format is the same as in PIL.Image.save.
            **params: Additional parameters to pass to PIL.Image.save.
        """
        img = self.to_raw()
        img.save(output_bytes, format=format, **params)


class AgentAudio(AgentType, str):
    # [CLASS DOC]
    # Audio wrapper that also subclasses str. Holds either a path/URL or a tensor of samples plus a sample
    #   rate; requires the optional soundfile and torch packages.
    # Inputs: value - str or pathlib path (local path or URL), torch tensor, or a tuple of (samplerate,
    #   samples) where samples is a numpy array or sequence; samplerate (int, default 16000).
    # Attributes: _path, _tensor, samplerate.
    # Exceptions: ModuleNotFoundError if soundfile or torch is missing; ValueError for unsupported value
    #   types.
    # Side effects: to_string writes WAV files to temp directories; to_raw may download over HTTP.
    """
    Audio type returned by the agent.
    """

    def __init__(self, value, samplerate=16_000):
        # [FUNCTION DOC]
        # Checks optional dependencies, initialises the base wrapper and stores the value in the path or
        #   tensor slot according to its type; a tuple sets the sample rate from its first element and
        #   converts its second element to a tensor.
        # Inputs: value (see class doc), samplerate (int, default 16000).
        # Outputs: none.
        # Exceptions: ModuleNotFoundError when soundfile or torch is unavailable; ValueError for unsupported
        #   types.
        # Side effects: imports numpy and torch.
        if not _is_package_available("soundfile") or not _is_package_available("torch"):
            raise ModuleNotFoundError(
                "Please install 'audio' extra to use AgentAudio: `pip install 'smolagents[audio]'`"
            )
        import numpy as np
        import torch

        super().__init__(value)

        self._path = None
        self._tensor = None

        self.samplerate = samplerate
        if isinstance(value, (str, pathlib.Path)):
            self._path = value
        elif isinstance(value, torch.Tensor):
            self._tensor = value
        elif isinstance(value, tuple):
            self.samplerate = value[0]
            if isinstance(value[1], np.ndarray):
                self._tensor = torch.from_numpy(value[1])
            else:
                self._tensor = torch.tensor(value[1])
        else:
            raise ValueError(f"Unsupported audio type: {type(value)}")

    def _ipython_display_(self, include=None, exclude=None):
        # [FUNCTION DOC]
        # HELPER: IPython display hook; renders an audio player for the file returned by to_string at the
        #   stored sample rate.
        # Inputs: include, exclude (ignored).
        # Outputs: none.
        # Exceptions: ImportError if IPython is missing; errors from to_string.
        # Side effects: may write a temp WAV file; emits notebook display output.
        """
        Displays correctly this type in an ipython notebook (ipython, colab, jupyter, ...)
        """
        from IPython.display import Audio, display

        display(Audio(self.to_string(), rate=self.samplerate))

    def to_raw(self):
        # [FUNCTION DOC]
        # Returns the audio as a torch tensor: the cached tensor if present, otherwise reads the path
        #   (downloading it with an HTTP GET if it contains "://") via soundfile, caches the tensor and
        #   updates the sample rate from the file.
        # Inputs: none.
        # Outputs: torch.Tensor (or None if neither path nor tensor is set).
        # Exceptions: ImportError for soundfile/torch; requests HTTPError for failed downloads; soundfile read
        #   errors.
        # Side effects: possible network request; mutates _tensor and samplerate.
        """
        Returns the "raw" version of that object. It is a `torch.Tensor` object.
        """
        import soundfile as sf

        if self._tensor is not None:
            return self._tensor

        import torch

        if self._path is not None:
            if "://" in str(self._path):
                response = requests.get(self._path)
                response.raise_for_status()
                tensor, self.samplerate = sf.read(BytesIO(response.content))
            else:
                tensor, self.samplerate = sf.read(self._path)
            self._tensor = torch.tensor(tensor)
            return self._tensor

    def to_string(self):
        # [FUNCTION DOC]
        # Returns a path for the audio: the stored path if any, otherwise writes the tensor as a WAV file
        #   (with the stored sample rate) to a new temp directory, caches and returns the path.
        # Inputs: none.
        # Outputs: path (str or pathlib path), or None if nothing is set.
        # Exceptions: ImportError for soundfile; soundfile write errors.
        # Side effects: creates temp directory and WAV file; sets _path.
        """
        Returns the stringified version of that object. In the case of an AgentAudio, it is a path to the serialized
        version of the audio.
        """
        import soundfile as sf

        if self._path is not None:
            return self._path

        if self._tensor is not None:
            directory = tempfile.mkdtemp()
            self._path = os.path.join(directory, str(uuid.uuid4()) + ".wav")
            sf.write(self._path, self._tensor, samplerate=self.samplerate)
            return self._path


_AGENT_TYPE_MAPPING = {"string": AgentText, "image": AgentImage, "audio": AgentAudio}


def handle_agent_input_types(*args, **kwargs):
    # [FUNCTION DOC]
    # Unwraps agent-type arguments before passing them to a tool: each positional or keyword value that is an
    #   AgentType is replaced by its to_raw result.
    # Inputs: *args, **kwargs (any values).
    # Outputs: a tuple of (list of positional values, dict of keyword values).
    # Exceptions: whatever to_raw raises.
    # Side effects: those of to_raw (e.g. file reads, downloads).
    # KNOWN ISSUE: positional args are returned as a list, not a tuple.
    args = [(arg.to_raw() if isinstance(arg, AgentType) else arg) for arg in args]
    kwargs = {k: (v.to_raw() if isinstance(v, AgentType) else v) for k, v in kwargs.items()}
    return args, kwargs


def handle_agent_output_types(output: Any, output_type: str | None = None) -> Any:
    # [FUNCTION DOC]
    # Wraps a tool/agent output in the appropriate agent type. If output_type is a known name ("string",
    #   "image", "audio") the matching wrapper is used; otherwise the wrapper is chosen from the runtime type
    #   (str to AgentText, PIL image to AgentImage, torch tensor to AgentAudio); anything else is returned
    #   unchanged.
    # Inputs: output (any), output_type (str or None, default None).
    # Outputs: the wrapped value or the original output.
    # Exceptions: errors from wrapper constructors (TypeError, ValueError, ModuleNotFoundError for audio
    #   deps).
    # Side effects: none beyond wrapper construction.
    if output_type in _AGENT_TYPE_MAPPING:
        # If the class has defined outputs, we can map directly according to the class definition
        decoded_outputs = _AGENT_TYPE_MAPPING[output_type](output)
        return decoded_outputs

    # If the class does not have defined output, then we map according to the type
    if isinstance(output, str):
        return AgentText(output)
    if isinstance(output, PIL.Image.Image):
        return AgentImage(output)
    try:
        import torch

        if isinstance(output, torch.Tensor):
            return AgentAudio(output)
    except ModuleNotFoundError:
        pass
    return output


__all__ = ["AgentType", "AgentImage", "AgentText", "AgentAudio"]
