#!/usr/bin/env python
# coding=utf-8

# Copyright 2024 The HuggingFace Inc. team. All rights reserved.
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
# [PACKAGE DOC]
# Description: Top-level package of smolagents. Defines the package version string (1.26.0) and re-exports,
#   via star imports, the public names (the __all__ lists) of the submodules agent_types, agents,
#   default_tools, gradio_ui, local_python_executor, mcp_client, memory, models, monitoring, remote_executors,
#   serialization, tools, utils and cli, so users can import everything from the package root. The import
#   order is significant: agent_types and agents are imported first and cli last to avoid a circular import
#   (cli imports from the package).
# Inputs: none (imported as a package).
# Outputs: module attribute __version__ plus every name re-exported from the listed submodules.
# Exceptions: none raised directly; ImportError from any submodule import propagates.
# Side effects: importing executes all listed submodules (their module-level code, including optional-
#   dependency checks).
__version__ = "1.26.0"

from .agent_types import *  # noqa: I001
from .agents import *  # Above noqa avoids a circular dependency due to cli.py
from .default_tools import *
from .gradio_ui import *
from .local_python_executor import *
from .mcp_client import *
from .memory import *
from .models import *
from .monitoring import *
from .remote_executors import *
from .serialization import *
from .tools import *
from .utils import *
from .cli import *
