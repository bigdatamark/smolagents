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
# Description: Executes the Python code blocks of every markdown file in docs/source/en as a script, one test
#   per doc file, to ensure the documentation examples run. Slow and needs API keys, so the test class is
#   gated by require_run_all.
# Inputs: depends on pytest, python-dotenv, subprocess and the docs tree; environment variables HF_TOKEN and
#   ANTHROPIC_API_KEY.
# Outputs: SubprocessCallException, run_command, DocCodeExtractor, TestDocs and the pytest_generate_tests
#   hook.
# Exceptions: none raised directly.
# Side effects: test collection may create a temporary directory and load .env (see pytest_generate_tests).

import ast
import os
import re
import shutil
import subprocess
import tempfile
import traceback
from pathlib import Path

import pytest
from dotenv import load_dotenv

from .utils.markers import require_run_all


class SubprocessCallException(Exception):
    # [CLASS DOC]
    # Description: HELPER: Exception raised by run_command when the subprocess exits non-zero.
    # Inputs: a message string.
    # Outputs: exception instance.
    # Exceptions: none raised directly.
    # Side effects: none.
    pass


def run_command(command: list[str], return_stdout=False, env=None):
    # [FUNCTION DOC]
    # Description: HELPER: Runs a command with stderr merged into stdout, converting Path items to strings,
    #   and optionally returns the decoded output.
    # Inputs: command (list of str or Path); return_stdout (bool, default False); env (dict or None, default a
    #   copy of os.environ).
    # Outputs: decoded UTF-8 output when return_stdout is True, otherwise None.
    # Exceptions: SubprocessCallException (chained from CalledProcessError) containing the command and its
    #   output on non-zero exit; FileNotFoundError if the executable is missing.
    # Side effects: spawns a subprocess; mutates the command list in place (Path to str).
    """
    Runs command with subprocess.check_output and returns stdout if requested.
    Properly captures and handles errors during command execution.
    """
    for i, c in enumerate(command):
        if isinstance(c, Path):
            command[i] = str(c)

    if env is None:
        env = os.environ.copy()

    try:
        output = subprocess.check_output(command, stderr=subprocess.STDOUT, env=env)
        if return_stdout:
            if hasattr(output, "decode"):
                output = output.decode("utf-8")
            return output
    except subprocess.CalledProcessError as e:
        raise SubprocessCallException(
            f"Command `{' '.join(command)}` failed with the following error:\n\n{e.output.decode()}"
        ) from e


class DocCodeExtractor:
    # [CLASS DOC]
    # Description: HELPER: Static helpers to pull Python code fences out of markdown and write them to a
    #   script file.
    # Inputs: no constructor args.
    # Outputs: instance.
    # Exceptions: none raised directly.
    # Side effects: none.
    """Handles extraction and validation of Python code from markdown files."""

    @staticmethod
    def extract_python_code(content: str) -> list[str]:
        # [FUNCTION DOC]
        # Description: HELPER: Finds all fenced blocks tagged python or py and returns their stripped bodies
        #   in order.
        # Inputs: content (str, markdown).
        # Outputs: list of str.
        # Exceptions: none raised directly.
        # Side effects: none.
        # KNOWN ISSUE: Fences must be immediately followed by a newline and closed by a newline plus fence;
        #   indented fences or fences with extra attributes are not matched.
        """Extract Python code blocks from markdown content."""
        pattern = r"```(?:python|py)\n(.*?)\n```"
        matches = re.finditer(pattern, content, re.DOTALL)
        return [match.group(1).strip() for match in matches]

    @staticmethod
    def create_test_script(code_blocks: list[str], tmp_dir: str) -> Path:
        # [FUNCTION DOC]
        # Description: HELPER: Joins the code blocks with blank lines and writes them to test_script.py in the
        #   given directory.
        # Inputs: code_blocks (list of str); tmp_dir (str).
        # Outputs: Path of the written file.
        # Exceptions: AssertionError if the combined code is empty; OSError on write failure.
        # Side effects: writes (overwrites) a file.
        """Create a temporary Python script from code blocks."""
        combined_code = "\n\n".join(code_blocks)
        assert len(combined_code) > 0, "Code is empty!"
        tmp_file = Path(tmp_dir) / "test_script.py"

        with open(tmp_file, "w", encoding="utf-8") as f:
            f.write(combined_code)

        return tmp_file


# Skip: slow tests + require API keys
@require_run_all
class TestDocs:
    # [CLASS DOC]
    # Description: Test class running each documentation file's code; gated by require_run_all.
    # Inputs: class attributes set by setup_class: _tmpdir, launch_args (python3), docs_dir, extractor,
    #   md_files.
    # Outputs: test outcomes.
    # Exceptions: AssertionError on failure.
    # Side effects: none.
    """Test case for documentation code testing."""

    @classmethod
    def setup_class(cls):
        # [FUNCTION DOC]
        # Description: Creates a temp dir, sets the docs directory, loads .env and collects all .md/.mdx files
        #   under docs/source/en.
        # Inputs: cls.
        # Outputs: None.
        # Exceptions: ValueError if the docs dir is missing or has no markdown files.
        # Side effects: creates a temp directory; loads .env into the environment; sets class attributes.
        cls._tmpdir = tempfile.mkdtemp()
        cls.launch_args = ["python3"]
        cls.docs_dir = Path(__file__).parent.parent / "docs" / "source" / "en"
        cls.extractor = DocCodeExtractor()

        if not cls.docs_dir.exists():
            raise ValueError(f"Docs directory not found at {cls.docs_dir}")

        load_dotenv()

        cls.md_files = list(cls.docs_dir.rglob("*.md")) + list(cls.docs_dir.rglob("*.mdx"))
        if not cls.md_files:
            raise ValueError(f"No markdown files found in {cls.docs_dir}")

    @classmethod
    def teardown_class(cls):
        # [FUNCTION DOC]
        # Description: Removes the class temp directory.
        # Inputs: cls.
        # Outputs: None.
        # Exceptions: OSError if removal fails.
        # Side effects: deletes files.
        shutil.rmtree(cls._tmpdir)

    @pytest.mark.timeout(100)
    def test_single_doc(self, doc_path: Path):
        # [FUNCTION DOC]
        # Description: Extracts the doc's Python blocks, drops blocks containing excluded snippets (expensive
        #   or non-runnable examples), skips if none remain, checks each block parses, substitutes HF token,
        #   Anthropic key and username placeholders, writes all blocks into one script and runs it with
        #   python3, failing the test on any error.
        # Inputs: doc_path (Path, parametrized per markdown file).
        # Outputs: none; skipped if no blocks.
        # Exceptions: SyntaxError from parsing; pytest failure (Failed) on execution errors; Skipped when no
        #   code.
        # Side effects: writes a script; spawns python3 that may make network/API calls; prints the code.
        # KNOWN ISSUE: If HF_TOKEN or ANTHROPIC_API_KEY is unset the replace call receives None and raises
        #   TypeError, reported only as an 'Unexpected error' failure.
        # KNOWN ISSUE: Unused loop index variable in the syntax-check loop.
        """Test a single documentation file."""
        with open(doc_path, "r", encoding="utf-8") as f:
            content = f.read()

        code_blocks = self.extractor.extract_python_code(content)
        excluded_snippets = [
            "ToolCollection",
            "image_generation_tool",  # We don't want to run this expensive operation
            "from_langchain",  # Langchain is not a dependency
            "while llm_should_continue(memory):",  # This is pseudo code
            "ollama_chat/llama3.2",  # Exclude ollama building in guided tour
            "model = TransformersModel(model_id=model_id)",  # Exclude testing with transformers model
            "SmolagentsInstrumentor",  # Exclude telemetry since it needs additional installs
        ]
        code_blocks = [
            block
            for block in code_blocks
            if not any(
                [snippet in block for snippet in excluded_snippets]
            )  # Exclude these tools that take longer to run and add dependencies
        ]
        if len(code_blocks) == 0:
            pytest.skip(f"No Python code blocks found in {doc_path.name}")

        # Validate syntax of each block individually by parsing it
        for i, block in enumerate(code_blocks, 1):
            ast.parse(block)

        # Create and execute test script
        print("\n\nCollected code block:==========\n".join(code_blocks))
        try:
            code_blocks = [
                (
                    block.replace("<YOUR_HUGGINGFACEHUB_API_TOKEN>", os.getenv("HF_TOKEN"))
                    .replace("YOUR_ANTHROPIC_API_KEY", os.getenv("ANTHROPIC_API_KEY"))
                    .replace("{your_username}", "m-ric")
                )
                for block in code_blocks
            ]
            test_script = self.extractor.create_test_script(code_blocks, self._tmpdir)
            run_command(self.launch_args + [str(test_script)])

        except SubprocessCallException as e:
            pytest.fail(f"\nError while testing {doc_path.name}:\n{str(e)}")
        except Exception:
            pytest.fail(f"\nUnexpected error while testing {doc_path.name}:\n{traceback.format_exc()}")

    @pytest.fixture(autouse=True)
    def _setup(self):
        # [FUNCTION DOC]
        # Description: HELPER: Autouse fixture: ensures the temp dir exists before each test and deletes the
        #   files in it afterwards.
        # Inputs: self.
        # Outputs: yields None.
        # Exceptions: OSError on file operations.
        # Side effects: creates/deletes files.
        """Fixture to ensure temporary directory exists for each test."""
        os.makedirs(self._tmpdir, exist_ok=True)
        yield
        # Clean up test files after each test
        for file in Path(self._tmpdir).glob("*"):
            file.unlink()


def pytest_generate_tests(metafunc):
    # [FUNCTION DOC]
    # Description: Pytest hook that parametrizes any test requesting doc_path with every collected markdown
    #   file (ids are file stems), calling the class's setup_class first if md_files is not yet set.
    # Inputs: metafunc (pytest Metafunc).
    # Outputs: None.
    # Exceptions: ValueError from setup_class.
    # Side effects: may create a temp dir and load .env during collection.
    # KNOWN ISSUE: The temp dir created here during collection is replaced by the one created again in
    #   setup_class and is never removed; collection also runs even when RUN_ALL is unset, so missing docs
    #   would error at collection time.
    """Generate test cases for each markdown file."""
    if "doc_path" in metafunc.fixturenames:
        test_class = metafunc.cls

        # Initialize the class if needed
        if not hasattr(test_class, "md_files"):
            test_class.setup_class()

        # Parameterize with the markdown files
        metafunc.parametrize("doc_path", test_class.md_files, ids=[f.stem for f in test_class.md_files])
