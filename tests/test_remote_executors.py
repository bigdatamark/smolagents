# [MODULE DOC]
# Purpose: tests for smolagents.remote_executors. Unit tests use mocks for RemotePythonExecutor (tool/variable
#   shipping, final-answer deserialization), E2BExecutor, DockerExecutor, ModalExecutor and BlaxelExecutor;
#   integration tests (marked require_run_all) exercise real E2B, Docker and Modal backends.
# Inputs/dependencies: pytest, unittest.mock, docker, PIL, rich Console, smolagents
#   tools/logger/serialization, tests.utils.markers.require_run_all. Exports: test classes and the
#   e2b_executor fixture.
# Exceptions: import fails if docker or PIL is not installed.
# Side effects: integration tests create real sandboxes/containers; unit tests patch modules (some patchers
#   are started manually).
import importlib
import io
from textwrap import dedent
from unittest.mock import MagicMock, patch

import docker
import PIL.Image
import pytest
from rich.console import Console

from smolagents.default_tools import FinalAnswerTool, WikipediaSearchTool
from smolagents.local_python_executor import CodeOutput
from smolagents.monitoring import AgentLogger, LogLevel
from smolagents.remote_executors import (
    BlaxelExecutor,
    DockerExecutor,
    E2BExecutor,
    ModalExecutor,
    RemotePythonExecutor,
)
from smolagents.serialization import SerializationError
from smolagents.utils import AgentError

from .utils.markers import require_run_all


class TestRemotePythonExecutor:
    # [CLASS DOC]
    # Unit tests for the base RemotePythonExecutor with run_code_raise_errors mocked.
    # Inputs: none. Attributes: none.
    def test_send_tools_empty_tools(self):
        # [FUNCTION DOC]
        # Sending an empty tool dict runs code exactly once and never issues a pip install.
        # Inputs: self.
        # Outputs: none (pytest assertions).
        # Exceptions: AssertionError on failure.
        # Side effects: none (run_code_raise_errors mocked).
        # KNOWN ISSUE: relies on get_tools_definition_code returning non-empty code (base imports) even for no
        #   tools.
        executor = RemotePythonExecutor(additional_imports=[], logger=MagicMock())
        executor.run_code_raise_errors = MagicMock()
        executor.send_tools({})
        assert executor.run_code_raise_errors.call_count == 1
        # No new packages should be installed
        assert "!pip install" not in executor.run_code_raise_errors.call_args.args[0]

    def test_send_variables_with_empty_dict_is_noop(self):
        # [FUNCTION DOC]
        # Empty variables dict causes no remote execution.
        # Inputs: self.
        # Outputs: none (pytest assertions).
        # Exceptions: AssertionError on failure.
        # Side effects: none.
        executor = RemotePythonExecutor(additional_imports=[], logger=MagicMock())
        executor.run_code_raise_errors = MagicMock()
        executor.send_variables({})
        assert executor.run_code_raise_errors.call_count == 0

    def test_send_variables_non_empty_generates_executable_deserializer_code(self):
        # [FUNCTION DOC]
        # Executes the generated code locally and checks int, tuple and bytes values round-trip in safe mode.
        # Inputs: self.
        # Outputs: none (pytest assertions).
        # Exceptions: AssertionError on failure.
        # Side effects: execs generated code into a local dict.
        executor = RemotePythonExecutor(additional_imports=[], logger=MagicMock(), allow_pickle=False)
        executor.run_code_raise_errors = MagicMock()

        variables = {
            "counter": 1,
            "tags": ("a", "b"),
            "blob": b"binary",
        }
        executor.send_variables(variables)

        sent_code = executor.run_code_raise_errors.call_args.args[0]
        remote_scope = {}
        exec(sent_code, remote_scope, remote_scope)

        assert remote_scope["counter"] == 1
        assert remote_scope["tags"] == ("a", "b")
        assert remote_scope["blob"] == b"binary"

    def test_send_variables_allow_pickle_handles_prefixed_payload(self):
        # [FUNCTION DOC]
        # With allow_pickle, an exception object round-trips through the generated code.
        # Inputs: self.
        # Outputs: none (pytest assertions).
        # Exceptions: AssertionError on failure.
        # Side effects: execs generated code (including unpickling) into a local dict.
        executor = RemotePythonExecutor(additional_imports=[], logger=MagicMock(), allow_pickle=True)
        executor.run_code_raise_errors = MagicMock()

        variables = {"error": ValueError("boom")}
        executor.send_variables(variables)

        sent_code = executor.run_code_raise_errors.call_args.args[0]
        remote_scope = {}
        exec(sent_code, remote_scope, remote_scope)

        assert isinstance(remote_scope["error"], ValueError)
        assert str(remote_scope["error"]) == "boom"

    def test_deserialize_final_answer_rejects_unprefixed_payload(self):
        # [FUNCTION DOC]
        # An unprefixed final-answer string raises SerializationError mentioning the unknown format, even with
        #   pickle allowed.
        # Inputs: self.
        # Outputs: none (pytest assertions).
        # Exceptions: AssertionError on failure.
        # Side effects: none.
        with pytest.raises(SerializationError, match="Unknown final answer format"):
            RemotePythonExecutor._deserialize_final_answer("legacy-unprefixed-payload", allow_pickle=True)

    @require_run_all
    def test_send_tools_with_default_wikipedia_search_tool(self):
        # [FUNCTION DOC]
        # (require_run_all) Sending WikipediaSearchTool first pip-installs wikipedia-api then sends the tool
        #   class code.
        # Inputs: self.
        # Outputs: none (pytest assertions).
        # Exceptions: AssertionError on failure.
        # Side effects: none (run_code_raise_errors mocked).
        tool = WikipediaSearchTool()
        executor = RemotePythonExecutor(additional_imports=[], logger=MagicMock())
        executor.run_code_raise_errors = MagicMock()
        executor.send_tools({"wikipedia_search": tool})
        assert executor.run_code_raise_errors.call_count == 2
        assert "!pip install wikipedia-api" == executor.run_code_raise_errors.call_args_list[0].args[0]
        assert "class WikipediaSearchTool(Tool)" in executor.run_code_raise_errors.call_args_list[1].args[0]


class TestE2BExecutorUnit:
    # [CLASS DOC]
    # Unit tests for E2BExecutor with the E2B Sandbox patched.
    # Inputs: none.
    def test_e2b_executor_instantiation(self):
        # [FUNCTION DOC]
        # Checks the executor stores the logger and sandbox, and that the Sandbox constructor or create
        #   factory (v1/v2) was called once with exactly the passed kwargs.
        # Inputs: self.
        # Outputs: none (pytest assertions).
        # Exceptions: AssertionError on failure.
        # Side effects: patches e2b_code_interpreter.Sandbox.
        logger = MagicMock()
        with patch("e2b_code_interpreter.Sandbox") as mock_sandbox:
            mock_sandbox.return_value.commands.run.return_value.error = None
            mock_sandbox.return_value.run_code.return_value.error = None
            # Also set up v2 path in case Sandbox.create is used
            mock_sandbox.create.return_value.commands.run.return_value.error = None
            mock_sandbox.create.return_value.run_code.return_value.error = None
            executor = E2BExecutor(
                additional_imports=[], logger=logger, api_key="dummy-api-key", template="dummy-template-id", timeout=60
            )
        assert isinstance(executor, E2BExecutor)
        assert executor.logger == logger
        # Support both e2b v1 (Sandbox(...)) and v2 (Sandbox.create(...))
        if mock_sandbox.create.called:
            sandbox_obj = mock_sandbox.create.return_value
            called_ctor = mock_sandbox.create
        else:
            sandbox_obj = mock_sandbox.return_value
            called_ctor = mock_sandbox
        assert executor.sandbox == sandbox_obj
        assert called_ctor.call_count == 1
        assert called_ctor.call_args.kwargs == {
            "api_key": "dummy-api-key",
            "template": "dummy-template-id",
            "timeout": 60,
        }

    def test_cleanup(self):
        # [FUNCTION DOC]
        # cleanup kills the sandbox once and logs at least twice.
        # Inputs: self.
        # Outputs: none (pytest assertions).
        # Exceptions: AssertionError on failure.
        # Side effects: patches e2b_code_interpreter.Sandbox.
        """Test that the cleanup method properly shuts down the sandbox"""
        logger = MagicMock()
        with patch("e2b_code_interpreter.Sandbox") as mock_sandbox:
            # Setup mock
            mock_sandbox.return_value.kill = MagicMock()
            # Also set up v2 path in case Sandbox.create is used
            mock_sandbox.create.return_value.kill = MagicMock()

            # Create executor
            executor = E2BExecutor(additional_imports=[], logger=logger, api_key="dummy-api-key")

            # Call cleanup
            executor.cleanup()

            # Verify sandbox was killed
            if mock_sandbox.create.called:
                mock_sandbox.create.return_value.kill.assert_called_once()
            else:
                mock_sandbox.return_value.kill.assert_called_once()
            assert logger.log.call_count >= 2  # Should log start and completion messages


@pytest.fixture
def e2b_executor():
    # [FUNCTION DOC]
    # HELPER: pytest fixture creating a real E2BExecutor (pillow and numpy installed) with an in-memory INFO
    #   logger, yielding it and cleaning up afterwards.
    # Inputs: none.
    # Outputs: yields E2BExecutor.
    # Exceptions: E2B creation errors propagate (requires credentials).
    # Side effects: creates and kills a real E2B sandbox.
    executor = E2BExecutor(
        additional_imports=["pillow", "numpy"],
        logger=AgentLogger(LogLevel.INFO, Console(force_terminal=False, file=io.StringIO())),
    )
    yield executor
    executor.cleanup()


@require_run_all
class TestE2BExecutorIntegration:
    @pytest.fixture(autouse=True)
    # [CLASS DOC]
    # (require_run_all) Integration tests against a real E2B sandbox.
    # Attributes: executor (set by autouse fixture).
    def set_executor(self, e2b_executor):
        # [FUNCTION DOC]
        # HELPER: Autouse fixture storing the e2b_executor fixture on self.executor.
        # Inputs: e2b_executor.
        # Outputs: none.
        # Exceptions: none raised directly.
        # Side effects: sets self.executor.
        self.executor = e2b_executor

    @pytest.mark.parametrize(
        "code_action, expected_result",
        [
            (
                dedent('''
                    final_answer("""This is
                    a multiline
                    final answer""")
                '''),
                "This is\na multiline\nfinal answer",
            ),
            (
                dedent("""
                    text = '''Text containing
                    final_answer(5)
                    '''
                    final_answer(text)
                """),
                "Text containing\nfinal_answer(5)\n",
            ),
            (
                dedent("""
                    num = 2
                    if num == 1:
                        final_answer("One")
                    elif num == 2:
                        final_answer("Two")
                """),
                "Two",
            ),
        ],
    )
    def test_final_answer_patterns(self, code_action, expected_result):
        # [FUNCTION DOC]
        # Parametrized: sends FinalAnswerTool, runs code calling final_answer in various forms (multiline
        #   string, text mentioning final_answer, conditional branch) and checks is_final_answer and the exact
        #   output.
        # Inputs: self, code_action (str), expected_result (str).
        # Outputs: none (pytest assertions).
        # Exceptions: AssertionError on failure.
        # Side effects: real remote execution (network/containers).
        self.executor.send_tools({"final_answer": FinalAnswerTool()})
        code_output = self.executor(code_action)
        assert code_output.is_final_answer is True
        assert code_output.output == expected_result

    def test_custom_final_answer(self):
        # [FUNCTION DOC]
        # Sends a FinalAnswerTool subclass whose forward prefixes 'CUSTOM' and checks the final answer is the
        #   transformed value.
        # Inputs: self.
        # Outputs: none (pytest assertions).
        # Exceptions: AssertionError on failure.
        # Side effects: real remote execution (network/containers).
        class CustomFinalAnswerTool(FinalAnswerTool):
            # [CLASS DOC]
            # HELPER: Test FinalAnswerTool subclass overriding forward.
            # Inputs: none (default tool constructor). Outputs/Exceptions/Side effects: none.
            def forward(self, answer: str) -> str:
                # [FUNCTION DOC]
                # HELPER: Returns 'CUSTOM' prepended to answer.
                # Inputs: self plus the answer argument(s) (str).
                # Outputs: the concatenated str.
                # Exceptions: none raised directly.
                # Side effects: none.
                return "CUSTOM" + answer

        self.executor.send_tools({"final_answer": CustomFinalAnswerTool()})
        code_action = dedent("""
            final_answer(answer="_answer")
        """)
        code_output = self.executor(code_action)
        assert code_output.is_final_answer is True
        assert code_output.output == "CUSTOM_answer"

    def test_custom_final_answer_with_custom_inputs(self):
        # [FUNCTION DOC]
        # Sends a FinalAnswerTool subclass with two string inputs and checks keyword-argument final answers
        #   are combined correctly.
        # Inputs: self.
        # Outputs: none (pytest assertions).
        # Exceptions: AssertionError on failure.
        # Side effects: real remote execution (network/containers).
        class CustomFinalAnswerToolWithCustomInputs(FinalAnswerTool):
            # [CLASS DOC]
            # HELPER: Test FinalAnswerTool subclass with inputs answer1 and answer2.
            # Inputs: none (default tool constructor). Outputs/Exceptions/Side effects: none.
            inputs = {
                "answer1": {"type": "string", "description": "First part of the answer."},
                "answer2": {"type": "string", "description": "Second part of the answer."},
            }

            def forward(self, answer1: str, answer2: str) -> str:
                # [FUNCTION DOC]
                # HELPER: Returns answer1 + 'CUSTOM' + answer2.
                # Inputs: self plus the answer argument(s) (str).
                # Outputs: the concatenated str.
                # Exceptions: none raised directly.
                # Side effects: none.
                return answer1 + "CUSTOM" + answer2

        self.executor.send_tools({"final_answer": CustomFinalAnswerToolWithCustomInputs()})
        code_action = dedent("""
            final_answer(
                answer1="answer1_",
                answer2="_answer2"
            )
        """)
        code_output = self.executor(code_action)
        assert code_output.is_final_answer is True
        assert code_output.output == "answer1_CUSTOM_answer2"


class TestDockerExecutorUnit:
    # [CLASS DOC]
    # Unit tests for DockerExecutor with docker, requests and websocket patched.
    # Inputs: none.
    def test_cleanup(self):
        # [FUNCTION DOC]
        # Builds a DockerExecutor against mocked Docker/HTTP (existing image, running container, kernel
        #   created) and checks cleanup stops and removes the container once.
        # Inputs: self.
        # Outputs: none (pytest assertions).
        # Exceptions: AssertionError on failure.
        # Side effects: patches docker.from_env, requests.get/post, websocket.create_connection.
        """Test that cleanup properly stops and removes the container"""
        logger = MagicMock()
        with (
            patch("docker.from_env") as mock_docker_client,
            patch("requests.get") as mock_get,
            patch("requests.post") as mock_post,
            patch("websocket.create_connection"),
        ):
            # Setup mocks
            mock_container = MagicMock()
            mock_container.status = "running"
            mock_container.short_id = "test123"

            mock_docker_client.return_value.containers.run.return_value = mock_container
            mock_docker_client.return_value.images.get.return_value = MagicMock()

            mock_get.return_value.status_code = 200
            mock_post.return_value.status_code = 201
            mock_post.return_value.json.return_value = {"id": "test-kernel-id"}

            # Create executor
            executor = DockerExecutor(additional_imports=[], logger=logger, build_new_image=False)

            # Call cleanup
            executor.cleanup()

            # Verify container was stopped and removed
            mock_container.stop.assert_called_once()
            mock_container.remove.assert_called_once()


class CommonDockerExecutorIntegration:
    @pytest.fixture(autouse=True)
    # [CLASS DOC]
    # HELPER: Mixin of integration tests shared by Docker and Modal executors; subclasses must provide a
    #   custom_executor fixture. Not collected itself (name lacks Test prefix).
    # Attributes: executor (set by autouse fixture).
    def set_executor(self, custom_executor):
        # [FUNCTION DOC]
        # HELPER: Autouse fixture storing custom_executor on self.executor.
        # Inputs: custom_executor.
        # Outputs: none.
        # Exceptions: none raised directly.
        # Side effects: sets self.executor.
        self.executor = custom_executor

    def test_state_persistence(self):
        # [FUNCTION DOC]
        # Imports and variables from one execution remain available in the next (prints sqrt(2)).
        # Inputs: none (uses self.executor from the autouse fixture).
        # Outputs: none (pytest assertions).
        # Exceptions: AssertionError on failure.
        # Side effects: real remote execution (network/containers).
        """Test that variables and imports form one snippet persist in the next"""
        code_action = "import numpy as np; a = 2"
        self.executor(code_action)

        code_action = "print(np.sqrt(a))"
        code_output = self.executor(code_action)
        assert "1.41421" in code_output.logs

    def test_execute_output(self):
        # [FUNCTION DOC]
        # final_answer with a literal string returns that string.
        # Inputs: none (uses self.executor from the autouse fixture).
        # Outputs: none (pytest assertions).
        # Exceptions: AssertionError on failure.
        # Side effects: real remote execution (network/containers).
        """Test execution that returns a string"""
        self.executor.send_tools({"final_answer": FinalAnswerTool()})
        code_action = 'final_answer("This is the final answer")'
        code_output = self.executor(code_action)
        assert code_output.output == "This is the final answer", "Result should be 'This is the final answer'"

    def test_execute_multiline_output(self):
        # [FUNCTION DOC]
        # final_answer of a variable set on a previous line returns its value.
        # Inputs: none (uses self.executor from the autouse fixture).
        # Outputs: none (pytest assertions).
        # Exceptions: AssertionError on failure.
        # Side effects: real remote execution (network/containers).
        """Test execution that returns a string"""
        self.executor.send_tools({"final_answer": FinalAnswerTool()})
        code_action = 'result = "This is the final answer"\nfinal_answer(result)'
        code_output = self.executor(code_action)
        assert code_output.output == "This is the final answer", "Result should be 'This is the final answer'"

    def test_execute_image_output(self):
        # [FUNCTION DOC]
        # final_answer with a PIL image returns a PIL image.
        # Inputs: none (uses self.executor from the autouse fixture).
        # Outputs: none (pytest assertions).
        # Exceptions: AssertionError on failure.
        # Side effects: real remote execution (network/containers).
        """Test execution that returns a base64 image"""
        self.executor.send_tools({"final_answer": FinalAnswerTool()})
        code_action = dedent("""
            import base64
            from PIL import Image
            from io import BytesIO
            image = Image.new("RGB", (10, 10), (255, 0, 0))
            final_answer(image)
        """)
        code_output = self.executor(code_action)
        assert isinstance(code_output.output, PIL.Image.Image), "Result should be a PIL Image"

    def test_syntax_error_handling(self):
        # [FUNCTION DOC]
        # A syntax error raises AgentError whose message contains SyntaxError.
        # Inputs: none (uses self.executor from the autouse fixture).
        # Outputs: none (pytest assertions).
        # Exceptions: AssertionError on failure.
        # Side effects: real remote execution (network/containers).
        """Test handling of syntax errors"""
        code_action = 'print("Missing Parenthesis'  # Syntax error
        with pytest.raises(AgentError) as exception_info:
            self.executor(code_action)
        assert "SyntaxError" in str(exception_info.value), "Should raise a syntax error"

    @pytest.mark.parametrize(
        "code_action, expected_result",
        [
            (
                dedent('''
                    final_answer("""This is
                    a multiline
                    final answer""")
                '''),
                "This is\na multiline\nfinal answer",
            ),
            (
                dedent("""
                    text = '''Text containing
                    final_answer(5)
                    '''
                    final_answer(text)
                """),
                "Text containing\nfinal_answer(5)\n",
            ),
            (
                dedent("""
                    num = 2
                    if num == 1:
                        final_answer("One")
                    elif num == 2:
                        final_answer("Two")
                """),
                "Two",
            ),
        ],
    )
    def test_final_answer_patterns(self, code_action, expected_result):
        # [FUNCTION DOC]
        # Parametrized: sends FinalAnswerTool, runs code calling final_answer in various forms (multiline
        #   string, text mentioning final_answer, conditional branch) and checks is_final_answer and the exact
        #   output.
        # Inputs: self, code_action (str), expected_result (str).
        # Outputs: none (pytest assertions).
        # Exceptions: AssertionError on failure.
        # Side effects: real remote execution (network/containers).
        self.executor.send_tools({"final_answer": FinalAnswerTool()})
        code_output = self.executor(code_action)
        assert code_output.is_final_answer is True
        assert code_output.output == expected_result

    def test_custom_final_answer(self):
        # [FUNCTION DOC]
        # Sends a FinalAnswerTool subclass whose forward prefixes 'CUSTOM' and checks the final answer is the
        #   transformed value.
        # Inputs: self.
        # Outputs: none (pytest assertions).
        # Exceptions: AssertionError on failure.
        # Side effects: real remote execution (network/containers).
        class CustomFinalAnswerTool(FinalAnswerTool):
            # [CLASS DOC]
            # HELPER: Test FinalAnswerTool subclass overriding forward.
            # Inputs: none (default tool constructor). Outputs/Exceptions/Side effects: none.
            def forward(self, answer: str) -> str:
                # [FUNCTION DOC]
                # HELPER: Returns 'CUSTOM' prepended to answer.
                # Inputs: self plus the answer argument(s) (str).
                # Outputs: the concatenated str.
                # Exceptions: none raised directly.
                # Side effects: none.
                return "CUSTOM" + answer

        self.executor.send_tools({"final_answer": CustomFinalAnswerTool()})
        code_action = dedent("""
            final_answer(answer="_answer")
        """)
        code_output = self.executor(code_action)
        assert code_output.is_final_answer is True
        assert code_output.output == "CUSTOM_answer"

    def test_custom_final_answer_with_custom_inputs(self):
        # [FUNCTION DOC]
        # Sends a FinalAnswerTool subclass with two string inputs and checks keyword-argument final answers
        #   are combined correctly.
        # Inputs: self.
        # Outputs: none (pytest assertions).
        # Exceptions: AssertionError on failure.
        # Side effects: real remote execution (network/containers).
        class CustomFinalAnswerToolWithCustomInputs(FinalAnswerTool):
            # [CLASS DOC]
            # HELPER: Test FinalAnswerTool subclass with inputs answer1 and answer2.
            # Inputs: none (default tool constructor). Outputs/Exceptions/Side effects: none.
            inputs = {
                "answer1": {"type": "string", "description": "First part of the answer."},
                "answer2": {"type": "string", "description": "Second part of the answer."},
            }

            def forward(self, answer1: str, answer2: str) -> str:
                # [FUNCTION DOC]
                # HELPER: Returns answer1 + 'CUSTOM' + answer2.
                # Inputs: self plus the answer argument(s) (str).
                # Outputs: the concatenated str.
                # Exceptions: none raised directly.
                # Side effects: none.
                return answer1 + "CUSTOM" + answer2

        self.executor.send_tools({"final_answer": CustomFinalAnswerToolWithCustomInputs()})
        code_action = dedent("""
            final_answer(
                answer1="answer1_",
                answer2="_answer2"
            )
        """)
        code_output = self.executor(code_action)
        assert code_output.is_final_answer is True
        assert code_output.output == "answer1_CUSTOM_answer2"


@require_run_all
class TestDockerExecutorIntegration(CommonDockerExecutorIntegration):
    @pytest.fixture
    # [CLASS DOC]
    # (require_run_all) Runs the common integration tests against a real DockerExecutor plus Docker-specific
    #   checks.
    # Attributes: executor.
    def custom_executor(self):
        # [FUNCTION DOC]
        # HELPER: Fixture creating a real DockerExecutor (pillow, numpy) and deleting it after the test.
        # Inputs: self.
        # Outputs: yields DockerExecutor.
        # Exceptions: Docker errors propagate.
        # Side effects: builds image, starts/stops a container.
        executor = DockerExecutor(
            additional_imports=["pillow", "numpy"],
            logger=AgentLogger(LogLevel.INFO, Console(force_terminal=False, file=io.StringIO())),
        )
        yield executor
        executor.delete()

    def test_initialization(self):
        # [FUNCTION DOC]
        # The container attribute is set.
        # Inputs: none (uses self.executor from the autouse fixture).
        # Outputs: none (pytest assertions).
        # Exceptions: AssertionError on failure.
        # Side effects: real remote execution (network/containers).
        """Check if DockerExecutor initializes without errors"""
        assert self.executor.container is not None, "Container should be initialized"

    def test_cleanup_on_deletion(self):
        # [FUNCTION DOC]
        # After delete, the container id is not listed by Docker.
        # Inputs: none (uses self.executor from the autouse fixture).
        # Outputs: none (pytest assertions).
        # Exceptions: AssertionError on failure.
        # Side effects: real remote execution (network/containers). Fixture teardown calls delete again (no-
        #   op).
        """Test if Docker container stops and removes on deletion"""
        container_id = self.executor.container.id
        self.executor.delete()  # Trigger cleanup

        client = docker.from_env()
        containers = [c.id for c in client.containers.list(all=True)]
        assert container_id not in containers, "Container should be removed"


@require_run_all
class TestModalExecutorIntegration(CommonDockerExecutorIntegration):
    @pytest.fixture
    # [CLASS DOC]
    # (require_run_all) Runs the common integration tests against a real ModalExecutor.
    # Attributes: executor.
    def custom_executor(self):
        # [FUNCTION DOC]
        # HELPER: Fixture creating a real ModalExecutor (pillow, numpy) and deleting it after the test.
        # Inputs: self.
        # Outputs: yields ModalExecutor.
        # Exceptions: Modal errors propagate.
        # Side effects: creates/terminates a Modal sandbox.
        executor = ModalExecutor(
            additional_imports=["pillow", "numpy"],
            logger=AgentLogger(LogLevel.INFO, Console(force_terminal=False, file=io.StringIO())),
        )
        yield executor
        executor.delete()


class TestModalExecutorUnit:
    @patch("smolagents.remote_executors._websocket_run_code_raise_errors")
    @patch("requests.post")
    @patch("requests.get")
    @patch("websocket.create_connection")
    @patch("modal.App.lookup")
    @patch("modal.Sandbox.create")
    # [CLASS DOC]
    # Unit tests for ModalExecutor with modal, requests and websocket patched.
    # Inputs: none.
    def test_sandbox_lifecycle(
        self, mock_sandbox_create, mock_app_lookup, mock_create_connection, mock_get, mock_post, mock_run_code_raises
    ):
        # [FUNCTION DOC]
        # Creates a ModalExecutor with custom app name, port and create_kwargs; checks the sandbox entrypoint
        #   args, that user timeout/cpu override defaults, that secrets are the user's plus the token secret,
        #   the app lookup call, then runs code and checks cleanup terminates the sandbox.
        # Inputs: self plus the six mocks injected by the patch decorators (bottom-up order).
        # Outputs: none (pytest assertions).
        # Exceptions: AssertionError on failure.
        # Side effects: patches modal, requests, websocket and the websocket runner helper; skips if modal
        #   missing.
        # KNOWN ISSUE: pytest.importorskip runs after the modal patch decorators, which already require modal
        #   to import, so the skip cannot take effect.
        """Test that sandbox is created with the correct kwargs and cleaned up correctly."""
        modal = pytest.importorskip("modal")
        port = 8889

        logger = MagicMock()
        mock_sandbox = MagicMock()
        tunnel_mock = MagicMock()
        tunnel_mock.host = "r4234.modal.host"
        mock_sandbox.tunnels.return_value = {port: tunnel_mock}

        mock_get.return_value.status_code = 200
        mock_post.return_value.status_code = 201
        mock_post.return_value.json.return_value = {"id": "test-kernel-id"}
        mock_run_code_raises.return_value = CodeOutput(output="3", logs="", is_final_answer=False)
        mock_sandbox_create.return_value = mock_sandbox

        executor = ModalExecutor(
            additional_imports=[],
            logger=logger,
            app_name="my-custom-app-name",
            port=port,
            create_kwargs={
                "secrets": [modal.Secret.from_dict({"MY_SECRET": "ABC"})],
                "timeout": 100,
                "cpu": 2,
            },
        )

        create_call = mock_sandbox_create.mock_calls[0]
        assert create_call.args == (
            "jupyter",
            "kernelgateway",
            "--KernelGatewayApp.ip=0.0.0.0",
            f"--KernelGatewayApp.port={port}",
        )
        assert create_call.kwargs["timeout"] == 100
        assert create_call.kwargs["cpu"] == 2
        assert len(create_call.kwargs["secrets"]) == 2
        mock_app_lookup.assert_called_with("my-custom-app-name", create_if_missing=True)

        executor.run_code_raise_errors("1 + 2")
        executor.cleanup()
        mock_sandbox.terminate.assert_called()


class TestBlaxelExecutorUnit:
    # [CLASS DOC]
    # Unit tests for BlaxelExecutor with the Blaxel SDK partially mocked.
    # Inputs: none.
    def test_blaxel_executor_instantiation_without_blaxel_sdk(self):
        # [FUNCTION DOC]
        # With blaxel.core blocked in sys.modules, construction raises ModuleNotFoundError with the install
        #   hint.
        # Inputs: self.
        # Outputs: none (pytest assertions).
        # Exceptions: AssertionError on failure.
        # Side effects: temporarily patches sys.modules.
        # KNOWN ISSUE: the executor imports the top-level blaxel package, so this only works if importing
        #   blaxel itself imports blaxel.core (or if blaxel is not cached).
        """Test that BlaxelExecutor raises appropriate error when blaxel SDK is not installed."""
        logger = MagicMock()
        with patch.dict("sys.modules", {"blaxel.core": None}):
            with pytest.raises(ModuleNotFoundError) as excinfo:
                BlaxelExecutor(additional_imports=[], logger=logger)
            assert "Please install 'blaxel' extra" in str(excinfo.value)

    @patch("smolagents.remote_executors._create_kernel_http")
    @patch("blaxel.core.SandboxInstance")
    @patch("blaxel.core.settings")
    def test_blaxel_executor_instantiation_with_blaxel_sdk(
        self, mock_settings, mock_sandbox_instance, mock_create_kernel
    ):
        # [FUNCTION DOC]
        # With mocked create_sandbox, SandboxInstance, settings and kernel creation, checks default sandbox
        #   name prefix, image, memory and region.
        # Inputs: self, mock_settings, mock_sandbox_instance, mock_create_kernel.
        # Outputs: none (pytest assertions).
        # Exceptions: AssertionError on failure.
        # Side effects: manually starts/stops a patcher on create_sandbox.
        # KNOWN ISSUE: patcher.stop is not in a finally block, so a failure leaks the patch to later tests.
        """Test BlaxelExecutor instantiation with mocked Blaxel SDK."""

        # patch manually for Python 3.10 compatibility
        from unittest.mock import patch

        mod = importlib.import_module("blaxel.core.client.api.compute")
        patcher = patch.object(mod, "create_sandbox")
        mock_create_sandbox = patcher.start()

        logger = MagicMock()
        mock_settings.headers = {}

        # Mock sandbox response
        mock_response = MagicMock()
        mock_create_sandbox.sync.return_value = mock_response

        # Mock SandboxInstance
        mock_sandbox = MagicMock()
        mock_metadata = MagicMock()
        mock_metadata.url = "https://test-sandbox.bl.run"
        mock_sandbox.metadata = mock_metadata
        mock_sandbox_instance.return_value = mock_sandbox

        # Mock kernel creation
        mock_create_kernel.return_value = "kernel-123"

        executor = BlaxelExecutor(additional_imports=[], logger=logger)

        patcher.stop()

        assert executor.sandbox_name.startswith("smolagent-executor-")
        assert executor.image == "blaxel/jupyter-notebook"
        assert executor.memory == 4096
        assert executor.region is None

    @patch("smolagents.remote_executors.BlaxelExecutor.install_packages")
    @patch("smolagents.remote_executors._create_kernel_http")
    @patch("blaxel.core.SandboxInstance")
    @patch("blaxel.core.settings")
    def test_blaxel_executor_custom_parameters(
        self, mock_settings, mock_sandbox_instance, mock_create_kernel, mock_install_packages
    ):
        # [FUNCTION DOC]
        # With mocks, checks custom sandbox name, image, memory and region are stored and install_packages was
        #   called.
        # Inputs: self, mock_settings, mock_sandbox_instance, mock_create_kernel, mock_install_packages.
        # Outputs: none (pytest assertions).
        # Exceptions: AssertionError on failure.
        # Side effects: manually starts/stops a patcher on create_sandbox.
        # KNOWN ISSUE: patcher not stopped on failure.
        """Test BlaxelExecutor with custom parameters."""
        logger = MagicMock()
        mock_settings.headers = {}
        mock_install_packages.return_value = ["numpy"]

        # Mock sandbox response
        mock_response = MagicMock()

        # patch manually for Python 3.10 compatibility
        mod = importlib.import_module("blaxel.core.client.api.compute")
        create_sandbox_patcher = patch.object(mod, "create_sandbox")
        mock_create_sandbox = create_sandbox_patcher.start()
        mock_create_sandbox.sync.return_value = mock_response

        # Mock SandboxInstance
        mock_sandbox = MagicMock()
        mock_metadata = MagicMock()
        mock_metadata.url = "https://test-sandbox.us-was-1.bl.run"
        mock_sandbox.metadata = mock_metadata
        mock_sandbox_instance.return_value = mock_sandbox

        # Mock kernel creation
        mock_create_kernel.return_value = "kernel-123"

        executor = BlaxelExecutor(
            additional_imports=["numpy"],
            logger=logger,
            sandbox_name="test-sandbox",
            image="custom-image:latest",
            memory=8192,
            region="us-was-1",
        )

        create_sandbox_patcher.stop()

        assert executor.sandbox_name == "test-sandbox"
        assert executor.image == "custom-image:latest"
        assert executor.memory == 8192
        assert executor.region == "us-was-1"
        assert mock_install_packages.called

    @patch("smolagents.remote_executors._create_kernel_http")
    @patch("blaxel.core.SandboxInstance")
    @patch("blaxel.core.settings")
    def test_blaxel_executor_cleanup(self, mock_settings, mock_sandbox_instance, mock_create_kernel):
        # [FUNCTION DOC]
        # With mocks, checks cleanup calls delete_sandbox.sync and removes the sandbox attribute.
        # Inputs: self, mock_settings, mock_sandbox_instance, mock_create_kernel.
        # Outputs: none (pytest assertions).
        # Exceptions: AssertionError on failure.
        # Side effects: manually starts/stops patchers on create_sandbox and delete_sandbox.
        # KNOWN ISSUE: patchers not stopped on failure.
        """Test BlaxelExecutor cleanup method."""

        # patch manually for Python 3.10 compatibility
        from unittest.mock import patch

        mod = importlib.import_module("blaxel.core.client.api.compute")
        create_sandbox_patcher = patch.object(mod, "create_sandbox")
        mock_create_sandbox = create_sandbox_patcher.start()
        delete_sandbox_patcher = patch.object(mod, "delete_sandbox")
        mock_delete_sandbox = delete_sandbox_patcher.start()

        logger = MagicMock()
        mock_settings.headers = {}

        # Mock sandbox response
        mock_response = MagicMock()
        mock_create_sandbox.sync.return_value = mock_response

        # Mock SandboxInstance
        mock_sandbox = MagicMock()
        mock_metadata = MagicMock()
        mock_metadata.url = "https://test-sandbox.bl.run"
        mock_sandbox.metadata = mock_metadata
        mock_sandbox_instance.return_value = mock_sandbox

        # Mock kernel creation
        mock_create_kernel.return_value = "kernel-123"

        executor = BlaxelExecutor(additional_imports=[], logger=logger)

        # Test cleanup
        executor.cleanup()
        create_sandbox_patcher.stop()
        delete_sandbox_patcher.stop()

        # Verify that delete_sandbox.sync was called
        assert mock_delete_sandbox.sync.called
        # Verify sandbox reference was cleaned up
        assert not hasattr(executor, "sandbox")
