# [MODULE DOC]
# Purpose: Data model for an agent's memory: dataclasses for each kind of step (task, action, planning, system
#   prompt, final answer), tool calls, the AgentMemory container and a CallbackRegistry for per-step
#   callbacks.
# Exports: AgentMemory in the public list; also defines ToolCall, MemoryStep, ActionStep, PlanningStep,
#   TaskStep, SystemPromptStep, FinalAnswerStep, CallbackRegistry, and re-exports imported
#   ChatMessage/MessageRole.
# Dependencies: smolagents.models (ChatMessage, MessageRole, nested dataclass to dict), smolagents.monitoring
#   (AgentLogger, LogLevel, Timing, TokenUsage), smolagents.utils (AgentError, JSON serialisation helper).
# Side effects: creates a module logger.
import inspect
from dataclasses import asdict, dataclass
from logging import getLogger
from typing import TYPE_CHECKING, Any, Callable, Type

from smolagents.models import ChatMessage, MessageRole, get_dict_from_nested_dataclasses
from smolagents.monitoring import AgentLogger, LogLevel, Timing, TokenUsage
from smolagents.utils import AgentError, make_json_serializable


if TYPE_CHECKING:
    import PIL.Image

    from smolagents.models import ChatMessage
    from smolagents.monitoring import AgentLogger


__all__ = ["AgentMemory"]


logger = getLogger(__name__)


@dataclass
class ToolCall:
    # [CLASS DOC]
    # Dataclass describing one tool call made by the model.
    # Inputs/attributes: name (str, tool name), arguments (any, tool arguments), id (str, call id).
    # Exceptions: none raised directly.
    # Side effects: none.
    name: str
    arguments: Any
    id: str

    def dict(self):
        # [FUNCTION DOC]
        # Serialises the tool call into an OpenAI-style dict with keys id, type (always "function") and
        #   function (holding name and JSON-serialisable arguments).
        # Inputs: none.
        # Outputs: dict.
        # Exceptions: errors from make_json_serializable, if any.
        # Side effects: none.
        return {
            "id": self.id,
            "type": "function",
            "function": {
                "name": self.name,
                "arguments": make_json_serializable(self.arguments),
            },
        }


@dataclass
class MemoryStep:
    # [CLASS DOC]
    # Abstract base dataclass for all memory steps; provides a default dict conversion and an abstract
    #   to_messages.
    # Inputs: none (no fields).
    # Exceptions: to_messages raises NotImplementedError.
    # Side effects: none.
    def dict(self):
        # [FUNCTION DOC]
        # Returns the dataclass converted recursively to a dict using the standard dataclass conversion.
        # Inputs: none.
        # Outputs: dict of field names to values.
        # Exceptions: errors from deep-copying non-copyable field values.
        # Side effects: none.
        return asdict(self)

    def to_messages(self, summary_mode: bool = False) -> list[ChatMessage]:
        # [FUNCTION DOC]
        # Abstract method to convert the step into chat messages for the model.
        # Inputs: summary_mode (bool, default False).
        # Outputs: none (never returns).
        # Exceptions: always raises NotImplementedError.
        # Side effects: none.
        raise NotImplementedError


@dataclass
class ActionStep(MemoryStep):
    # [CLASS DOC]
    # Dataclass recording one action step of an agent: model input/output, tool calls, code, observations,
    #   errors, timing and token usage.
    # Inputs/attributes: step_number (int), timing (Timing), model_input_messages (list of ChatMessage or
    #   None), tool_calls (list of ToolCall or None), error (AgentError or None), model_output_message
    #   (ChatMessage or None), model_output (str, list of dicts, or None), code_action (str or None),
    #   observations (str or None), observations_images (list of PIL images or None), action_output (any),
    #   token_usage (TokenUsage or None), is_final_answer (bool, default False).
    # Exceptions: none raised directly.
    # Side effects: none.
    step_number: int
    timing: Timing
    model_input_messages: list[ChatMessage] | None = None
    tool_calls: list[ToolCall] | None = None
    error: AgentError | None = None
    model_output_message: ChatMessage | None = None
    model_output: str | list[dict[str, Any]] | None = None
    code_action: str | None = None
    observations: str | None = None
    observations_images: list["PIL.Image.Image"] | None = None
    action_output: Any = None
    token_usage: TokenUsage | None = None
    is_final_answer: bool = False

    def dict(self):
        # [FUNCTION DOC]
        # Serialises the step into a JSON-friendly dict: messages converted via nested-dataclass-to-dict plus
        #   JSON serialisation, tool calls via ToolCall.dict (empty list when none), error via its dict
        #   method, timing via Timing.dict, images as raw bytes from tobytes, token usage as a dict, other
        #   fields copied.
        # Inputs: none.
        # Outputs: dict with keys step_number, timing, model_input_messages, tool_calls, error,
        #   model_output_message, model_output, code_action, observations, observations_images, action_output,
        #   token_usage, is_final_answer.
        # Exceptions: AttributeError if timing is not a Timing.
        # Side effects: none.
        # KNOWN ISSUE: observations_images are emitted as raw bytes, which are not JSON-serialisable despite
        #   the method's purpose.
        # We overwrite the method to parse the tool_calls and action_output manually
        return {
            "step_number": self.step_number,
            "timing": self.timing.dict(),
            "model_input_messages": [
                make_json_serializable(get_dict_from_nested_dataclasses(msg)) for msg in self.model_input_messages
            ]
            if self.model_input_messages
            else None,
            "tool_calls": [tc.dict() for tc in self.tool_calls] if self.tool_calls else [],
            "error": self.error.dict() if self.error else None,
            "model_output_message": make_json_serializable(get_dict_from_nested_dataclasses(self.model_output_message))
            if self.model_output_message
            else None,
            "model_output": self.model_output,
            "code_action": self.code_action,
            "observations": self.observations,
            "observations_images": [image.tobytes() for image in self.observations_images]
            if self.observations_images
            else None,
            "action_output": make_json_serializable(self.action_output),
            "token_usage": asdict(self.token_usage) if self.token_usage else None,
            "is_final_answer": self.is_final_answer,
        }

    def to_messages(self, summary_mode: bool = False) -> list[ChatMessage]:
        # [FUNCTION DOC]
        # Converts the step into chat messages to feed back to the model, in order: assistant message with
        #   stripped model_output (skipped in summary mode), a TOOL_CALL message listing tool call dicts, a
        #   USER message with observation images, a TOOL_RESPONSE message with "Observation:" text, and a
        #   TOOL_RESPONSE error message (prefixed by the first tool call id when present) asking the model to
        #   retry.
        # Inputs: summary_mode (bool, default False).
        # Outputs: list of ChatMessage (possibly empty).
        # Exceptions: AttributeError if model_output is a list (strip is called on it).
        # Side effects: none.
        # KNOWN ISSUE: model_output is typed as possibly a list of dicts, but the method assumes a str.
        messages = []
        if self.model_output is not None and not summary_mode:
            messages.append(
                ChatMessage(role=MessageRole.ASSISTANT, content=[{"type": "text", "text": self.model_output.strip()}])
            )

        if self.tool_calls is not None:
            messages.append(
                ChatMessage(
                    role=MessageRole.TOOL_CALL,
                    content=[
                        {
                            "type": "text",
                            "text": "Calling tools:\n" + str([tc.dict() for tc in self.tool_calls]),
                        }
                    ],
                )
            )

        if self.observations_images:
            messages.append(
                ChatMessage(
                    role=MessageRole.USER,
                    content=[
                        {
                            "type": "image",
                            "image": image,
                        }
                        for image in self.observations_images
                    ],
                )
            )

        if self.observations is not None:
            messages.append(
                ChatMessage(
                    role=MessageRole.TOOL_RESPONSE,
                    content=[
                        {
                            "type": "text",
                            "text": f"Observation:\n{self.observations}",
                        }
                    ],
                )
            )
        if self.error is not None:
            error_message = (
                "Error:\n"
                + str(self.error)
                + "\nNow let's retry: take care not to repeat previous errors! If you have retried several times, try a completely different approach.\n"
            )
            message_content = f"Call id: {self.tool_calls[0].id}\n" if self.tool_calls else ""
            message_content += error_message
            messages.append(
                ChatMessage(role=MessageRole.TOOL_RESPONSE, content=[{"type": "text", "text": message_content}])
            )

        return messages


@dataclass
class PlanningStep(MemoryStep):
    # [CLASS DOC]
    # Dataclass recording a planning step: the messages sent to the model, its output message, the extracted
    #   plan text, timing and token usage.
    # Inputs/attributes: model_input_messages (list of ChatMessage), model_output_message (ChatMessage), plan
    #   (str), timing (Timing), token_usage (TokenUsage or None).
    # Exceptions: none raised directly.
    # Side effects: none.
    model_input_messages: list[ChatMessage]
    model_output_message: ChatMessage
    plan: str
    timing: Timing
    token_usage: TokenUsage | None = None

    def dict(self):
        # [FUNCTION DOC]
        # Serialises the step into a JSON-friendly dict with model_input_messages, model_output_message (both
        #   JSON-serialised), plan, timing dict and token_usage dict or None.
        # Inputs: none.
        # Outputs: dict.
        # Exceptions: errors if the message fields are not iterable/convertible.
        # Side effects: none.
        return {
            "model_input_messages": [
                make_json_serializable(get_dict_from_nested_dataclasses(msg)) for msg in self.model_input_messages
            ],
            "model_output_message": make_json_serializable(
                get_dict_from_nested_dataclasses(self.model_output_message)
            ),
            "plan": self.plan,
            "timing": self.timing.dict(),
            "token_usage": asdict(self.token_usage) if self.token_usage else None,
        }

    def to_messages(self, summary_mode: bool = False) -> list[ChatMessage]:
        # [FUNCTION DOC]
        # Returns an assistant message with the stripped plan followed by a user message instructing to carry
        #   out the plan (creating a role change); returns an empty list in summary mode.
        # Inputs: summary_mode (bool, default False).
        # Outputs: list of ChatMessage (0 or 2 items).
        # Exceptions: none raised directly.
        # Side effects: none.
        if summary_mode:
            return []
        return [
            ChatMessage(role=MessageRole.ASSISTANT, content=[{"type": "text", "text": self.plan.strip()}]),
            ChatMessage(
                role=MessageRole.USER, content=[{"type": "text", "text": "Now proceed and carry out this plan."}]
            ),
            # This second message creates a role change to prevent models models from simply continuing the plan message
        ]


@dataclass
class TaskStep(MemoryStep):
    # [CLASS DOC]
    # Dataclass recording the task given to the agent, optionally with images.
    # Inputs/attributes: task (str), task_images (list of PIL images or None).
    # Exceptions: none raised directly.
    # Side effects: none.
    task: str
    task_images: list["PIL.Image.Image"] | None = None

    def to_messages(self, summary_mode: bool = False) -> list[ChatMessage]:
        # [FUNCTION DOC]
        # Returns a single user message whose content is a "New task:" text part followed by one image part
        #   per task image. Summary mode is ignored.
        # Inputs: summary_mode (bool, default False, unused).
        # Outputs: list with one ChatMessage.
        # Exceptions: none raised directly.
        # Side effects: none.
        content = [{"type": "text", "text": f"New task:\n{self.task}"}]
        if self.task_images:
            content.extend([{"type": "image", "image": image} for image in self.task_images])

        return [ChatMessage(role=MessageRole.USER, content=content)]


@dataclass
class SystemPromptStep(MemoryStep):
    # [CLASS DOC]
    # Dataclass holding the agent's system prompt.
    # Inputs/attributes: system_prompt (str).
    # Exceptions: none raised directly.
    # Side effects: none.
    system_prompt: str

    def to_messages(self, summary_mode: bool = False) -> list[ChatMessage]:
        # [FUNCTION DOC]
        # Returns a single system message containing the prompt text, or an empty list in summary mode.
        # Inputs: summary_mode (bool, default False).
        # Outputs: list of ChatMessage (0 or 1 item).
        # Exceptions: none raised directly.
        # Side effects: none.
        if summary_mode:
            return []
        return [ChatMessage(role=MessageRole.SYSTEM, content=[{"type": "text", "text": self.system_prompt}])]


@dataclass
class FinalAnswerStep(MemoryStep):
    # [CLASS DOC]
    # Dataclass wrapping the agent's final answer; has no to_messages implementation (inherits the abstract
    #   one).
    # Inputs/attributes: output (any).
    # Exceptions: to_messages raises NotImplementedError.
    # Side effects: none.
    output: Any


class AgentMemory:
    # [CLASS DOC]
    # Container for an agent's memory: the system prompt step and the ordered list of steps; offers reset,
    #   serialisation, replay and code extraction.
    # Inputs: system_prompt (str).
    # Attributes: system_prompt (SystemPromptStep), steps (list of TaskStep, ActionStep, PlanningStep).
    # Exceptions: none raised directly.
    # Side effects: replay writes to the logger console.
    """Memory for the agent, containing the system prompt and all steps taken by the agent.

    This class is used to store the agent's steps, including tasks, actions, and planning steps.
    It allows for resetting the memory, retrieving succinct or full step information, and replaying the agent's steps.

    Args:
        system_prompt (`str`): System prompt for the agent, which sets the context and instructions for the agent's behavior.

    **Attributes**:
        - **system_prompt** (`SystemPromptStep`) -- System prompt step for the agent.
        - **steps** (`list[TaskStep | ActionStep | PlanningStep]`) -- List of steps taken by the agent, which can include tasks, actions, and planning steps.
    """

    def __init__(self, system_prompt: str):
        # [FUNCTION DOC]
        # Creates the system prompt step and an empty steps list.
        # Inputs: system_prompt (str).
        # Outputs: none.
        # Exceptions: none raised directly.
        # Side effects: sets instance attributes.
        self.system_prompt: SystemPromptStep = SystemPromptStep(system_prompt=system_prompt)
        self.steps: list[TaskStep | ActionStep | PlanningStep] = []

    def reset(self):
        # [FUNCTION DOC]
        # Clears all steps, keeping the system prompt.
        # Inputs: none.
        # Outputs: none.
        # Exceptions: none raised directly.
        # Side effects: replaces steps with a new empty list.
        """Reset the agent's memory, clearing all steps and keeping the system prompt."""
        self.steps = []

    def get_succinct_steps(self) -> list[dict]:
        # [FUNCTION DOC]
        # Returns each step's dict form with the model_input_messages key removed.
        # Inputs: none.
        # Outputs: list of dicts.
        # Exceptions: errors from step dict methods.
        # Side effects: none.
        """Return a succinct representation of the agent's steps, excluding model input messages."""
        return [
            {key: value for key, value in step.dict().items() if key != "model_input_messages"} for step in self.steps
        ]

    def get_full_steps(self) -> list[dict]:
        # [FUNCTION DOC]
        # Returns each step's full dict form (empty list when there are no steps).
        # Inputs: none.
        # Outputs: list of dicts.
        # Exceptions: errors from step dict methods.
        # Side effects: none.
        """Return a full representation of the agent's steps, including model input messages."""
        if len(self.steps) == 0:
            return []
        return [step.dict() for step in self.steps]

    def replay(self, logger: AgentLogger, detailed: bool = False):
        # [FUNCTION DOC]
        # Pretty-prints the memory through the logger at ERROR level: the system prompt, then for each step a
        #   task panel, a step rule plus optional input messages and model output for action steps, or a
        #   planning rule plus optional input messages and plan.
        # Inputs: logger (AgentLogger), detailed (bool, default False; also logs model input messages for each
        #   step).
        # Outputs: none.
        # Exceptions: errors from logger methods (e.g. messages lacking a dict method).
        # Side effects: console output.
        """Prints a pretty replay of the agent's steps.

        Args:
            logger (`AgentLogger`): The logger to print replay logs to.
            detailed (`bool`, default `False`): If True, also displays the memory at each step. Defaults to False.
                Careful: will increase log length exponentially. Use only for debugging.
        """
        logger.console.log("Replaying the agent's steps:")
        logger.log_markdown(title="System prompt", content=self.system_prompt.system_prompt, level=LogLevel.ERROR)
        for step in self.steps:
            if isinstance(step, TaskStep):
                logger.log_task(step.task, "", level=LogLevel.ERROR)
            elif isinstance(step, ActionStep):
                logger.log_rule(f"Step {step.step_number}", level=LogLevel.ERROR)
                if detailed and step.model_input_messages is not None:
                    logger.log_messages(step.model_input_messages, level=LogLevel.ERROR)
                if step.model_output is not None:
                    logger.log_markdown(title="Agent output:", content=step.model_output, level=LogLevel.ERROR)
            elif isinstance(step, PlanningStep):
                logger.log_rule("Planning step", level=LogLevel.ERROR)
                if detailed and step.model_input_messages is not None:
                    logger.log_messages(step.model_input_messages, level=LogLevel.ERROR)
                logger.log_markdown(title="Agent output:", content=step.plan, level=LogLevel.ERROR)

    def return_full_code(self) -> str:
        # [FUNCTION DOC]
        # Concatenates the code_action of all action steps that have one, separated by blank lines.
        # Inputs: none.
        # Outputs: str (empty if no code).
        # Exceptions: none raised directly.
        # Side effects: none.
        """Returns all code actions from the agent's steps, concatenated as a single script."""
        return "\n\n".join(
            [step.code_action for step in self.steps if isinstance(step, ActionStep) and step.code_action is not None]
        )


class CallbackRegistry:
    # [CLASS DOC]
    # Registry mapping step classes to lists of callbacks, invoked for a step and for each class in the step's
    #   MRO.
    # Inputs: none.
    # Attributes: _callbacks (dict from step class to list of callables).
    # Exceptions: none raised directly.
    # Side effects: callbacks may have arbitrary side effects.
    """Registry for callbacks that are called at each step of the agent's execution.

    Callbacks are registered by passing a step class and a callback function.
    """

    def __init__(self):
        # [FUNCTION DOC]
        # Initialises an empty callback mapping.
        # Inputs: none.
        # Outputs: none.
        # Exceptions: none raised directly.
        # Side effects: sets _callbacks.
        self._callbacks: dict[Type[MemoryStep], list[Callable]] = {}

    def register(self, step_cls: Type[MemoryStep], callback: Callable):
        # [FUNCTION DOC]
        # Appends a callback to the list for the given step class, creating the list if needed.
        # Inputs: step_cls (a MemoryStep subclass), callback (callable).
        # Outputs: none.
        # Exceptions: none raised directly.
        # Side effects: mutates the registry.
        """Register a callback for a step class.

        Args:
            step_cls (Type[MemoryStep]): Step class to register the callback for.
            callback (Callable): Callback function to register.
        """
        if step_cls not in self._callbacks:
            self._callbacks[step_cls] = []
        self._callbacks[step_cls].append(callback)

    def callback(self, memory_step, **kwargs):
        # [FUNCTION DOC]
        # Invokes every callback registered for any class in the step's MRO (most specific first). Callbacks
        #   with exactly one parameter receive only the step; others receive the step plus the keyword
        #   arguments.
        # Inputs: memory_step (MemoryStep instance), **kwargs (e.g. agent).
        # Outputs: none.
        # Exceptions: propagates exceptions from callbacks; ValueError/TypeError from signature inspection of
        #   unusual callables.
        # Side effects: those of the callbacks.
        """Call callbacks registered for a step type.

        Args:
            memory_step (MemoryStep): Step to call the callbacks for.
            **kwargs: Additional arguments to pass to callbacks that accept them.
                Typically, includes the agent instance.

        Notes:
            For backwards compatibility, callbacks with a single parameter signature
            receive only the memory_step, while callbacks with multiple parameters
            receive both the memory_step and any additional kwargs.
        """
        # For compatibility with old callbacks that only take the step as an argument
        for cls in memory_step.__class__.__mro__:
            for cb in self._callbacks.get(cls, []):
                cb(memory_step) if len(inspect.signature(cb).parameters) == 1 else cb(memory_step, **kwargs)
