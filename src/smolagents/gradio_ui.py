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
# [MODULE DOC]
# Purpose: Gradio integration: converts agent memory steps and streaming deltas into gradio ChatMessage
#   objects, streams agent runs, and provides the GradioUI chat application with optional file uploads.
# Exports: stream_to_gradio, GradioUI (public); also pull_messages_from_step, get_step_footnote_content and
#   private formatting helpers.
# Dependencies: smolagents agent_types, agents, memory, models; gradio is imported lazily and checked via
#   _is_package_available.
# Side effects: none at import.
import os
import re
import shutil
from pathlib import Path
from typing import Generator

from smolagents.agent_types import AgentAudio, AgentImage, AgentText
from smolagents.agents import MultiStepAgent, PlanningStep
from smolagents.memory import ActionStep, FinalAnswerStep
from smolagents.models import ChatMessageStreamDelta, MessageRole, agglomerate_stream_deltas
from smolagents.utils import _is_package_available


def get_step_footnote_content(step_log: ActionStep | PlanningStep, step_name: str) -> str:
    # [FUNCTION DOC]
    # Builds a small grey HTML span footnote: bold step name, then input/output token counts (thousands
    #   separators) if token usage exists, then duration rounded to 2 decimals if non-zero.
    # Inputs: step_log (ActionStep or PlanningStep), step_name (str).
    # Outputs: str (HTML).
    # Exceptions: AttributeError if the step lacks timing/token_usage.
    # Side effects: none.
    """Get a footnote string for a step log with duration and token information"""
    step_footnote = f"**{step_name}**"
    if step_log.token_usage is not None:
        step_footnote += f" | Input tokens: {step_log.token_usage.input_tokens:,} | Output tokens: {step_log.token_usage.output_tokens:,}"
    step_footnote += f" | Duration: {round(float(step_log.timing.duration), 2)}s" if step_log.timing.duration else ""
    step_footnote_content = f"""<span style="color: #bbbbc2; font-size: 12px;">{step_footnote}</span> """
    return step_footnote_content


def _clean_model_output(model_output: str) -> str:
    # [FUNCTION DOC]
    # HELPER: Strips whitespace and removes "<end_code>" tags adjacent to triple backticks in the three common
    #   layouts, leaving the backticks.
    # Inputs: model_output (str).
    # Outputs: cleaned str (empty if input is falsy).
    # Exceptions: TypeError if a non-string is passed.
    # Side effects: none.
    """
    Clean up model output by removing trailing tags and extra backticks.

    Args:
        model_output (`str`): Raw model output.

    Returns:
        `str`: Cleaned model output.
    """
    if not model_output:
        return ""
    model_output = model_output.strip()
    # Remove any trailing <end_code> and extra backticks, handling multiple possible formats
    model_output = re.sub(r"```\s*<end_code>", "```", model_output)  # handles ```<end_code>
    model_output = re.sub(r"<end_code>\s*```", "```", model_output)  # handles <end_code>```
    model_output = re.sub(r"```\s*\n\s*<end_code>", "```", model_output)  # handles ```\n<end_code>
    return model_output.strip()


def _format_code_content(content: str) -> str:
    # [FUNCTION DOC]
    # HELPER: Removes any opening code fences (a triple backtick plus the rest of its line) and "<end_code>"
    #   tags, then wraps the result in a python code fence.
    # Inputs: content (str).
    # Outputs: str fenced as python code.
    # Exceptions: none raised directly.
    # Side effects: none.
    # KNOWN ISSUE: a closing fence without a trailing newline is not removed, and the final check for an
    #   existing python fence is almost never true after the stripping.
    """
    Format code content as Python code block if it's not already formatted.

    Args:
        content (`str`): Code content to format.

    Returns:
        `str`: Code content formatted as a Python code block.
    """
    content = content.strip()
    # Remove existing code blocks and end_code tags
    content = re.sub(r"```.*?\n", "", content)
    content = re.sub(r"\s*<end_code>\s*", "", content)
    content = content.strip()
    # Add Python code block formatting if not already present
    if not content.startswith("```python"):
        content = f"```python\n{content}\n```"
    return content


def _process_action_step(step_log: ActionStep, skip_model_outputs: bool = False) -> Generator:
    # [FUNCTION DOC]
    # HELPER: Generator turning an ActionStep into gradio messages: step number header and cleaned model
    #   output (unless skipped), a "Used tool" message for the first tool call (dict arguments use their
    #   "answer" entry or the whole dict; python_interpreter content is code-fenced), execution logs, one
    #   message per observation image (saved via AgentImage), error, footnote and a "-----" separator.
    # Inputs: step_log (ActionStep), skip_model_outputs (bool, default False).
    # Outputs: yields gradio ChatMessage objects.
    # Exceptions: ImportError if gradio is missing.
    # Side effects: observation images are written to temp files.
    # KNOWN ISSUE: only the first tool call is shown; the execution-logs code fence is opened but never
    #   closed.
    """
    Process an [`ActionStep`] and yield appropriate Gradio ChatMessage objects.

    Args:
        step_log ([`ActionStep`]): ActionStep to process.
        skip_model_outputs (`bool`): Whether to skip model outputs.

    Yields:
        `gradio.ChatMessage`: Gradio ChatMessages representing the action step.
    """
    import gradio as gr

    # Output the step number
    step_number = f"Step {step_log.step_number}"
    if not skip_model_outputs:
        yield gr.ChatMessage(role=MessageRole.ASSISTANT, content=f"**{step_number}**", metadata={"status": "done"})

    # First yield the thought/reasoning from the LLM
    if not skip_model_outputs and getattr(step_log, "model_output", ""):
        model_output = _clean_model_output(step_log.model_output)
        yield gr.ChatMessage(role=MessageRole.ASSISTANT, content=model_output, metadata={"status": "done"})

    # For tool calls, create a parent message
    if getattr(step_log, "tool_calls", []):
        first_tool_call = step_log.tool_calls[0]
        used_code = first_tool_call.name == "python_interpreter"

        # Process arguments based on type
        args = first_tool_call.arguments
        if isinstance(args, dict):
            content = str(args.get("answer", str(args)))
        else:
            content = str(args).strip()

        # Format code content if needed
        if used_code:
            content = _format_code_content(content)

        # Create the tool call message
        parent_message_tool = gr.ChatMessage(
            role=MessageRole.ASSISTANT,
            content=content,
            metadata={
                "title": f"🛠️ Used tool {first_tool_call.name}",
                "status": "done",
            },
        )
        yield parent_message_tool

    # Display execution logs if they exist
    if getattr(step_log, "observations", "") and step_log.observations.strip():
        log_content = step_log.observations.strip()
        if log_content:
            log_content = re.sub(r"^Execution logs:\s*", "", log_content)
            yield gr.ChatMessage(
                role=MessageRole.ASSISTANT,
                content=f"```bash\n{log_content}\n",
                metadata={"title": "📝 Execution Logs", "status": "done"},
            )

    # Display any images in observations
    if getattr(step_log, "observations_images", []):
        for image in step_log.observations_images:
            path_image = AgentImage(image).to_string()
            yield gr.ChatMessage(
                role=MessageRole.ASSISTANT,
                content={"path": path_image, "mime_type": f"image/{path_image.split('.')[-1]}"},
                metadata={"title": "🖼️ Output Image", "status": "done"},
            )

    # Handle errors
    if getattr(step_log, "error", None):
        yield gr.ChatMessage(
            role=MessageRole.ASSISTANT, content=str(step_log.error), metadata={"title": "💥 Error", "status": "done"}
        )

    # Add step footnote and separator
    yield gr.ChatMessage(
        role=MessageRole.ASSISTANT,
        content=get_step_footnote_content(step_log, step_number),
        metadata={"status": "done"},
    )
    yield gr.ChatMessage(role=MessageRole.ASSISTANT, content="-----", metadata={"status": "done"})


def _process_planning_step(step_log: PlanningStep, skip_model_outputs: bool = False) -> Generator:
    # [FUNCTION DOC]
    # HELPER: Generator turning a PlanningStep into gradio messages: "Planning step" header and plan text
    #   (unless skipped), then footnote and separator.
    # Inputs: step_log (PlanningStep), skip_model_outputs (bool, default False).
    # Outputs: yields gradio ChatMessage objects (2 or 4).
    # Exceptions: ImportError if gradio is missing.
    # Side effects: none.
    """
    Process a [`PlanningStep`] and yield appropriate gradio.ChatMessage objects.

    Args:
        step_log ([`PlanningStep`]): PlanningStep to process.

    Yields:
        `gradio.ChatMessage`: Gradio ChatMessages representing the planning step.
    """
    import gradio as gr

    if not skip_model_outputs:
        yield gr.ChatMessage(role=MessageRole.ASSISTANT, content="**Planning step**", metadata={"status": "done"})
        yield gr.ChatMessage(role=MessageRole.ASSISTANT, content=step_log.plan, metadata={"status": "done"})
    yield gr.ChatMessage(
        role=MessageRole.ASSISTANT,
        content=get_step_footnote_content(step_log, "Planning step"),
        metadata={"status": "done"},
    )
    yield gr.ChatMessage(role=MessageRole.ASSISTANT, content="-----", metadata={"status": "done"})


def _process_final_answer_step(step_log: FinalAnswerStep) -> Generator:
    # [FUNCTION DOC]
    # HELPER: Generator yielding one gradio message for a final answer: formatted text for AgentText, a file
    #   dict with image/png mime for AgentImage, a file dict with audio/wav mime for AgentAudio, or the str of
    #   any other value.
    # Inputs: step_log (FinalAnswerStep).
    # Outputs: yields one gradio ChatMessage.
    # Exceptions: ImportError if gradio is missing.
    # Side effects: to_string on images/audio may write temp files.
    """
    Process a [`FinalAnswerStep`] and yield appropriate gradio.ChatMessage objects.

    Args:
        step_log ([`FinalAnswerStep`]): FinalAnswerStep to process.

    Yields:
        `gradio.ChatMessage`: Gradio ChatMessages representing the final answer.
    """
    import gradio as gr

    final_answer = step_log.output
    if isinstance(final_answer, AgentText):
        yield gr.ChatMessage(
            role=MessageRole.ASSISTANT,
            content=f"**Final answer:**\n{final_answer.to_string()}\n",
            metadata={"status": "done"},
        )
    elif isinstance(final_answer, AgentImage):
        yield gr.ChatMessage(
            role=MessageRole.ASSISTANT,
            content={"path": final_answer.to_string(), "mime_type": "image/png"},
            metadata={"status": "done"},
        )
    elif isinstance(final_answer, AgentAudio):
        yield gr.ChatMessage(
            role=MessageRole.ASSISTANT,
            content={"path": final_answer.to_string(), "mime_type": "audio/wav"},
            metadata={"status": "done"},
        )
    else:
        yield gr.ChatMessage(
            role=MessageRole.ASSISTANT, content=f"**Final answer:** {str(final_answer)}", metadata={"status": "done"}
        )


def pull_messages_from_step(step_log: ActionStep | PlanningStep | FinalAnswerStep, skip_model_outputs: bool = False):
    # [FUNCTION DOC]
    # Generator dispatching a memory step to the matching processor (action, planning, final answer) and
    #   yielding its gradio messages.
    # Inputs: step_log (ActionStep, PlanningStep or FinalAnswerStep), skip_model_outputs (bool, default False;
    #   used when model output was already streamed).
    # Outputs: yields gradio ChatMessage objects.
    # Exceptions: ModuleNotFoundError if gradio is not installed; ValueError for unsupported step types (both
    #   raised on first iteration since this is a generator).
    # Side effects: those of the processors.
    """Extract Gradio ChatMessage objects from agent steps with proper nesting.

    Args:
        step_log: The step log to display as gr.ChatMessage objects.
        skip_model_outputs: If True, skip the model outputs when creating the gr.ChatMessage objects:
            This is used for instance when streaming model outputs have already been displayed.
    """
    if not _is_package_available("gradio"):
        raise ModuleNotFoundError(
            "Please install 'gradio' extra to use the GradioUI: `pip install 'smolagents[gradio]'`"
        )
    if isinstance(step_log, ActionStep):
        yield from _process_action_step(step_log, skip_model_outputs)
    elif isinstance(step_log, PlanningStep):
        yield from _process_planning_step(step_log, skip_model_outputs)
    elif isinstance(step_log, FinalAnswerStep):
        yield from _process_final_answer_step(step_log)
    else:
        raise ValueError(f"Unsupported step type: {type(step_log)}")


def stream_to_gradio(
    agent,
    task: str,
    task_images: list | None = None,
    reset_agent_memory: bool = False,
    additional_args: dict | None = None,
) -> Generator:
    # [FUNCTION DOC]
    # Generator that runs the agent in streaming mode and yields gradio messages for each completed step, and
    #   the accumulated markdown text for each stream delta (reset after every step).
    # Inputs: agent (MultiStepAgent), task (str), task_images (list or None), reset_agent_memory (bool,
    #   default False), additional_args (dict or None).
    # Outputs: yields gradio ChatMessage objects or str (markdown of the partial model output).
    # Exceptions: ModuleNotFoundError if gradio is missing; errors raised by the agent run.
    # Side effects: runs the agent (model calls, tool execution).
    """Runs an agent with the given task and streams the messages from the agent as gradio ChatMessages."""

    if not _is_package_available("gradio"):
        raise ModuleNotFoundError(
            "Please install 'gradio' extra to use the GradioUI: `pip install 'smolagents[gradio]'`"
        )
    accumulated_events: list[ChatMessageStreamDelta] = []
    for event in agent.run(
        task, images=task_images, stream=True, reset=reset_agent_memory, additional_args=additional_args
    ):
        if isinstance(event, ActionStep | PlanningStep | FinalAnswerStep):
            for message in pull_messages_from_step(
                event,
                # If we're streaming model outputs, no need to display them twice
                skip_model_outputs=getattr(agent, "stream_outputs", False),
            ):
                yield message
            accumulated_events = []
        elif isinstance(event, ChatMessageStreamDelta):
            accumulated_events.append(event)
            text = agglomerate_stream_deltas(accumulated_events).render_as_markdown()
            yield text


class GradioUI:
    # [CLASS DOC]
    # Gradio chat application around a MultiStepAgent, built on gradio ChatInterface with optional multimodal
    #   file uploads and optional memory reset per message.
    # Inputs: agent (MultiStepAgent), file_upload_folder (str or None; enables uploads and is created if
    #   missing), reset_agent_memory (bool, default False).
    # Attributes: agent, file_upload_folder (Path or None), reset_agent_memory, name (agent name or "Agent
    #   interface"), description (agent description or None).
    # Exceptions: ModuleNotFoundError if gradio is missing.
    # Side effects: creates the upload folder; launching starts a web server.
    """
    Gradio interface for interacting with a [`MultiStepAgent`].

    This class provides a web interface to interact with the agent in real-time, allowing users to submit prompts, upload files, and receive responses in a chat-like format.
    It uses the modern [`gradio.ChatInterface`] component for a native chatbot experience.
    It can reset the agent's memory at the start of each interaction if desired.
    It supports file uploads via multimodal input.
    This class requires the `gradio` extra to be installed: `pip install 'smolagents[gradio]'`.

    Args:
        agent ([`MultiStepAgent`]): The agent to interact with.
        file_upload_folder (`str`, *optional*): The folder where uploaded files will be saved.
            If not provided, file uploads are disabled.
        reset_agent_memory (`bool`, *optional*, defaults to `False`): Whether to reset the agent's memory at the start of each interaction.
            If `True`, the agent will not remember previous interactions.

    Raises:
        ModuleNotFoundError: If the `gradio` extra is not installed.

    Example:
        ```python
        from smolagents import CodeAgent, GradioUI, InferenceClientModel

        model = InferenceClientModel(model_id="meta-llama/Meta-Llama-3.1-8B-Instruct")
        agent = CodeAgent(tools=[], model=model)
        gradio_ui = GradioUI(agent, file_upload_folder="uploads", reset_agent_memory=True)
        gradio_ui.launch()
        ```
    """

    def __init__(self, agent: MultiStepAgent, file_upload_folder: str | None = None, reset_agent_memory: bool = False):
        # [FUNCTION DOC]
        # Checks gradio availability, stores configuration, derives name and description from the agent and
        #   creates the upload folder when needed.
        # Inputs: agent, file_upload_folder (str or None), reset_agent_memory (bool).
        # Outputs: none.
        # Exceptions: ModuleNotFoundError if gradio is missing; AttributeError if the agent has no name
        #   attribute.
        # Side effects: may create directories.
        # KNOWN ISSUE: the name is read without a default, so agents lacking a name attribute raise instead of
        #   falling back to "Agent interface".
        if not _is_package_available("gradio"):
            raise ModuleNotFoundError(
                "Please install 'gradio' extra to use the GradioUI: `pip install 'smolagents[gradio]'`"
            )
        self.agent = agent
        self.file_upload_folder = Path(file_upload_folder) if file_upload_folder is not None else None
        self.reset_agent_memory = reset_agent_memory
        self.name = getattr(agent, "name") or "Agent interface"
        self.description = getattr(agent, "description", None)
        if self.file_upload_folder is not None:
            if not self.file_upload_folder.exists():
                self.file_upload_folder.mkdir(parents=True, exist_ok=True)

    def _save_uploaded_file(self, file_path: str) -> str:
        # [FUNCTION DOC]
        # HELPER: Copies a file into the upload folder under a sanitised basename (characters other than word
        #   chars, dash and dot become underscores); returns the original path if uploads are disabled.
        # Inputs: file_path (str).
        # Outputs: destination path (str).
        # Exceptions: OSError from copying.
        # Side effects: writes a file (overwrites same-named files).
        """Save an uploaded file to the upload folder and return the new path."""
        if self.file_upload_folder is None:
            return file_path

        original_name = os.path.basename(file_path)
        sanitized_name = re.sub(r"[^\w\-.]", "_", original_name)
        dest_path = os.path.join(self.file_upload_folder, sanitized_name)
        shutil.copy(file_path, dest_path)
        return dest_path

    def upload_file(self, file, file_uploads_log: list, allowed_file_types: list | None = None):
        # [FUNCTION DOC]
        # Validates and saves an uploaded file: reports "No file uploaded" for None, "File type disallowed"
        #   for extensions outside the allowed list, otherwise saves it and appends the new path to the log.
        # Inputs: file (object with a name path or None), file_uploads_log (list), allowed_file_types (list of
        #   extensions or None; defaults to pdf, docx, txt).
        # Outputs: tuple (gradio Textbox with status, new uploads list).
        # Exceptions: ImportError if gradio is missing; OSError from copying.
        # Side effects: writes the file into the upload folder.
        """
        Handle file upload with validation.

        Args:
            file: The uploaded file object.
            file_uploads_log: List to track uploaded files.
            allowed_file_types: List of allowed extensions. Defaults to [".pdf", ".docx", ".txt"].

        Returns:
            Tuple of (status textbox, updated file log).
        """
        import gradio as gr

        if file is None:
            return gr.Textbox(value="No file uploaded", visible=True), file_uploads_log

        if allowed_file_types is None:
            allowed_file_types = [".pdf", ".docx", ".txt"]

        file_ext = os.path.splitext(file.name)[1].lower()
        if file_ext not in allowed_file_types:
            return gr.Textbox(value="File type disallowed", visible=True), file_uploads_log

        file_path = self._save_uploaded_file(file.name)
        return gr.Textbox(value=f"File uploaded: {file_path}", visible=True), file_uploads_log + [file_path]

    def _process_message(self, message: str | dict) -> tuple[str, list[str] | None]:
        # [FUNCTION DOC]
        # HELPER: Splits a chat message into task text and files. Strings return no files. For dicts, files
        #   are saved to the upload folder (when enabled) and a sentence listing them is appended to the text.
        # Inputs: message (str or dict with text and files).
        # Outputs: tuple (text, list of file paths or None).
        # Exceptions: OSError from saving files.
        # Side effects: may copy files.
        # KNOWN ISSUE: the returned files are passed to the agent as images by the caller, even when they are
        #   non-image documents.
        """Process incoming message and extract text and files."""
        if isinstance(message, str):
            return message, None

        text = message.get("text", "")
        files = message.get("files", [])

        if files and self.file_upload_folder:
            saved_files = [self._save_uploaded_file(f) for f in files]
            if saved_files:
                text += f"\nYou have been provided with these files: {saved_files}"
            return text, saved_files

        return text, files if files else None

    def _stream_response(self, message: str | dict, history: list[dict]) -> Generator:  # noqa: ARG002
        # [FUNCTION DOC]
        # HELPER: ChatInterface callback generator: runs the agent with the message's text and files,
        #   maintains a list of messages where a single temporary streaming message (with angle brackets
        #   escaped) is updated per delta and removed when a step completes, appending the step's gradio
        #   messages; yields the full list after each change.
        # Inputs: message (str or dict), history (list, unused).
        # Outputs: yields the growing list of gradio ChatMessage objects.
        # Exceptions: errors raised by the agent run.
        # Side effects: runs the agent; may save uploaded files.
        """Stream agent responses for ChatInterface."""
        import gradio as gr

        task, task_files = self._process_message(message)

        all_messages: list[gr.ChatMessage] = []
        accumulated_events: list[ChatMessageStreamDelta] = []
        streaming_msg_idx: int | None = None

        for event in self.agent.run(
            task, images=task_files, stream=True, reset=self.reset_agent_memory, additional_args=None
        ):
            if isinstance(event, ActionStep | PlanningStep | FinalAnswerStep):
                # Remove streaming message if present
                if streaming_msg_idx is not None:
                    all_messages.pop(streaming_msg_idx)
                    streaming_msg_idx = None

                for msg in pull_messages_from_step(
                    event,
                    skip_model_outputs=getattr(self.agent, "stream_outputs", False),
                ):
                    all_messages.append(
                        gr.ChatMessage(
                            role=msg.role,
                            content=msg.content,
                            metadata=msg.metadata,
                        )
                    )
                    yield all_messages
                accumulated_events = []
            elif isinstance(event, ChatMessageStreamDelta):
                accumulated_events.append(event)
                text = agglomerate_stream_deltas(accumulated_events).render_as_markdown()
                text = text.replace("<", r"\<").replace(">", r"\>")
                msg = gr.ChatMessage(role="assistant", content=text)
                if streaming_msg_idx is None:
                    streaming_msg_idx = len(all_messages)
                    all_messages.append(msg)
                else:
                    all_messages[streaming_msg_idx] = msg
                yield all_messages

    def launch(self, share: bool = True, **kwargs):
        # [FUNCTION DOC]
        # Builds the app and launches it with debug enabled.
        # Inputs: share (bool, default True; creates a public share link), **kwargs (forwarded to gradio
        #   launch).
        # Outputs: none (blocks while serving).
        # Exceptions: gradio launch errors.
        # Side effects: starts a web server, possibly a public tunnel.
        """
        Launch the Gradio app with the agent interface.

        Args:
            share (`bool`, defaults to `True`): Whether to share the app publicly.
            **kwargs: Additional keyword arguments to pass to the Gradio launch method.
        """
        self.create_app().launch(debug=True, share=share, **kwargs)

    def create_app(self):
        # [FUNCTION DOC]
        # Builds the gradio ChatInterface: a Chatbot with the smolagents avatar and LaTeX delimiters, using
        #   the messages type only on gradio 5, multimodal input when uploads are enabled, history saving, and
        #   a title derived from the agent name.
        # Inputs: none.
        # Outputs: gradio ChatInterface (Blocks) instance.
        # Exceptions: ImportError if gradio is missing.
        # Side effects: none until launched.
        import gradio as gr

        # Gradio 5.x requires type="messages", but Gradio 6 removed this parameter
        type_messages_kwarg = {"type": "messages"} if gr.__version__.startswith("5") else {}

        chatbot = gr.Chatbot(
            label="Agent",
            avatar_images=(
                None,
                "https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/smolagents/mascot_smol.png",
            ),
            latex_delimiters=[
                {"left": r"$$", "right": r"$$", "display": True},
                {"left": r"$", "right": r"$", "display": False},
                {"left": r"\[", "right": r"\]", "display": True},
                {"left": r"\(", "right": r"\)", "display": False},
            ],
            **type_messages_kwarg,
        )

        demo = gr.ChatInterface(
            fn=self._stream_response,
            chatbot=chatbot,
            title=self.name.replace("_", " ").capitalize(),
            multimodal=self.file_upload_folder is not None,
            save_history=True,
            **type_messages_kwarg,
        )
        return demo


__all__ = ["stream_to_gradio", "GradioUI"]
