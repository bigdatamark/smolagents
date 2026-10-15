# [MODULE DOC]
# HELPER: Utilities for describing GAIA attachments to an agent and for resuming runs. Exports
#   serialize_agent_error, image/document/single-file/zip description helpers and get_tasks_to_run.
# Inputs: depends on json, os, shutil, textwrap, pathlib and smolagents.utils.AgentError.
# Outputs: the functions listed above.
# Exceptions: none at import.
# Side effects: none at import.
import json
import os
import shutil
import textwrap
from pathlib import Path

# import tqdm.asyncio
from smolagents.utils import AgentError


def serialize_agent_error(obj):
    # [FUNCTION DOC]
    # HELPER: JSON default hook for agent errors.
    # Inputs: obj (any).
    # Outputs: dict with error_type (class name) and message for AgentError instances, otherwise the str of
    #   obj.
    # Exceptions: none raised directly.
    # Side effects: none.
    if isinstance(obj, AgentError):
        return {"error_type": obj.__class__.__name__, "message": obj.message}
    else:
        return str(obj)


def get_image_description(file_name: str, question: str, visual_inspection_tool) -> str:
    # [FUNCTION DOC]
    # HELPER: Asks a visual tool for a 5-sentence caption of an image focused on details relevant to a
    #   question (without answering it).
    # Inputs: file_name (str image path), question (str), visual_inspection_tool (callable tool taking
    #   image_path and question).
    # Outputs: the caption string returned by the tool.
    # Exceptions: tool errors propagate.
    # Side effects: whatever the tool does (network calls).
    prompt = f"""Write a caption of 5 sentences for this image. Pay special attention to any details that might be useful for someone answering the following question:
{question}. But do not try to answer the question directly!
Do not add any information that is not present in the image."""
    return visual_inspection_tool(image_path=file_name, question=prompt)


def get_document_description(file_path: str, question: str, document_inspection_tool) -> str:
    # [FUNCTION DOC]
    # HELPER: Asks the text inspector for a 5-sentence caption of a document in initial-exam mode.
    # Inputs: file_path (str), question (str), document_inspection_tool (TextInspectorTool).
    # Outputs: caption or raw content string returned by forward_initial_exam_mode.
    # Exceptions: tool errors propagate.
    # Side effects: file reads and model calls via the tool.
    prompt = f"""Write a caption of 5 sentences for this document. Pay special attention to any details that might be useful for someone answering the following question:
{question}. But do not try to answer the question directly!
Do not add any information that is not present in the document."""
    return document_inspection_tool.forward_initial_exam_mode(file_path=file_path, question=prompt)


def get_single_file_description(file_path: str, question: str, visual_inspection_tool, document_inspection_tool):
    # [FUNCTION DOC]
    # HELPER: Builds a bullet describing one attached file by extension: images (png/jpg/jpeg) get a visual
    #   caption; documents (pdf/xls/xlsx/docx/doc/xml) get a caption of a same-named .png if it exists, else a
    #   document caption; audio (mp3/m4a/wav) and other files are only listed by path.
    # Inputs: file_path (str), question (str), visual_inspection_tool, document_inspection_tool.
    # Outputs: multi-line description string.
    # Exceptions: tool errors propagate.
    # Side effects: tool calls (network), file reads.
    # KNOWN ISSUE: the sibling .png path is built by splitting on the first dot, which breaks for paths
    #   containing dots in directory names or relative prefixes such as "./".
    file_extension = file_path.split(".")[-1]
    if file_extension in ["png", "jpg", "jpeg"]:
        file_description = f" - Attached image: {file_path}"
        file_description += (
            f"\n     -> Image description: {get_image_description(file_path, question, visual_inspection_tool)}"
        )
        return file_description
    elif file_extension in ["pdf", "xls", "xlsx", "docx", "doc", "xml"]:
        image_path = file_path.split(".")[0] + ".png"
        if os.path.exists(image_path):
            description = get_image_description(image_path, question, visual_inspection_tool)
            file_path = image_path
        else:
            description = get_document_description(file_path, question, document_inspection_tool)
        file_description = f" - Attached document: {file_path}"
        file_description += f"\n     -> File description: {description}"
        return file_description
    elif file_extension in ["mp3", "m4a", "wav"]:
        return f" - Attached audio: {file_path}"
    else:
        return f" - Attached file: {file_path}"


def get_zip_description(file_path: str, question: str, visual_inspection_tool, document_inspection_tool):
    # [FUNCTION DOC]
    # HELPER: Extracts a zip archive into a folder named like the archive without ".zip" and describes every
    #   contained file with get_single_file_description, indented by four spaces.
    # Inputs: file_path (str .zip path), question (str), visual_inspection_tool, document_inspection_tool.
    # Outputs: concatenated description string (one block per file, each preceded by a newline).
    # Exceptions: archive/tool errors propagate.
    # Side effects: creates the folder and writes extracted files to disk; tool calls.
    folder_path = file_path.replace(".zip", "")
    os.makedirs(folder_path, exist_ok=True)
    shutil.unpack_archive(file_path, folder_path)

    prompt_use_files = ""
    for root, dirs, files in os.walk(folder_path):
        for file in files:
            file_path = os.path.join(root, file)
            prompt_use_files += "\n" + textwrap.indent(
                get_single_file_description(file_path, question, visual_inspection_tool, document_inspection_tool),
                prefix="    ",
            )
    return prompt_use_files


def get_tasks_to_run(data, total: int, base_filename: Path, tasks_ids: list[int]):
    # [FUNCTION DOC]
    # HELPER: Selects tasks not yet answered, based on a sibling "<stem>_answers.jsonl" file next to
    #   base_filename.
    # Inputs: data (indexable sequence of dicts with task_id), total (int, number of leading items to
    #   consider), base_filename (Path), tasks_ids (list of int or None; when given, restricts to those ids).
    # Outputs: list of task dicts to run, in original order.
    # Exceptions: JSON decode errors, KeyError and ValueError (non-integer task_id) propagate.
    # Side effects: reads the answers file if present.
    f = base_filename.parent / f"{base_filename.stem}_answers.jsonl"
    done = set()
    if f.exists():
        with open(f, encoding="utf-8") as fh:
            done = {json.loads(line)["task_id"] for line in fh if line.strip()}

    tasks = []
    for i in range(total):
        task_id = int(data[i]["task_id"])
        if task_id not in done:
            if tasks_ids is not None:
                if task_id in tasks_ids:
                    tasks.append(data[i])
            else:
                tasks.append(data[i])
    return tasks
