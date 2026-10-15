# [MODULE DOC]
# HELPER: Defines TextInspectorTool, a smolagents tool that converts files to markdown (via the sibling
#   mdconvert module) and optionally asks an LLM about them.
# Inputs: depends on smolagents (Tool, Model, MessageRole) and .mdconvert.MarkdownConverter.
# Outputs: class TextInspectorTool.
# Exceptions: none at import.
# Side effects: none at import.
from smolagents import Tool
from smolagents.models import Model


class TextInspectorTool(Tool):
    # [CLASS DOC]
    # Tool "inspect_file_as_text": reads a non-image file as markdown text and either returns it or answers a
    #   question about it with an LLM.
    # Inputs: constructor model (smolagents Model, default None; required whenever a question is asked) and
    #   text_limit (int, default 100000, max characters of file text sent to the model). Tool inputs:
    #   file_path (str) and optional question (str).
    # Key attributes: model, text_limit, md_converter (MarkdownConverter).
    # Exceptions: see methods.
    # Side effects: see methods.
    name = "inspect_file_as_text"
    description = """
You cannot load files yourself: instead call this tool to read a file as markdown text and ask questions about it.
This tool handles the following file extensions: [".html", ".htm", ".xlsx", ".pptx", ".wav", ".mp3", ".m4a", ".flac", ".pdf", ".docx"], and all other types of text files. IT DOES NOT HANDLE IMAGES."""

    inputs = {
        "file_path": {
            "description": "The path to the file you want to read as text. Must be a '.something' file, like '.pdf'. If it is an image, use the visualizer tool instead! DO NOT use this tool for an HTML webpage: use the web_search tool instead!",
            "type": "string",
        },
        "question": {
            "description": "[Optional]: Your question, as a natural language sentence. Provide as much context as possible. Do not pass this parameter if you just want to directly return the content of the file.",
            "type": "string",
            "nullable": True,
        },
    }
    output_type = "string"

    def __init__(self, model: Model = None, text_limit: int = 100000):
        # [FUNCTION DOC]
        # Initializes the Tool base, stores model and text_limit, and creates a MarkdownConverter (imported
        #   lazily).
        # Inputs: model (Model or None, default None), text_limit (int, default 100000).
        # Outputs: none.
        # Exceptions: ImportError if mdconvert or its dependencies are missing.
        # Side effects: sets instance attributes.
        super().__init__()
        self.model = model
        self.text_limit = text_limit
        from .mdconvert import MarkdownConverter

        self.md_converter = MarkdownConverter()

    def forward_initial_exam_mode(self, file_path, question):
        # [FUNCTION DOC]
        # Converts the file and returns either its raw text or a short LLM-written caption. Returns the full
        #   text for zip paths or when no question is given; returns "Document content: " plus the text when
        #   it is under 4000 characters; otherwise asks the model for a 5-sentence caption helpful for the
        #   question (without answering it), giving it the title and the first text_limit characters.
        # Inputs: file_path (str), question (str or None).
        # Outputs: string.
        # Exceptions: generic Exception for .png/.jpg paths; conversion and model errors propagate
        #   (AttributeError if model is None).
        # Side effects: reads the file; one model call in the caption case.
        # KNOWN ISSUE: the image check happens after conversion, and only checks the last 4 characters, so
        #   ".jpeg" and upper-case extensions slip through.
        from smolagents.models import MessageRole

        result = self.md_converter.convert(file_path)

        if file_path[-4:] in [".png", ".jpg"]:
            raise Exception("Cannot use inspect_file_as_text tool with images: use visualizer instead!")

        if ".zip" in file_path:
            return result.text_content

        if not question:
            return result.text_content

        if len(result.text_content) < 4000:
            return "Document content: " + result.text_content

        messages = [
            {
                "role": MessageRole.SYSTEM,
                "content": [
                    {
                        "type": "text",
                        "text": "Here is a file:\n### "
                        + str(result.title)
                        + "\n\n"
                        + result.text_content[: self.text_limit],
                    }
                ],
            },
            {
                "role": MessageRole.USER,
                "content": [
                    {
                        "type": "text",
                        "text": "Now please write a short, 5 sentence caption for this document, that could help someone asking this question: "
                        + question
                        + "\n\nDon't answer the question yourself! Just provide useful notes on the document",
                    }
                ],
            },
        ]
        return self.model(messages).content

    def forward(self, file_path, question: str | None = None) -> str:
        # [FUNCTION DOC]
        # Tool entry point. Converts the file; returns the full text for zip paths or when no question is
        #   given; otherwise asks the model to caption the file and answer the question under three headings
        #   (short answer, detailed answer, additional context), giving it the title and the first text_limit
        #   characters.
        # Inputs: file_path (str), question (str or None, default None).
        # Outputs: string (file text or model answer).
        # Exceptions: generic Exception for .png/.jpg paths; conversion and model errors propagate.
        # Side effects: reads the file; one model call when a question is given.
        # KNOWN ISSUE: same late, case-sensitive 4-character image check as forward_initial_exam_mode; the
        #   question is concatenated to the prompts without a separator.
        from smolagents.models import MessageRole

        result = self.md_converter.convert(file_path)

        if file_path[-4:] in [".png", ".jpg"]:
            raise Exception("Cannot use inspect_file_as_text tool with images: use visualizer instead!")

        if ".zip" in file_path:
            return result.text_content

        if not question:
            return result.text_content

        messages = [
            {
                "role": MessageRole.SYSTEM,
                "content": [
                    {
                        "type": "text",
                        "text": "You will have to write a short caption for this file, then answer this question:"
                        + question,
                    }
                ],
            },
            {
                "role": MessageRole.USER,
                "content": [
                    {
                        "type": "text",
                        "text": "Here is the complete file:\n### "
                        + str(result.title)
                        + "\n\n"
                        + result.text_content[: self.text_limit],
                    }
                ],
            },
            {
                "role": MessageRole.USER,
                "content": [
                    {
                        "type": "text",
                        "text": "Now answer the question below. Use these three headings: '1. Short answer', '2. Extremely detailed answer', '3. Additional Context on the document and question asked'."
                        + question,
                    }
                ],
            },
        ]
        return self.model(messages).content
