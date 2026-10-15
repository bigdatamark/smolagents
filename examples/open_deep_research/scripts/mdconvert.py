# This is copied from Magentic-one's great repo: https://github.com/microsoft/autogen/blob/v0.4.4/python/packages/autogen-magentic-one/src/autogen_magentic_one/markdown_browser/mdconvert.py
# Thanks to Microsoft researchers for open-sourcing this!
# type: ignore
# [MODULE DOC]
# Module: Markdown conversion utilities (vendored from Magentic-One's mdconvert) used by the
#   open_deep_research example's text browser to turn local files, URLs and HTTP responses into Markdown text
#   for LLM consumption.
# Exports: DocumentConverterResult (title + text_content), the DocumentConverter base class and concrete
#   converters (plain text, HTML, Wikipedia, YouTube, PDF, DOCX, XLSX, PPTX, WAV, MP3/M4A, ZIP, images),
#   FileConversionException, UnsupportedFormatException, and the MarkdownConverter facade that tries
#   registered converters by priority.
# Dependencies: mammoth, markdownify, pandas, pdfminer, python-pptx, puremagic, pydub, requests,
#   speech_recognition, BeautifulSoup, youtube_transcript_api; optionally the exiftool binary and an OpenAI-
#   compatible multimodal client.
# Side effects: importing has none beyond loading dependencies; converters read files, may perform network
#   calls, spawn exiftool, write temp files and extract ZIPs to disk.
import base64
import copy
import html
import json
import mimetypes
import os
import re
import shutil
import subprocess
import sys
import tempfile
import traceback
import zipfile
from typing import Any
from urllib.parse import parse_qs, quote, unquote, urlparse, urlunparse

import mammoth
import markdownify
import pandas as pd
import pdfminer
import pdfminer.high_level
import pptx

# File-format detection
import puremagic
import pydub
import requests
import speech_recognition as sr
from bs4 import BeautifulSoup
from youtube_transcript_api import YouTubeTranscriptApi
from youtube_transcript_api.formatters import SRTFormatter


class _CustomMarkdownify(markdownify.MarkdownConverter):
    # [CLASS DOC]
    # Description: HELPER: markdownify MarkdownConverter subclass that defaults to ATX ('#') headings, forces
    #   headings to start on a new line, drops non-http/https/file links (e.g. javascript:) while keeping
    #   their text, percent-escapes link paths, and truncates data: URI image sources.
    # Inputs: constructor accepts arbitrary markdownify options as keyword arguments; heading_style defaults
    #   to ATX.
    # Outputs: used via convert_soup to produce a Markdown string.
    # Exceptions: none raised directly.
    # Side effects: none
    # KNOWN ISSUE: the convert_hn/convert_a/convert_img signatures (with convert_as_inline) target older
    #   markdownify versions; newer markdownify releases changed the converter method signature (parent_tags),
    #   which would break these overrides.
    """
    A custom version of markdownify's MarkdownConverter. Changes include:

    - Altering the default heading style to use '#', '##', etc.
    - Removing javascript hyperlinks.
    - Truncating images with large data:uri sources.
    - Ensuring URIs are properly escaped, and do not conflict with Markdown syntax
    """

    def __init__(self, **options: Any):
        # [FUNCTION DOC]
        # Description: HELPER: sets heading_style to markdownify ATX unless the caller provided one, then
        #   delegates to the parent constructor.
        # Inputs: options (keyword arguments, markdownify converter options).
        # Outputs: none (initializer).
        # Exceptions: none raised directly; parent constructor errors propagate.
        # Side effects: mutates the local options mapping before passing it on.
        options["heading_style"] = options.get("heading_style", markdownify.ATX)
        # Explicitly cast options to the expected type if necessary
        super().__init__(**options)

    def convert_hn(self, n: int, el: Any, text: str, convert_as_inline: bool) -> str:
        # [FUNCTION DOC]
        # Description: HELPER: converts a heading element; when not converting inline and the text does not
        #   already start with a newline, prefixes the parent's heading output with a newline; otherwise
        #   returns the parent output unchanged.
        # Inputs: n (int, heading level), el (element), text (str, converted inner text), convert_as_inline
        #   (bool).
        # Outputs: Markdown heading string.
        # Exceptions: none raised directly.
        # Side effects: none
        """Same as usual, but be sure to start with a new line"""
        if not convert_as_inline:
            if not re.search(r"^\n", text):
                return "\n" + super().convert_hn(n, el, text, convert_as_inline)  # type: ignore

        return super().convert_hn(n, el, text, convert_as_inline)  # type: ignore

    def convert_a(self, el: Any, text: str, convert_as_inline: bool):
        # [FUNCTION DOC]
        # Description: HELPER: converts an anchor element. Splits surrounding whitespace from text (chomp);
        #   returns empty string if no text. If href has a scheme other than http/https/file, returns just the
        #   text; otherwise rebuilds the URL with the path unquoted then re-quoted. Emits autolink form (angle
        #   brackets) when autolinks are enabled, text equals href (ignoring escaped underscores) and no
        #   title/default_title; else emits [text](href "title") with quotes in the title escaped, using href
        #   as title when default_title is set; returns plain text when no href.
        # Inputs: el (anchor element), text (str), convert_as_inline (bool, unused).
        # Outputs: Markdown link string, or plain/empty text.
        # Exceptions: ValueError from URL parsing is caught and the text is returned instead.
        # Side effects: none
        """Same as usual converter, but removes Javascript links and escapes URIs."""
        prefix, suffix, text = markdownify.chomp(text)  # type: ignore
        if not text:
            return ""
        href = el.get("href")
        title = el.get("title")

        # Escape URIs and skip non-http or file schemes
        if href:
            try:
                parsed_url = urlparse(href)  # type: ignore
                if parsed_url.scheme and parsed_url.scheme.lower() not in ["http", "https", "file"]:  # type: ignore
                    return "%s%s%s" % (prefix, text, suffix)
                href = urlunparse(parsed_url._replace(path=quote(unquote(parsed_url.path))))  # type: ignore
            except ValueError:  # It's not clear if this ever gets thrown
                return "%s%s%s" % (prefix, text, suffix)

        # For the replacement see #29: text nodes underscores are escaped
        if (
            self.options["autolinks"]
            and text.replace(r"\_", "_") == href
            and not title
            and not self.options["default_title"]
        ):
            # Shortcut syntax
            return "<%s>" % href
        if self.options["default_title"] and not title:
            title = href
        title_part = ' "%s"' % title.replace('"', r"\"") if title else ""
        return "%s[%s](%s%s)%s" % (prefix, text, href, title_part, suffix) if href else text

    def convert_img(self, el: Any, text: str, convert_as_inline: bool) -> str:
        # [FUNCTION DOC]
        # Description: HELPER: converts an img element to Markdown image syntax with alt, src and optional
        #   quoted title. When converting inline and the parent tag is not in keep_inline_images_in, returns
        #   only the alt text. data: URI sources are truncated to their header (before the first comma)
        #   followed by '...'.
        # Inputs: el (img element), text (str, unused), convert_as_inline (bool).
        # Outputs: Markdown image string or alt text.
        # Exceptions: none raised directly.
        # Side effects: none
        """Same as usual converter, but removes data URIs"""

        alt = el.attrs.get("alt", None) or ""
        src = el.attrs.get("src", None) or ""
        title = el.attrs.get("title", None) or ""
        title_part = ' "%s"' % title.replace('"', r"\"") if title else ""
        if convert_as_inline and el.parent.name not in self.options["keep_inline_images_in"]:
            return alt

        # Remove dataURIs
        if src.startswith("data:"):
            src = src.split(",")[0] + "..."

        return "![%s](%s%s)" % (alt, src, title_part)

    def convert_soup(self, soup: Any) -> str:
        # [FUNCTION DOC]
        # Description: HELPER: thin pass-through to the parent convert_soup.
        # Inputs: soup (BeautifulSoup object or element).
        # Outputs: Markdown string.
        # Exceptions: none raised directly.
        # Side effects: none
        return super().convert_soup(soup)  # type: ignore


class DocumentConverterResult:
    # [CLASS DOC]
    # Description: Plain value object holding the result of a document conversion.
    # Inputs: title (str or None, default None), text_content (str, default empty) stored as same-named
    #   attributes.
    # Outputs: n/a.
    # Exceptions: none raised directly.
    # Side effects: none
    """The result of converting a document to text."""

    def __init__(self, title: str | None = None, text_content: str = ""):
        # [FUNCTION DOC]
        # Description: Stores title and text_content on the instance.
        # Inputs: title (str or None, default None), text_content (str, default empty string).
        # Outputs: none (initializer).
        # Exceptions: none raised directly.
        # Side effects: sets instance attributes.
        self.title: str | None = title
        self.text_content: str = text_content


class DocumentConverter:
    # [CLASS DOC]
    # Description: Abstract base for converters. Subclasses implement convert, returning a
    #   DocumentConverterResult when they accept the input or None to signal 'not my format' so the next
    #   converter is tried.
    # Inputs: no constructor arguments.
    # Outputs: n/a.
    # Exceptions: convert raises NotImplementedError on the base class.
    # Side effects: none
    """Abstract superclass of all DocumentConverters."""

    def convert(self, local_path: str, **kwargs: Any) -> None | DocumentConverterResult:
        # [FUNCTION DOC]
        # Description: Abstract conversion hook.
        # Inputs: local_path (str, path to file), kwargs (options such as file_extension, url, mlm_client,
        #   mlm_model).
        # Outputs: never returns on the base class.
        # Exceptions: always raises NotImplementedError.
        # Side effects: none
        raise NotImplementedError()


class PlainTextConverter(DocumentConverter):
    # [CLASS DOC]
    # Description: Converter for plain text: accepts any file whose file_extension maps to a known mimetype
    #   and returns its UTF-8 contents unchanged with no title.
    # Inputs: no constructor arguments.
    # Outputs: n/a.
    # Exceptions: see convert.
    # Side effects: none
    # KNOWN ISSUE: the docstring says text/plain, but the text/ content-type check is commented out, so any
    #   extension with a guessable mimetype (e.g. .png, .zip) is accepted and read as UTF-8 text, which can
    #   raise UnicodeDecodeError; it is the lowest-priority converter so this mostly only matters as a
    #   fallback.
    """Anything with content type text/plain"""

    def convert(self, local_path: str, **kwargs: Any) -> None | DocumentConverterResult:
        # [FUNCTION DOC]
        # Description: Guesses the mimetype from the file_extension kwarg; returns None if unknown, else reads
        #   the whole file as UTF-8 text.
        # Inputs: local_path (str), kwargs with optional file_extension (str, default empty).
        # Outputs: DocumentConverterResult with title None and the file text, or None.
        # Exceptions: OSError / UnicodeDecodeError from reading propagate.
        # Side effects: reads the file from disk.
        # Guess the content type from any file extension that might be around
        content_type, _ = mimetypes.guess_type("__placeholder" + kwargs.get("file_extension", ""))

        # Only accept text files
        if content_type is None:
            return None
        # elif "text/" not in content_type.lower():
        #     return None

        text_content = ""
        with open(local_path, "rt", encoding="utf-8") as fh:
            text_content = fh.read()
        return DocumentConverterResult(
            title=None,
            text_content=text_content,
        )


class HtmlConverter(DocumentConverter):
    # [CLASS DOC]
    # Description: Converter for HTML files (.html/.htm): strips script/style, converts the body (or whole
    #   document) to Markdown with _CustomMarkdownify and uses the <title> as the title. Also used as a base
    #   by DOCX/XLSX/PPTX converters via its _convert helper.
    # Inputs: no constructor arguments.
    # Outputs: n/a.
    # Exceptions: see methods.
    # Side effects: none
    """Anything with content type text/html"""

    def convert(self, local_path: str, **kwargs: Any) -> None | DocumentConverterResult:
        # [FUNCTION DOC]
        # Description: Returns None unless file_extension is .html or .htm (case-insensitive); otherwise reads
        #   the file as UTF-8 and delegates to _convert.
        # Inputs: local_path (str), kwargs with file_extension (str, default empty).
        # Outputs: DocumentConverterResult or None.
        # Exceptions: file read/decoding errors propagate.
        # Side effects: reads the file from disk.
        # Bail if not html
        extension = kwargs.get("file_extension", "")
        if extension.lower() not in [".html", ".htm"]:
            return None

        result = None
        with open(local_path, "rt", encoding="utf-8") as fh:
            result = self._convert(fh.read())

        return result

    def _convert(self, html_content: str) -> None | DocumentConverterResult:
        # [FUNCTION DOC]
        # Description: HELPER: parses an HTML string with html.parser, removes script and style elements,
        #   converts the body element (or the whole soup if no body) to Markdown, and takes the title from the
        #   <title> tag.
        # Inputs: html_content (str).
        # Outputs: DocumentConverterResult with title (str or None) and Markdown text.
        # Exceptions: AssertionError if the Markdown output is not a string.
        # Side effects: mutates the parsed soup only (local).
        """Helper function that converts and HTML string."""

        # Parse the string
        soup = BeautifulSoup(html_content, "html.parser")

        # Remove javascript and style blocks
        for script in soup(["script", "style"]):
            script.extract()

        # Print only the main content
        body_elm = soup.find("body")
        webpage_text = ""
        if body_elm:
            webpage_text = _CustomMarkdownify().convert_soup(body_elm)
        else:
            webpage_text = _CustomMarkdownify().convert_soup(soup)

        assert isinstance(webpage_text, str)

        return DocumentConverterResult(
            title=None if soup.title is None else soup.title.string, text_content=webpage_text
        )


class WikipediaConverter(DocumentConverter):
    # [CLASS DOC]
    # Description: Converter for Wikipedia HTML pages: only the mw-content-text div is converted, prefixed
    #   with a level-1 heading of the page title.
    # Inputs: no constructor arguments.
    # Outputs: n/a.
    # Exceptions: see convert.
    # Side effects: none
    """Handle Wikipedia pages separately, focusing only on the main document content."""

    def convert(self, local_path: str, **kwargs: Any) -> None | DocumentConverterResult:
        # [FUNCTION DOC]
        # Description: Returns None unless file_extension is .html/.htm and url matches http(s)://<2-3
        #   letters>.wikipedia.org/. Parses the file, removes script/style, finds the content div and the mw-
        #   page-title-main span. Title defaults to the <title> text; if the content div exists and the title
        #   span is present and non-empty, uses the span's string. Output is a '# title' heading plus Markdown
        #   of the content div, or Markdown of the whole page when the div is missing.
        # Inputs: local_path (str), kwargs with file_extension (str) and url (str, default empty).
        # Outputs: DocumentConverterResult with the main title and Markdown text, or None.
        # Exceptions: AssertionError if the title span's string is not a str (e.g. nested markup); read errors
        #   propagate.
        # Side effects: reads the file from disk.
        # KNOWN ISSUE: the 'len(title_elm) > 0' check counts child nodes rather than text length.
        # Bail if not Wikipedia
        extension = kwargs.get("file_extension", "")
        if extension.lower() not in [".html", ".htm"]:
            return None
        url = kwargs.get("url", "")
        if not re.search(r"^https?:\/\/[a-zA-Z]{2,3}\.wikipedia.org\/", url):
            return None

        # Parse the file
        soup = None
        with open(local_path, "rt", encoding="utf-8") as fh:
            soup = BeautifulSoup(fh.read(), "html.parser")

        # Remove javascript and style blocks
        for script in soup(["script", "style"]):
            script.extract()

        # Print only the main content
        body_elm = soup.find("div", {"id": "mw-content-text"})
        title_elm = soup.find("span", {"class": "mw-page-title-main"})

        webpage_text = ""
        main_title = None if soup.title is None else soup.title.string

        if body_elm:
            # What's the title
            if title_elm and len(title_elm) > 0:
                main_title = title_elm.string  # type: ignore
                assert isinstance(main_title, str)

            # Convert the page
            webpage_text = f"# {main_title}\n\n" + _CustomMarkdownify().convert_soup(body_elm)
        else:
            webpage_text = _CustomMarkdownify().convert_soup(soup)

        return DocumentConverterResult(
            title=main_title,
            text_content=webpage_text,
        )


class YouTubeConverter(DocumentConverter):
    # [CLASS DOC]
    # Description: Converter for YouTube watch pages: builds Markdown from page meta tags (title, views,
    #   keywords, duration, description), the full description extracted from ytInitialData JSON when
    #   possible, and the SRT-formatted transcript fetched via youtube_transcript_api.
    # Inputs: no constructor arguments.
    # Outputs: n/a.
    # Exceptions: see convert.
    # Side effects: network access for transcripts.
    """Handle YouTube specially, focusing on the video title, description, and transcript."""

    def convert(self, local_path: str, **kwargs: Any) -> None | DocumentConverterResult:
        # [FUNCTION DOC]
        # Description: Returns None unless file_extension is .html/.htm and url starts with
        #   https://www.youtube.com/watch?. Collects meta tags keyed by their itemprop/property/name attribute
        #   into a metadata dict (seeded with the <title>). Best-effort parses the first line of the script
        #   containing ytInitialData to find attributedDescriptionBodyText content as the description. Builds
        #   '# YouTube', '## title', a '### Video Metadata' bullet list (views, keywords, runtime) if any,
        #   '### Description' and '### Transcript' (from the v query parameter) sections.
        # Inputs: local_path (str), kwargs with file_extension (str) and url (str).
        # Outputs: DocumentConverterResult with title (metadata title or page title) and Markdown text, or
        #   None.
        # Exceptions: AssertionError if the page has no <title> string; description and transcript failures
        #   are swallowed.
        # Side effects: reads the file; network call to YouTube for the transcript.
        # KNOWN ISSUE: YouTubeTranscriptApi.get_transcript is a static API removed in newer
        #   youtube_transcript_api versions; the broad except would then silently omit transcripts.
        # Bail if not YouTube
        extension = kwargs.get("file_extension", "")
        if extension.lower() not in [".html", ".htm"]:
            return None
        url = kwargs.get("url", "")
        if not url.startswith("https://www.youtube.com/watch?"):
            return None

        # Parse the file
        soup = None
        with open(local_path, "rt", encoding="utf-8") as fh:
            soup = BeautifulSoup(fh.read(), "html.parser")

        # Read the meta tags
        assert soup.title is not None and soup.title.string is not None
        metadata: dict[str, str] = {"title": soup.title.string}
        for meta in soup(["meta"]):
            for a in meta.attrs:
                if a in ["itemprop", "property", "name"]:
                    metadata[meta[a]] = meta.get("content", "")
                    break

        # We can also try to read the full description. This is more prone to breaking, since it reaches into the page implementation
        try:
            for script in soup(["script"]):
                content = script.text
                if "ytInitialData" in content:
                    lines = re.split(r"\r?\n", content)
                    obj_start = lines[0].find("{")
                    obj_end = lines[0].rfind("}")
                    if obj_start >= 0 and obj_end >= 0:
                        data = json.loads(lines[0][obj_start : obj_end + 1])
                        attrdesc = self._findKey(data, "attributedDescriptionBodyText")  # type: ignore
                        if attrdesc:
                            metadata["description"] = str(attrdesc["content"])
                    break
        except Exception:
            pass

        # Start preparing the page
        webpage_text = "# YouTube\n"

        title = self._get(metadata, ["title", "og:title", "name"])  # type: ignore
        assert isinstance(title, str)

        if title:
            webpage_text += f"\n## {title}\n"

        stats = ""
        views = self._get(metadata, ["interactionCount"])  # type: ignore
        if views:
            stats += f"- **Views:** {views}\n"

        keywords = self._get(metadata, ["keywords"])  # type: ignore
        if keywords:
            stats += f"- **Keywords:** {keywords}\n"

        runtime = self._get(metadata, ["duration"])  # type: ignore
        if runtime:
            stats += f"- **Runtime:** {runtime}\n"

        if len(stats) > 0:
            webpage_text += f"\n### Video Metadata\n{stats}\n"

        description = self._get(metadata, ["description", "og:description"])  # type: ignore
        if description:
            webpage_text += f"\n### Description\n{description}\n"

        transcript_text = ""
        parsed_url = urlparse(url)  # type: ignore
        params = parse_qs(parsed_url.query)  # type: ignore
        if "v" in params:
            assert isinstance(params["v"][0], str)
            video_id = str(params["v"][0])
            try:
                # Must be a single transcript.
                transcript = YouTubeTranscriptApi.get_transcript(video_id)  # type: ignore
                # transcript_text = " ".join([part["text"] for part in transcript])  # type: ignore
                # Alternative formatting:
                transcript_text = SRTFormatter().format_transcript(transcript)
            except Exception:
                pass
        if transcript_text:
            webpage_text += f"\n### Transcript\n{transcript_text}\n"

        title = title if title else soup.title.string
        assert isinstance(title, str)

        return DocumentConverterResult(
            title=title,
            text_content=webpage_text,
        )

    def _get(self, metadata: dict[str, str], keys: list[str], default: str | None = None) -> str | None:
        # [FUNCTION DOC]
        # Description: HELPER: returns the value of the first key from keys present in metadata, else default.
        # Inputs: metadata (dict of str to str), keys (list of str, in priority order), default (str or None,
        #   default None).
        # Outputs: the matching value or default.
        # Exceptions: none raised directly.
        # Side effects: none
        for k in keys:
            if k in metadata:
                return metadata[k]
        return default

    def _findKey(self, json: Any, key: str) -> str | None:  # TODO: Fix json type
        # [FUNCTION DOC]
        # Description: HELPER: recursive depth-first search through nested lists/dicts for the first dict
        #   entry with the given key; returns its value.
        # Inputs: json (any decoded JSON value), key (str).
        # Outputs: the found value (any type despite the str annotation) or None.
        # Exceptions: RecursionError possible for extremely deep structures.
        # Side effects: none
        if isinstance(json, list):
            for elm in json:
                ret = self._findKey(elm, key)
                if ret is not None:
                    return ret
        elif isinstance(json, dict):
            for k in json:
                if k == key:
                    return json[k]
                else:
                    ret = self._findKey(json[k], key)
                    if ret is not None:
                        return ret
        return None


class PdfConverter(DocumentConverter):
    # [CLASS DOC]
    # Description: Converter for .pdf files using pdfminer's extract_text; output is essentially plain text.
    # Inputs: no constructor arguments.
    # Outputs: n/a.
    # Exceptions: see convert.
    # Side effects: none
    """
    Converts PDFs to Markdown. Most style information is ignored, so the results are essentially plain-text.
    """

    def convert(self, local_path, **kwargs) -> None | DocumentConverterResult:
        # [FUNCTION DOC]
        # Description: Returns None unless file_extension is .pdf; otherwise extracts all text from the PDF.
        # Inputs: local_path (str), kwargs with file_extension.
        # Outputs: DocumentConverterResult with title None and the extracted text, or None.
        # Exceptions: pdfminer errors propagate.
        # Side effects: reads the file.
        # Bail if not a PDF
        extension = kwargs.get("file_extension", "")
        if extension.lower() != ".pdf":
            return None

        return DocumentConverterResult(
            title=None,
            text_content=pdfminer.high_level.extract_text(local_path),
        )


class DocxConverter(HtmlConverter):
    # [CLASS DOC]
    # Description: Converter for .docx files: mammoth converts to HTML, then HtmlConverter._convert turns it
    #   into Markdown.
    # Inputs: no constructor arguments.
    # Outputs: n/a.
    # Exceptions: see convert.
    # Side effects: none
    """
    Converts DOCX files to Markdown. Style information (e.g.m headings) and tables are preserved where possible.
    """

    def convert(self, local_path, **kwargs) -> None | DocumentConverterResult:
        # [FUNCTION DOC]
        # Description: Returns None unless file_extension is .docx; otherwise opens the file in binary mode,
        #   converts to HTML with mammoth and then to Markdown.
        # Inputs: local_path (str), kwargs with file_extension.
        # Outputs: DocumentConverterResult (title from HTML, usually None) or None.
        # Exceptions: mammoth / read errors propagate.
        # Side effects: reads the file.
        # Bail if not a DOCX
        extension = kwargs.get("file_extension", "")
        if extension.lower() != ".docx":
            return None

        result = None
        with open(local_path, "rb") as docx_file:
            result = mammoth.convert_to_html(docx_file)
            html_content = result.value
            result = self._convert(html_content)

        return result


class XlsxConverter(HtmlConverter):
    # [CLASS DOC]
    # Description: Converter for .xlsx/.xls workbooks: every sheet becomes a '## sheet name' heading followed
    #   by a Markdown table.
    # Inputs: no constructor arguments.
    # Outputs: n/a.
    # Exceptions: see convert.
    # Side effects: none
    """
    Converts XLSX files to Markdown, with each sheet presented as a separate Markdown table.
    """

    def convert(self, local_path, **kwargs) -> None | DocumentConverterResult:
        # [FUNCTION DOC]
        # Description: Returns None unless file_extension is .xlsx or .xls; reads all sheets with pandas,
        #   renders each to HTML without index and converts to Markdown, joining sections with blank lines.
        # Inputs: local_path (str), kwargs with file_extension.
        # Outputs: DocumentConverterResult with title None and stripped Markdown, or None.
        # Exceptions: pandas/engine errors propagate (e.g. missing openpyxl/xlrd).
        # Side effects: reads the file.
        # Bail if not a XLSX
        extension = kwargs.get("file_extension", "")
        if extension.lower() not in [".xlsx", ".xls"]:
            return None

        sheets = pd.read_excel(local_path, sheet_name=None)
        md_content = ""
        for s in sheets:
            md_content += f"## {s}\n"
            html_content = sheets[s].to_html(index=False)
            md_content += self._convert(html_content).text_content.strip() + "\n\n"

        return DocumentConverterResult(
            title=None,
            text_content=md_content.strip(),
        )


class PptxConverter(HtmlConverter):
    # [CLASS DOC]
    # Description: Converter for .pptx presentations: per-slide sections with an HTML comment slide marker,
    #   title as '# ' heading, other text frames, pictures as image links with alt text, tables as Markdown
    #   tables and speaker notes under '### Notes:'.
    # Inputs: no constructor arguments.
    # Outputs: n/a.
    # Exceptions: see convert.
    # Side effects: none
    """
    Converts PPTX files to Markdown. Supports heading, tables and images with alt text.
    """

    def convert(self, local_path, **kwargs) -> None | DocumentConverterResult:
        # [FUNCTION DOC]
        # Description: Returns None unless file_extension is .pptx. Iterates slides (numbered from 1) and
        #   shapes: pictures emit an image link with alt text (descr attribute, falling back to shape name)
        #   and a filename built from the shape name stripped of non-word chars plus .jpg; tables are built as
        #   HTML (first row as header cells, cell text escaped) and converted to Markdown; otherwise text
        #   frames are emitted, the slide title prefixed with '# '. Notes are appended if present. Content is
        #   stripped after each slide.
        # Inputs: local_path (str), kwargs with file_extension.
        # Outputs: DocumentConverterResult with title None and Markdown, or None.
        # Exceptions: python-pptx errors propagate; alt-text lookup errors are swallowed.
        # Side effects: reads the file.
        # Bail if not a PPTX
        extension = kwargs.get("file_extension", "")
        if extension.lower() != ".pptx":
            return None

        md_content = ""

        presentation = pptx.Presentation(local_path)
        slide_num = 0
        for slide in presentation.slides:
            slide_num += 1

            md_content += f"\n\n<!-- Slide number: {slide_num} -->\n"

            title = slide.shapes.title
            for shape in slide.shapes:
                # Pictures
                if self._is_picture(shape):
                    # https://github.com/scanny/python-pptx/pull/512#issuecomment-1713100069
                    alt_text = ""
                    try:
                        alt_text = shape._element._nvXxPr.cNvPr.attrib.get("descr", "")
                    except Exception:
                        pass

                    # A placeholder name
                    filename = re.sub(r"\W", "", shape.name) + ".jpg"
                    md_content += "\n![" + (alt_text if alt_text else shape.name) + "](" + filename + ")\n"

                # Tables
                if self._is_table(shape):
                    html_table = "<html><body><table>"
                    first_row = True
                    for row in shape.table.rows:
                        html_table += "<tr>"
                        for cell in row.cells:
                            if first_row:
                                html_table += "<th>" + html.escape(cell.text) + "</th>"
                            else:
                                html_table += "<td>" + html.escape(cell.text) + "</td>"
                        html_table += "</tr>"
                        first_row = False
                    html_table += "</table></body></html>"
                    md_content += "\n" + self._convert(html_table).text_content.strip() + "\n"

                # Text areas
                elif shape.has_text_frame:
                    if shape == title:
                        md_content += "# " + shape.text.lstrip() + "\n"
                    else:
                        md_content += shape.text + "\n"

            md_content = md_content.strip()

            if slide.has_notes_slide:
                md_content += "\n\n### Notes:\n"
                notes_frame = slide.notes_slide.notes_text_frame
                if notes_frame is not None:
                    md_content += notes_frame.text
                md_content = md_content.strip()

        return DocumentConverterResult(
            title=None,
            text_content=md_content.strip(),
        )

    def _is_picture(self, shape):
        # [FUNCTION DOC]
        # Description: HELPER: true if the shape is a picture, or a placeholder that has an image attribute.
        # Inputs: shape (python-pptx shape).
        # Outputs: bool.
        # Exceptions: none raised directly.
        # Side effects: none
        if shape.shape_type == pptx.enum.shapes.MSO_SHAPE_TYPE.PICTURE:
            return True
        if shape.shape_type == pptx.enum.shapes.MSO_SHAPE_TYPE.PLACEHOLDER:
            if hasattr(shape, "image"):
                return True
        return False

    def _is_table(self, shape):
        # [FUNCTION DOC]
        # Description: HELPER: true if the shape type is TABLE.
        # Inputs: shape (python-pptx shape).
        # Outputs: bool.
        # Exceptions: none raised directly.
        # Side effects: none
        if shape.shape_type == pptx.enum.shapes.MSO_SHAPE_TYPE.TABLE:
            return True
        return False


class MediaConverter(DocumentConverter):
    # [CLASS DOC]
    # Description: Abstract base for media converters providing exiftool-based metadata extraction.
    # Inputs: no constructor arguments.
    # Outputs: n/a.
    # Exceptions: convert is not implemented here (inherits NotImplementedError).
    # Side effects: none
    """
    Abstract class for multi-modal media (e.g., images and audio)
    """

    def _get_metadata(self, local_path):
        # [FUNCTION DOC]
        # Description: HELPER: if exiftool is on PATH, runs it with -json on the file and returns the first
        #   object of the parsed output; returns None if exiftool is absent or anything fails.
        # Inputs: local_path (str).
        # Outputs: dict of metadata or None.
        # Exceptions: none (all errors swallowed).
        # Side effects: spawns an exiftool subprocess.
        exiftool = shutil.which("exiftool")
        if not exiftool:
            return None
        else:
            try:
                result = subprocess.run([exiftool, "-json", local_path], capture_output=True, text=True).stdout
                return json.loads(result)[0]
            except Exception:
                return None


class WavConverter(MediaConverter):
    # [CLASS DOC]
    # Description: Converter for .wav audio: selected exiftool metadata fields plus a Google speech-
    #   recognition transcript.
    # Inputs: no constructor arguments.
    # Outputs: n/a.
    # Exceptions: see convert.
    # Side effects: network access for transcription.
    """
    Converts WAV files to markdown via extraction of metadata (if `exiftool` is installed), and speech transcription (if `speech_recognition` is installed).
    """

    def convert(self, local_path, **kwargs) -> None | DocumentConverterResult:
        # [FUNCTION DOC]
        # Description: Returns None unless file_extension is .wav. Emits 'Field: value' lines for Title,
        #   Artist, Author, Band, Album, Genre, Track, DateTimeOriginal, CreateDate, Duration when present,
        #   then a '### Audio Transcript:' section with the transcript, '[No speech detected]' if empty, or an
        #   error message if transcription fails.
        # Inputs: local_path (str), kwargs with file_extension.
        # Outputs: DocumentConverterResult with title None, or None.
        # Exceptions: none raised directly (transcription errors swallowed).
        # Side effects: exiftool subprocess; network call to Google speech API.
        # KNOWN ISSUE: the inline comment says 'Bail if not a XLSX' but the check is for .wav (copy-paste).
        # Bail if not a XLSX
        extension = kwargs.get("file_extension", "")
        if extension.lower() != ".wav":
            return None

        md_content = ""

        # Add metadata
        metadata = self._get_metadata(local_path)
        if metadata:
            for f in [
                "Title",
                "Artist",
                "Author",
                "Band",
                "Album",
                "Genre",
                "Track",
                "DateTimeOriginal",
                "CreateDate",
                "Duration",
            ]:
                if f in metadata:
                    md_content += f"{f}: {metadata[f]}\n"

        # Transcribe
        try:
            transcript = self._transcribe_audio(local_path)
            md_content += "\n\n### Audio Transcript:\n" + ("[No speech detected]" if transcript == "" else transcript)
        except Exception:
            md_content += "\n\n### Audio Transcript:\nError. Could not transcribe this audio."

        return DocumentConverterResult(
            title=None,
            text_content=md_content.strip(),
        )

    def _transcribe_audio(self, local_path) -> str:
        # [FUNCTION DOC]
        # Description: HELPER: loads the WAV file with speech_recognition, records the whole file and
        #   transcribes it with recognize_google.
        # Inputs: local_path (str, WAV file).
        # Outputs: stripped transcript string.
        # Exceptions: speech_recognition errors (UnknownValueError, RequestError) and file errors propagate.
        # Side effects: reads the file; network call to Google.
        recognizer = sr.Recognizer()
        with sr.AudioFile(local_path) as source:
            audio = recognizer.record(source)
            return recognizer.recognize_google(audio).strip()


class Mp3Converter(WavConverter):
    # [CLASS DOC]
    # Description: Converter for .mp3/.m4a audio: metadata like WavConverter, then converts to a temp WAV with
    #   pydub (needs ffmpeg) and transcribes it.
    # Inputs: no constructor arguments.
    # Outputs: n/a.
    # Exceptions: see convert.
    # Side effects: temp files, subprocess, network.
    """
    Converts MP3 and M4A files to markdown via extraction of metadata (if `exiftool` is installed), and speech transcription (if `speech_recognition` AND `pydub` are installed).
    """

    def convert(self, local_path, **kwargs) -> None | DocumentConverterResult:
        # [FUNCTION DOC]
        # Description: Returns None unless file_extension is .mp3 or .m4a. Adds the same metadata fields as
        #   WavConverter, decodes the audio with pydub (mp3 or m4a format), exports to a temporary .wav file,
        #   transcribes it and appends the transcript section; the temp file is always deleted.
        # Inputs: local_path (str), kwargs with file_extension.
        # Outputs: DocumentConverterResult with title None, or None.
        # Exceptions: pydub decode/export errors propagate (only transcription errors are swallowed).
        # Side effects: creates and deletes a temp file; ffmpeg subprocess; exiftool; network.
        # KNOWN ISSUE: the _args dict is built but never used (dead code).
        # Bail if not a MP3
        extension = kwargs.get("file_extension", "")
        if extension.lower() not in [".mp3", ".m4a"]:
            return None

        md_content = ""

        # Add metadata
        metadata = self._get_metadata(local_path)
        if metadata:
            for f in [
                "Title",
                "Artist",
                "Author",
                "Band",
                "Album",
                "Genre",
                "Track",
                "DateTimeOriginal",
                "CreateDate",
                "Duration",
            ]:
                if f in metadata:
                    md_content += f"{f}: {metadata[f]}\n"

        # Transcribe
        handle, temp_path = tempfile.mkstemp(suffix=".wav")
        os.close(handle)
        try:
            if extension.lower() == ".mp3":
                sound = pydub.AudioSegment.from_mp3(local_path)
            else:
                sound = pydub.AudioSegment.from_file(local_path, format="m4a")
            sound.export(temp_path, format="wav")

            _args = dict()
            _args.update(kwargs)
            _args["file_extension"] = ".wav"

            try:
                transcript = super()._transcribe_audio(temp_path).strip()
                md_content += "\n\n### Audio Transcript:\n" + (
                    "[No speech detected]" if transcript == "" else transcript
                )
            except Exception:
                md_content += "\n\n### Audio Transcript:\nError. Could not transcribe this audio."

        finally:
            os.unlink(temp_path)

        # Return the result
        return DocumentConverterResult(
            title=None,
            text_content=md_content.strip(),
        )


class ZipConverter(DocumentConverter):
    # [CLASS DOC]
    # Description: Converter for .zip archives: extracts everything into a persistent directory and returns a
    #   Markdown bullet list of extracted file paths.
    # Inputs: extract_dir (str, default 'downloads', relative to the CWD) stored as extract_dir.
    # Outputs: n/a.
    # Exceptions: OSError if the directory cannot be created.
    # Side effects: creates extract_dir at construction time.
    # KNOWN ISSUE: extraction uses extractall without path sanitization, relying on zipfile's own handling;
    #   extracted files persist and may overwrite earlier ones. MarkdownConverter always constructs one, so
    #   creating a 'downloads' directory in the CWD is a side effect of building a MarkdownConverter.
    """
    Extracts ZIP files to a permanent local directory and returns a listing of extracted files.
    """

    def __init__(self, extract_dir: str = "downloads"):
        # [FUNCTION DOC]
        # Description: Stores the extraction directory and creates it (including parents) if missing.
        # Inputs: extract_dir (str, default 'downloads').
        # Outputs: none (initializer).
        # Exceptions: OSError from directory creation propagates.
        # Side effects: creates a directory on disk.
        """
        Initialize with path to extraction directory.

        Args:
            extract_dir: The directory where files will be extracted. Defaults to "downloads"
        """
        self.extract_dir = extract_dir
        # Create the extraction directory if it doesn't exist
        os.makedirs(self.extract_dir, exist_ok=True)

    def convert(self, local_path: str, **kwargs: Any) -> None | DocumentConverterResult:
        # [FUNCTION DOC]
        # Description: Returns None unless file_extension is .zip and the file is a valid ZIP. Extracts all
        #   members into extract_dir, collects non-directory member paths joined with extract_dir and '/',
        #   sorts them and returns 'Downloaded the following files:' followed by '* path' lines.
        # Inputs: local_path (str), kwargs with file_extension.
        # Outputs: DocumentConverterResult titled 'Extracted Files', or None.
        # Exceptions: zipfile/OSError errors propagate.
        # Side effects: writes extracted files to disk.
        # Bail if not a ZIP file
        extension = kwargs.get("file_extension", "")
        if extension.lower() != ".zip":
            return None

        # Verify it's actually a ZIP file
        if not zipfile.is_zipfile(local_path):
            return None

        # Extract all files and build list
        extracted_files = []
        with zipfile.ZipFile(local_path, "r") as zip_ref:
            # Extract all files
            zip_ref.extractall(self.extract_dir)
            # Get list of all files
            for file_path in zip_ref.namelist():
                # Skip directories
                if not file_path.endswith("/"):
                    extracted_files.append(self.extract_dir + "/" + file_path)

        # Sort files for consistent output
        extracted_files.sort()

        # Build the markdown content
        md_content = "Downloaded the following files:\n"
        for file in extracted_files:
            md_content += f"* {file}\n"

        return DocumentConverterResult(title="Extracted Files", text_content=md_content.strip())


class ImageConverter(MediaConverter):
    # [CLASS DOC]
    # Description: Converter for .jpg/.jpeg/.png images: selected exiftool metadata plus an optional
    #   description from a multimodal LLM client.
    # Inputs: no constructor arguments.
    # Outputs: n/a.
    # Exceptions: see convert.
    # Side effects: subprocess and network.
    # KNOWN ISSUE: the docstring mentions OCR via easyocr, but no OCR is implemented.
    """
    Converts images to markdown via extraction of metadata (if `exiftool` is installed), OCR (if `easyocr` is installed), and description via a multimodal LLM (if an mlm_client is configured).
    """

    def convert(self, local_path, **kwargs) -> None | DocumentConverterResult:
        # [FUNCTION DOC]
        # Description: Returns None unless file_extension is .jpg, .jpeg or .png. Emits 'Field: value' lines
        #   for ImageSize, Title, Caption, Description, Keywords, Artist, Author, DateTimeOriginal,
        #   CreateDate, GPSPosition when present. If both mlm_client and mlm_model kwargs are given, appends a
        #   '# Description:' section from _get_mlm_description using the optional mlm_prompt.
        # Inputs: local_path (str), kwargs with file_extension, mlm_client, mlm_model, mlm_prompt (all
        #   optional).
        # Outputs: DocumentConverterResult with title None (text not stripped), or None.
        # Exceptions: LLM client errors propagate.
        # Side effects: exiftool subprocess; network call to the LLM; writes the prompt to stderr.
        # KNOWN ISSUE: the inline comment says 'Bail if not a XLSX' but the check is for image extensions.
        # Bail if not a XLSX
        extension = kwargs.get("file_extension", "")
        if extension.lower() not in [".jpg", ".jpeg", ".png"]:
            return None

        md_content = ""

        # Add metadata
        metadata = self._get_metadata(local_path)
        if metadata:
            for f in [
                "ImageSize",
                "Title",
                "Caption",
                "Description",
                "Keywords",
                "Artist",
                "Author",
                "DateTimeOriginal",
                "CreateDate",
                "GPSPosition",
            ]:
                if f in metadata:
                    md_content += f"{f}: {metadata[f]}\n"

        # Try describing the image with GPTV
        mlm_client = kwargs.get("mlm_client")
        mlm_model = kwargs.get("mlm_model")
        if mlm_client is not None and mlm_model is not None:
            md_content += (
                "\n# Description:\n"
                + self._get_mlm_description(
                    local_path, extension, mlm_client, mlm_model, prompt=kwargs.get("mlm_prompt")
                ).strip()
                + "\n"
            )

        return DocumentConverterResult(
            title=None,
            text_content=md_content,
        )

    def _get_mlm_description(self, local_path, extension, client, model, prompt=None):
        # [FUNCTION DOC]
        # Description: HELPER: base64-encodes the image as a data URI (mimetype guessed from extension,
        #   default image/jpeg) and sends a single user message with the prompt text and image_url to an
        #   OpenAI-style chat completions client.
        # Inputs: local_path (str), extension (str), client (OpenAI-compatible client), model (str), prompt
        #   (str or None; blank/None becomes 'Write a detailed caption for this image.').
        # Outputs: the first choice's message content (str).
        # Exceptions: file and client errors propagate.
        # Side effects: reads the file; prints the prompt to stderr; network call.
        if prompt is None or prompt.strip() == "":
            prompt = "Write a detailed caption for this image."

        sys.stderr.write(f"MLM Prompt:\n{prompt}\n")

        data_uri = ""
        with open(local_path, "rb") as image_file:
            content_type, encoding = mimetypes.guess_type("_dummy" + extension)
            if content_type is None:
                content_type = "image/jpeg"
            image_base64 = base64.b64encode(image_file.read()).decode("utf-8")
            data_uri = f"data:{content_type};base64,{image_base64}"

        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": data_uri,
                        },
                    },
                ],
            }
        ]

        response = client.chat.completions.create(model=model, messages=messages)
        return response.choices[0].message.content


class FileConversionException(Exception):
    # [CLASS DOC]
    # Description: Raised when at least one converter accepted the file but errored and none succeeded.
    # Inputs: standard Exception arguments.
    # Outputs: n/a.
    # Exceptions: n/a.
    # Side effects: none
    pass


class UnsupportedFormatException(Exception):
    # [CLASS DOC]
    # Description: Raised when no registered converter accepts the file.
    # Inputs: standard Exception arguments.
    # Outputs: n/a.
    # Exceptions: n/a.
    # Side effects: none
    pass


class MarkdownConverter:
    # [CLASS DOC]
    # Description: Facade that converts local paths, URLs (http/https/file), streams and requests Responses to
    #   Markdown by trying registered converters, most recently registered first, for each candidate
    #   extension.
    # Inputs: requests_session (requests.Session or None; a new Session is created if None), mlm_client and
    #   mlm_model (optional, injected into converter kwargs). Key attributes: _requests_session, _mlm_client,
    #   _mlm_model, _page_converters (list, highest priority first).
    # Outputs: n/a.
    # Exceptions: see methods.
    # Side effects: constructing it creates a ZipConverter, which creates the 'downloads' directory.
    """(In preview) An extremely simple text-based document reader, suitable for LLM use.
    This reader will convert common file-types or webpages to Markdown."""

    def __init__(
        self,
        requests_session: requests.Session | None = None,
        mlm_client: Any | None = None,
        mlm_model: Any | None = None,
    ):
        # [FUNCTION DOC]
        # Description: Stores session and LLM settings and registers the default converters in order
        #   PlainText, Html, Wikipedia, YouTube, Docx, Xlsx, Pptx, Wav, Mp3, Image, Zip, Pdf; since
        #   registration prepends, Pdf ends up tried first and PlainText last.
        # Inputs: requests_session (requests.Session or None, default None), mlm_client (any, default None),
        #   mlm_model (any, default None).
        # Outputs: none (initializer).
        # Exceptions: OSError if the ZIP extraction directory cannot be created.
        # Side effects: creates a 'downloads' directory in the CWD; sets instance attributes.
        if requests_session is None:
            self._requests_session = requests.Session()
        else:
            self._requests_session = requests_session

        self._mlm_client = mlm_client
        self._mlm_model = mlm_model

        self._page_converters: list[DocumentConverter] = []

        # Register converters for successful browsing operations
        # Later registrations are tried first / take higher priority than earlier registrations
        # To this end, the most specific converters should appear below the most generic converters
        self.register_page_converter(PlainTextConverter())
        self.register_page_converter(HtmlConverter())
        self.register_page_converter(WikipediaConverter())
        self.register_page_converter(YouTubeConverter())
        self.register_page_converter(DocxConverter())
        self.register_page_converter(XlsxConverter())
        self.register_page_converter(PptxConverter())
        self.register_page_converter(WavConverter())
        self.register_page_converter(Mp3Converter())
        self.register_page_converter(ImageConverter())
        self.register_page_converter(ZipConverter())
        self.register_page_converter(PdfConverter())

    def convert(
        self, source: str | requests.Response, **kwargs: Any
    ) -> DocumentConverterResult:  # TODO: deal with kwargs
        # [FUNCTION DOC]
        # Description: Dispatches by source type: strings starting with http://, https:// or file:// go to
        #   convert_url, other strings to convert_local, requests.Response objects to convert_response.
        # Inputs: source (str path/URL or requests.Response), kwargs forwarded (e.g. file_extension, mlm_*
        #   options).
        # Outputs: DocumentConverterResult.
        # Exceptions: whatever the delegated method raises.
        # Side effects: as for the delegated method.
        # KNOWN ISSUE: the docstring documents an 'extension' argument but the kwarg actually read is
        #   'file_extension'; any other source type silently returns None. file:// URLs are sent to requests,
        #   which cannot fetch them.
        """
        Args:
            - source: can be a string representing a path or url, or a requests.response object
            - extension: specifies the file extension to use when interpreting the file. If None, infer from source (path, uri, content-type, etc.)
        """

        # Local path or url
        if isinstance(source, str):
            if source.startswith("http://") or source.startswith("https://") or source.startswith("file://"):
                return self.convert_url(source, **kwargs)
            else:
                return self.convert_local(source, **kwargs)
        # Request response
        elif isinstance(source, requests.Response):
            return self.convert_response(source, **kwargs)

    def convert_local(self, path: str, **kwargs: Any) -> DocumentConverterResult:  # TODO: deal with kwargs
        # [FUNCTION DOC]
        # Description: Builds a candidate extension list: explicit file_extension kwarg first, then the path's
        #   extension, then puremagic's guess, and runs _convert on the path.
        # Inputs: path (str), kwargs (file_extension optional plus converter options).
        # Outputs: DocumentConverterResult.
        # Exceptions: FileConversionException or UnsupportedFormatException from _convert.
        # Side effects: reads the file.
        # Prepare a list of extensions to try (in order of priority)
        ext = kwargs.get("file_extension")
        extensions = [ext] if ext is not None else []

        # Get extension alternatives from the path and puremagic
        base, ext = os.path.splitext(path)
        self._append_ext(extensions, ext)
        self._append_ext(extensions, self._guess_ext_magic(path))

        # Convert
        return self._convert(path, extensions, **kwargs)

    # TODO what should stream's type be?
    def convert_stream(self, stream: Any, **kwargs: Any) -> DocumentConverterResult:  # TODO: deal with kwargs
        # [FUNCTION DOC]
        # Description: Writes the stream's full content (str encoded as UTF-8, or bytes) to a temp file, adds
        #   puremagic's extension guess after any explicit file_extension, converts it and deletes the temp
        #   file.
        # Inputs: stream (object with read()), kwargs (file_extension optional plus converter options).
        # Outputs: DocumentConverterResult.
        # Exceptions: conversion exceptions propagate (temp file still removed).
        # Side effects: creates and deletes a temporary file.
        # Prepare a list of extensions to try (in order of priority)
        ext = kwargs.get("file_extension")
        extensions = [ext] if ext is not None else []

        # Save the file locally to a temporary file. It will be deleted before this method exits
        handle, temp_path = tempfile.mkstemp()
        fh = os.fdopen(handle, "wb")
        result = None
        try:
            # Write to the temporary file
            content = stream.read()
            if isinstance(content, str):
                fh.write(content.encode("utf-8"))
            else:
                fh.write(content)
            fh.close()

            # Use puremagic to check for more extension options
            self._append_ext(extensions, self._guess_ext_magic(temp_path))

            # Convert
            result = self._convert(temp_path, extensions, **kwargs)
        # Clean up
        finally:
            try:
                fh.close()
            except Exception:
                pass
            os.unlink(temp_path)

        return result

    def convert_url(self, url: str, **kwargs: Any) -> DocumentConverterResult:  # TODO: fix kwargs type
        # [FUNCTION DOC]
        # Description: GETs the URL with streaming and a fixed desktop Edge User-Agent via the session, raises
        #   for HTTP errors and delegates to convert_response.
        # Inputs: url (str), kwargs forwarded.
        # Outputs: DocumentConverterResult (or None, see convert_response).
        # Exceptions: requests exceptions including HTTPError propagate.
        # Side effects: network request.
        # Send a HTTP request to the URL
        user_agent = "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/119.0.0.0 Safari/537.36 Edg/119.0.0.0"
        response = self._requests_session.get(url, stream=True, headers={"User-Agent": user_agent})
        response.raise_for_status()
        return self.convert_response(response, **kwargs)

    def convert_response(
        self, response: requests.Response, **kwargs: Any
    ) -> DocumentConverterResult:  # TODO fix kwargs type
        # [FUNCTION DOC]
        # Description: Builds candidate extensions from explicit file_extension, the content-type header's
        #   guessed extension, the content-disposition filename and the URL path extension; streams the body
        #   in 512-byte chunks to a temp file, adds puremagic's guess, and runs _convert with url set to the
        #   response URL.
        # Inputs: response (requests.Response), kwargs (only file_extension is used).
        # Outputs: DocumentConverterResult, or None if conversion failed.
        # Exceptions: none propagate from conversion (caught and printed).
        # Side effects: reads the network stream; creates/deletes a temp file; prints errors to stdout.
        # KNOWN ISSUE: conversion errors are swallowed and None returned despite the return annotation; kwargs
        #   other than file_extension (e.g. mlm_prompt) are not forwarded to _convert.
        # Prepare a list of extensions to try (in order of priority)
        ext = kwargs.get("file_extension")
        extensions = [ext] if ext is not None else []

        # Guess from the mimetype
        content_type = response.headers.get("content-type", "").split(";")[0]
        self._append_ext(extensions, mimetypes.guess_extension(content_type))

        # Read the content disposition if there is one
        content_disposition = response.headers.get("content-disposition", "")
        m = re.search(r"filename=([^;]+)", content_disposition)
        if m:
            base, ext = os.path.splitext(m.group(1).strip("\"'"))
            self._append_ext(extensions, ext)

        # Read from the extension from the path
        base, ext = os.path.splitext(urlparse(response.url).path)
        self._append_ext(extensions, ext)

        # Save the file locally to a temporary file. It will be deleted before this method exits
        handle, temp_path = tempfile.mkstemp()
        fh = os.fdopen(handle, "wb")
        result = None
        try:
            # Download the file
            for chunk in response.iter_content(chunk_size=512):
                fh.write(chunk)
            fh.close()

            # Use puremagic to check for more extension options
            self._append_ext(extensions, self._guess_ext_magic(temp_path))

            # Convert
            result = self._convert(temp_path, extensions, url=response.url)
        except Exception as e:
            print(f"Error in converting: {e}")

        # Clean up
        finally:
            try:
                fh.close()
            except Exception:
                pass
            os.unlink(temp_path)

        return result

    def _convert(self, local_path: str, extensions: list[str | None], **kwargs) -> DocumentConverterResult:
        # [FUNCTION DOC]
        # Description: HELPER: for each candidate extension plus a final None, tries every converter in
        #   priority order with a deep copy of kwargs where file_extension is set (or removed for None) and
        #   the instance's mlm_client / mlm_model are injected if absent. The first non-None result has
        #   trailing whitespace stripped from each line and runs of 3+ newlines collapsed to 2, and is
        #   returned.
        # Inputs: local_path (str), extensions (list of str or None), kwargs (converter options).
        # Outputs: normalized DocumentConverterResult.
        # Exceptions: FileConversionException if any converter raised and none succeeded; otherwise
        #   UnsupportedFormatException.
        # Side effects: as performed by the converters.
        # KNOWN ISSUE: if the very first converter call raises, res is never assigned and the subsequent check
        #   raises UnboundLocalError instead of continuing; only the last error trace is kept, and the leading
        #   blank lines added to it are immediately stripped.
        error_trace = ""
        for ext in extensions + [None]:  # Try last with no extension
            for converter in self._page_converters:
                _kwargs = copy.deepcopy(kwargs)

                # Overwrite file_extension appropriately
                if ext is None:
                    if "file_extension" in _kwargs:
                        del _kwargs["file_extension"]
                else:
                    _kwargs.update({"file_extension": ext})

                # Copy any additional global options
                if "mlm_client" not in _kwargs and self._mlm_client is not None:
                    _kwargs["mlm_client"] = self._mlm_client

                if "mlm_model" not in _kwargs and self._mlm_model is not None:
                    _kwargs["mlm_model"] = self._mlm_model

                # If we hit an error log it and keep trying
                try:
                    res = converter.convert(local_path, **_kwargs)
                except Exception:
                    error_trace = ("\n\n" + traceback.format_exc()).strip()

                if res is not None:
                    # Normalize the content
                    res.text_content = "\n".join([line.rstrip() for line in re.split(r"\r?\n", res.text_content)])
                    res.text_content = re.sub(r"\n{3,}", "\n\n", res.text_content)

                    # Todo
                    return res

        # If we got this far without success, report any exceptions
        if len(error_trace) > 0:
            raise FileConversionException(
                f"Could not convert '{local_path}' to Markdown. File type was recognized as {extensions}. While converting the file, the following error was encountered:\n\n{error_trace}"
            )

        # Nothing can handle it!
        raise UnsupportedFormatException(
            f"Could not convert '{local_path}' to Markdown. The formats {extensions} are not supported."
        )

    def _append_ext(self, extensions, ext):
        # [FUNCTION DOC]
        # Description: HELPER: appends ext (whitespace-stripped) to the extensions list unless it is None or
        #   empty.
        # Inputs: extensions (list, mutated), ext (str or None).
        # Outputs: None.
        # Exceptions: none raised directly.
        # Side effects: mutates the extensions list.
        # KNOWN ISSUE: the docstring promises a unique extension, but the uniqueness check is replaced with an
        #   always-true condition, so duplicates are appended (causing repeated conversion attempts).
        """Append a unique non-None, non-empty extension to a list of extensions."""
        if ext is None:
            return
        ext = ext.strip()
        if ext == "":
            return
        # if ext not in extensions:
        if True:
            extensions.append(ext)

    def _guess_ext_magic(self, path):
        # [FUNCTION DOC]
        # Description: HELPER: uses puremagic to guess the file's extension from its header bytes; returns the
        #   first guess's stripped extension if non-empty.
        # Inputs: path (str).
        # Outputs: extension string (with leading dot) or None.
        # Exceptions: FileNotFoundError, IsADirectoryError and PermissionError are swallowed; other puremagic
        #   errors (e.g. PureError for unknown/empty files) propagate.
        # Side effects: reads the file header.
        """Use puremagic (a Python implementation of libmagic) to guess a file's extension based on the first few bytes."""
        # Use puremagic to guess
        try:
            guesses = puremagic.magic_file(path)
            if len(guesses) > 0:
                ext = guesses[0].extension.strip()
                if len(ext) > 0:
                    return ext
        except FileNotFoundError:
            pass
        except IsADirectoryError:
            pass
        except PermissionError:
            pass
        return None

    def register_page_converter(self, converter: DocumentConverter) -> None:
        # [FUNCTION DOC]
        # Description: Registers a converter at the front of the list, giving it the highest priority.
        # Inputs: converter (DocumentConverter).
        # Outputs: None.
        # Exceptions: none raised directly.
        # Side effects: mutates _page_converters.
        """Register a page text converter."""
        self._page_converters.insert(0, converter)
