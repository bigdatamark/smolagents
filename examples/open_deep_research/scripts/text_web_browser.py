# Shamelessly stolen from Microsoft Autogen team: thanks to them for this great resource!
# https://github.com/microsoft/autogen/blob/gaia_multiagent_v01_march_1st/autogen/browser_utils.py
# [MODULE DOC]
# HELPER: Text-based web browser and the smolagents tools that drive it (adapted from Microsoft Autogen). The
#   browser keeps a history, renders pages/files to markdown via mdconvert, paginates content into viewports,
#   supports SerpAPI Google search and Ctrl+F style search; each tool returns a header (address, title,
#   previous-visit note, viewport position) plus the current viewport text.
# Inputs: depends on requests, pathvalidate, serpapi (GoogleSearch), smolagents.Tool, sibling cookies
#   (COOKIES) and mdconvert (MarkdownConverter and its exceptions).
# Outputs: SimpleTextBrowser, SearchInformationTool, VisitTool, DownloadTool, ArchiveSearchTool, PageUpTool,
#   PageDownTool, FinderTool, FindNextTool.
# Exceptions: none at import.
# Side effects: none at import.
import mimetypes
import os
import pathlib
import re
import time
import uuid
from typing import Any
from urllib.parse import unquote, urljoin, urlparse

import pathvalidate
import requests
from serpapi import GoogleSearch

from smolagents import Tool

from .cookies import COOKIES
from .mdconvert import FileConversionException, MarkdownConverter, UnsupportedFormatException


class SimpleTextBrowser:
    # [CLASS DOC]
    # Minimal Lynx-like browser for agents. Visiting an address records it in history and loads content:
    #   "about:blank" is empty, "google: <query>" runs a SerpAPI search, relative addresses are resolved
    #   against the previous address, other URLs/file URIs are fetched and converted to markdown; non-text
    #   downloads are saved to downloads_folder and then rendered from the local file.
    # Inputs: start_page (str or None, default about:blank), viewport_size (int, default 8192 characters per
    #   viewport), downloads_folder (str or None), serpapi_key (str or None), request_kwargs (dict passed to
    #   requests.get; must not be None).
    # Key attributes: history (list of address and timestamp pairs), page_title, viewport_current_page,
    #   viewport_pages (list of start/end offsets), serpapi_key, request_kwargs (with cookies injected),
    #   private page content and find state.
    # Exceptions: see methods.
    # Side effects: mutates the given request_kwargs dict by adding the module COOKIES.
    """(In preview) An extremely simple text-based web browser comparable to Lynx. Suitable for Agentic use."""

    def __init__(
        self,
        start_page: str | None = None,
        viewport_size: int | None = 1024 * 8,
        downloads_folder: str | None | None = None,
        serpapi_key: str | None | None = None,
        request_kwargs: dict[str, Any] | None | None = None,
    ):
        # [FUNCTION DOC]
        # Initializes state, navigates to the start page, then stores serpapi_key and request_kwargs (adding
        #   cookies) and creates the MarkdownConverter.
        # Inputs: see class doc.
        # Outputs: none.
        # Exceptions: TypeError if request_kwargs is None; AttributeError if start_page is anything but
        #   about:blank (see KNOWN ISSUE).
        # Side effects: mutates request_kwargs.
        # KNOWN ISSUE: set_address is called before serpapi_key, request_kwargs and the converter exist, so a
        #   non-blank start_page fails; request_kwargs None is advertised but crashes.
        self.start_page: str = start_page if start_page else "about:blank"
        self.viewport_size = viewport_size  # Applies only to the standard uri types
        self.downloads_folder = downloads_folder
        self.history: list[tuple[str, float]] = list()
        self.page_title: str | None = None
        self.viewport_current_page = 0
        self.viewport_pages: list[tuple[int, int]] = list()
        self.set_address(self.start_page)
        self.serpapi_key = serpapi_key
        self.request_kwargs = request_kwargs
        self.request_kwargs["cookies"] = COOKIES
        self._mdconvert = MarkdownConverter()
        self._page_content: str = ""

        self._find_on_page_query: str | None = None
        self._find_on_page_last_result: int | None = None  # Location of the last result

    @property
    def address(self) -> str:
        # [FUNCTION DOC]
        # Property returning the address of the current page (last history entry).
        # Inputs: none.
        # Outputs: str.
        # Exceptions: IndexError if history is empty.
        # Side effects: none.
        """Return the address of the current page."""
        return self.history[-1][0]

    def set_address(self, uri_or_path: str, filter_year: int | None = None) -> None:
        # [FUNCTION DOC]
        # Navigates to an address: appends it with the current time to history, dispatches on the address kind
        #   (blank, google search, relative/absolute URL or file URI), then resets to the first viewport.
        # Inputs: uri_or_path (str), filter_year (int or None; only used for google searches).
        # Outputs: None.
        # Exceptions: ValueError if searching without a SerpAPI key; generic Exception when SerpAPI returns no
        #   organic results key; network errors are mostly converted to error pages by the fetch helper.
        # Side effects: mutates history, page content, title and viewport; network I/O; file writes for
        #   downloads.
        # KNOWN ISSUE: it resets find_on_page_query and find_on_page_viewport, which are not the private
        #   attributes used by the find methods, so the previous search state is not actually cleared.
        # TODO: Handle anchors
        self.history.append((uri_or_path, time.time()))

        # Handle special URIs
        if uri_or_path == "about:blank":
            self._set_page_content("")
        elif uri_or_path.startswith("google:"):
            self._serpapi_search(uri_or_path[len("google:") :].strip(), filter_year=filter_year)
        else:
            if (
                not uri_or_path.startswith("http:")
                and not uri_or_path.startswith("https:")
                and not uri_or_path.startswith("file:")
            ):
                if len(self.history) > 1:
                    prior_address = self.history[-2][0]
                    uri_or_path = urljoin(prior_address, uri_or_path)
                    # Update the address with the fully-qualified path
                    self.history[-1] = (uri_or_path, self.history[-1][1])
            self._fetch_page(uri_or_path)

        self.viewport_current_page = 0
        self.find_on_page_query = None
        self.find_on_page_viewport = None

    @property
    def viewport(self) -> str:
        # [FUNCTION DOC]
        # Property returning the text of the current viewport slice.
        # Inputs: none.
        # Outputs: str.
        # Exceptions: IndexError if the viewport index is out of range.
        # Side effects: none.
        """Return the content of the current viewport."""
        bounds = self.viewport_pages[self.viewport_current_page]
        return self.page_content[bounds[0] : bounds[1]]

    @property
    def page_content(self) -> str:
        # [FUNCTION DOC]
        # Property returning the full text content of the current page.
        # Inputs: none.
        # Outputs: str.
        # Exceptions: none raised directly.
        # Side effects: none.
        """Return the full contents of the current page."""
        return self._page_content

    def _set_page_content(self, content: str) -> None:
        # [FUNCTION DOC]
        # HELPER: Replaces the page content, recomputes viewport boundaries and clamps the current viewport
        #   index to the last page.
        # Inputs: content (str).
        # Outputs: None.
        # Exceptions: none raised directly.
        # Side effects: mutates page content, viewport_pages and viewport_current_page.
        """Sets the text content of the current page."""
        self._page_content = content
        self._split_pages()
        if self.viewport_current_page >= len(self.viewport_pages):
            self.viewport_current_page = len(self.viewport_pages) - 1

    def page_down(self) -> None:
        # [FUNCTION DOC]
        # Moves to the next viewport, stopping at the last one.
        # Inputs: none.
        # Outputs: None.
        # Exceptions: none raised directly.
        # Side effects: mutates viewport_current_page.
        self.viewport_current_page = min(self.viewport_current_page + 1, len(self.viewport_pages) - 1)

    def page_up(self) -> None:
        # [FUNCTION DOC]
        # Moves to the previous viewport, stopping at the first one.
        # Inputs: none.
        # Outputs: None.
        # Exceptions: none raised directly.
        # Side effects: mutates viewport_current_page.
        self.viewport_current_page = max(self.viewport_current_page - 1, 0)

    def find_on_page(self, query: str) -> str | None:
        # [FUNCTION DOC]
        # Searches for the query starting at the current viewport and wrapping around; if the same query is
        #   repeated while still on the last match, behaves like find_next.
        # Inputs: query (str; supports star wildcards, matching is case-insensitive on word tokens).
        # Outputs: the matching viewport text, or None if not found.
        # Exceptions: none raised directly.
        # Side effects: stores the query and last match index; moves the current viewport to the match.
        """Searches for the query from the current viewport forward, looping back to the start if necessary."""

        # Did we get here via a previous find_on_page search with the same query?
        # If so, map to find_next
        if query == self._find_on_page_query and self.viewport_current_page == self._find_on_page_last_result:
            return self.find_next()

        # Ok it's a new search start from the current viewport
        self._find_on_page_query = query
        viewport_match = self._find_next_viewport(query, self.viewport_current_page)
        if viewport_match is None:
            self._find_on_page_last_result = None
            return None
        else:
            self.viewport_current_page = viewport_match
            self._find_on_page_last_result = viewport_match
            return self.viewport

    def find_next(self) -> str | None:
        # [FUNCTION DOC]
        # Continues the last search from the viewport after the previous match (wrapping to the start).
        # Inputs: none.
        # Outputs: the matching viewport text, or None if no search is active or nothing matches.
        # Exceptions: none raised directly.
        # Side effects: updates the last match index and current viewport.
        """Scroll to the next viewport that matches the query"""

        if self._find_on_page_query is None:
            return None

        starting_viewport = self._find_on_page_last_result
        if starting_viewport is None:
            starting_viewport = 0
        else:
            starting_viewport += 1
            if starting_viewport >= len(self.viewport_pages):
                starting_viewport = 0

        viewport_match = self._find_next_viewport(self._find_on_page_query, starting_viewport)
        if viewport_match is None:
            self._find_on_page_last_result = None
            return None
        else:
            self.viewport_current_page = viewport_match
            self._find_on_page_last_result = viewport_match
            return self.viewport

    def _find_next_viewport(self, query: str, starting_viewport: int) -> int | None:
        # [FUNCTION DOC]
        # HELPER: Finds the first viewport index, starting from starting_viewport and wrapping, whose
        #   word-normalized lowercase text matches the normalized query regex (non-word runs collapsed to
        #   spaces, star turned into a match-anything pattern merged with the preceding word).
        # Inputs: query (str or None), starting_viewport (int).
        # Outputs: viewport index or None (also None for empty/None queries).
        # Exceptions: none raised directly.
        # Side effects: none.
        # KNOWN ISSUE: the normalized query is lowercased but not regex-escaped, and the content is lowercased
        #   after stripping while the query keeps its surrounding spaces, so some matches near content
        #   boundaries are missed.
        """Search for matches between the starting viewport looping when reaching the end."""

        if query is None:
            return None

        # Normalize the query, and convert to a regular expression
        nquery = re.sub(r"\*", "__STAR__", query)
        nquery = " " + (" ".join(re.split(r"\W+", nquery))).strip() + " "
        nquery = nquery.replace(" __STAR__ ", "__STAR__ ")  # Merge isolated stars with prior word
        nquery = nquery.replace("__STAR__", ".*").lower()

        if nquery.strip() == "":
            return None

        idxs = list()
        idxs.extend(range(starting_viewport, len(self.viewport_pages)))
        idxs.extend(range(0, starting_viewport))

        for i in idxs:
            bounds = self.viewport_pages[i]
            content = self.page_content[bounds[0] : bounds[1]]

            # TODO: Remove markdown links and images
            ncontent = " " + (" ".join(re.split(r"\W+", content))).strip().lower() + " "
            if re.search(nquery, ncontent):
                return i

        return None

    def visit_page(self, path_or_uri: str, filter_year: int | None = None) -> str:
        # [FUNCTION DOC]
        # Navigates to an address and returns the first viewport.
        # Inputs: path_or_uri (str), filter_year (int or None).
        # Outputs: viewport text (str).
        # Exceptions: as set_address.
        # Side effects: as set_address.
        """Update the address, visit the page, and return the content of the viewport."""
        self.set_address(path_or_uri, filter_year=filter_year)
        return self.viewport

    def _split_pages(self) -> None:
        # [FUNCTION DOC]
        # HELPER: Computes viewport boundaries: a single viewport for google search results, one empty
        #   viewport for empty content, otherwise chunks of viewport_size characters each extended to end just
        #   after whitespace.
        # Inputs: none (uses current address and content).
        # Outputs: None.
        # Exceptions: TypeError if viewport_size is None.
        # Side effects: replaces viewport_pages.
        # Do not split search results
        if self.address.startswith("google:"):
            self.viewport_pages = [(0, len(self._page_content))]
            return

        # Handle empty pages
        if len(self._page_content) == 0:
            self.viewport_pages = [(0, 0)]
            return

        # Break the viewport into pages
        self.viewport_pages = []
        start_idx = 0
        while start_idx < len(self._page_content):
            end_idx = min(start_idx + self.viewport_size, len(self._page_content))  # type: ignore[operator]
            # Adjust to end on a space
            while end_idx < len(self._page_content) and self._page_content[end_idx - 1] not in [" ", "\t", "\r", "\n"]:
                end_idx += 1
            self.viewport_pages.append((start_idx, end_idx))
            start_idx = end_idx

    def _serpapi_search(self, query: str, filter_year: int | None = None) -> None:
        # [FUNCTION DOC]
        # HELPER: Runs a Google search via SerpAPI (optionally restricted to a calendar year) and sets the
        #   page content to a numbered markdown list of results with title/link, date, source, a
        #   previous-visit note and snippet.
        # Inputs: query (str), filter_year (int or None).
        # Outputs: None.
        # Exceptions: ValueError if serpapi_key is None; generic Exception when the response has no
        #   organic_results key.
        # Side effects: network call; sets page_title and page content (a "no results" message when the list
        #   is empty).
        if self.serpapi_key is None:
            raise ValueError("Missing SerpAPI key.")

        params = {
            "engine": "google",
            "q": query,
            "api_key": self.serpapi_key,
        }
        if filter_year is not None:
            params["tbs"] = f"cdr:1,cd_min:01/01/{filter_year},cd_max:12/31/{filter_year}"

        search = GoogleSearch(params)
        results = search.get_dict()
        self.page_title = f"{query} - Search"
        if "organic_results" not in results.keys():
            raise Exception(f"No results found for query: '{query}'. Use a less specific query.")
        if len(results["organic_results"]) == 0:
            year_filter_message = f" with filter year={filter_year}" if filter_year is not None else ""
            self._set_page_content(
                f"No results found for '{query}'{year_filter_message}. Try with a more general query, or remove the year filter."
            )
            return

        def _prev_visit(url):
            # [FUNCTION DOC]
            # HELPER: Returns a note saying how many seconds ago a URL was last visited, based on history.
            # Inputs: url (str).
            # Outputs: note string ending with a newline, or an empty string if never visited.
            # Exceptions: none raised directly.
            # Side effects: none.
            for i in range(len(self.history) - 1, -1, -1):
                if self.history[i][0] == url:
                    return f"You previously visited this page {round(time.time() - self.history[i][1])} seconds ago.\n"
            return ""

        web_snippets: list[str] = list()
        idx = 0
        if "organic_results" in results:
            for page in results["organic_results"]:
                idx += 1
                date_published = ""
                if "date" in page:
                    date_published = "\nDate published: " + page["date"]

                source = ""
                if "source" in page:
                    source = "\nSource: " + page["source"]

                snippet = ""
                if "snippet" in page:
                    snippet = "\n" + page["snippet"]

                redacted_version = f"{idx}. [{page['title']}]({page['link']}){date_published}{source}\n{_prev_visit(page['link'])}{snippet}"

                redacted_version = redacted_version.replace("Your browser can't play this video.", "")
                web_snippets.append(redacted_version)

        content = (
            f"A Google search for '{query}' found {len(web_snippets)} results:\n\n## Web Results\n"
            + "\n\n".join(web_snippets)
        )

        self._set_page_content(content)

    def _fetch_page(self, url: str) -> None:
        # [FUNCTION DOC]
        # HELPER: Loads a URL. file:// URIs are converted locally; otherwise does a streamed GET with
        #   request_kwargs: text content types are converted to markdown; anything else is saved to
        #   downloads_folder under a sanitized (de-duplicated with numeric suffixes, or uuid-based) file name
        #   and then rendered by navigating to its file URI.
        # Inputs: url (str).
        # Outputs: None.
        # Exceptions: conversion exceptions, missing files and request errors are caught and rendered as
        #   "Download complete" or "Error ..." pages; other errors propagate (e.g. TypeError when
        #   downloads_folder is None).
        # Side effects: network I/O, file writes, history/page state mutation, prints conversion errors.
        # KNOWN ISSUE: page_title is set to a one-element tuple instead of a string in the download-complete
        #   handlers; the NameError guard around file naming is ineffective; the HTML error branch passes the
        #   response object to convert.
        download_path = ""
        try:
            if url.startswith("file://"):
                download_path = os.path.normcase(os.path.normpath(unquote(url[7:])))
                res = self._mdconvert.convert_local(download_path)
                self.page_title = res.title
                self._set_page_content(res.text_content)
            else:
                # Prepare the request parameters
                request_kwargs = self.request_kwargs.copy() if self.request_kwargs is not None else {}
                request_kwargs["stream"] = True

                # Send a HTTP request to the URL
                response = requests.get(url, **request_kwargs)
                response.raise_for_status()

                # If the HTTP request was successful
                content_type = response.headers.get("content-type", "")

                # Text or HTML
                if "text/" in content_type.lower():
                    res = self._mdconvert.convert_response(response)
                    self.page_title = res.title
                    self._set_page_content(res.text_content)
                # A download
                else:
                    # Try producing a safe filename
                    fname = None
                    download_path = None
                    try:
                        fname = pathvalidate.sanitize_filename(os.path.basename(urlparse(url).path)).strip()
                        download_path = os.path.abspath(os.path.join(self.downloads_folder, fname))

                        suffix = 0
                        while os.path.exists(download_path) and suffix < 1000:
                            suffix += 1
                            base, ext = os.path.splitext(fname)
                            new_fname = f"{base}__{suffix}{ext}"
                            download_path = os.path.abspath(os.path.join(self.downloads_folder, new_fname))

                    except NameError:
                        pass

                    # No suitable name, so make one
                    if fname is None:
                        extension = mimetypes.guess_extension(content_type)
                        if extension is None:
                            extension = ".download"
                        fname = str(uuid.uuid4()) + extension
                        download_path = os.path.abspath(os.path.join(self.downloads_folder, fname))

                    # Open a file for writing
                    with open(download_path, "wb") as fh:
                        for chunk in response.iter_content(chunk_size=512):
                            fh.write(chunk)

                    # Render it
                    local_uri = pathlib.Path(download_path).as_uri()
                    self.set_address(local_uri)

        except UnsupportedFormatException as e:
            print(e)
            self.page_title = ("Download complete.",)
            self._set_page_content(f"# Download complete\n\nSaved file to '{download_path}'")
        except FileConversionException as e:
            print(e)
            self.page_title = ("Download complete.",)
            self._set_page_content(f"# Download complete\n\nSaved file to '{download_path}'")
        except FileNotFoundError:
            self.page_title = "Error 404"
            self._set_page_content(f"## Error 404\n\nFile not found: {download_path}")
        except requests.exceptions.RequestException as request_exception:
            try:
                self.page_title = f"Error {response.status_code}"

                # If the error was rendered in HTML we might as well render it
                content_type = response.headers.get("content-type", "")
                if content_type is not None and "text/html" in content_type.lower():
                    res = self._mdconvert.convert(response)
                    self.page_title = f"Error {response.status_code}"
                    self._set_page_content(f"## Error {response.status_code}\n\n{res.text_content}")
                else:
                    text = ""
                    for chunk in response.iter_content(chunk_size=512, decode_unicode=True):
                        text += chunk
                    self.page_title = f"Error {response.status_code}"
                    self._set_page_content(f"## Error {response.status_code}\n\n{text}")
            except NameError:
                self.page_title = "Error"
                self._set_page_content(f"## Error\n\n{str(request_exception)}")

    def _state(self) -> tuple[str, str]:
        # [FUNCTION DOC]
        # HELPER: Builds the header describing the current page (address, title if any, previous-visit note,
        #   viewport position) and returns it with the current viewport.
        # Inputs: none.
        # Outputs: tuple of header string and viewport string.
        # Exceptions: none raised directly.
        # Side effects: none.
        header = f"Address: {self.address}\n"
        if self.page_title is not None:
            header += f"Title: {self.page_title}\n"

        current_page = self.viewport_current_page
        total_pages = len(self.viewport_pages)

        address = self.address
        for i in range(len(self.history) - 2, -1, -1):  # Start from the second last
            if self.history[i][0] == address:
                header += f"You previously visited this page {round(time.time() - self.history[i][1])} seconds ago.\n"
                break

        header += f"Viewport position: Showing page {current_page + 1} of {total_pages}.\n"
        return (header, self.viewport)


class SearchInformationTool(Tool):
    # [CLASS DOC]
    # Tool "web_search": runs a Google search through the browser (SerpAPI) and returns results.
    # Inputs: constructor browser (SimpleTextBrowser). Tool inputs query (str) and optional filter_year.
    # Key attributes: browser.
    # Exceptions: see forward.
    # Side effects: none at construction.
    name = "web_search"
    description = "Perform a web search query (think a google search) and returns the search results."
    inputs = {"query": {"type": "string", "description": "The web search query to perform."}}
    inputs["filter_year"] = {
        "type": "string",
        "description": "[Optional parameter]: filter the search results to only include pages from a specific year. For example, '2020' will only include pages from 2020. Make sure to use this parameter if you're trying to search for articles from a specific date!",
        "nullable": True,
    }
    output_type = "string"

    def __init__(self, browser):
        # [FUNCTION DOC]
        # Initializes the Tool base and stores the browser.
        # Inputs: browser (SimpleTextBrowser).
        # Outputs: none.
        # Exceptions: Tool validation errors propagate.
        # Side effects: none.
        super().__init__()
        self.browser = browser

    def forward(self, query: str, filter_year: int | None = None) -> str:
        # [FUNCTION DOC]
        # Visits "google: <query>" with the year filter and returns the header, a separator line and the
        #   results.
        # Inputs: query (str), filter_year (int or None).
        # Outputs: str.
        # Exceptions: browser search errors propagate.
        # Side effects: network call, browser state change.
        # KNOWN ISSUE: filter_year is declared as a string input but annotated int; it is interpolated as-is
        #   so both work.
        self.browser.visit_page(f"google: {query}", filter_year=filter_year)
        header, content = self.browser._state()
        return header.strip() + "\n=======================\n" + content


class VisitTool(Tool):
    # [CLASS DOC]
    # Tool "visit_page": loads a URL in the browser and returns its first viewport.
    # Inputs: constructor browser (SimpleTextBrowser, default None but required at call time). Tool input url
    #   (str).
    # Key attributes: browser.
    # Exceptions: see forward.
    # Side effects: none at construction.
    name = "visit_page"
    description = "Visit a webpage at a given URL and return its text. Given a url to a YouTube video, this returns the transcript."
    inputs = {"url": {"type": "string", "description": "The relative or absolute url of the webpage to visit."}}
    output_type = "string"

    def __init__(self, browser=None):
        # [FUNCTION DOC]
        # Initializes the Tool base and stores the browser.
        # Inputs: browser (default None).
        # Outputs: none.
        # Exceptions: Tool validation errors propagate.
        # Side effects: none.
        super().__init__()
        self.browser = browser

    def forward(self, url: str) -> str:
        # [FUNCTION DOC]
        # Visits the URL and returns header, separator and viewport text.
        # Inputs: url (str, relative or absolute).
        # Outputs: str.
        # Exceptions: AttributeError if no browser was given; browser errors propagate.
        # Side effects: network I/O, browser state change.
        self.browser.visit_page(url)
        header, content = self.browser._state()
        return header.strip() + "\n=======================\n" + content


class DownloadTool(Tool):
    # [CLASS DOC]
    # Tool "download_file": downloads a binary file (xlsx, pptx, audio, png, docx...) to ./downloads.
    # Inputs: constructor browser (stored but unused). Tool input url (str).
    # Key attributes: browser.
    # Exceptions: see forward.
    # Side effects: none at construction.
    name = "download_file"
    description = """
Download a file at a given URL. The file should be of this format: [".xlsx", ".pptx", ".wav", ".mp3", ".m4a", ".png", ".docx"]
After using this tool, for further inspection of this page you should return the download path to your manager via final_answer, and they will be able to inspect it.
DO NOT use this tool for .pdf or .txt or .htm files: for these types of files use visit_page with the file url instead."""
    inputs = {"url": {"type": "string", "description": "The relative or absolute url of the file to be downloaded."}}
    output_type = "string"

    def __init__(self, browser):
        # [FUNCTION DOC]
        # Initializes the Tool base and stores the browser.
        # Inputs: browser.
        # Outputs: none.
        # Exceptions: Tool validation errors propagate.
        # Side effects: none.
        super().__init__()
        self.browser = browser

    def forward(self, url: str) -> str:
        # [FUNCTION DOC]
        # Downloads the URL (arxiv "abs" links rewritten to "pdf") and writes it to ./downloads/file<guessed
        #   extension> (or file.object), overwriting any previous download.
        # Inputs: url (str).
        # Outputs: message with the saved path.
        # Exceptions: generic Exception if the extension is pdf/txt/htm (after the file was written);
        #   TypeError when no extension can be guessed; I/O error if ./downloads does not exist.
        # Side effects: network GET, file write.
        # KNOWN ISSUE: checks membership in a None extension (TypeError instead of the intended path), raises
        #   only after saving, ignores HTTP error statuses, and replaces every "abs" substring in the URL.
        import requests

        if "arxiv" in url:
            url = url.replace("abs", "pdf")
        response = requests.get(url)
        content_type = response.headers.get("content-type", "")
        extension = mimetypes.guess_extension(content_type)
        if extension and isinstance(extension, str):
            new_path = f"./downloads/file{extension}"
        else:
            new_path = "./downloads/file.object"

        with open(new_path, "wb") as f:
            f.write(response.content)

        if "pdf" in extension or "txt" in extension or "htm" in extension:
            raise Exception("Do not use this tool for pdf or txt or html files: use visit_page instead.")

        return f"File was downloaded and saved under path {new_path}."


class ArchiveSearchTool(Tool):
    # [CLASS DOC]
    # Tool "find_archived_url": finds the Wayback Machine snapshot of a URL closest to a date and opens it in
    #   the browser.
    # Inputs: constructor browser (SimpleTextBrowser, default None but required). Tool inputs url (str), date
    #   (str YYYYMMDD).
    # Key attributes: browser.
    # Exceptions: see forward.
    # Side effects: none at construction.
    name = "find_archived_url"
    description = "Given a url, searches the Wayback Machine and returns the archived version of the url that's closest in time to the desired date."
    inputs = {
        "url": {"type": "string", "description": "The url you need the archive for."},
        "date": {
            "type": "string",
            "description": "The date that you want to find the archive for. Give this date in the format 'YYYYMMDD', for instance '27 June 2008' is written as '20080627'.",
        },
    }
    output_type = "string"

    def __init__(self, browser=None):
        # [FUNCTION DOC]
        # Initializes the Tool base and stores the browser.
        # Inputs: browser (default None).
        # Outputs: none.
        # Exceptions: Tool validation errors propagate.
        # Side effects: none.
        super().__init__()
        self.browser = browser

    def forward(self, url, date) -> str:
        # [FUNCTION DOC]
        # Queries the Wayback availability API with and without the timestamp, prefers the timestamped closest
        #   snapshot, visits it and returns a line naming the snapshot date followed by header, separator and
        #   viewport.
        # Inputs: url (str), date (str YYYYMMDD).
        # Outputs: str.
        # Exceptions: generic Exception if no snapshot exists; network/JSON errors propagate.
        # Side effects: two network GETs (always both), browser navigation, prints the snapshot found.
        # KNOWN ISSUE: the url is not URL-encoded when inserted into the query string.
        import requests

        no_timestamp_url = f"https://archive.org/wayback/available?url={url}"
        archive_url = no_timestamp_url + f"&timestamp={date}"
        response = requests.get(archive_url).json()
        response_notimestamp = requests.get(no_timestamp_url).json()
        if "archived_snapshots" in response and "closest" in response["archived_snapshots"]:
            closest = response["archived_snapshots"]["closest"]
            print("Archive found!", closest)

        elif "archived_snapshots" in response_notimestamp and "closest" in response_notimestamp["archived_snapshots"]:
            closest = response_notimestamp["archived_snapshots"]["closest"]
            print("Archive found!", closest)
        else:
            raise Exception(f"Your {url=} was not archived on Wayback Machine, try a different url.")
        target_url = closest["url"]
        self.browser.visit_page(target_url)
        header, content = self.browser._state()
        return (
            f"Web archive for url {url}, snapshot taken at date {closest['timestamp'][:8]}:\n"
            + header.strip()
            + "\n=======================\n"
            + content
        )


class PageUpTool(Tool):
    # [CLASS DOC]
    # Tool "page_up": scrolls the browser up one viewport.
    # Inputs: constructor browser (default None but required). No tool inputs.
    # Key attributes: browser.
    # Exceptions: see forward.
    # Side effects: none at construction.
    name = "page_up"
    description = "Scroll the viewport UP one page-length in the current webpage and return the new viewport content."
    inputs = {}
    output_type = "string"

    def __init__(self, browser=None):
        # [FUNCTION DOC]
        # Initializes the Tool base and stores the browser.
        # Inputs: browser (default None).
        # Outputs: none.
        # Exceptions: Tool validation errors propagate.
        # Side effects: none.
        super().__init__()
        self.browser = browser

    def forward(self) -> str:
        # [FUNCTION DOC]
        # Moves one viewport up and returns header, separator and viewport text.
        # Inputs: none.
        # Outputs: str.
        # Exceptions: AttributeError if no browser.
        # Side effects: browser viewport change.
        self.browser.page_up()
        header, content = self.browser._state()
        return header.strip() + "\n=======================\n" + content


class PageDownTool(Tool):
    # [CLASS DOC]
    # Tool "page_down": scrolls the browser down one viewport.
    # Inputs: constructor browser (default None but required). No tool inputs.
    # Key attributes: browser.
    # Exceptions: see forward.
    # Side effects: none at construction.
    name = "page_down"
    description = (
        "Scroll the viewport DOWN one page-length in the current webpage and return the new viewport content."
    )
    inputs = {}
    output_type = "string"

    def __init__(self, browser=None):
        # [FUNCTION DOC]
        # Initializes the Tool base and stores the browser.
        # Inputs: browser (default None).
        # Outputs: none.
        # Exceptions: Tool validation errors propagate.
        # Side effects: none.
        super().__init__()
        self.browser = browser

    def forward(self) -> str:
        # [FUNCTION DOC]
        # Moves one viewport down and returns header, separator and viewport text.
        # Inputs: none.
        # Outputs: str.
        # Exceptions: AttributeError if no browser.
        # Side effects: browser viewport change.
        self.browser.page_down()
        header, content = self.browser._state()
        return header.strip() + "\n=======================\n" + content


class FinderTool(Tool):
    # [CLASS DOC]
    # Tool "find_on_page_ctrl_f": Ctrl+F search on the current page with star wildcards.
    # Inputs: constructor browser (default None but required). Tool input search_string (str).
    # Key attributes: browser.
    # Exceptions: see forward.
    # Side effects: none at construction.
    name = "find_on_page_ctrl_f"
    description = "Scroll the viewport to the first occurrence of the search string. This is equivalent to Ctrl+F."
    inputs = {
        "search_string": {
            "type": "string",
            "description": "The string to search for on the page. This search string supports wildcards like '*'",
        }
    }
    output_type = "string"

    def __init__(self, browser=None):
        # [FUNCTION DOC]
        # Initializes the Tool base and stores the browser.
        # Inputs: browser (default None).
        # Outputs: none.
        # Exceptions: Tool validation errors propagate.
        # Side effects: none.
        super().__init__()
        self.browser = browser

    def forward(self, search_string: str) -> str:
        # [FUNCTION DOC]
        # Searches the page; returns header plus the matching viewport, or header plus a not-found message
        #   naming the string.
        # Inputs: search_string (str).
        # Outputs: str.
        # Exceptions: AttributeError if no browser.
        # Side effects: browser search state and viewport change.
        find_result = self.browser.find_on_page(search_string)
        header, content = self.browser._state()

        if find_result is None:
            return (
                header.strip()
                + f"\n=======================\nThe search string '{search_string}' was not found on this page."
            )
        else:
            return header.strip() + "\n=======================\n" + content


class FindNextTool(Tool):
    # [CLASS DOC]
    # Tool "find_next": jumps to the next match of the last Ctrl+F search.
    # Inputs: constructor browser (default None but required). No tool inputs.
    # Key attributes: browser.
    # Exceptions: see forward.
    # Side effects: none at construction.
    name = "find_next"
    description = "Scroll the viewport to next occurrence of the search string. This is equivalent to finding the next match in a Ctrl+F search."
    inputs = {}
    output_type = "string"

    def __init__(self, browser=None):
        # [FUNCTION DOC]
        # Initializes the Tool base and stores the browser.
        # Inputs: browser (default None).
        # Outputs: none.
        # Exceptions: Tool validation errors propagate.
        # Side effects: none.
        super().__init__()
        self.browser = browser

    def forward(self) -> str:
        # [FUNCTION DOC]
        # Finds the next match; returns header plus the matching viewport, or header plus a generic not-found
        #   message.
        # Inputs: none.
        # Outputs: str.
        # Exceptions: AttributeError if no browser.
        # Side effects: browser search state and viewport change.
        find_result = self.browser.find_next()
        header, content = self.browser._state()

        if find_result is None:
            return header.strip() + "\n=======================\nThe search string was not found on this page."
        else:
            return header.strip() + "\n=======================\n" + content
