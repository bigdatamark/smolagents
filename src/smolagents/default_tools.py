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
# Purpose: Built-in, ready-to-use Tool subclasses shipped with smolagents (python interpreter, final answer,
#   user input, several web search backends, webpage visiting, Wikipedia lookup, speech-to-text).
# Exports (via __all__): ApiWebSearchTool, PythonInterpreterTool, FinalAnswerTool, UserInputTool, WebSearchTool,
#   DuckDuckGoSearchTool, GoogleSearchTool, VisitWebpageTool, WikipediaSearchTool, SpeechToTextTool.
#   Also defines the PreTool dataclass and TOOL_MAPPING, a dict from tool name to class for
#   PythonInterpreterTool, DuckDuckGoSearchTool and VisitWebpageTool.
# Dependencies: smolagents.tools (Tool, PipelineTool), smolagents.local_python_executor (sandboxed evaluator and
#   its constants). Optional third-party packages (ddgs, requests, markdownify, wikipediaapi, transformers) are
#   imported lazily inside the tools that need them.
# Side effects on import: none beyond defining classes and the two module-level containers.
# KNOWN ISSUE: TOOL_MAPPING keys by tool name; DuckDuckGoSearchTool is registered under "web_search", which is
#   the same name GoogleSearchTool, ApiWebSearchTool and WebSearchTool use, so names are not unique across tools.
from dataclasses import dataclass
from typing import Any

from .local_python_executor import (
    BASE_BUILTIN_MODULES,
    BASE_PYTHON_TOOLS,
    MAX_EXECUTION_TIME_SECONDS,
    evaluate_python_code,
)
from .tools import PipelineTool, Tool


@dataclass
class PreTool:
    # [CLASS DOC]
    # HELPER: Plain dataclass describing a tool's metadata (not a Tool itself; not used elsewhere in this module).
    # Fields: name (str), inputs (dict of input name to type string), output_type (a type), task (str),
    #   description (str), repo_id (str, Hub repository id). All fields are required; no defaults.
    # Outputs: dataclass-generated init, repr and equality.
    # Exceptions: TypeError if a field is missing at construction.
    # Side effects: none.
    name: str
    inputs: dict[str, str]
    output_type: type
    task: str
    description: str
    repo_id: str


class PythonInterpreterTool(Tool):
    # [CLASS DOC]
    # Tool named "python_interpreter" that runs a Python code string in smolagents' restricted local evaluator and
    #   returns captured stdout plus the value of the last expression.
    # Constructor args: see __init__ (authorized_imports, timeout_seconds, passthrough args to Tool).
    # Key attributes: authorized_imports (list of module names allowed for import), inputs (rebuilt per instance so
    #   the "code" input description lists the allowed imports), base_python_tools (allowed builtin callables),
    #   python_evaluator (the evaluate function), timeout_seconds. output_type is "string".
    name = "python_interpreter"
    description = "This is a tool that evaluates python code. It can be used to perform calculations."
    inputs = {
        "code": {
            "type": "string",
            "description": "The python code to run in interpreter",
        }
    }
    output_type = "string"

    def __init__(self, *args, authorized_imports=None, timeout_seconds=MAX_EXECUTION_TIME_SECONDS, **kwargs):
        # [FUNCTION DOC]
        # Builds the allowed-import list and the instance-specific input schema, then calls Tool.__init__.
        # Inputs: *args/**kwargs forwarded to Tool.__init__; authorized_imports (iterable of module names or None,
        #   default None meaning only BASE_BUILTIN_MODULES); timeout_seconds (number or None, default
        #   MAX_EXECUTION_TIME_SECONDS; None disables the timeout).
        # Behavior: authorized_imports becomes the de-duplicated union of BASE_BUILTIN_MODULES and the given names
        #   (order is not deterministic since it goes through a set). The "code" input description embeds that list.
        # Outputs: none (initializes the instance).
        # Exceptions: whatever Tool.__init__ raises when validating the tool attributes.
        # Side effects: sets instance attributes; overrides the class-level inputs with an instance-level dict.
        if authorized_imports is None:
            self.authorized_imports = list(set(BASE_BUILTIN_MODULES))
        else:
            self.authorized_imports = list(set(BASE_BUILTIN_MODULES) | set(authorized_imports))
        self.inputs = {
            "code": {
                "type": "string",
                "description": (
                    "The code snippet to evaluate. All variables used in this snippet must be defined in this same snippet, "
                    f"else you will get an error. This code can only import the following python libraries: {self.authorized_imports}."
                ),
            }
        }
        self.base_python_tools = BASE_PYTHON_TOOLS
        self.python_evaluator = evaluate_python_code
        self.timeout_seconds = timeout_seconds
        super().__init__(*args, **kwargs)

    def forward(self, code: str) -> str:
        # [FUNCTION DOC]
        # Evaluates the code with a fresh empty state, the base python tools as static tools, the authorized imports
        #   and the configured timeout.
        # Inputs: code (str), the Python source to run.
        # Outputs: a string of the form "Stdout:" newline, the printed output captured in the state under the
        #   "_print_outputs" key, newline, "Output: " followed by the string form of the evaluator's result value (the
        #   first element of the evaluator's returned tuple; the is-final-answer flag is ignored).
        # Exceptions: propagates evaluator errors (e.g. InterpreterError for forbidden imports or runtime errors,
        #   ExecutionTimeoutError when the timeout is exceeded).
        # Side effects: executes user code (which may print, sleep, etc.) inside the restricted interpreter.
        state = {}
        output = str(
            self.python_evaluator(
                code,
                state=state,
                static_tools=self.base_python_tools,
                authorized_imports=self.authorized_imports,
                timeout_seconds=self.timeout_seconds,
            )[0]  # The second element is boolean is_final_answer
        )
        return f"Stdout:\n{str(state['_print_outputs'])}\nOutput: {output}"


class FinalAnswerTool(Tool):
    # [CLASS DOC]
    # Tool named "final_answer" used by agents to return their final result; it is an identity function.
    # Inputs schema: "answer" of type "any". output_type is "any". No constructor customization.
    name = "final_answer"
    description = "Provides a final answer to the given problem."
    inputs = {"answer": {"type": "any", "description": "The final answer to the problem"}}
    output_type = "any"

    def forward(self, answer: Any) -> Any:
        # [FUNCTION DOC]
        # Returns the given answer unchanged.
        # Inputs: answer (any value).
        # Outputs: the same object.
        # Exceptions: none raised directly.
        # Side effects: none.
        return answer


class UserInputTool(Tool):
    # [CLASS DOC]
    # Tool named "user_input" that asks a human a question on the console and returns the typed reply.
    # Inputs schema: "question" (string). output_type is "string". No constructor customization.
    name = "user_input"
    description = "Asks for user's input on a specific question"
    inputs = {"question": {"type": "string", "description": "The question to ask the user"}}
    output_type = "string"

    def forward(self, question):
        # [FUNCTION DOC]
        # Prompts on stdin with the question followed by " => Type your answer here:" and returns the entered line.
        # Inputs: question (str).
        # Outputs: the user's input string (without trailing newline).
        # Exceptions: EOFError / KeyboardInterrupt propagate from the builtin input function.
        # Side effects: writes to stdout and blocks reading stdin.
        user_input = input(f"{question} => Type your answer here:")
        return user_input


class DuckDuckGoSearchTool(Tool):
    # [CLASS DOC]
    # Tool named "web_search" that queries DuckDuckGo through the third-party ddgs package, with optional
    #   client-side rate limiting.
    # Constructor args: max_results (int, default 10), rate_limit (float queries per second or None, default 1.0),
    #   **kwargs forwarded to the DDGS client constructor (e.g. timeout, proxy).
    # Key attributes: max_results, rate_limit, _min_interval (seconds between requests, 0 if disabled),
    #   _last_request_time (epoch seconds of the previous request), ddgs (the DDGS client).
    """Web search tool that performs searches using the DuckDuckGo search engine.

    Args:
        max_results (`int`, default `10`): Maximum number of search results to return.
        rate_limit (`float`, default `1.0`): Maximum queries per second. Set to `None` to disable rate limiting.
        **kwargs: Additional keyword arguments for the `DDGS` client.

    Examples:
        ```python
        >>> from smolagents import DuckDuckGoSearchTool
        >>> web_search_tool = DuckDuckGoSearchTool(max_results=5, rate_limit=2.0)
        >>> results = web_search_tool("Hugging Face")
        >>> print(results)
        ```
    """

    name = "web_search"
    description = """Performs a duckduckgo web search based on your query (think a Google search) then returns the top search results."""
    inputs = {"query": {"type": "string", "description": "The search query to perform."}}
    output_type = "string"

    def __init__(self, max_results: int = 10, rate_limit: float | None = 1.0, **kwargs):
        # [FUNCTION DOC]
        # Stores settings, computes the minimum interval between requests and creates the DDGS client.
        # Inputs: max_results (int, default 10); rate_limit (float or None, default 1.0; a falsy value disables
        #   limiting); **kwargs passed to the DDGS constructor.
        # Outputs: none.
        # Exceptions: ImportError with installation hint if the ddgs package is missing; ZeroDivisionError is avoided
        #   because a zero rate is treated as disabled.
        # Side effects: imports ddgs and instantiates a client. Note that Tool.__init__ is called before the attributes
        #   are set and without the kwargs.
        super().__init__()
        self.max_results = max_results
        self.rate_limit = rate_limit
        self._min_interval = 1.0 / rate_limit if rate_limit else 0.0
        self._last_request_time = 0.0
        try:
            from ddgs import DDGS
        except ImportError as e:
            raise ImportError(
                "You must install package `ddgs` to run this tool: for instance run `pip install ddgs`."
            ) from e
        self.ddgs = DDGS(**kwargs)

    def forward(self, query: str) -> str:
        # [FUNCTION DOC]
        # Runs a text search and formats the hits as markdown.
        # Inputs: query (str).
        # Outputs: a string starting with "## Search Results" followed by a blank line, then one entry per result made of
        #   a markdown link built from the result title and href, a newline and the result body; entries are separated
        #   by blank lines.
        # Exceptions: generic Exception "No results found! ..." when the search returns nothing; network/ddgs errors
        #   propagate.
        # Side effects: may sleep to honor the rate limit; performs a network request; updates _last_request_time.
        self._enforce_rate_limit()
        results = self.ddgs.text(query, max_results=self.max_results)
        if len(results) == 0:
            raise Exception("No results found! Try a less restrictive/shorter query.")
        postprocessed_results = [f"[{result['title']}]({result['href']})\n{result['body']}" for result in results]
        return "## Search Results\n\n" + "\n\n".join(postprocessed_results)

    def _enforce_rate_limit(self) -> None:
        # [FUNCTION DOC]
        # HELPER: Blocks until at least _min_interval seconds have passed since the previous request.
        # Inputs: none (uses rate_limit, _min_interval, _last_request_time).
        # Outputs: none.
        # Exceptions: none raised directly.
        # Side effects: returns immediately if rate limiting is disabled; otherwise may call time.sleep and then records
        #   the current time as _last_request_time.
        import time

        # No rate limit enforced
        if not self.rate_limit:
            return

        now = time.time()
        elapsed = now - self._last_request_time
        if elapsed < self._min_interval:
            time.sleep(self._min_interval - elapsed)
        self._last_request_time = time.time()


class GoogleSearchTool(Tool):
    # [CLASS DOC]
    # Tool named "web_search" that performs Google searches through a paid proxy API: SerpAPI (default) or Serper.
    # Inputs schema: query (string) and optional nullable filter_year (integer).
    # Constructor args: provider (str, default "serpapi"; any other value selects Serper).
    # Key attributes: provider, organic_key (the JSON key holding organic results: "organic_results" for SerpAPI,
    #   "organic" for Serper), api_key (read from SERPAPI_API_KEY or SERPER_API_KEY).
    name = "web_search"
    description = """Performs a google web search for your query then returns a string of the top search results."""
    inputs = {
        "query": {"type": "string", "description": "The search query to perform."},
        "filter_year": {
            "type": "integer",
            "description": "Optionally restrict results to a certain year",
            "nullable": True,
        },
    }
    output_type = "string"

    def __init__(self, provider: str = "serpapi"):
        # [FUNCTION DOC]
        # Selects the provider configuration and loads the API key from the environment.
        # Inputs: provider (str, default "serpapi").
        # Outputs: none.
        # Exceptions: ValueError if the relevant API key environment variable is not set.
        # Side effects: reads environment variables.
        super().__init__()
        import os

        self.provider = provider
        if provider == "serpapi":
            self.organic_key = "organic_results"
            api_key_env_name = "SERPAPI_API_KEY"
        else:
            self.organic_key = "organic"
            api_key_env_name = "SERPER_API_KEY"
        self.api_key = os.getenv(api_key_env_name)
        if self.api_key is None:
            raise ValueError(f"Missing API key. Make sure you have '{api_key_env_name}' in your env variables.")

    def forward(self, query: str, filter_year: int | None = None) -> str:
        # [FUNCTION DOC]
        # Calls the provider's search endpoint and formats organic results as numbered markdown entries.
        # Inputs: query (str); filter_year (int or None, default None) which, when given, adds a Google "tbs" custom date
        #   range parameter covering January 1 through December 31 of that year.
        # Behavior: SerpAPI requests go to serpapi.com search.json with engine google and google.com domain; Serper
        #   requests go to google.serper.dev/search. Both send the API key as a query parameter via HTTP GET.
        # Outputs: "## Search Results" then a newline, then entries joined by blank lines; each entry is the zero-based
        #   index, a markdown link of title and link, optional "Date published:" and "Source:" lines, then a newline and
        #   the optional snippet. If the organic list exists but is empty, returns a "No results found for ..." message
        #   (mentioning the year filter if used) instead of raising.
        # Exceptions: ValueError carrying the response JSON on any non-200 status; generic Exception when the organic
        #   key is absent from the response (message varies with filter_year); network errors from requests propagate.
        # Side effects: network request.
        # KNOWN ISSUE: Serper's API expects a POST with the key in an X-API-KEY header; sending it as a GET query
        #   parameter likely fails for the Serper provider. The membership check inside the loop is redundant.
        import requests

        if self.provider == "serpapi":
            params = {
                "q": query,
                "api_key": self.api_key,
                "engine": "google",
                "google_domain": "google.com",
            }
            base_url = "https://serpapi.com/search.json"
        else:
            params = {
                "q": query,
                "api_key": self.api_key,
            }
            base_url = "https://google.serper.dev/search"
        if filter_year is not None:
            params["tbs"] = f"cdr:1,cd_min:01/01/{filter_year},cd_max:12/31/{filter_year}"

        response = requests.get(base_url, params=params)

        if response.status_code == 200:
            results = response.json()
        else:
            raise ValueError(response.json())

        if self.organic_key not in results.keys():
            if filter_year is not None:
                raise Exception(
                    f"No results found for query: '{query}' with filtering on year={filter_year}. Use a less restrictive query or do not filter on year."
                )
            else:
                raise Exception(f"No results found for query: '{query}'. Use a less restrictive query.")
        if len(results[self.organic_key]) == 0:
            year_filter_message = f" with filter year={filter_year}" if filter_year is not None else ""
            return f"No results found for '{query}'{year_filter_message}. Try with a more general query, or remove the year filter."

        web_snippets = []
        if self.organic_key in results:
            for idx, page in enumerate(results[self.organic_key]):
                date_published = ""
                if "date" in page:
                    date_published = "\nDate published: " + page["date"]

                source = ""
                if "source" in page:
                    source = "\nSource: " + page["source"]

                snippet = ""
                if "snippet" in page:
                    snippet = "\n" + page["snippet"]

                redacted_version = f"{idx}. [{page['title']}]({page['link']}){date_published}{source}\n{snippet}"
                web_snippets.append(redacted_version)

        return "## Search Results\n" + "\n\n".join(web_snippets)


class ApiWebSearchTool(Tool):
    # [CLASS DOC]
    # Tool named "web_search" that queries a JSON web-search API (Brave Search by default) with client-side rate
    #   limiting and formats the results as markdown.
    # Constructor args: endpoint, api_key, api_key_name, headers, params, rate_limit (see __init__).
    # Key attributes: endpoint, api_key_name, api_key, headers, params, rate_limit, _min_interval,
    #   _last_request_time. Subclasses may override extract_results/format_markdown for other APIs.
    """Web search tool that performs API-based searches.
    By default, it uses the Brave Search API.

    This tool implements a rate limiting mechanism to ensure compliance with API usage policies.
    By default, it limits requests to 1 query per second.

    Args:
        endpoint (`str`): API endpoint URL. Defaults to Brave Search API.
        api_key (`str`): API key for authentication.
        api_key_name (`str`): Environment variable name containing the API key. Defaults to "BRAVE_API_KEY".
        headers (`dict`, *optional*): Headers for API requests.
        params (`dict`, *optional*): Parameters for API requests.
        rate_limit (`float`, default `1.0`): Maximum queries per second. Set to `None` to disable rate limiting.

    Examples:
        ```python
        >>> from smolagents import ApiWebSearchTool
        >>> web_search_tool = ApiWebSearchTool(rate_limit=50.0)
        >>> results = web_search_tool("Hugging Face")
        >>> print(results)
        ```
    """

    name = "web_search"
    description = "Performs a web search for a query and returns a string of the top search results formatted as markdown with titles, URLs, and descriptions."
    inputs = {"query": {"type": "string", "description": "The search query to perform."}}
    output_type = "string"

    def __init__(
        self,
        endpoint: str = "",
        api_key: str = "",
        api_key_name: str = "",
        headers: dict = None,
        params: dict = None,
        rate_limit: float | None = 1.0,
    ):
        # [FUNCTION DOC]
        # Resolves defaults for the endpoint, credentials, headers and parameters.
        # Inputs: endpoint (str, default "" meaning the Brave web search URL); api_key (str, default "" meaning read from
        #   the environment variable named by api_key_name); api_key_name (str, default "" meaning "BRAVE_API_KEY");
        #   headers (dict or None, default a dict with the X-Subscription-Token header set to the api key);
        #   params (dict or None, default count of 10); rate_limit (float queries per second or None, default 1.0).
        # Outputs: none.
        # Exceptions: none raised directly (a missing key is not detected here).
        # Side effects: reads an environment variable.
        # KNOWN ISSUE: if no key is found, the default header value is None, which surfaces only later as a request
        #   failure; empty dicts passed for headers/params are silently replaced by the defaults.
        import os

        super().__init__()
        self.endpoint = endpoint or "https://api.search.brave.com/res/v1/web/search"
        self.api_key_name = api_key_name or "BRAVE_API_KEY"
        self.api_key = api_key or os.getenv(self.api_key_name)
        self.headers = headers or {"X-Subscription-Token": self.api_key}
        self.params = params or {"count": 10}
        self.rate_limit = rate_limit
        self._min_interval = 1.0 / rate_limit if rate_limit else 0.0
        self._last_request_time = 0.0

    def _enforce_rate_limit(self) -> None:
        # [FUNCTION DOC]
        # HELPER: Same algorithm as DuckDuckGoSearchTool's rate limiter: sleeps until _min_interval seconds have passed
        #   since the last request, then records the current time.
        # Inputs: none. Outputs: none. Exceptions: none raised directly.
        # Side effects: may call time.sleep; updates _last_request_time. No-op when rate_limit is falsy.
        import time

        # No rate limit enforced
        if not self.rate_limit:
            return

        now = time.time()
        elapsed = now - self._last_request_time
        if elapsed < self._min_interval:
            time.sleep(self._min_interval - elapsed)
        self._last_request_time = time.time()

    def forward(self, query: str) -> str:
        # [FUNCTION DOC]
        # Performs a rate-limited GET request with the configured headers and params plus the query as "q", then
        #   extracts and formats the results.
        # Inputs: query (str).
        # Outputs: the markdown string produced by format_markdown.
        # Exceptions: requests HTTPError on non-2xx status (via raise_for_status); JSON decode or network errors
        #   propagate; KeyError from extract_results if a result lacks title or url.
        # Side effects: may sleep; network request; updates _last_request_time.
        import requests

        self._enforce_rate_limit()
        params = {**self.params, "q": query}
        response = requests.get(self.endpoint, headers=self.headers, params=params)
        response.raise_for_status()
        data = response.json()
        results = self.extract_results(data)
        return self.format_markdown(results)

    def extract_results(self, data: dict) -> list:
        # [FUNCTION DOC]
        # Extracts results from a Brave-style response: the list under the "web" then "results" keys.
        # Inputs: data (dict, decoded JSON response).
        # Outputs: list of dicts with "title", "url" and "description" (empty string when missing); empty list if the
        #   keys are absent.
        # Exceptions: KeyError if a result lacks "title" or "url".
        # Side effects: none.
        results = []
        for result in data.get("web", {}).get("results", []):
            results.append(
                {"title": result["title"], "url": result["url"], "description": result.get("description", "")}
            )
        return results

    def format_markdown(self, results: list) -> str:
        # [FUNCTION DOC]
        # Formats extracted results as a numbered markdown list.
        # Inputs: results (list of dicts with title, url, description).
        # Outputs: "No results found." for an empty list; otherwise "## Search Results", a blank line, then entries
        #   joined by blank lines, each being a 1-based index, a markdown link of title and url, a newline and the
        #   description.
        # Exceptions: KeyError if an entry is missing a key.
        # Side effects: none.
        if not results:
            return "No results found."
        return "## Search Results\n\n" + "\n\n".join(
            [
                f"{idx}. [{result['title']}]({result['url']})\n{result['description']}"
                for idx, result in enumerate(results, start=1)
            ]
        )


class WebSearchTool(Tool):
    # [CLASS DOC]
    # Dependency-light tool named "web_search" that supports several engines: DuckDuckGo lite HTML scraping, Bing
    #   RSS, or the Exa API.
    # Constructor args: max_results (int, default 10), engine (str, default "duckduckgo"; also "bing" or "exa").
    # Key attributes: max_results, engine.
    name = "web_search"
    description = "Performs a web search for a query and returns a string of the top search results formatted as markdown with titles, links, and descriptions."
    inputs = {"query": {"type": "string", "description": "The search query to perform."}}
    output_type = "string"

    def __init__(self, max_results: int = 10, engine: str = "duckduckgo"):
        # [FUNCTION DOC]
        # Stores the result limit and engine name after initializing the Tool base.
        # Inputs: max_results (int, default 10); engine (str, default "duckduckgo").
        # Outputs: none. Exceptions: none raised directly (engine is validated lazily in search).
        # Side effects: none beyond setting attributes.
        super().__init__()
        self.max_results = max_results
        self.engine = engine

    def forward(self, query: str) -> str:
        # [FUNCTION DOC]
        # Runs search for the query and formats the results.
        # Inputs: query (str).
        # Outputs: markdown string from parse_results.
        # Exceptions: generic Exception "No results found! ..." when the engine returns an empty list; ValueError for an
        #   unsupported engine or missing Exa key; network errors propagate.
        # Side effects: network request(s).
        results = self.search(query)
        if len(results) == 0:
            raise Exception("No results found! Try a less restrictive/shorter query.")
        return self.parse_results(results)

    def search(self, query: str) -> list:
        # [FUNCTION DOC]
        # Dispatches to the engine-specific search method based on the engine attribute.
        # Inputs: query (str).
        # Outputs: list of result dicts with "title", "link" and "description" keys.
        # Exceptions: ValueError "Unsupported engine: ..." for unknown engines; errors from the engine method propagate.
        # Side effects: network request via the chosen engine.
        if self.engine == "duckduckgo":
            return self.search_duckduckgo(query)
        elif self.engine == "bing":
            return self.search_bing(query)
        elif self.engine == "exa":
            return self.search_exa(query)
        else:
            raise ValueError(f"Unsupported engine: {self.engine}")

    def parse_results(self, results: list) -> str:
        # [FUNCTION DOC]
        # Formats result dicts as markdown.
        # Inputs: results (list of dicts with title, link, description).
        # Outputs: "## Search Results", a blank line, then entries joined by blank lines, each a markdown link of title
        #   and link, a newline and the description.
        # Exceptions: KeyError if a key is missing.
        # Side effects: none.
        return "## Search Results\n\n" + "\n\n".join(
            [f"[{result['title']}]({result['link']})\n{result['description']}" for result in results]
        )

    def search_duckduckgo(self, query: str) -> list:
        # [FUNCTION DOC]
        # Fetches the DuckDuckGo lite HTML page for the query (GET with a basic Mozilla user agent) and parses results
        #   with the helper HTML parser.
        # Inputs: query (str).
        # Outputs: list of result dicts (title, link, description) as gathered by the parser.
        # Exceptions: requests HTTPError on a bad status; network errors propagate.
        # Side effects: network request.
        # KNOWN ISSUE: max_results is not applied for this engine, unlike bing and exa; no request timeout is set.
        import requests

        response = requests.get(
            "https://lite.duckduckgo.com/lite/",
            params={"q": query},
            headers={"User-Agent": "Mozilla/5.0"},
        )
        response.raise_for_status()
        parser = self._create_duckduckgo_parser()
        parser.feed(response.text)
        return parser.results

    def _create_duckduckgo_parser(self):
        # [FUNCTION DOC]
        # HELPER: Builds and returns a fresh HTMLParser subclass instance that extracts results from DuckDuckGo lite
        #   markup.
        # Inputs: none.
        # Outputs: a SimpleResultParser instance (its results attribute fills as HTML is fed).
        # Exceptions: none raised directly.
        # Side effects: defines a local class on each call.
        from html.parser import HTMLParser

        class SimpleResultParser(HTMLParser):
            # [CLASS DOC]
            # HELPER: Stateful HTML parser. Text inside an anchor with class "result-link" becomes the title, text inside a
            #   table cell with class "result-snippet" is accumulated as description fragments, and text inside a span with
            #   class "link-text" becomes the link (prefixed with "https://"). At each closing table row, if the current
            #   record has title, description and link, the fragments are joined with spaces and the record is appended to
            #   results.
            # Key attributes: results (list of dicts), current (dict being built), capture_title / capture_description /
            #   capture_link (booleans marking which element is open).
            def __init__(self):
                # [FUNCTION DOC]
                # HELPER: Initializes the base HTMLParser and resets results, current record and capture flags.
                # Inputs: none. Outputs: none. Exceptions: none raised directly. Side effects: sets attributes.
                super().__init__()
                self.results = []
                self.current = {}
                self.capture_title = False
                self.capture_description = False
                self.capture_link = False

            def handle_starttag(self, tag, attrs):
                # [FUNCTION DOC]
                # HELPER: Turns on the matching capture flag when a result-link anchor, result-snippet cell or link-text span
                #   opens (class attribute must equal the name exactly).
                # Inputs: tag (str), attrs (list of name/value pairs). Outputs: none. Exceptions: none raised directly.
                # Side effects: mutates capture flags.
                attrs = dict(attrs)
                if tag == "a" and attrs.get("class") == "result-link":
                    self.capture_title = True
                elif tag == "td" and attrs.get("class") == "result-snippet":
                    self.capture_description = True
                elif tag == "span" and attrs.get("class") == "link-text":
                    self.capture_link = True

            def handle_endtag(self, tag):
                # [FUNCTION DOC]
                # HELPER: Turns off the matching capture flag when the captured element closes; on a closing table row,
                #   finalizes and stores the current record if it is complete, then starts a new one.
                # Inputs: tag (str). Outputs: none. Exceptions: none raised directly.
                # Side effects: mutates flags, current and results.
                # KNOWN ISSUE: incomplete records are not reset at a row end, so partial data may leak into the next result.
                if tag == "a" and self.capture_title:
                    self.capture_title = False
                elif tag == "td" and self.capture_description:
                    self.capture_description = False
                elif tag == "span" and self.capture_link:
                    self.capture_link = False
                elif tag == "tr":
                    # Store current result if all parts are present
                    if {"title", "description", "link"} <= self.current.keys():
                        self.current["description"] = " ".join(self.current["description"])
                        self.results.append(self.current)
                        self.current = {}

            def handle_data(self, data):
                # [FUNCTION DOC]
                # HELPER: Stores stripped text into the field selected by the active capture flag (title overwrite,
                #   description append, link overwrite with "https://" prefix).
                # Inputs: data (str). Outputs: none. Exceptions: none raised directly. Side effects: mutates current.
                if self.capture_title:
                    self.current["title"] = data.strip()
                elif self.capture_description:
                    self.current.setdefault("description", [])
                    self.current["description"].append(data.strip())
                elif self.capture_link:
                    self.current["link"] = "https://" + data.strip()

        return SimpleResultParser()

    def search_bing(self, query: str) -> list:
        # [FUNCTION DOC]
        # Fetches Bing search results as RSS and parses the items.
        # Inputs: query (str).
        # Outputs: up to max_results dicts with "title", "link" and "description" taken from each RSS item (values may be
        #   None when a child element is missing).
        # Exceptions: requests HTTPError on a bad status; xml ParseError if the response is not valid XML.
        # Side effects: network request (no timeout set).
        import xml.etree.ElementTree as ET

        import requests

        response = requests.get(
            "https://www.bing.com/search",
            params={"q": query, "format": "rss"},
        )
        response.raise_for_status()
        root = ET.fromstring(response.text)
        items = root.findall(".//item")
        results = [
            {
                "title": item.findtext("title"),
                "link": item.findtext("link"),
                "description": item.findtext("description"),
            }
            for item in items[: self.max_results]
        ]
        return results

    def search_exa(self, query: str) -> list:
        # [FUNCTION DOC]
        # Queries the Exa search API with highlights enabled.
        # Inputs: query (str). Requires EXA_API_KEY in the environment.
        # Behavior: POSTs JSON with the query, numResults equal to max_results and contents requesting highlights,
        #   with headers x-api-key, JSON content type and x-exa-integration set to "smolagents"; timeout is the
        #   instance's timeout attribute if present, else 30 seconds.
        # Outputs: list of dicts: title (empty string if missing), link (the result url), description (highlights joined
        #   by spaces, empty when highlights are missing or null).
        # Exceptions: ValueError if EXA_API_KEY is missing/empty; requests HTTPError on bad status; KeyError if a result
        #   lacks "url".
        # Side effects: reads an environment variable; network request.
        """Search using the Exa API. Requires an EXA_API_KEY environment variable."""
        import os

        import requests

        api_key = os.getenv("EXA_API_KEY")
        if not api_key:
            raise ValueError("Missing API key. Make sure you have 'EXA_API_KEY' in your env variables.")

        response = requests.post(
            "https://api.exa.ai/search",
            headers={
                "x-api-key": api_key,
                "Content-Type": "application/json",
                "x-exa-integration": "smolagents",
            },
            json={
                "query": query,
                "numResults": self.max_results,
                "contents": {"highlights": True},
            },
            timeout=getattr(self, "timeout", 30),
        )
        response.raise_for_status()
        data = response.json()
        return [
            {
                "title": result.get("title", ""),
                "link": result["url"],
                "description": " ".join(result.get("highlights") or []),
            }
            for result in data.get("results", [])
        ]


class VisitWebpageTool(Tool):
    # [CLASS DOC]
    # Tool named "visit_webpage" that downloads a URL and returns its content converted to markdown, truncated to a
    #   maximum length.
    # Constructor args: max_output_length (int, default 40000).
    # Key attributes: max_output_length.
    name = "visit_webpage"
    description = (
        "Visits a webpage at the given url and reads its content as a markdown string. Use this to browse webpages."
    )
    inputs = {
        "url": {
            "type": "string",
            "description": "The url of the webpage to visit.",
        }
    }
    output_type = "string"

    def __init__(self, max_output_length: int = 40000):
        # [FUNCTION DOC]
        # Initializes the Tool base and stores the maximum output length.
        # Inputs: max_output_length (int, default 40000). Outputs: none. Exceptions: none raised directly.
        # Side effects: none beyond setting attributes.
        super().__init__()
        self.max_output_length = max_output_length

    def _truncate_content(self, content: str, max_length: int) -> str:
        # [FUNCTION DOC]
        # HELPER: Truncates text to a maximum number of characters, keeping the beginning.
        # Inputs: content (str), max_length (int).
        # Outputs: content unchanged if short enough; otherwise its first max_length characters followed by a notice line
        #   saying the content was truncated to stay below max_length characters.
        # Exceptions: none raised directly. Side effects: none.
        if len(content) <= max_length:
            return content
        return (
            content[:max_length] + f"\n..._This content has been truncated to stay below {max_length} characters_...\n"
        )

    def forward(self, url: str) -> str:
        # [FUNCTION DOC]
        # Fetches the URL (20 second timeout), converts HTML to markdown with markdownify, collapses runs of three or more
        #   newlines into two, and truncates to max_output_length.
        # Inputs: url (str).
        # Outputs: the markdown string, or an error message string: a timeout message, "Error fetching the webpage: ..."
        #   for request errors (including bad HTTP status), or "An unexpected error occurred: ..." for anything else.
        # Exceptions: ImportError if markdownify or requests is not installed; all other errors are converted to strings.
        # Side effects: network request.
        try:
            import re

            import requests
            from markdownify import markdownify
            from requests.exceptions import RequestException
        except ImportError as e:
            raise ImportError(
                "You must install packages `markdownify` and `requests` to run this tool: for instance run `pip install markdownify requests`."
            ) from e
        try:
            # Send a GET request to the URL with a 20-second timeout
            response = requests.get(url, timeout=20)
            response.raise_for_status()  # Raise an exception for bad status codes

            # Convert the HTML content to Markdown
            markdown_content = markdownify(response.text).strip()

            # Remove multiple line breaks
            markdown_content = re.sub(r"\n{3,}", "\n\n", markdown_content)

            return self._truncate_content(markdown_content, self.max_output_length)

        except requests.exceptions.Timeout:
            return "The request timed out. Please try again later or check the URL."
        except RequestException as e:
            return f"Error fetching the webpage: {str(e)}"
        except Exception as e:
            return f"An unexpected error occurred: {str(e)}"


class WikipediaSearchTool(Tool):
    # [CLASS DOC]
    # Tool named "wikipedia_search" that fetches a Wikipedia page through the wikipedia-api package and returns its
    #   summary or full text with its URL.
    # Constructor args: user_agent, language, content_type, extract_format (see __init__).
    # Key attributes: user_agent, language, content_type ("summary" or "text"), extract_format (a wikipediaapi
    #   ExtractFormat value), wiki (the wikipediaapi client).
    """
    Search Wikipedia and return the summary or full text of the requested article, along with the page URL.

    Attributes:
        user_agent (`str`): Custom user-agent string to identify the project. This is required as per Wikipedia API policies.
            See: https://foundation.wikimedia.org/wiki/Policy:Wikimedia_Foundation_User-Agent_Policy
        language (`str`, default `"en"`): Language in which to retrieve Wikipedia article.
            See: http://meta.wikimedia.org/wiki/List_of_Wikipedias
        content_type (`Literal["summary", "text"]`, default `"text"`): Type of content to fetch. Can be "summary" for a short summary or "text" for the full article.
        extract_format (`Literal["HTML", "WIKI"]`, default `"WIKI"`): Extraction format of the output. Can be `"WIKI"` or `"HTML"`.

    Example:
        ```python
        >>> from smolagents import CodeAgent, InferenceClientModel, WikipediaSearchTool
        >>> agent = CodeAgent(
        >>>     tools=[
        >>>            WikipediaSearchTool(
        >>>                user_agent="MyResearchBot (myemail@example.com)",
        >>>                language="en",
        >>>                content_type="summary",  # or "text"
        >>>                extract_format="WIKI",
        >>>            )
        >>>        ],
        >>>     model=InferenceClientModel(),
        >>> )
        >>> agent.run("Python_(programming_language)")
        ```
    """

    name = "wikipedia_search"
    description = "Searches Wikipedia and returns a summary or full text of the given topic, along with the page URL."
    inputs = {
        "query": {
            "type": "string",
            "description": "The topic to search on Wikipedia.",
        }
    }
    output_type = "string"

    def __init__(
        self,
        user_agent: str = "Smolagents (myemail@example.com)",
        language: str = "en",
        content_type: str = "text",
        extract_format: str = "WIKI",
    ):
        # [FUNCTION DOC]
        # Validates options and creates the Wikipedia client.
        # Inputs: user_agent (str, default "Smolagents (myemail@example.com)"; must be non-empty); language (str, default
        #   "en"); content_type (str, default "text"; not validated here); extract_format (str "WIKI" or "HTML",
        #   default "WIKI").
        # Outputs: none.
        # Exceptions: ImportError if wikipedia-api is missing; ValueError for an empty user agent or an invalid
        #   extract_format.
        # Side effects: imports wikipediaapi and builds a client (no network call yet).
        super().__init__()
        try:
            import wikipediaapi
        except ImportError as e:
            raise ImportError(
                "You must install `wikipedia-api` to run this tool: for instance run `pip install wikipedia-api`"
            ) from e
        if not user_agent:
            raise ValueError("User-agent is required. Provide a meaningful identifier for your project.")

        self.user_agent = user_agent
        self.language = language
        self.content_type = content_type

        # Map string format to wikipediaapi.ExtractFormat
        extract_format_map = {
            "WIKI": wikipediaapi.ExtractFormat.WIKI,
            "HTML": wikipediaapi.ExtractFormat.HTML,
        }

        if extract_format not in extract_format_map:
            raise ValueError("Invalid extract_format. Choose between 'WIKI' or 'HTML'.")

        self.extract_format = extract_format_map[extract_format]

        self.wiki = wikipediaapi.Wikipedia(
            user_agent=self.user_agent, language=self.language, extract_format=self.extract_format
        )

    def forward(self, query: str) -> str:
        # [FUNCTION DOC]
        # Looks up the page whose title equals the query.
        # Inputs: query (str, page title).
        # Outputs: a formatted string with the page title, the summary or full text depending on content_type, and the
        #   page URL; "No Wikipedia page found for ..." if the page does not exist; a warning string if content_type is
        #   invalid; "Error fetching Wikipedia summary: ..." for any exception.
        # Exceptions: none propagate (all are converted to strings).
        # Side effects: network requests to the Wikipedia API.
        try:
            page = self.wiki.page(query)

            if not page.exists():
                return f"No Wikipedia page found for '{query}'. Try a different query."

            title = page.title
            url = page.fullurl

            if self.content_type == "summary":
                text = page.summary
            elif self.content_type == "text":
                text = page.text
            else:
                return "⚠️ Invalid `content_type`. Use either 'summary' or 'text'."

            return f"✅ **Wikipedia Page:** {title}\n\n**Content:** {text}\n\n🔗 **Read more:** {url}"

        except Exception as e:
            return f"Error fetching Wikipedia summary: {str(e)}"


class SpeechToTextTool(PipelineTool):
    # [CLASS DOC]
    # PipelineTool named "transcriber" that transcribes audio to text using a Whisper model
    #   (default checkpoint openai/whisper-large-v3-turbo).
    # Inputs schema: "audio" (audio type: path, URL or tensor). output_type is "string".
    # Constructor args: those of PipelineTool (e.g. model, device). Class attributes pre_processor_class and
    #   model_class are set lazily in __new__ to the Whisper processor and conditional-generation model classes.
    default_checkpoint = "openai/whisper-large-v3-turbo"
    description = "This is a tool that transcribes an audio into text. It returns the transcribed text."
    name = "transcriber"
    inputs = {
        "audio": {
            "type": "audio",
            "description": "The audio to transcribe. Can be a local path, an url, or a tensor.",
        }
    }
    output_type = "string"

    def __new__(cls, *args, **kwargs):
        # [FUNCTION DOC]
        # Imports the Whisper classes from transformers and assigns them to the class before instance creation.
        # Inputs: cls; *args and **kwargs (ignored here; consumed later by __init__).
        # Outputs: a new, uninitialized instance.
        # Exceptions: ImportError if transformers (with Whisper) is not installed.
        # Side effects: mutates class attributes pre_processor_class and model_class (shared by all instances).
        from transformers.models.whisper import WhisperForConditionalGeneration, WhisperProcessor

        cls.pre_processor_class = WhisperProcessor
        cls.model_class = WhisperForConditionalGeneration
        return super().__new__(cls)

    def encode(self, audio):
        # [FUNCTION DOC]
        # Converts the audio input to a raw waveform via AgentAudio and runs the Whisper processor on it.
        # Inputs: audio (path, URL, tensor, or AgentAudio-compatible value).
        # Outputs: processor output (PyTorch tensors including "input_features").
        # Exceptions: propagates AgentAudio or processor errors.
        # Side effects: may read a file or download audio.
        # KNOWN ISSUE: no sampling rate is passed to the processor, so audio at a rate other than the processor default
        #   may be misinterpreted.
        from .agent_types import AgentAudio

        audio = AgentAudio(audio).to_raw()
        return self.pre_processor(audio, return_tensors="pt")

    def forward(self, inputs):
        # [FUNCTION DOC]
        # Runs Whisper generation on the encoded input features.
        # Inputs: inputs (mapping with "input_features").
        # Outputs: generated token id tensor.
        # Exceptions: propagates model errors. Side effects: model computation on the configured device.
        return self.model.generate(inputs["input_features"])

    def decode(self, outputs):
        # [FUNCTION DOC]
        # Decodes generated token ids into text, skipping special tokens.
        # Inputs: outputs (token id tensor).
        # Outputs: the first decoded string.
        # Exceptions: IndexError if the batch is empty. Side effects: none.
        return self.pre_processor.batch_decode(outputs, skip_special_tokens=True)[0]


TOOL_MAPPING = {
    tool_class.name: tool_class
    for tool_class in [
        PythonInterpreterTool,
        DuckDuckGoSearchTool,
        VisitWebpageTool,
    ]
}

__all__ = [
    "ApiWebSearchTool",
    "PythonInterpreterTool",
    "FinalAnswerTool",
    "UserInputTool",
    "WebSearchTool",
    "DuckDuckGoSearchTool",
    "GoogleSearchTool",
    "VisitWebpageTool",
    "WikipediaSearchTool",
    "SpeechToTextTool",
]
