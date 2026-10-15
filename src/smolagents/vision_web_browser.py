# [MODULE DOC]
# Description: Example/CLI script that drives a real Chrome browser with helium and a vision-capable
#   CodeAgent. Defines browser tools (search_item_ctrl_f, go_back, close_popups), a screenshot step callback,
#   driver/agent setup and the run_webagent/main entry points, plus prompt constants github_request,
#   search_request and helium_instructions.
# Inputs: depends on helium, PIL, python-dotenv, selenium and smolagents (CodeAgent, WebSearchTool, tool,
#   ActionStep, load_model).
# Outputs: module functions and constants listed above; a module-global driver is created only by
#   run_webagent.
# Exceptions: ImportError at import time if helium/selenium/dotenv are missing.
# Side effects: none at import beyond creating tool objects; running main launches a browser.
# KNOWN ISSUE: github_request is defined but never used.
# KNOWN ISSUE: The tools rely on a global driver that only exists after run_webagent runs, so using them
#   otherwise raises NameError.
import argparse
from io import BytesIO
from time import sleep

import helium
import PIL.Image
from dotenv import load_dotenv
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.common.keys import Keys

from smolagents import CodeAgent, WebSearchTool, tool
from smolagents.agents import ActionStep
from smolagents.cli import load_model


github_request = """
I'm trying to find how hard I have to work to get a repo in github.com/trending.
Can you navigate to the profile for the top author of the top trending repo, and give me their total number of commits over the last year?
"""  # The agent is able to achieve this request only when powered by GPT-4o or Claude-3.5-sonnet.

search_request = """
Please navigate to https://en.wikipedia.org/wiki/Chicago and give me a sentence containing the word "1992" that mentions a construction accident.
"""


def parse_arguments():
    # [FUNCTION DOC]
    # Description: Builds the command-line parser for the web-browser agent script and parses sys.argv.
    # Inputs: none (reads process arguments). Defines optional positional prompt (default: the Wikipedia
    #   search_request text), --model-type (default LiteLLMModel), --model-id (default gpt-4o), --provider,
    #   --api-base and --api-key (all default None).
    # Outputs: argparse.Namespace with attributes prompt, model_type, model_id, provider, api_base, api_key.
    # Exceptions: SystemExit from argparse on invalid arguments or --help.
    # Side effects: reads command-line arguments; argparse may print usage to stdout/stderr.
    parser = argparse.ArgumentParser(description="Run a web browser automation script with a specified model.")
    parser.add_argument(
        "prompt",
        type=str,
        nargs="?",  # Makes it optional
        default=search_request,
        help="The prompt to run with the agent",
    )
    parser.add_argument(
        "--model-type",
        type=str,
        default="LiteLLMModel",
        help="The model type to use (e.g., OpenAIModel, LiteLLMModel, TransformersModel, InferenceClientModel)",
    )
    parser.add_argument(
        "--model-id",
        type=str,
        default="gpt-4o",
        help="The model ID to use for the specified model type",
    )
    parser.add_argument(
        "--provider",
        type=str,
        help="The inference provider to use for the model",
    )
    parser.add_argument(
        "--api-base",
        type=str,
        help="The API base to use for the model",
    )
    parser.add_argument(
        "--api-key",
        type=str,
        help="The API key to use for the model",
    )
    return parser.parse_args()


def save_screenshot(memory_step: ActionStep, agent: CodeAgent) -> None:
    # [FUNCTION DOC]
    # Description: Agent step callback. After waiting one second, takes a PNG screenshot of the current helium
    #   browser, attaches a copy of it as the step's observation images, clears observation images of steps
    #   that are two or more steps older to keep memory lean, and appends a 'Current url: ...' line to the
    #   step observations (or sets it if observations were empty).
    # Inputs: memory_step (ActionStep, the step just completed); agent (CodeAgent whose memory steps are
    #   scanned).
    # Outputs: None.
    # Exceptions: none raised directly.
    # Side effects: sleeps 1 second; mutates memory_step and older ActionSteps in agent memory; prints the
    #   screenshot size to stdout; talks to the browser driver.
    # KNOWN ISSUE: If helium has no driver, the screenshot branch is skipped but the URL update still
    #   dereferences the driver and raises AttributeError.
    sleep(1.0)  # Let JavaScript animations happen before taking the screenshot
    driver = helium.get_driver()
    current_step = memory_step.step_number
    if driver is not None:
        for previous_memory_step in agent.memory.steps:  # Remove previous screenshots from logs for lean processing
            if isinstance(previous_memory_step, ActionStep) and previous_memory_step.step_number <= current_step - 2:
                previous_memory_step.observations_images = None
        png_bytes = driver.get_screenshot_as_png()
        image = PIL.Image.open(BytesIO(png_bytes))
        print(f"Captured a browser screenshot: {image.size} pixels")
        memory_step.observations_images = [image.copy()]  # Create a copy to ensure it persists, important!

    # Update observations with current URL
    url_info = f"Current url: {driver.current_url}"
    memory_step.observations = (
        url_info if memory_step.observations is None else memory_step.observations + "\n" + url_info
    )
    return


def _escape_xpath_string(s: str) -> str:
    # [FUNCTION DOC]
    # Description: HELPER: Turns an arbitrary string into an XPath string literal: wraps it in single quotes
    #   if it has no single quote, otherwise in double quotes if it has no double quote, otherwise builds a
    #   concat expression of single-quoted pieces split on single quotes, joined with a double-quoted single
    #   quote.
    # Inputs: s (str).
    # Outputs: str, an XPath expression evaluating to s.
    # Exceptions: none raised directly.
    # Side effects: none.
    """
    Escapes a string for safe use in an XPath expression.

    Args:
        s (`str`): Arbitrary input string to escape.

    Returns:
        `str`: Valid XPath expression representing the literal value of `s`.
    """
    if "'" not in s:
        return f"'{s}'"
    if '"' not in s:
        return f'"{s}"'
    parts = s.split("'")
    return "concat(" + ', "\'", '.join(f"'{p}'" for p in parts) + ")"


@tool
def search_item_ctrl_f(text: str, nth_result: int = 1) -> str:
    # [FUNCTION DOC]
    # Description: Agent tool (wrapped by the tool decorator) that finds all elements whose text contains the
    #   given text using an escaped XPath query, scrolls the nth match into view, and reports the match count.
    # Inputs: text (str, text to search); nth_result (int, default 1, 1-based occurrence).
    # Outputs: str like Found N matches for '<text>'.Focused on element K of N.
    # Exceptions: Exception if nth_result exceeds the number of matches; NameError if the module-level driver
    #   has not been set yet.
    # Side effects: executes JavaScript in the browser (scrolls the page); uses the global driver.
    # KNOWN ISSUE: nth_result of zero or negative is not rejected and selects elements from the end of the
    #   list (and with no matches it raises IndexError).
    # KNOWN ISSUE: The two sentences in the result string are concatenated without a space.
    """
    Searches for text on the current page via Ctrl + F and jumps to the nth occurrence.
    Args:
        text: The text to search for
        nth_result: Which occurrence to jump to (default: 1)
    """
    escaped_text = _escape_xpath_string(text)
    elements = driver.find_elements(By.XPATH, f"//*[contains(text(), {escaped_text})]")
    if nth_result > len(elements):
        raise Exception(f"Match n°{nth_result} not found (only {len(elements)} matches found)")
    result = f"Found {len(elements)} matches for '{text}'."
    elem = elements[nth_result - 1]
    driver.execute_script("arguments[0].scrollIntoView(true);", elem)
    result += f"Focused on element {nth_result} of {len(elements)}"
    return result


@tool
def go_back() -> None:
    # [FUNCTION DOC]
    # Description: Agent tool that navigates the browser back to the previous page.
    # Inputs: none.
    # Outputs: None.
    # Exceptions: NameError if the global driver is not set; driver errors propagate.
    # Side effects: browser navigation via the global driver.
    """Goes back to previous page."""
    driver.back()


@tool
def close_popups() -> str:
    # [FUNCTION DOC]
    # Description: Agent tool that sends an Escape key press to the browser to dismiss modals/pop-ups.
    # Inputs: none.
    # Outputs: None.
    # Exceptions: NameError if the global driver is not set.
    # Side effects: sends a key event to the browser through the global driver.
    # KNOWN ISSUE: Annotated to return str but always returns None.
    """
    Closes any visible modal or pop-up on the page. Use this to dismiss pop-up windows! This does not work on cookie consent banners.
    """
    webdriver.ActionChains(driver).send_keys(Keys.ESCAPE).perform()


def initialize_driver():
    # [FUNCTION DOC]
    # Description: Starts a visible (non-headless) Chrome browser through helium with fixed window size
    #   1000x1350 at position 0,0, device scale factor 1 and the PDF viewer disabled.
    # Inputs: none.
    # Outputs: the Selenium WebDriver returned by helium.start_chrome.
    # Exceptions: Selenium/helium errors propagate (e.g. Chrome not installed).
    # Side effects: launches a Chrome process.
    """Initialize the Selenium WebDriver."""
    chrome_options = webdriver.ChromeOptions()
    chrome_options.add_argument("--force-device-scale-factor=1")
    chrome_options.add_argument("--window-size=1000,1350")
    chrome_options.add_argument("--disable-pdf-viewer")
    chrome_options.add_argument("--window-position=0,0")
    return helium.start_chrome(headless=False, options=chrome_options)


def initialize_agent(model):
    # [FUNCTION DOC]
    # Description: Creates the CodeAgent used for web browsing with tools WebSearchTool, go_back, close_popups
    #   and search_item_ctrl_f, helium as an authorized import, save_screenshot as step callback, 20 max steps
    #   and verbosity level 2.
    # Inputs: model (a smolagents Model).
    # Outputs: CodeAgent instance.
    # Exceptions: none raised directly.
    # Side effects: none beyond agent construction.
    """Initialize the CodeAgent with the specified model."""
    return CodeAgent(
        tools=[WebSearchTool(), go_back, close_popups, search_item_ctrl_f],
        model=model,
        additional_authorized_imports=["helium"],
        step_callbacks=[save_screenshot],
        max_steps=20,
        verbosity_level=2,
    )


helium_instructions = """
Use your web_search tool when you want to get Google search results.
Then you can use helium to access websites. Don't use helium for Google search, only for navigating websites!
Don't bother about the helium driver, it's already managed.
We've already ran "from helium import *"
Then you can go to pages!
<code>
go_to('github.com/trending')
</code>

You can directly click clickable elements by inputting the text that appears on them.
<code>
click("Top products")
</code>

If it's a link:
<code>
click(Link("Top products"))
</code>

If you try to interact with an element and it's not found, you'll get a LookupError.
In general stop your action after each button click to see what happens on your screenshot.
Never try to login in a page.

To scroll up or down, use scroll_down or scroll_up with as an argument the number of pixels to scroll from.
<code>
scroll_down(num_pixels=1200) # This will scroll one viewport down
</code>

When you have pop-ups with a cross icon to close, don't try to click the close icon by finding its element or targeting an 'X' element (this most often fails).
Just use your built-in tool `close_popups` to close them:
<code>
close_popups()
</code>

You can use .exists() to check for the existence of an element. For example:
<code>
if Text('Accept cookies?').exists():
    click('I accept')
</code>

Proceed in several steps rather than trying to solve the task in one shot.
And at the end, only when you have your answer, return your final answer.
<code>
final_answer("YOUR_ANSWER_HERE")
</code>

If pages seem stuck on loading, you might have to wait, for instance `import time` and run `time.sleep(5.0)`. But don't overuse this!
To list elements on page, DO NOT try code-based element searches like 'contributors = find_all(S("ol > li"))': just look at the latest screenshot you have and read it visually, or use your tool search_item_ctrl_f.
Of course, you can act on buttons like a user would do when navigating.
After each code blob you write, you will be automatically provided with an updated screenshot of the browser and the current browser url.
But beware that the screenshot will only be taken at the end of the whole action, it won't see intermediate states.
Don't kill the browser.
When you have modals or cookie banners on screen, you should get rid of them before you can click anything else.
"""


def run_webagent(
    prompt: str,
    model_type: str,
    model_id: str,
    provider: str | None = None,
    api_base: str | None = None,
    api_key: str | None = None,
) -> None:
    # [FUNCTION DOC]
    # Description: End-to-end run: loads .env variables, builds the model via load_model, starts the browser
    #   (stored in the module-global driver used by the tools), creates the agent, pre-imports helium into the
    #   agent's Python executor and runs the agent on the prompt followed by the helium usage instructions.
    # Inputs: prompt (str); model_type (str, model class name); model_id (str); provider, api_base, api_key
    #   (str or None, default None).
    # Outputs: None (the agent result is discarded).
    # Exceptions: errors from model loading, browser start or agent run propagate.
    # Side effects: reads .env into environment variables; sets the global driver; launches Chrome; makes
    #   model API calls; leaves the browser running.
    # Load environment variables
    load_dotenv()

    # Initialize the model based on the provided arguments
    model = load_model(model_type, model_id, provider=provider, api_base=api_base, api_key=api_key)

    global driver
    driver = initialize_driver()
    agent = initialize_agent(model)

    # Run the agent with the provided prompt
    agent.python_executor("from helium import *")
    agent.run(prompt + helium_instructions)


def main() -> None:
    # [FUNCTION DOC]
    # Description: CLI entry point: parses arguments and calls run_webagent with them.
    # Inputs: none (command-line arguments).
    # Outputs: None.
    # Exceptions: SystemExit from argument parsing; errors from run_webagent propagate.
    # Side effects: see run_webagent.
    # Parse command line arguments
    args = parse_arguments()
    run_webagent(args.prompt, args.model_type, args.model_id, args.provider, args.api_base, args.api_key)


if __name__ == "__main__":
    main()
