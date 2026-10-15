# [MODULE DOC]
# HELPER: Notebook-generation config for the English docs build (consumed by the HF doc-builder tooling).
# Inputs: none; no imports.
# Outputs: module globals INSTALL_CONTENT (markdown/shell snippet that pip-installs smolagents, with a
#   commented alternative to install from GitHub source), notebook_first_cells (list with one code cell
#   holding INSTALL_CONTENT, prepended to every generated notebook) and black_avoid_patterns
#   (placeholder-to-fake-name map so code formatting does not choke on template placeholders).
# Exceptions: none raised directly.
# Side effects: none.
# docstyle-ignore
INSTALL_CONTENT = """
# Installation
! pip install smolagents
# To install from source instead of the last release, comment the command above and uncomment the following one.
# ! pip install git+https://github.com/huggingface/smolagents.git
"""

notebook_first_cells = [{"type": "code", "content": INSTALL_CONTENT}]
black_avoid_patterns = {
    "{processor_class}": "FakeProcessorClass",
    "{model_class}": "FakeModelClass",
    "{object_class}": "FakeObjectClass",
}
