# [MODULE DOC]
# HELPER: Notebook-generation config for the Spanish docs build (consumed by the HF doc-builder tooling).
# Inputs: none; no imports.
# Outputs: module globals INSTALL_CONTENT (pip-install snippet for smolagents, English text, with a commented
#   install-from-source alternative), notebook_first_cells (single code cell with INSTALL_CONTENT prepended to
#   every generated notebook) and black_avoid_patterns (placeholder-to-fake-name map used during code
#   formatting).
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
