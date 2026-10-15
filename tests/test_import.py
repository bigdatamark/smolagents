# [MODULE DOC]
# Description: Checks smolagents can be imported in a fresh virtual environment with no optional extras
#   installed.
# Inputs: depends on os, subprocess, tempfile and the uv CLI.
# Outputs: test_import_smolagents_without_extras.
# Exceptions: none raised directly.
# Side effects: none at import.
import os
import subprocess
import tempfile


def test_import_smolagents_without_extras(monkeypatch):
    # [FUNCTION DOC]
    # Description: Unsets VIRTUAL_ENV, creates a uv venv in a temp dir, installs the local package (current
    #   directory) into it, runs import smolagents with that interpreter and asserts exit code 0, showing the
    #   last stderr line on failure.
    # Inputs: monkeypatch (pytest fixture).
    # Outputs: none (passes or fails via assertions).
    # Exceptions: AssertionError on import failure; CalledProcessError if venv creation or install fails;
    #   FileNotFoundError if uv is missing.
    # Side effects: spawns processes, may download packages over the network, creates and deletes a temp dir;
    #   must be run from the repo root.
    monkeypatch.delenv("VIRTUAL_ENV", raising=False)
    with tempfile.TemporaryDirectory() as temp_dir:
        # Create a virtual environment
        venv_dir = os.path.join(temp_dir, "venv")
        subprocess.run(["uv", "venv", venv_dir], check=True)

        # Install smolagents in the virtual environment
        subprocess.run(
            ["uv", "pip", "install", "--python", os.path.join(venv_dir, "bin", "python"), "smolagents @ ."], check=True
        )

        # Run the import test in the virtual environment
        result = subprocess.run(
            [os.path.join(venv_dir, "bin", "python"), "-c", "import smolagents"],
            capture_output=True,
            text=True,
        )

    # Check if the import was successful
    assert result.returncode == 0, (
        "Import failed with error: "
        + (result.stderr.splitlines()[-1] if result.stderr else "No error message")
        + "\n"
        + result.stderr
    )
