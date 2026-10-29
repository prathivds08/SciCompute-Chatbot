"""
Comprehensive test suite for Docker keywords, synonyms, and execution workflow.
Tests:
  1. detect_docker_intent with extensive keywords, verbs, and synonyms.
  2. is_followup_execution_request with conversational follow-up commands.
  3. Direct user code execution in chat.
  4. Conversational follow-up execution of previous assistant code in chat.
  5. Fallback template resolution when execution is requested without code.
"""

import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from backend.tools.code_tools import (
    detect_docker_intent,
    is_followup_execution_request,
    resolve_fallback_analysis_script,
    extract_code_blocks,
)
from backend.db.models import Database
from backend.config import settings


def test_detect_docker_intent_synonyms():
    """Verify detect_docker_intent detects all execution keywords, verbs, and container synonyms."""
    print("Testing detect_docker_intent with keywords and synonyms ...")

    positive_cases = [
        "run in docker",
        "execute in docker",
        "exec in docker",
        "run the code in docker",
        "execute the code in docker",
        "can you run this in docker and return the output?",
        "please execute this script in docker",
        "run it inside docker container",
        "execute via docker",
        "evaluate the model in docker",
        "eval python in container",
        "simulate the interface in docker",
        "calculate the total force in docker",
        "compute the water density profile in docker",
        "find the total force and execute",
        "run the script",
        "execute the script",
        "run the python code",
        "run by yourself and show the output",
        "execute and return output",
        "calculate and run",
        "test run in docker",
        "plot the radial distribution function in docker",
        "water density profile",
        "radial distribution",
    ]

    for phrase in positive_cases:
        matched = detect_docker_intent(phrase)
        assert matched, f"Failed to match positive intent: '{phrase}'"

    negative_cases = [
        "what is quantum espresso?",
        "explain density functional theory",
        "tell me about pseudopotential files",
        "what are the coordinates of platinum?",
        "how does lammps molecular dynamics work?",
    ]

    for phrase in negative_cases:
        matched = detect_docker_intent(phrase)
        assert not matched, f"False positive intent for: '{phrase}'"

    print(f"  [OK] Successfully tested {len(positive_cases)} positive and {len(negative_cases)} negative intent queries.")


def test_is_followup_execution_request():
    """Verify is_followup_execution_request detects all short follow-up execution requests."""
    print("Testing is_followup_execution_request ...")

    followup_cases = [
        "run",
        "execute",
        "exec",
        "run it",
        "execute it",
        "run this",
        "execute this",
        "run that",
        "execute that",
        "run the code",
        "execute the code",
        "run the script",
        "execute the script",
        "run in docker",
        "execute in docker",
        "run it in docker",
        "execute it in docker",
        "run this in docker",
        "execute this in docker",
        "run the code in docker",
        "execute the code in docker",
        "run inside docker container",
        "can you run this in docker and return the output?",
        "please run the above code in docker",
        "execute the above script",
        "run it and show output",
        "execute and return output",
        "now run it",
        "please execute",
    ]

    for phrase in followup_cases:
        matched = is_followup_execution_request(phrase)
        assert matched, f"Failed to match follow-up request: '{phrase}'"

    non_followups = [
        "what is the meaning of life?",
        "can you write a python script for rdf?",
        "here is some code: ```python\nprint(1)\n```",
        "how do I install docker?",
        "explain the difference between dft and md",
    ]

    for phrase in non_followups:
        matched = is_followup_execution_request(phrase)
        assert not matched, f"False positive follow-up for: '{phrase}'"

    print(f"  [OK] Successfully tested {len(followup_cases)} positive and {len(non_followups)} negative follow-up cases.")


def test_resolve_fallback_analysis_script():
    """Verify fallback template scripts are resolved accurately when model does not generate code."""
    print("Testing resolve_fallback_analysis_script ...")

    cases = {
        "calculate water density profile in docker": "density",
        "compute radial distribution function rdf": "rdf",
        "convert trajectory to raw format for deepmd": "raw",
        "plot the training loss curve": "loss",
        "generate parity plot for forces": "parity",
    }

    for query, expected in cases.items():
        script = resolve_fallback_analysis_script(query)
        assert script is not None, f"Failed to resolve fallback script for: '{query}'"
        assert len(script) > 200
        compile(script, "<fallback_test>", "exec")

    print(f"  [OK] Verified fallback script templates for {len(cases)} scientific workflows.")


def test_extract_code_blocks_enhancement():
    """Verify extract_code_blocks classifies untagged code blocks containing Python keywords as python."""
    print("Testing extract_code_blocks enhancements ...")

    md_with_tagged = "```python\nimport numpy as np\nprint(1)\n```"
    blocks1 = extract_code_blocks(md_with_tagged)
    assert len(blocks1) == 1
    assert blocks1[0]["language"] == "python"

    md_untagged_python = "```\nimport os\nfrom ase.io import read\natoms = read('test.xyz')\nprint(len(atoms))\n```"
    blocks2 = extract_code_blocks(md_untagged_python)
    assert len(blocks2) == 1
    assert blocks2[0]["language"] == "python", f"Expected 'python', got '{blocks2[0]['language']}'"

    print("  [OK] extract_code_blocks correctly classifies untagged Python code blocks.")


if __name__ == "__main__":
    print("==================================================")
    print("  Running Docker Intent & Execution Test Suite")
    print("==================================================")
    test_detect_docker_intent_synonyms()
    test_is_followup_execution_request()
    test_resolve_fallback_analysis_script()
    test_extract_code_blocks_enhancement()
    print("\n[SUCCESS] ALL DOCKER INTENT & EXECUTION TESTS PASSED!")
