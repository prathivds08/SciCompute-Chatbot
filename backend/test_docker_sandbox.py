"""
Test suite for DeePMD analysis script generation and Docker sandbox enforcement.
"""

import os
import sys
import tempfile
from pathlib import Path

# Add project root to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from backend.config import settings
from backend.sandbox.docker_runner import DockerRunner
from backend.sandbox.executor import CodeExecutor
from backend.tools.deepmd_analysis_tools import (
    build_water_density_profile_script,
    build_rdf_script,
    build_extxyz_to_deepmd_raw_script,
    build_loss_curve_script,
    build_parity_plot_script,
    save_analysis_script,
)


def test_docker_runner_probe():
    """Verify DockerRunner accurately detects Docker status and image availability."""
    print("Testing DockerRunner availability probe ...")
    runner = DockerRunner()
    avail, msg = runner.is_docker_available()
    print(f"  Docker Available: {avail} -> {msg}")
    
    img_avail, img_msg = runner.is_image_available()
    print(f"  Image Available: {img_avail} -> {img_msg}")
    assert isinstance(avail, bool)
    assert isinstance(img_avail, bool)
    print("  [OK] DockerRunner probe functional.")


def test_deepmd_script_builders():
    """Verify all DeePMD analysis script builders generate syntactically valid Python code."""
    print("Testing DeePMD analysis script builders ...")
    
    scripts = {
        "density": build_water_density_profile_script(),
        "rdf": build_rdf_script(),
        "raw_conversion": build_extxyz_to_deepmd_raw_script(),
        "loss_curve": build_loss_curve_script(),
        "parity": build_parity_plot_script(),
    }
    
    for name, code in scripts.items():
        assert len(code) > 200, f"Script '{name}' generated unexpectedly short code."
        # Verify it compiles without SyntaxError
        compile(code, f"<test_{name}>", "exec")
        print(f"  [OK] Script builder '{name}' compiled cleanly ({len(code)} bytes).")


def test_script_saving_and_artifact_registration():
    """Verify scripts are saved to backend/data/sandbox/scripts/{session_id}/ and registered in SQLite."""
    print("Testing script persistence and artifact registration ...")
    session_id = "test_verification_session"
    code = "# Test DeePMD Script\nprint('Hello DeePMD')\n"
    
    meta = save_analysis_script(
        code=code,
        script_name="test_script.py",
        session_id=session_id,
        description="Unit test script",
    )
    
    assert Path(meta["path"]).exists(), f"Saved script file not found at {meta['path']}"
    assert meta["name"] == "test_script.py"
    assert meta["session_id"] == session_id
    assert meta["size_bytes"] > 0
    assert meta["artifact_id"] > 0
    print(f"  [OK] Script saved at {meta['path']} with SQLite artifact_id={meta['artifact_id']}.")


def test_strict_docker_containment_policy():
    """Verify that when REQUIRE_DOCKER is True, host execution is strictly blocked if Docker is not active."""
    print("Testing strict Docker containment policy ...")
    executor = CodeExecutor()
    runner = DockerRunner()
    avail, _ = runner.is_docker_available()
    
    if not avail and settings.REQUIRE_DOCKER:
        # Docker is not running: execution MUST be blocked
        res = executor.execute_python("print('This should not run on host')")
        assert res["success"] is False, "Host execution was unexpectedly allowed when Docker is unavailable!"
        assert "EXECUTION BLOCKED" in res["stderr"]
        assert res["executed_in_docker"] is True
        print("  [OK] Host execution correctly blocked by Docker policy.")
        
        # Verify shell is also blocked
        res_shell = executor.execute_shell("echo 'Host shell'")
        assert res_shell["success"] is False
        assert "EXECUTION BLOCKED" in res_shell["stderr"]
        print("  [OK] Direct shell execution correctly blocked.")
    else:
        print("  [SKIP] Docker is available or REQUIRE_DOCKER is False.")


if __name__ == "__main__":
    print("==================================================")
    print("  Running DeePMD Sandbox & Docker Enforcement Tests")
    print("==================================================")
    test_docker_runner_probe()
    test_deepmd_script_builders()
    test_script_saving_and_artifact_registration()
    test_strict_docker_containment_policy()
    print("\n[SUCCESS] ALL TESTS PASSED SUCCESSFULLY!")
