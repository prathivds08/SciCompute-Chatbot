"""
Safe Code Execution Sandbox.

Executes scientific Python (and bash/shell) scripts in isolated run
directories, capturing stdout, stderr, execution duration, return codes,
and generated artifacts (such as Matplotlib PNG plots converted to base64).
"""

import base64
import mimetypes
import os
import subprocess
import sys
import time
import uuid
from pathlib import Path
from typing import Any

from backend.config import settings
from backend.sandbox.docker_runner import DockerRunner


class CodeExecutor:
    """Manages isolated script execution with Docker container enforcement and artifact capture."""

    def __init__(self, sandbox_base_dir: Path | None = None, default_timeout: int | None = None) -> None:
        self.base_dir = sandbox_base_dir or settings.SANDBOX_DIR
        self.base_dir.mkdir(parents=True, exist_ok=True)
        self.default_timeout = default_timeout or settings.SANDBOX_TIMEOUT
        self.max_output_chars = settings.SANDBOX_MAX_OUTPUT_CHARS
        self.docker_runner = DockerRunner(default_timeout=self.default_timeout)

    def execute(
        self,
        code: str,
        language: str = "python",
        timeout: int | None = None,
        session_id: str | None = None,
    ) -> dict[str, Any]:
        """Execute code in the requested language under container isolation rules."""
        lang = language.strip().lower()
        if lang in ("python", "py"):
            return self.execute_python(code, timeout=timeout, session_id=session_id)
        elif lang in ("bash", "sh", "powershell", "shell"):
            return self.execute_shell(code, timeout=timeout, session_id=session_id)
        else:
            return {
                "success": False,
                "stdout": "",
                "stderr": f"Unsupported language '{language}'. Supported languages: python, bash, powershell.",
                "returncode": -1,
                "duration": 0.0,
                "artifacts": [],
            }

    def execute_python(
        self,
        code: str,
        timeout: int | None = None,
        session_id: str | None = None,
    ) -> dict[str, Any]:
        """Execute a Python script inside the isolated Docker container."""
        timeout_val = timeout or self.default_timeout
        run_id = f"run_{int(time.time())}_{uuid.uuid4().hex[:8]}"
        run_dir = self.base_dir / (session_id or "default") / run_id
        run_dir.mkdir(parents=True, exist_ok=True)

        # Wrap script to configure headless matplotlib, path resolution, and auto-save figures
        wrapped_code = self._prepare_python_script(code)
        script_file = run_dir / "script.py"
        script_file.write_text(wrapped_code, encoding="utf-8")

        # Strict Container Execution Policy
        if settings.REQUIRE_DOCKER:
            docker_res = self.docker_runner.run_script_in_container(
                script_file=script_file,
                run_dir=run_dir,
                timeout=timeout_val,
            )
            stdout = docker_res["stdout"]
            stderr = docker_res["stderr"]
            returncode = docker_res["returncode"]
            duration = docker_res["duration"]
            timed_out = docker_res.get("timed_out", False)
        else:
            # Local host fallback (only if REQUIRE_DOCKER is explicitly turned off for testing)
            env = os.environ.copy()
            env["MPLBACKEND"] = "Agg"
            env["PYTHONIOENCODING"] = "utf-8"
            existing_pythonpath = env.get("PYTHONPATH", "")
            paths_to_add = [str(settings.WORKSPACE_ROOT), str(run_dir)]
            env["PYTHONPATH"] = os.pathsep.join(paths_to_add + ([existing_pythonpath] if existing_pythonpath else []))

            start_time = time.perf_counter()
            python_bin = str(settings.CONDA_PYTHON_PATH) if settings.CONDA_PYTHON_PATH.exists() else sys.executable
            try:
                proc = subprocess.run(
                    [python_bin, str(script_file)],
                    cwd=str(run_dir),
                    env=env,
                    capture_output=True,
                    text=True,
                    timeout=timeout_val,
                )
                duration = time.perf_counter() - start_time
                stdout = proc.stdout or ""
                stderr = proc.stderr or ""
                returncode = proc.returncode
                timed_out = False
            except subprocess.TimeoutExpired as exc:
                duration = time.perf_counter() - start_time
                stdout = (exc.stdout or "").decode("utf-8", errors="replace") if isinstance(exc.stdout, bytes) else (exc.stdout or "")
                stderr = f"Execution timed out after {timeout_val} seconds."
                returncode = -1
                timed_out = True
            except Exception as exc:
                duration = time.perf_counter() - start_time
                stdout = ""
                stderr = f"Execution error: {str(exc)}"
                returncode = -1
                timed_out = False

        # Truncate output if necessary
        if len(stdout) > self.max_output_chars:
            stdout = stdout[: self.max_output_chars] + f"\n... [Output truncated after {self.max_output_chars} chars]"
        if len(stderr) > self.max_output_chars:
            stderr = stderr[: self.max_output_chars] + f"\n... [Stderr truncated after {self.max_output_chars} chars]"

        # Collect artifacts generated in run_dir (plots, data files)
        artifacts = self._collect_artifacts(run_dir, ignore_files={"script.py"})

        return {
            "success": returncode == 0 and not timed_out,
            "stdout": stdout,
            "stderr": stderr,
            "returncode": returncode,
            "duration": round(duration, 3),
            "artifacts": artifacts,
            "run_id": run_id,
            "run_dir": str(run_dir),
            "executed_in_docker": settings.REQUIRE_DOCKER,
        }

    def execute_shell(
        self,
        command: str,
        timeout: int | None = None,
        session_id: str | None = None,
    ) -> dict[str, Any]:
        """Execute a shell command. Refused if REQUIRE_DOCKER is True to protect host."""
        if settings.REQUIRE_DOCKER:
            return {
                "success": False,
                "stdout": "",
                "stderr": (
                    "❌ EXECUTION BLOCKED:\n"
                    "Direct host shell execution is disabled by policy. "
                    "The AI can only execute verified Python analysis code inside the isolated Docker container."
                ),
                "returncode": -1,
                "duration": 0.0,
                "artifacts": [],
            }

        timeout_val = timeout or self.default_timeout
        run_id = f"shell_{int(time.time())}_{uuid.uuid4().hex[:8]}"
        run_dir = self.base_dir / (session_id or "default") / run_id
        run_dir.mkdir(parents=True, exist_ok=True)

        start_time = time.perf_counter()
        try:
            # On Windows, run in powershell
            if os.name == "nt":
                cmd = ["powershell", "-NoProfile", "-NonInteractive", "-Command", command]
            else:
                cmd = ["/bin/bash", "-c", command]

            proc = subprocess.run(
                cmd,
                cwd=str(run_dir),
                capture_output=True,
                text=True,
                timeout=timeout_val,
            )
            duration = time.perf_counter() - start_time
            stdout = proc.stdout or ""
            stderr = proc.stderr or ""
            returncode = proc.returncode
            timed_out = False

        except subprocess.TimeoutExpired:
            duration = time.perf_counter() - start_time
            stdout = ""
            stderr = f"Command timed out after {timeout_val} seconds."
            returncode = -1
            timed_out = True
        except Exception as exc:
            duration = time.perf_counter() - start_time
            stdout = ""
            stderr = f"Shell execution error: {str(exc)}"
            returncode = -1
            timed_out = False

        artifacts = self._collect_artifacts(run_dir)

        return {
            "success": returncode == 0 and not timed_out,
            "stdout": stdout,
            "stderr": stderr,
            "returncode": returncode,
            "duration": round(duration, 3),
            "artifacts": artifacts,
            "run_id": run_id,
            "run_dir": str(run_dir),
        }

    def _prepare_python_script(self, raw_code: str) -> str:
        """Inject headless matplotlib initialization, path resolution, and automatic plot saving."""
        header = (
            "# --- SciChat Execution Sandbox Wrapper ---\n"
            "import os\n"
            "import sys\n"
            "from pathlib import Path\n"
            "os.environ['MPLBACKEND'] = 'Agg'\n"
            "try:\n"
            "    import matplotlib\n"
            "    matplotlib.use('Agg')\n"
            "except Exception:\n"
            "    pass\n"
            "# Set DATA_DIR to /data if in container, else local data directory\n"
            "DATA_DIR = Path('/data') if os.path.exists('/data') else Path(r'" + str(settings.backend_dir / "data").replace("\\", "/") + "')\n"
            "# --- User Code ---\n"
        )

        footer = (
            "\n# --- SciChat Auto-Plot Capture ---\n"
            "try:\n"
            "    import matplotlib.pyplot as plt\n"
            "    if plt.get_fignums():\n"
            "        plt.tight_layout()\n"
            "        plt.savefig('figure_output.png', dpi=150, bbox_inches='tight')\n"
            "        plt.close('all')\n"
            "except Exception:\n"
            "    pass\n"
        )

        return header + raw_code + footer

    def _collect_artifacts(self, directory: Path, ignore_files: set[str] | None = None) -> list[dict[str, Any]]:
        """Find and encode generated files such as PNG/JPG plots or data tables."""
        ignore = ignore_files or set()
        artifacts = []

        if not directory.exists():
            return artifacts

        image_extensions = {".png", ".jpg", ".jpeg", ".svg", ".webp"}

        for entry in sorted(directory.iterdir(), key=lambda p: p.name):
            if entry.is_file() and entry.name not in ignore:
                suffix = entry.suffix.lower()
                size_bytes = entry.stat().st_size

                if suffix in image_extensions:
                    try:
                        mime_type, _ = mimetypes.guess_type(str(entry))
                        mime_type = mime_type or "image/png"
                        with open(entry, "rb") as f:
                            b64_data = base64.b64encode(f.read()).decode("utf-8")
                        artifacts.append({
                            "name": entry.name,
                            "type": "image",
                            "mime": mime_type,
                            "data_url": f"data:{mime_type};base64,{b64_data}",
                            "size_bytes": size_bytes,
                        })
                    except Exception as err:
                        print(f"[Sandbox] Failed to encode image {entry.name}: {err}")
                else:
                    # Data / text file
                    preview = ""
                    if suffix in (".extxyz", ".xyz", ".raw", ".npy", ".pb"):
                        preview = f"[Dataset file: {entry.name} ({size_bytes} bytes)]"
                    elif size_bytes < 4096:
                        try:
                            preview = entry.read_text(encoding="utf-8", errors="replace")
                        except Exception:
                            pass
                    artifacts.append({
                        "name": entry.name,
                        "type": "file",
                        "size_bytes": size_bytes,
                        "preview": preview,
                    })

        return artifacts
