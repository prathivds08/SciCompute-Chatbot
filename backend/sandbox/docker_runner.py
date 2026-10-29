"""
Docker Container Execution Runner for DeePMD & Scientific Workflows.

Executes scientific Python scripts strictly inside an isolated Docker container
with network isolation, resource limits, and read-only dataset volume mounts.
"""

import os
import shutil
import subprocess
import time
import uuid
from pathlib import Path
from typing import Any, Tuple

from backend.config import settings


class DockerRunner:
    """Manages isolated containerized execution of scientific scripts."""

    def __init__(
        self,
        image_name: str | None = None,
        default_timeout: int | None = None,
        memory_limit: str | None = None,
        cpu_limit: str | None = None,
    ) -> None:
        self.image_name = image_name or settings.DOCKER_IMAGE_NAME
        self.default_timeout = default_timeout or settings.DOCKER_CONTAINER_TIMEOUT
        self.memory_limit = memory_limit or settings.DOCKER_MEMORY_LIMIT
        self.cpu_limit = cpu_limit or settings.DOCKER_CPU_LIMIT
        self.data_dir = settings.backend_dir / "data"

    def _find_docker_bin(self) -> str | None:
        """Find Docker executable path with fallback to standard Docker Desktop locations."""
        docker_bin = shutil.which("docker")
        if docker_bin:
            return docker_bin
        
        # Check standard installation locations on Windows
        candidate_paths = [
            Path(r"E:\Docker\Docker\resources\bin\docker.exe"),
            Path(os.environ.get("LOCALAPPDATA", "")) / "Programs" / "DockerDesktop" / "resources" / "bin" / "docker.exe",
            Path(r"C:\Program Files\Docker\Docker\resources\bin\docker.exe"),
            Path(os.environ.get("PROGRAMFILES", r"C:\Program Files")) / "Docker" / "Docker" / "resources" / "bin" / "docker.exe",
        ]
        for candidate in candidate_paths:
            try:
                if candidate.is_file():
                    return str(candidate)
            except Exception:
                pass
        return None

    def is_docker_available(self) -> Tuple[bool, str]:
        """Check if Docker CLI exists and the Docker daemon is responding."""
        docker_bin = self._find_docker_bin()
        if not docker_bin:
            return False, "Docker CLI not found on PATH or standard locations. Please install Docker Desktop or enable Docker."

        try:
            res = subprocess.run(
                [docker_bin, "info", "--format", "{{.ServerVersion}}"],
                capture_output=True,
                text=True,
                timeout=5,
            )
            if res.returncode != 0:
                err_msg = res.stderr.strip() or "Docker daemon is not running."
                return False, f"Docker daemon is not reachable: {err_msg}"
            
            server_version = res.stdout.strip()
            return True, f"Docker is running (Server version: {server_version})"
        except subprocess.TimeoutExpired:
            return False, "Docker daemon connection timed out."
        except Exception as exc:
            return False, f"Failed to probe Docker: {str(exc)}"

    def is_image_available(self) -> Tuple[bool, str]:
        """Check if the target sandbox image is built and available locally."""
        is_avail, msg = self.is_docker_available()
        if not is_avail:
            return False, msg

        docker_bin = self._find_docker_bin() or "docker"
        try:
            res = subprocess.run(
                [docker_bin, "image", "inspect", self.image_name],
                capture_output=True,
                text=True,
                timeout=5,
            )
            if res.returncode == 0:
                return True, f"Image '{self.image_name}' is available."
            return False, f"Image '{self.image_name}' not found locally. Please build it using docker/build_container.ps1."
        except Exception as exc:
            return False, f"Error inspecting image '{self.image_name}': {str(exc)}"

    def run_script_in_container(
        self,
        script_file: Path,
        run_dir: Path,
        timeout: int | None = None,
    ) -> dict[str, Any]:
        """Execute a Python script inside the isolated Docker container."""
        timeout_val = timeout or self.default_timeout
        container_name = f"deepmd_run_{int(time.time())}_{uuid.uuid4().hex[:6]}"

        is_avail, status_msg = self.is_docker_available()
        if not is_avail:
            return {
                "success": False,
                "stdout": "",
                "stderr": (
                    "❌ EXECUTION BLOCKED (Docker Required):\n"
                    f"{status_msg}\n\n"
                    "Policy Enforcement: Untrusted AI-generated code is strictly prohibited from executing "
                    "on the host operating system. Execution must occur inside an isolated Docker container.\n"
                    "Please ensure Docker Desktop is running and the container image is built:\n"
                    "  powershell -File docker/build_container.ps1"
                ),
                "returncode": -1,
                "duration": 0.0,
                "container_name": None,
                "timed_out": False,
            }

        is_img, img_msg = self.is_image_available()
        if not is_img:
            return {
                "success": False,
                "stdout": "",
                "stderr": (
                    f"❌ EXECUTION BLOCKED (Image Missing):\n{img_msg}\n\n"
                    f"To build the required container, run:\n"
                    f"  powershell -File docker/build_container.ps1\n"
                    f"or:\n"
                    f"  docker build -t {self.image_name} -f docker/Dockerfile.deepmd ."
                ),
                "returncode": -1,
                "duration": 0.0,
                "container_name": None,
                "timed_out": False,
            }

        docker_bin = self._find_docker_bin() or "docker"

        # Format paths for Docker volume mounting (Windows compatible)
        run_dir_abs = str(run_dir.resolve()).replace("\\", "/")
        data_dir_abs = str(self.data_dir.resolve()).replace("\\", "/")
        script_rel_name = script_file.name

        # Highly secure container execution arguments:
        # 1. --network none: Total network isolation (no internet, no local subnet)
        # 2. --security-opt=no-new-privileges:true: Prevent privilege escalation
        # 3. --cap-drop=ALL: Drop all Linux kernel capabilities
        # 4. --pids-limit=100: Prevent fork bombs and thread starvation
        # 5. --tmpfs /tmp:rw,noexec,nosuid,size=64m: Ephemeral non-executable /tmp
        # 6. -v ...:/data:ro: Read-only dataset access (host data cannot be tampered with)
        # 7. -v ...:/workspace:rw: Isolated run directory (only generated plots/summaries output here)
        # 8. --user 1000:1000: Strictly unprivileged user execution
        cmd = [
            docker_bin, "run",
            "--name", container_name,
            "--rm",
            "--network", "none",
            "--security-opt=no-new-privileges:true",
            "--cap-drop=ALL",
            "--pids-limit=100",
            f"--memory={self.memory_limit}",
            f"--cpus={self.cpu_limit}",
            "--tmpfs", "/tmp:rw,noexec,nosuid,size=64m",
            "--user", "1000:1000",
            "-v", f"{run_dir_abs}:/workspace:rw",
            "-v", f"{data_dir_abs}:/data:ro",
            "-w", "/workspace",
            "-e", "MPLBACKEND=Agg",
            "-e", "PYTHONIOENCODING=utf-8",
            self.image_name,
            "python", f"/workspace/{script_rel_name}",
        ]

        start_time = time.perf_counter()
        timed_out = False
        stdout = ""
        stderr = ""
        returncode = -1

        try:
            proc = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=timeout_val,
            )
            duration = time.perf_counter() - start_time
            stdout = proc.stdout or ""
            stderr = proc.stderr or ""
            returncode = proc.returncode

        except subprocess.TimeoutExpired:
            duration = time.perf_counter() - start_time
            timed_out = True
            stderr = f"Execution timed out inside Docker container after {timeout_val} seconds."
            # Attempt to kill running container
            try:
                subprocess.run([docker_bin, "kill", container_name], capture_output=True, timeout=5)
            except Exception:
                pass

        except Exception as exc:
            duration = time.perf_counter() - start_time
            stderr = f"Docker execution error: {str(exc)}"

        return {
            "success": returncode == 0 and not timed_out,
            "stdout": stdout,
            "stderr": stderr,
            "returncode": returncode,
            "duration": round(duration, 3),
            "container_name": container_name,
            "timed_out": timed_out,
        }
