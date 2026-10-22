"""
Centralised application settings.

All configurable values live here so every module imports from one place.
Values are read from environment variables (or a .env file) with sensible
defaults for local development.
"""

from pathlib import Path
from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    """Application-wide configuration backed by env vars / .env."""

    # ------------------------------------------------------------------ #
    #  LLM / Ollama
    # ------------------------------------------------------------------ #
    OLLAMA_BASE_URL: str = "http://localhost:11434"
    MODEL_NAME: str = "qwen2.5-coder:7b"               # Primary reasoning / code-gen model
    EMBEDDING_MODEL: str = "nomic-embed-text"         # Embedding model for ChromaDB

    # ------------------------------------------------------------------ #
    #  Workspace paths
    # ------------------------------------------------------------------ #
    WORKSPACE_ROOT: Path = Path(r"d:\Chatbox Lab\Chatbox_for_analysis_AL")

    # Derived convenience paths
    @property
    def dft_dir(self) -> Path:
        return self.WORKSPACE_ROOT / "DFT"

    @property
    def training_dir(self) -> Path:
        return self.WORKSPACE_ROOT / "Training"

    @property
    def md_dir(self) -> Path:
        return self.WORKSPACE_ROOT / "Molecular Dynamics"

    @property
    def analysis_dir(self) -> Path:
        return self.WORKSPACE_ROOT / "Analysis"

    @property
    def backend_dir(self) -> Path:
        return self.WORKSPACE_ROOT / "backend"

    # ------------------------------------------------------------------ #
    #  Storage backends
    # ------------------------------------------------------------------ #
    CHROMA_PERSIST_DIR: Path = Path(r"d:\Chatbox Lab\Chatbox_for_analysis_AL\backend\data\chromadb")
    KUZU_DB_DIR: Path = Path(r"d:\Chatbox Lab\Chatbox_for_analysis_AL\backend\data\kuzu_db")
    SQLITE_PATH: Path = Path(r"d:\Chatbox Lab\Chatbox_for_analysis_AL\backend\data\chatbox.db")
    SANDBOX_DIR: Path = Path(r"d:\Chatbox Lab\Chatbox_for_analysis_AL\backend\data\sandbox")

    # ------------------------------------------------------------------ #
    #  Sandbox & Execution (Docker Container Isolation)
    # ------------------------------------------------------------------ #
    SANDBOX_TIMEOUT: int = 60                        # Timeout per execution in seconds
    SANDBOX_MAX_OUTPUT_CHARS: int = 12000            # Max characters captured from stdout/stderr
    SCRIPTS_DIR: Path = Path(r"d:\Chatbox Lab\Chatbox_for_analysis_AL\backend\data\sandbox\scripts")

    # Docker container execution policies
    REQUIRE_DOCKER: bool = True                      # Enforce strict container-only execution (block host execution)
    DOCKER_IMAGE_NAME: str = "deepmd-analysis-sandbox:latest"
    DOCKER_CONTAINER_TIMEOUT: int = 120              # Max seconds for container execution
    DOCKER_MEMORY_LIMIT: str = "4g"                  # Memory ceiling
    DOCKER_CPU_LIMIT: str = "2.0"                    # CPU core limit
    # Python interpreter path for local scientific tasks
    CONDA_PYTHON_PATH: Path = Path(r"C:\Users\prath\miniconda3\envs\chatbot\python.exe")


    # ------------------------------------------------------------------ #
    #  HPC / SLURM  (configure when ready to submit jobs)
    # ------------------------------------------------------------------ #
    HPC_HOST: str = ""
    HPC_USER: str = ""
    HPC_KEY_PATH: str = ""                          # Path to SSH private key
    HPC_MAIL_DOMAIN: str = "@smail.iitm.ac.in"      # For SLURM --mail-user

    # ------------------------------------------------------------------ #
    #  Server
    # ------------------------------------------------------------------ #
    HOST: str = "0.0.0.0"
    PORT: int = 8000

    model_config = {
        "env_prefix": "CHATBOX_",
        "env_file": ".env",
        "env_file_encoding": "utf-8",
    }


# Singleton – import this everywhere
settings = Settings()
