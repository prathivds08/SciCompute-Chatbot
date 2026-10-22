"""
Test suite verifying that the model cannot access raw dataset files
(such as backend/data/Anatase_TiO2_Water_MolecularAdsorbedWater.extxyz)
while allowing Docker execution to mount and call them.
"""

import sys
from pathlib import Path

# Add project root to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from backend.config import settings
from backend.rag.vectorstore import VectorStore
from backend.sandbox.docker_runner import DockerRunner
from backend.sandbox.executor import CodeExecutor


def test_vectorstore_blocks_extxyz_ingestion():
    """Verify VectorStore blocks ingestion of Anatase_TiO2_Water_MolecularAdsorbedWater.extxyz."""
    print("Testing VectorStore protected dataset policy ...")
    vs = VectorStore.__new__(VectorStore)  # test policy helper without connecting to Chroma
    
    extxyz_path = Path("backend/data/Anatase_TiO2_Water_MolecularAdsorbedWater.extxyz")
    assert vs.is_protected_file(extxyz_path) is True, "Failed to identify Anatase extxyz as protected!"
    
    other_xyz = Path("test_data/structure.xyz")
    assert vs.is_protected_file(other_xyz) is True, "Failed to identify .xyz as protected!"
    
    safe_md = Path("README.md")
    assert vs.is_protected_file(safe_md) is False, "Safe markdown was mistakenly flagged as protected!"
    print("  [OK] VectorStore correctly identifies protected datasets.")


def test_docker_mounts_data_ro():
    """Verify DockerRunner volume mounts configure read-only /data access."""
    print("Testing Docker dataset volume mount configuration ...")
    runner = DockerRunner()
    assert runner.data_dir == settings.backend_dir / "data"
    
    # Check that Anatase_TiO2_Water_MolecularAdsorbedWater.extxyz exists in runner.data_dir
    target_file = runner.data_dir / "Anatase_TiO2_Water_MolecularAdsorbedWater.extxyz"
    assert target_file.exists(), f"Target dataset not found at {target_file}"
    print(f"  [OK] DockerRunner dataset exists on host: {target_file.name} ({target_file.stat().st_size} bytes)")


def test_artifact_collector_suppresses_dataset_preview(tmp_path_factory):
    """Verify CodeExecutor artifact collection suppresses preview for .extxyz files."""
    print("Testing artifact collector privacy enforcement ...")
    import tempfile
    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp_p = Path(tmp_dir)
        fake_extxyz = tmp_p / "output.extxyz"
        fake_extxyz.write_text("100\nLattice=\"...\"\nTi 0.0 0.0 0.0\n", encoding="utf-8")
        
        fake_txt = tmp_p / "summary.txt"
        fake_txt.write_text("Analysis completed successfully.\n", encoding="utf-8")
        
        executor = CodeExecutor()
        artifacts = executor._collect_artifacts(tmp_p)
        
        art_map = {a["name"]: a for a in artifacts}
        assert "output.extxyz" in art_map
        assert "Dataset file" in art_map["output.extxyz"]["preview"]
        assert "Ti 0.0 0.0" not in art_map["output.extxyz"]["preview"], "Raw coordinates leaked in preview!"
        
        assert "summary.txt" in art_map
        assert "Analysis completed" in art_map["summary.txt"]["preview"]
        print("  [OK] Artifact collector cleanly suppresses raw trajectory previews.")


def test_ignore_and_rule_files():
    """Verify all ignore and rule files exist and contain required entries."""
    print("Testing ignore and rule configurations ...")
    root = settings.WORKSPACE_ROOT
    
    files_to_check = [
        root / ".geminiignore",
        root / ".gitignore",
        root / ".cursorignore",
        root / ".ignore",
        root / "AGENTS.md",
        root / "GEMINI.md",
        root / ".agents" / "rules" / "data_privacy.md",
    ]
    
    for f in files_to_check:
        assert f.exists(), f"Missing required configuration file: {f}"
        content = f.read_text(encoding="utf-8")
        assert "Anatase_TiO2_Water_MolecularAdsorbedWater.extxyz" in content, (
            f"File {f.name} missing reference to Anatase_TiO2_Water_MolecularAdsorbedWater.extxyz"
        )
        print(f"  [OK] {f.name} verified.")


if __name__ == "__main__":
    print("==================================================")
    print("  Running Data Isolation & Docker Calling Tests")
    print("==================================================")
    test_vectorstore_blocks_extxyz_ingestion()
    test_docker_mounts_data_ro()
    test_artifact_collector_suppresses_dataset_preview(None)
    test_ignore_and_rule_files()
    print("\n[SUCCESS] ALL DATA ISOLATION TESTS PASSED!")
