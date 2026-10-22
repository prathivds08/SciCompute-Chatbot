"""
ChromaDB vector store — persistent, embedded instance.

Ingests your project READMEs + optional external documentation into a
semantically searchable collection.  Uses Ollama nomic-embed-text for
embedding via the EmbeddingModel wrapper.
"""

from pathlib import Path
from typing import Optional

import chromadb
from chromadb.config import Settings as ChromaSettings

from backend.config import settings
from backend.rag.embeddings import EmbeddingModel


class OllamaEmbeddingFunction(chromadb.EmbeddingFunction):
    """Adapter so ChromaDB calls our Ollama-backed EmbeddingModel."""

    def __init__(self, embedding_model: EmbeddingModel):
        self._model = embedding_model

    def __call__(self, input: list[str]) -> list[list[float]]:  # noqa: A002
        return self._model.embed_batch(input)


class VectorStore:
    """Manages a ChromaDB collection for the chatbox knowledge base."""

    COLLECTION_NAME = "scientific_docs"

    def __init__(
        self,
        persist_dir: Optional[Path] = None,
        embedding_model: Optional[EmbeddingModel] = None,
    ):
        self.persist_dir = persist_dir or settings.CHROMA_PERSIST_DIR
        self.persist_dir.mkdir(parents=True, exist_ok=True)

        self._emb_model = embedding_model or EmbeddingModel()
        self._emb_fn = OllamaEmbeddingFunction(self._emb_model)

        self._client = chromadb.PersistentClient(
            path=str(self.persist_dir),
            settings=ChromaSettings(anonymized_telemetry=False),
        )
        self._collection = self._client.get_or_create_collection(
            name=self.COLLECTION_NAME,
            embedding_function=self._emb_fn,
            metadata={"hnsw:space": "cosine"},
        )

    # ------------------------------------------------------------------ #
    #  Ingestion
    # ------------------------------------------------------------------ #

    def ingest_document(
        self,
        doc_id: str,
        text: str,
        metadata: Optional[dict] = None,
    ) -> None:
        """Add (or update) a single document chunk."""
        self._collection.upsert(
            ids=[doc_id],
            documents=[text],
            metadatas=[metadata or {}],
        )

    PROTECTED_EXTENSIONS = {".extxyz", ".xyz", ".raw", ".npy", ".pb", ".db", ".sqlite", ".wal", ".shm"}
    PROTECTED_FILENAMES = {"Anatase_TiO2_Water_MolecularAdsorbedWater.extxyz"}

    def is_protected_file(self, filepath: Path) -> bool:
        """Check if a file contains raw simulation or coordinate data that must not be ingested into the model's context."""
        name = filepath.name
        suffix = filepath.suffix.lower()
        return name in self.PROTECTED_FILENAMES or suffix in self.PROTECTED_EXTENSIONS

    def ingest_file(
        self,
        filepath: Path,
        chunk_size: int = 1000,
        chunk_overlap: int = 200,
        metadata_extra: Optional[dict] = None,
    ) -> int:
        """Read a text file, chunk it, and ingest every chunk.

        Returns the number of chunks created. Refuses to ingest protected datasets.
        """
        if self.is_protected_file(filepath):
            print(
                f"[vectorstore] BLOCKED: '{filepath.name}' is a protected dataset. "
                "The model must not access raw trajectory data directly; execution occurs in Docker."
            )
            return 0

        text = filepath.read_text(encoding="utf-8", errors="replace")
        chunks = self._chunk_text(text, chunk_size, chunk_overlap)

        ids = []
        documents = []
        metadatas = []

        for i, chunk in enumerate(chunks):
            doc_id = f"{filepath.stem}__chunk_{i:04d}"
            meta = {
                "source_file": str(filepath),
                "chunk_index": i,
                "total_chunks": len(chunks),
            }
            if metadata_extra:
                meta.update(metadata_extra)

            ids.append(doc_id)
            documents.append(chunk)
            metadatas.append(meta)

        if ids:
            self._collection.upsert(ids=ids, documents=documents, metadatas=metadatas)

        return len(chunks)

    def ingest_project_readmes(self) -> dict[str, int]:
        """Auto-discover and ingest all README*.md files in the workspace.

        Returns {filepath: num_chunks} for each ingested file.
        """
        results: dict[str, int] = {}
        readme_paths = [
            settings.dft_dir / "README_EnergyCutoffAndKpoints.md",
            settings.training_dir / "README_Training_Script.md",
            settings.md_dir / "README_MD_Simulation_Script.md",
            settings.analysis_dir / "README_Water_Interface_Analysis.md",
        ]
        workflow_labels = ["dft", "training", "md", "analysis"]

        for path, wf_label in zip(readme_paths, workflow_labels):
            if path.exists():
                n = self.ingest_file(
                    path,
                    metadata_extra={"workflow_type": wf_label},
                )
                results[str(path)] = n

        # Ingest Python & scientific library documentation from backend/docs/
        docs_dir = settings.backend_dir / "docs"
        if docs_dir.exists():
            for doc_path in sorted(docs_dir.glob("*.md")):
                n = self.ingest_file(
                    doc_path,
                    metadata_extra={"workflow_type": "python_docs", "doc_type": "scientific_python_reference"},
                )
                results[str(doc_path)] = n

        return results

    # ------------------------------------------------------------------ #
    #  Retrieval
    # ------------------------------------------------------------------ #

    def search(
        self,
        query: str,
        n_results: int = 5,
        where: Optional[dict] = None,
    ) -> list[dict]:
        """Semantic search. Returns list of {id, document, metadata, distance}."""
        kwargs: dict = {"query_texts": [query], "n_results": n_results}
        if where:
            kwargs["where"] = where

        results = self._collection.query(**kwargs)

        docs = []
        for i in range(len(results["ids"][0])):
            docs.append({
                "id": results["ids"][0][i],
                "document": results["documents"][0][i],
                "metadata": results["metadatas"][0][i],
                "distance": results["distances"][0][i] if results.get("distances") else None,
            })
        return docs

    @property
    def count(self) -> int:
        return self._collection.count()

    # ------------------------------------------------------------------ #
    #  Internal helpers
    # ------------------------------------------------------------------ #

    @staticmethod
    def _chunk_text(text: str, chunk_size: int, overlap: int) -> list[str]:
        """Split text into overlapping chunks at paragraph / sentence boundaries."""
        # Simple paragraph-aware chunker
        paragraphs = text.split("\n\n")
        chunks: list[str] = []
        current = ""

        for para in paragraphs:
            if len(current) + len(para) + 2 > chunk_size and current:
                chunks.append(current.strip())
                # Keep overlap from the end of the previous chunk
                current = current[-overlap:] if len(current) > overlap else current
            current += "\n\n" + para

        if current.strip():
            chunks.append(current.strip())

        return chunks if chunks else [text]
