"""
ChromaDB vector store — persistent, embedded instance.

Ingests your project READMEs + optional external documentation into a
semantically searchable collection.  Uses Ollama nomic-embed-text for
embedding via the EmbeddingModel wrapper.

Upgrade 1 (partial): search() now calls embed_query() so queries get
  the 'search_query: ' task prefix required by nomic-embed-text.

Upgrade 2: _chunk_text() is replaced with a Markdown-header & code-block-
  aware chunker that:
  - Never splits a fenced code block (```...```) across two chunks.
  - Attaches a "[Document: <name> | Section: <heading>]" breadcrumb to
    every chunk so qwen2.5-coder:7b knows which system/library it describes.
  - Uses line-aligned overlap (never slices mid-word or mid-token).
  - Accepts a source_name argument for the breadcrumb.

Upgrade 4 (partial): ingest_project_readmes() also ingests the canonical
  DeePMD analysis script templates from backend/tools/deepmd_analysis_tools.py
  as doc_type="canonical_code_template" so the model always retrieves a
  working reference script from RAG rather than hallucinating one.
"""

import re
from pathlib import Path
from typing import Optional

import chromadb
from chromadb.config import Settings as ChromaSettings

from backend.config import settings
from backend.rag.embeddings import EmbeddingModel


class OllamaEmbeddingFunction(chromadb.EmbeddingFunction):
    """Adapter so ChromaDB calls our Ollama-backed EmbeddingModel.

    Ingestion uses embed_documents() (→ 'search_document: ' prefix).
    """

    def __init__(self, embedding_model: EmbeddingModel):
        self._model = embedding_model

    def __call__(self, input: list[str]) -> list[list[float]]:  # noqa: A002
        return self._model.embed_documents(input)


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
        """Check if a file contains raw simulation or coordinate data that must not be ingested."""
        name = filepath.name
        suffix = filepath.suffix.lower()
        return name in self.PROTECTED_FILENAMES or suffix in self.PROTECTED_EXTENSIONS

    def ingest_file(
        self,
        filepath: Path,
        chunk_size: int = 1200,
        chunk_overlap: int = 150,
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
        chunks = self._chunk_text(text, chunk_size, chunk_overlap, source_name=filepath.name)

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
        """Auto-discover and ingest all workflow READMEs and canonical code templates.

        Returns {filepath: num_chunks} for each ingested file.

        Upgrade 4 (partial): Also ingests deepmd_analysis_tools.py as a
        canonical_code_template so qwen2.5-coder:7b retrieves verified scripts
        directly from RAG rather than re-generating them from scratch.
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

        # Upgrade 4: Ingest canonical DeePMD analysis script templates
        # This ensures qwen2.5-coder:7b retrieves verified, working script patterns from RAG.
        deepmd_tools_path = settings.backend_dir / "tools" / "deepmd_analysis_tools.py"
        if deepmd_tools_path.exists():
            n = self.ingest_file(
                deepmd_tools_path,
                chunk_size=1400,  # Slightly larger to keep full function bodies intact
                metadata_extra={
                    "workflow_type": "analysis",
                    "doc_type": "canonical_code_template",
                },
            )
            results[str(deepmd_tools_path)] = n

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
        """Semantic search using the 'search_query: ' task prefix for nomic-embed-text.

        Returns list of {id, document, metadata, distance}.
        """
        # Upgrade 1: use embed_query() to apply the 'search_query: ' prefix
        query_embedding = self._emb_model.embed_query(query)

        kwargs: dict = {
            "query_embeddings": [query_embedding],
            "n_results": n_results,
            "include": ["documents", "metadatas", "distances"],
        }
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
    def _chunk_text(text: str, chunk_size: int = 1200, overlap: int = 150, source_name: str = "") -> list[str]:
        """Split Markdown text along headings and paragraphs while keeping fenced code blocks atomic.

        Upgrade 2: Replaces the naive double-newline splitter with a structure-aware
        Markdown chunker that:
          - Splits first on ## / ### section headings to preserve semantic sections.
          - Never splits a fenced ``` code block ``` across chunks.
          - Prepends a contextual breadcrumb to each chunk:
              [Document: <source> | Section: <heading>]
          - Uses line-aligned overlap (never slices mid-word or mid-token).
        """
        # Split by Markdown headings (## and ### level) to preserve semantic sections
        sections = re.split(r'(?m)(?=^#{1,3}\s+)', text)
        chunks: list[str] = []

        for section in sections:
            section = section.strip()
            if not section:
                continue

            lines = section.splitlines()
            header = lines[0].strip() if lines and lines[0].startswith("#") else ""
            prefix = (
                f"[Document: {source_name} | Section: {header.lstrip('#').strip()}]\n"
                if header
                else (f"[Document: {source_name}]\n" if source_name else "")
            )

            # If the whole section fits in one chunk, emit it directly
            if len(prefix) + len(section) <= chunk_size:
                chunks.append(prefix + section)
                continue

            # Split on code fences or double-newlines, keeping code blocks atomic
            # The regex captures fenced blocks (```...```) as indivisible units
            blocks = re.split(r'(```[\s\S]*?```|\n\n+)', section)
            current = prefix

            for block in blocks:
                if not block or block.isspace():
                    continue

                # If adding this block overflows, flush and start a new chunk with overlap
                if len(current) + len(block) + 2 > chunk_size and current.strip() != prefix.strip():
                    chunks.append(current.strip())

                    # Line-aligned overlap: take the last few lines (never slice mid-word)
                    tail_lines = current.splitlines()[-4:]
                    overlap_text = "\n".join(tail_lines)
                    if len(overlap_text) > overlap:
                        # Trim to overlap length but only at a newline boundary
                        overlap_text = overlap_text[-overlap:].partition("\n")[2]

                    current = prefix + (overlap_text + "\n\n" if overlap_text else "") + block.strip()
                else:
                    separator = "\n\n" if current != prefix else ""
                    current += separator + block.strip()

            if current.strip() and current.strip() != prefix.strip():
                chunks.append(current.strip())

        return chunks if chunks else [text]
