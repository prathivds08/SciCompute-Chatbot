"""
Hybrid retriever — fuses results from ChromaDB, Kùzu, and SQLite.

Uses Reciprocal Rank Fusion (RRF) to merge ranked lists from different
retrieval backends into a single, unified context for the agent's prompt.

Upgrades applied:
  Upgrade 3: True hybrid retrieval — dense ChromaDB + sparse BM25 lexical
    search, fused via RRF, with a cosine distance gate (distance <= 0.55)
    to drop irrelevant chunks rather than polluting the 7B context window.
  Upgrade 4 (partial): _graph_lookup() uses word-boundary regex matching
    to prevent short element symbols (H, O, C, Si, Pt) from matching inside
    normal English words (e.g. "script", "simulation", "run").
  Upgrade 5: format_context_for_prompt() uses XML tags and U-shaped
    "sandwich" chunk ordering to counter 7B lost-in-the-middle attention decay.
"""

import math
import re
from pathlib import Path
from typing import Any, Optional

from backend.db.models import Database
from backend.rag.knowledge_graph import KnowledgeGraph
from backend.rag.vectorstore import VectorStore


class HybridRetriever:
    """Combines vector search, BM25 lexical search, graph lookup, and SQL into one context."""

    def __init__(
        self,
        vectorstore: VectorStore,
        knowledge_graph: KnowledgeGraph,
        database: Database,
    ):
        self.vs = vectorstore
        self.kg = knowledge_graph
        self.db = database

    # ------------------------------------------------------------------ #
    #  Core Retrieval — Upgrade 3
    # ------------------------------------------------------------------ #

    def retrieve(
        self,
        query: str,
        workflow_filter: Optional[str] = None,
        n_vector_results: int = 5,
        max_distance: float = 0.55,
        include_job_history: bool = False,
    ) -> dict[str, Any]:
        """Retrieve and fuse context across Dense Vector, Sparse BM25, Kùzu KG, and SQLite.

        Returns a dict with keys:
            - vector_docs: list of fused, distance-gated document chunks
            - graph_context: structured domain facts relevant to the query
            - job_history: recent SLURM jobs (only if query is job-related)
        """
        where = {"workflow_type": workflow_filter} if workflow_filter else None

        # 1. Dense semantic search — fetch 2x requested so the distance gate has headroom
        raw_dense = self.vs.search(query=query, n_results=n_vector_results * 2, where=where)

        # Apply cosine distance gate: drop chunks that are clearly off-topic,
        # but always keep at least 2 results (fallback) to avoid empty context.
        dense_docs = [
            d for d in raw_dense
            if d.get("distance") is None or d["distance"] <= max_distance
        ] or raw_dense[:2]

        # 2. Sparse BM25 lexical search — catches exact scientific identifiers
        #    (ecutwfc, ibrav, se_e2_a, get_scaled_positions, neighbor_list, etc.)
        lexical_docs = self._lexical_bm25_search(query=query, n_results=n_vector_results * 2)
        if workflow_filter:
            lexical_docs = [
                d for d in lexical_docs
                if d["metadata"].get("workflow_type") in (workflow_filter, "python_docs", "analysis")
            ]

        # 3. Fuse rankings via Reciprocal Rank Fusion (RRF)
        fused_docs = reciprocal_rank_fusion(
            [dense_docs, lexical_docs], id_key="id", k=60
        )[:n_vector_results]

        # 4. Knowledge Graph entity lookup (with word-boundary regexes — Upgrade 4)
        graph_context = self._graph_lookup(query)

        # 5. Job history — only when the query is explicitly about jobs/status/SLURM
        q_low = query.lower()
        wants_jobs = include_job_history or any(
            k in q_low for k in ("job", "slurm", "status", "running", "submitted", "history", "queue")
        )
        jobs = self.db.get_jobs(status=None)[:5] if wants_jobs else []

        return {
            "vector_docs": fused_docs,
            "graph_context": graph_context,
            "job_history": jobs,
        }

    # ------------------------------------------------------------------ #
    #  Sparse BM25 Lexical Search — Upgrade 3
    # ------------------------------------------------------------------ #

    def _lexical_bm25_search(self, query: str, n_results: int = 8) -> list[dict]:
        """Lightweight BM25 + exact identifier scoring over ChromaDB documents.

        Uses zero external dependencies (pure Python math).  Particularly
        effective for exact scientific API tokens:
          ecutwfc, ibrav, se_e2_a, rcut_smth, get_scaled_positions,
          neighbor_list, Pt_ONCV_PBE-1.2.upf, etc.
        """
        all_items = self.vs._collection.get(include=["documents", "metadatas"])
        ids = all_items.get("ids", [])
        docs = all_items.get("documents", [])
        metas = all_items.get("metadatas", [])
        if not ids:
            return []

        tokenize = lambda s: re.findall(r"[a-zA-Z0-9_]+", s.lower())  # noqa: E731
        q_tokens = tokenize(query)
        if not q_tokens:
            return []

        doc_tokens = [tokenize(d) for d in docs]
        N = len(docs)
        avgdl = sum(len(dt) for dt in doc_tokens) / max(N, 1)
        k1, b = 1.5, 0.75

        # Compute inverse document frequency per query token
        idf: dict[str, float] = {}
        for qt in set(q_tokens):
            df = sum(1 for dt in doc_tokens if qt in dt)
            idf[qt] = math.log(1.0 + (N - df + 0.5) / (df + 0.5))

        scored = []
        for i, dt in enumerate(doc_tokens):
            dl = len(dt)
            score = 0.0
            for qt in q_tokens:
                tf = dt.count(qt)
                if tf > 0:
                    score += idf[qt] * (tf * (k1 + 1.0)) / (tf + k1 * (1.0 - b + b * dl / avgdl))
            if score > 0:
                scored.append({
                    "id": ids[i],
                    "document": docs[i],
                    "metadata": metas[i] or {},
                    "bm25_score": score,
                })

        scored.sort(key=lambda x: x["bm25_score"], reverse=True)
        return scored[:n_results]

    # ------------------------------------------------------------------ #
    #  Context Formatting — Upgrade 5
    # ------------------------------------------------------------------ #

    def format_context_for_prompt(self, context: dict[str, Any]) -> str:
        """Format retrieved context using XML tags and U-shaped attention ordering.

        Upgrade 5:
          - XML tags (<retrieved_context>, <knowledge_graph_facts>, <doc>, etc.)
            are parsed with higher structural adherence by qwen2.5-coder:7b
            than flat markdown ### headers.
          - U-shaped "sandwich" ordering places the highest-ranked chunk first
            and the second-highest chunk last, so they anchor both ends of
            the attention window (counters 7B lost-in-the-middle decay).
        """
        parts: list[str] = ["<retrieved_context>"]

        # 1. Knowledge Graph facts FIRST — highest precision, most compact
        if context.get("graph_context"):
            parts.append("  <knowledge_graph_facts>")
            for key, value in context["graph_context"].items():
                if value:
                    parts.append(f'    <fact key="{key}">{value}</fact>')
            parts.append("  </knowledge_graph_facts>")

        # 2. U-shaped sandwich ordering: [rank1, rank3, rank5, ..., rank4, rank2]
        docs = context.get("vector_docs") or []
        if docs:
            # Build reordered list: fill from left (odd ranks) and right (even ranks)
            reordered: list[tuple[int, dict]] = [None] * len(docs)  # type: ignore[list-item]
            left, right = 0, len(docs) - 1
            for idx, doc in enumerate(docs):
                if idx % 2 == 0:
                    reordered[left] = (idx + 1, doc)
                    left += 1
                else:
                    reordered[right] = (idx + 1, doc)
                    right -= 1

            parts.append("  <documentation_excerpts>")
            for rank, doc in reordered:
                source = Path(doc["metadata"].get("source_file", "unknown")).name
                wf = doc["metadata"].get("workflow_type", "general")
                doc_type = doc["metadata"].get("doc_type", "")
                type_attr = f' type="{doc_type}"' if doc_type else ""
                parts.append(
                    f'    <doc rank="{rank}" source="{source}" domain="{wf}"{type_attr}>\n'
                    f'{doc["document"]}\n'
                    f"    </doc>"
                )
            parts.append("  </documentation_excerpts>")

        # 3. Job history — only when present (query was job-related)
        if context.get("job_history"):
            parts.append("  <recent_jobs>")
            for job in context["job_history"][:5]:
                parts.append(
                    f'    <job workflow="{job.get("workflow_type", "?")}" '
                    f'status="{job.get("status", "?")}" '
                    f'slurm_id="{job.get("slurm_job_id", "N/A")}" />'
                )
            parts.append("  </recent_jobs>")

        parts.append("</retrieved_context>")
        return "\n".join(parts) if len(parts) > 2 else '<retrieved_context status="empty" />'

    # ------------------------------------------------------------------ #
    #  Graph lookup heuristics — Upgrade 4 (word-boundary fix)
    # ------------------------------------------------------------------ #

    def _graph_lookup(self, query: str) -> dict[str, Any]:
        """Extract structured facts from the KG based on query keywords.

        Uses compiled word-boundary regexes and chemical/domain synonyms to prevent
        single-character and two-character symbols (H, O, C, Pt, Ir, Si, Ti, Ru)
        from falsely matching inside English words like 'script', 'simulation',
        'run', 'directory', etc.
        """
        result: dict[str, Any] = {}
        q_lower = query.lower()

        element_patterns = {
            "Pt": r"\b(?:pt|platinum)\b",
            "Ir": r"\b(?:ir|iridium|iro2)\b",
            "Ti": r"\b(?:ti|titanium|tio2|anatase)\b",
            "H":  r"\b(?:h|h2o|hydrogen|water|hydroxyl|oh)\b",
            "O":  r"\b(?:o|o2|h2o|tio2|iro2|oxygen|oxide)\b",
            "C":  r"\b(?:c|carbon|graphene)\b",
            "Si": r"\b(?:si|silicon)\b",
            "Mg": r"\b(?:mg|magnesium|mgo)\b",
            "Fe": r"\b(?:fe|iron)\b",
            "Ru": r"\b(?:ru|ruthenium|ruo2)\b",
        }
        for elem, pat in element_patterns.items():
            if re.search(pat, q_lower):
                pp = self.kg.get_pseudopotential(elem)
                if pp:
                    result[f"{elem}_pseudopotential"] = pp

        system_patterns = {
            "Water": r"\b(?:water|h2o)\b",
            "Interface": r"\b(?:interface|graphene)\b",
            "IrO2_Water": r"\b(?:iro2|iridium)\b",
            "Anatase_TiO2_Water": r"\b(?:anatase|tio2|titanium)\b",
        }
        for sys_name, pat in system_patterns.items():
            if re.search(pat, q_lower):
                info = self.kg.get_system_info(sys_name)
                if info:
                    result[f"system_{sys_name}"] = info
                materials = self.kg.get_system_materials(sys_name)
                if materials:
                    result[f"{sys_name}_elements"] = materials

        workflow_patterns = {
            "dft": r"\b(?:dft|espresso|pw\.x|ecutwfc|kpoints|k-points|pseudopotential)\b",
            "training": r"\b(?:train|training|deepmd|se_e2_a|raw_to_set|lcurve)\b",
            "md": r"\b(?:md|lammps|molecular dynamics|nvt|npt|pair_style)\b",
            "analysis": r"\b(?:analysis|rdf|density|survival|hydroxyl|extxyz)\b",
        }
        for wf_name, pat in workflow_patterns.items():
            if re.search(pat, q_lower):
                steps = self.kg.get_workflow_steps(wf_name)
                if steps:
                    result[f"{wf_name}_steps"] = [s["name"] for s in steps]

        return result


# ------------------------------------------------------------------ #
#  Reciprocal Rank Fusion — Upgrade 3 (now actually called)
# ------------------------------------------------------------------ #

def reciprocal_rank_fusion(
    ranked_lists: list[list[dict]],
    id_key: str = "id",
    k: int = 60,
) -> list[dict]:
    """Merge multiple ranked lists using Reciprocal Rank Fusion (RRF).

    Each item must have a unique `id_key` field. Items are scored by
    sum(1 / (k + rank)) across all lists they appear in, then sorted
    descending by fused score.

    Args:
        ranked_lists: List of ranked result lists (each result is a dict).
        id_key: The key used to identify unique items across lists.
        k: RRF constant (default 60, per the original paper).

    Returns:
        A single list sorted by fused RRF score (best first).
    """
    scores: dict[str, float] = {}
    items: dict[str, dict] = {}

    for ranked in ranked_lists:
        for rank, item in enumerate(ranked):
            item_id = str(item.get(id_key, rank))
            scores[item_id] = scores.get(item_id, 0.0) + 1.0 / (k + rank + 1)
            items[item_id] = item

    sorted_ids = sorted(scores, key=lambda x: scores[x], reverse=True)
    return [items[i] for i in sorted_ids]
