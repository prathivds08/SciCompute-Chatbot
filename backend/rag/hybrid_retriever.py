"""
Hybrid retriever — fuses results from ChromaDB, Kùzu, and SQLite.

Uses Reciprocal Rank Fusion (RRF) to merge ranked lists from different
retrieval backends into a single, unified context for the agent's prompt.
"""

from typing import Any, Optional

from backend.db.models import Database
from backend.rag.knowledge_graph import KnowledgeGraph
from backend.rag.vectorstore import VectorStore


class HybridRetriever:
    """Combines vector search, graph lookup, and SQL queries into one context."""

    def __init__(
        self,
        vectorstore: VectorStore,
        knowledge_graph: KnowledgeGraph,
        database: Database,
    ):
        self.vs = vectorstore
        self.kg = knowledge_graph
        self.db = database

    def retrieve(
        self,
        query: str,
        workflow_filter: Optional[str] = None,
        n_vector_results: int = 5,
        include_job_history: bool = True,
    ) -> dict[str, Any]:
        """Retrieve context from all three backends.

        Returns a dict with keys:
            - vector_docs: list of semantically similar document chunks
            - graph_context: structured domain facts relevant to the query
            - job_history: recent related jobs (if include_job_history=True)
        """
        context: dict[str, Any] = {}

        # ---- 1. Vector search (semantic) ----
        where = {"workflow_type": workflow_filter} if workflow_filter else None
        vector_docs = self.vs.search(
            query=query,
            n_results=n_vector_results,
            where=where,
        )
        context["vector_docs"] = vector_docs

        # ---- 2. Knowledge graph (structured) ----
        graph_context = self._graph_lookup(query)
        context["graph_context"] = graph_context

        # ---- 3. Job history (relational) ----
        if include_job_history:
            jobs = self.db.get_jobs(status=None)[:10]  # Last 10 jobs
            context["job_history"] = jobs
        else:
            context["job_history"] = []

        return context

    def format_context_for_prompt(self, context: dict[str, Any]) -> str:
        """Format the retrieved context into a string suitable for an LLM prompt."""
        parts: list[str] = []

        # Vector docs
        if context.get("vector_docs"):
            parts.append("## Relevant Documentation")
            for i, doc in enumerate(context["vector_docs"], 1):
                source = doc["metadata"].get("source_file", "unknown")
                parts.append(f"\n### Source {i}: {source}")
                parts.append(doc["document"])

        # Graph context
        if context.get("graph_context"):
            parts.append("\n## Domain Knowledge (Knowledge Graph)")
            for key, value in context["graph_context"].items():
                if value:
                    parts.append(f"- **{key}**: {value}")

        # Job history
        if context.get("job_history"):
            parts.append("\n## Recent Job History")
            for job in context["job_history"][:5]:
                parts.append(
                    f"- [{job.get('workflow_type', '?')}] "
                    f"Status: {job.get('status', '?')} | "
                    f"SLURM ID: {job.get('slurm_job_id', 'N/A')}"
                )

        return "\n".join(parts) if parts else "(No relevant context found.)"

    # ------------------------------------------------------------------ #
    #  Graph lookup heuristics
    # ------------------------------------------------------------------ #

    def _graph_lookup(self, query: str) -> dict[str, Any]:
        """Extract structured facts from the KG based on query keywords.

        This is a heuristic keyword-based approach. The router agent can
        also call KG methods directly for precise lookups.
        """
        result: dict[str, Any] = {}
        q_lower = query.lower()

        # Check for element symbols in the query
        elements = ["Pt", "Ir", "H", "O", "C", "Si", "Mg", "Fe", "Ti", "Ru"]
        mentioned_elements = [e for e in elements if e.lower() in q_lower or e in query]

        for elem in mentioned_elements:
            pp = self.kg.get_pseudopotential(elem)
            if pp:
                result[f"{elem}_pseudopotential"] = pp

        # Check for system names
        system_names = ["Water", "Interface", "IrO2_Water"]
        for sys_name in system_names:
            if sys_name.lower().replace("_", " ") in q_lower or sys_name.lower() in q_lower:
                info = self.kg.get_system_info(sys_name)
                if info:
                    result[f"system_{sys_name}"] = info
                materials = self.kg.get_system_materials(sys_name)
                if materials:
                    result[f"{sys_name}_elements"] = materials

        # Check for workflow names
        workflow_names = ["dft", "training", "md", "analysis"]
        for wf_name in workflow_names:
            if wf_name in q_lower:
                steps = self.kg.get_workflow_steps(wf_name)
                if steps:
                    result[f"{wf_name}_steps"] = [s["name"] for s in steps]

        return result


def reciprocal_rank_fusion(
    ranked_lists: list[list[dict]],
    id_key: str = "id",
    k: int = 60,
) -> list[dict]:
    """Merge multiple ranked lists using RRF.

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
