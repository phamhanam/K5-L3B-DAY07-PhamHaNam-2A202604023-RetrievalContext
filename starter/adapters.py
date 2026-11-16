import json
import os
import re
from typing import Any

import psycopg2
from psycopg2.extras import RealDictCursor
from openai import OpenAI
from sentence_transformers import SentenceTransformer
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import CrossEncoder

ALLOWED_FILTER_KEYS = {"chapter", "section", "year", "category", "source", "document_id"}

class PgDenseRetriever:
    """Method #1: Dense retrieval against Day 6 PostgreSQL + pgvector store."""

    def __init__(
        self,
        database_url: str | None = None,
        model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
        strategy: str | None = None,
    ) -> None:
        if not strategy or not strategy.strip():
            raise ValueError("strategy is required to avoid duplicate IDs across strategies")
        self.strategy = strategy
        self.database_url = database_url or os.environ["DATABASE_URL"]
        self.embedder = SentenceTransformer(model_name)

    def _embed_query(self, query: str) -> str:
        vec = self.embedder.encode([query], normalize_embeddings=True)[0].tolist()
        return "[" + ",".join(f"{x:.8f}" for x in vec) + "]"

    def search(
        self,
        query: str,
        *,
        filters: dict[str, Any] | None,
        top_k: int,
    ) -> list[dict[str, Any]]:
        qvec = self._embed_query(query)
        where_clauses: list[str] = ["strategy = %s"]
        params: list[Any] = [qvec, self.strategy]

        if filters:
            for k, v in filters.items():
                if k not in ALLOWED_FILTER_KEYS:
                    raise ValueError(f"Unsupported filter key: {k}")
                where_clauses.append("metadata->>%s = %s")
                params.extend([k, str(v)])

        where_sql = f"WHERE {' AND '.join(where_clauses)}" if where_clauses else ""
        params.extend([qvec, top_k])

        sql = f"""
            SELECT
                id,
                text,
                document_id,
                source,
                page,
                metadata,
                1 - (embedding <=> %s::vector) AS score
            FROM chunks
            {where_sql}
            ORDER BY embedding <=> %s::vector
            LIMIT %s
        """

        with psycopg2.connect(self.database_url) as conn:
            with conn.cursor(cursor_factory=RealDictCursor) as cur:
                cur.execute(sql, params)
                rows = cur.fetchall()

        results: list[dict[str, Any]] = []
        for row in rows:
            meta = row.get("metadata") or {}
            results.append(
                {
                    "id": str(row["id"]),
                    "text": row["text"],
                    "source_id": row["document_id"],
                    "source": row["source"],
                    "page": row["page"],
                    "metadata": meta,
                    "score": float(row["score"]),
                }
            )
        return results

SYSTEM_PROMPT = """You are a grounded RAG agent. You MUST respond with a single valid JSON object and nothing else.

Available actions:
1) Search for evidence:
   {"action": "search", "query": "<concise search query>"}
2) Answer using ONLY the retrieved context:
   {"action": "answer", "answer": "<answer with inline [chunk-id] citations>", "citations": ["chunk-id", ...]}
3) Abstain when the retrieved context does NOT contain enough evidence (after searching):
   {"action": "answer", "answer": "", "citations": []}

Rules:
- Always search first if you have no context yet.
- Never use outside knowledge. Only cite chunk IDs present in context_ids.
- Every ID in `citations` MUST appear verbatim as `[chunk-id]` in `answer`.
- If the specific answer is NOT in the context (even if related context is present), DO NOT explain what is missing and DO NOT include citations. You MUST return exactly:
  {"action": "answer", "answer": "", "citations": []}
"""


class LLMController:
    """Real LLM controller using an OpenAI-compatible chat completions API."""

    def __init__(
        self,
        model: str | None = None,
        api_key: str | None = None,
        base_url: str | None = None,
        temperature: float = 0.0,
    ) -> None:
        self.model = model or os.environ.get("LLM_MODEL", "gpt-4o-mini")
        self.temperature = temperature
        self.client = OpenAI(
            api_key=api_key or os.environ["OPENAI_API_KEY"],
            base_url=base_url or os.environ.get("OPENAI_BASE_URL"),
        )

    @staticmethod
    def _parse_json(raw: str) -> dict[str, Any]:
        raw = raw.strip()
        # Strip markdown fences if the model wraps JSON in ```json ... ```
        match = re.search(r"\{.*\}", raw, flags=re.DOTALL)
        if not match:
            raise ValueError(f"Model did not return JSON: {raw[:200]}")
        return json.loads(match.group(0))

    def decide(self, question: str, history: list[dict[str, Any]]) -> dict[str, Any]:
        user_payload = {
            "question": question,
            "history": history,
        }
        resp = self.client.chat.completions.create(
            model=self.model,
            temperature=self.temperature,
            response_format={"type": "json_object"},
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": json.dumps(user_payload, ensure_ascii=False)},
            ],
        )
        content = resp.choices[0].message.content or ""
        action = self._parse_json(content)
        if action.get("action") not in {"search", "answer"}:
            raise ValueError(f"Invalid action from model: {action}")
        return action

class PgLexicalRetriever:
    """TF-IDF lexical retriever over the exact same Day 6 Postgres `strategy` corpus."""

    def __init__(self, strategy: str, database_url: str | None = None) -> None:
        if not strategy or not strategy.strip():
            raise ValueError("strategy is required")
        self.strategy = strategy
        self.database_url = database_url or os.environ["DATABASE_URL"]
        with psycopg2.connect(self.database_url) as conn:
            with conn.cursor(cursor_factory=RealDictCursor) as cur:
                cur.execute(
                    """
                    SELECT id, text, document_id, source, page, metadata
                    FROM chunks
                    WHERE strategy = %s
                    ORDER BY id
                    """,
                    (self.strategy,),
                )
                rows = cur.fetchall()

        self.chunks: list[dict[str, Any]] = [
            {
                "id": str(r["id"]),
                "text": r["text"],
                "source_id": r["document_id"],
                "source": r["source"],
                "page": r["page"],
                "metadata": r.get("metadata") or {},
            }
            for r in rows
        ]
        self.vectorizer = TfidfVectorizer()
        self.matrix = self.vectorizer.fit_transform([c["text"] for c in self.chunks])

    def search(
        self,
        query: str,
        *,
        filters: dict[str, Any] | None,
        top_k: int,
    ) -> list[dict[str, Any]]:
        q_vec = self.vectorizer.transform([query])
        scores = cosine_similarity(q_vec, self.matrix).ravel()
        order = np.argsort(-scores)

        results: list[dict[str, Any]] = []
        for idx in order:
            c = self.chunks[int(idx)]
            if filters:
                meta = c["metadata"]
                if any(str(meta.get(k)) != str(v) for k, v in filters.items()):
                    continue
            results.append({**c, "score": float(scores[int(idx)])})
            if len(results) >= top_k:
                break
        return results


class HybridRRFRetriever:
    """Method #3: Reciprocal Rank Fusion (RRF) of Day 6 Dense + TF-IDF Lexical."""

    def __init__(
        self,
        strategy: str,
        database_url: str | None = None,
        model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
        rrf_k: int = 60,
        candidate_multiplier: int = 4,
    ) -> None:
        self.dense = PgDenseRetriever(strategy=strategy, database_url=database_url, model_name=model_name)
        self.lexical = PgLexicalRetriever(strategy=strategy, database_url=database_url)
        self.rrf_k = rrf_k
        self.candidate_multiplier = candidate_multiplier

    def search(
        self,
        query: str,
        *,
        filters: dict[str, Any] | None,
        top_k: int,
    ) -> list[dict[str, Any]]:
        fetch_k = max(top_k * self.candidate_multiplier, 10)
        dense_hits = self.dense.search(query, filters=filters, top_k=fetch_k)
        lex_hits = self.lexical.search(query, filters=filters, top_k=fetch_k)

        fused_scores: dict[str, float] = {}
        chunk_by_id: dict[str, dict[str, Any]] = {}

        for rank, hit in enumerate(dense_hits, start=1):
            cid = hit["id"]
            chunk_by_id[cid] = hit
            fused_scores[cid] = fused_scores.get(cid, 0.0) + 1.0 / (self.rrf_k + rank)

        for rank, hit in enumerate(lex_hits, start=1):
            cid = hit["id"]
            chunk_by_id.setdefault(cid, hit)
            fused_scores[cid] = fused_scores.get(cid, 0.0) + 1.0 / (self.rrf_k + rank)

        ranked_ids = sorted(fused_scores, key=lambda cid: fused_scores[cid], reverse=True)[:top_k]
        return [{**chunk_by_id[cid], "score": float(fused_scores[cid])} for cid in ranked_ids]


class RerankRetriever:
    """Improvement / Method #4: Retrieve candidates then rerank with a local CrossEncoder."""

    def __init__(
        self,
        base_retriever: Any,
        reranker_model: str = "cross-encoder/ms-marco-MiniLM-L-6-v2",
        fetch_k: int = 12,
    ) -> None:
        self.base_retriever = base_retriever
        self.reranker = CrossEncoder(reranker_model)
        self.fetch_k = fetch_k

    def search(
        self,
        query: str,
        *,
        filters: dict[str, Any] | None,
        top_k: int,
    ) -> list[dict[str, Any]]:
        candidates = self.base_retriever.search(query, filters=filters, top_k=max(self.fetch_k, top_k))
        if not candidates:
            return []
        pairs = [(query, c["text"]) for c in candidates]
        scores = self.reranker.predict(pairs)
        ranked = sorted(zip(candidates, scores), key=lambda x: float(x[1]), reverse=True)[:top_k]
        return [{**c, "score": float(s)} for c, s in ranked]
    