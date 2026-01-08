import logging
from typing import List, Dict, Any, Optional


class ResponseGenerator:
    """
    ResponseGenerator produces the final user-facing answer using retrieved context.

    Responsibilities:
    - Build a controlled, instruction-heavy prompt using:
        1) user query
        2) retrieved document content
        3) optional chat history
    - Generate an answer using an LLM while constraining it to retrieved context only
    - Optionally append source citations and reference image links
    - Compute a simple confidence score based on retrieval / reranking scores

    Notes:
    - This class is designed for use in a medical RAG pipeline, where factual grounding
      and non-hallucination are critical.
    - It assumes retrieved_docs contain "content" and may include "source"/"source_path".
    """

    def __init__(self, config):
        """
        Initialize the response generator.

        Args:
            config: Application configuration object providing:
                - config.rag.response_generator_model: LLM used to write final answers
                - config.rag.include_sources: Whether to append source links (default: True)
        """
        self.logger = logging.getLogger(__name__)
        self.response_generator_model = config.rag.response_generator_model
        self.include_sources = getattr(config.rag, "include_sources", True)

    def _build_prompt(
        self,
        query: str,
        context: str,
        chat_history: Optional[List[Dict[str, str]]] = None
    ) -> str:
        """
        Build the prompt provided to the response generation LLM.

        The prompt enforces:
        - Grounding: answer only from retrieved context
        - Safe fallback: explicit refusal when context is insufficient
        - Formatting: clean markdown with small headings and optional tables
        - Non-hallucination: do not invent links or values

        Args:
            query: The user's question.
            context: Retrieved context text (concatenated documents).
            chat_history: Optional prior conversation messages (for continuity).

        Returns:
            A fully formatted prompt string.
        """
        table_instructions = """
Some of the retrieved information may be presented in table format. When using tabular data:
1. Present tables using proper markdown formatting with headers:
   | Column1 | Column2 |
   |---------|---------|
   | Value1  | Value2  |
2. Re-format tables for readability if needed.
3. If you introduce new structure (renaming columns, merging cells), explicitly note it.
4. Interpret table values in plain language in addition to showing the table.
5. Reference the relevant table when citing specific data points.
6. Summarize trends or patterns shown in tables when appropriate.
7. If reference numbers appear and the corresponding values (e.g., title/authors) exist in context,
   replace the reference numbers with the actual values.
""".strip()

        response_format_instructions = """
Instructions:
1. Answer ONLY using the information in the provided context.
2. If the context does not contain relevant information, respond exactly:
   "I don't have enough information to answer this question based on the provided context."
3. Do NOT use prior knowledge outside the context.
4. Be concise, accurate, and well-structured.
5. Use small headings and sub-headings in markdown when helpful.
6. Include tables only when relevant and supported by the context.
7. Do not invent or approximate values: use exact values present in context.
8. Do not repeat the question in the answer.
""".strip()

        prompt = f"""
You are a medical assistant providing accurate information based on verified retrieved sources.

Conversation context (recent messages):
{chat_history}

User question:
{query}

Retrieved context:
{context}

{table_instructions}

{response_format_instructions}

Answer the user thoroughly but concisely using only the retrieved context.
If the answer is not contained in the context, acknowledge the limitation.

Do not provide any source link that is not present in the context.
Do not fabricate citations or URLs.

Medical Assistant Response:
""".strip()

        return prompt

    def generate_response(
        self,
        query: str,
        retrieved_docs: List[Dict[str, Any]],
        picture_paths: List[str],
        chat_history: Optional[List[Dict[str, str]]] = None,
    ) -> Dict[str, Any]:
        """
        Generate a grounded response using retrieved documents.

        Workflow:
        1) Concatenate retrieved document content into a single context string
        2) Build a strict prompt (context-only answers)
        3) Invoke the response generation LLM
        4) Extract unique sources (optional)
        5) Compute a confidence score from retrieval/reranking scores
        6) Append source links and reference image links (optional)

        Args:
            query: User question.
            retrieved_docs: List of retrieved documents, each containing at minimum "content".
            picture_paths: List of image URLs/paths referenced by retrieved chunks.
            chat_history: Optional conversation history for continuity.

        Returns:
            A dictionary containing:
            - response: Final response text (possibly with sources and image links)
            - sources: List of source objects (title/path)
            - confidence: Float confidence score (0.0–1.0 approximation)
        """
        try:
            # --------------------------------------------------------------
            # Build context from retrieved documents
            # --------------------------------------------------------------
            doc_texts = [doc["content"] for doc in retrieved_docs]
            context = "\n\n===DOCUMENT SECTION===\n\n".join(doc_texts)

            # --------------------------------------------------------------
            # Generate LLM response
            # --------------------------------------------------------------
            prompt = self._build_prompt(query, context, chat_history)
            response = self.response_generator_model.invoke(prompt)

            # --------------------------------------------------------------
            # Collect citations/sources (optional)
            # --------------------------------------------------------------
            sources = self._extract_sources(retrieved_docs) if self.include_sources else []

            # --------------------------------------------------------------
            # Compute confidence score from top retrieved docs
            # --------------------------------------------------------------
            confidence = self._calculate_confidence(retrieved_docs)

            # --------------------------------------------------------------
            # Append sources to the response (optional)
            # --------------------------------------------------------------
            if self.include_sources:
                response_with_source = response.content + "\n\n##### Source documents:"
                for current_source in sources:
                    source_path = current_source["path"]
                    source_title = current_source["title"]
                    response_with_source += f"\n- [{source_title}]({source_path})"
            else:
                response_with_source = response.content

            # --------------------------------------------------------------
            # Append reference images (if any were detected in retrieved chunks)
            # --------------------------------------------------------------
            response_with_source_and_picture_paths = (
                response_with_source + "\n\n##### Reference images:"
            )
            for picture_path in picture_paths:
                response_with_source_and_picture_paths += (
                    f"\n- [{picture_path.split('/')[-1]}]({picture_path})"
                )

            return {
                "response": response_with_source_and_picture_paths,
                "sources": sources,
                "confidence": confidence,
            }

        except Exception as e:
            # Fail closed with a safe message instead of crashing upstream services
            self.logger.error(f"Error generating response: {e}")
            return {
                "response": (
                    "I apologize, but I encountered an error while generating a response. "
                    "Please try rephrasing your question."
                ),
                "sources": [],
                "confidence": 0.0,
            }

    def _extract_sources(self, documents: List[Dict[str, Any]]) -> List[Dict[str, str]]:
        """
        Extract unique source metadata from retrieved documents.

        Deduplicates sources and sorts them by document score to display
        the most relevant sources first.

        Expected document fields (if available):
        - source: human-readable title or filename
        - source_path: URL or local path to the source artifact
        - combined_score / rerank_score / score: ranking score

        Args:
            documents: Retrieved document dictionaries.

        Returns:
            List of unique sources with:
            - title
            - path
        """
        sources = []
        seen_sources = set()

        for doc in documents:
            source = doc.get("source")
            source_path = doc.get("source_path")

            # Skip if source metadata is missing
            if not source:
                continue

            source_id = f"{source}|{source_path}"
            if source_id in seen_sources:
                continue

            sources.append({
                "title": source,
                "path": source_path,
                "score": doc.get("combined_score", doc.get("rerank_score", doc.get("score", 0.0))),
            })
            seen_sources.add(source_id)

        # Sort by relevance score
        sources.sort(key=lambda x: x.get("score", 0), reverse=True)

        # Return only fields intended for display
        return [{"title": s["title"], "path": s["path"]} for s in sources]

    def _calculate_confidence(self, documents: List[Dict[str, Any]]) -> float:
        """
        Compute a simple confidence score based on retrieval ranking.

        This is a heuristic measure meant to support routing decisions
        (e.g., whether to fallback to web search).

        Logic:
        - Prefer combined_score (retrieval + rerank)
        - Else use rerank_score
        - Else use original score
        - Average top-3 available scores

        Args:
            documents: Retrieved document dictionaries.

        Returns:
            Float confidence score (0.0–1.0 approximation depending on scoring scale).
        """
        if not documents:
            return 0.0

        # Choose best available score type
        if "combined_score" in documents[0]:
            scores = [doc.get("combined_score", 0.0) for doc in documents[:3]]
        elif "rerank_score" in documents[0]:
            scores = [doc.get("rerank_score", 0.0) for doc in documents[:3]]
        else:
            scores = [doc.get("score", 0.0) for doc in documents[:3]]

        return sum(scores) / len(scores) if scores else 0.0
