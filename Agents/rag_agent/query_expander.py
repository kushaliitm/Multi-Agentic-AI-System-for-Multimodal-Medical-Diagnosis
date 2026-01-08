import logging
from typing import Dict, Any


class QueryExpander:
    """
    QueryExpander enhances user queries with medically relevant terminology
    to improve retrieval quality in a Retrieval-Augmented Generation (RAG) pipeline.

    Purpose:
    - Enrich short or ambiguous user queries
    - Add clinically relevant synonyms and related concepts
    - Preserve the user's original intent and domain
    - Improve recall during vector search without changing meaning
    """

    def __init__(self, config):
        """
        Initialize the query expander.

        Args:
            config: Application configuration object containing:
                - config.rag.llm: LLM used for query expansion
        """
        self.logger = logging.getLogger(self.__module__)
        self.config = config
        self.model = config.rag.llm

    def expand_query(self, original_query: str) -> Dict[str, Any]:
        """
        Expand a user query with relevant medical terminology.

        This method keeps the original query intact and produces
        an expanded version that may include:
        - Medical synonyms
        - Related clinical terms
        - Domain-specific phrasing useful for retrieval

        Args:
            original_query: Raw user query text

        Returns:
            A dictionary containing:
            - original_query: The original user query
            - expanded_query: The LLM-expanded query text
        """
        self.logger.info(f"Expanding query: {original_query}")

        # Generate expanded query using LLM
        expanded_query = self._generate_expansions(original_query)

        return {
            "original_query": original_query,
            "expanded_query": expanded_query.content
        }

    def _generate_expansions(self, query: str):
        """
        Use an LLM to expand a query with medically relevant terminology.

        Expansion rules enforced via prompt:
        - Expand ONLY if beneficial
        - Preserve the original domain (medical or otherwise)
        - Do NOT introduce unrelated medical domains
        - Preserve user-requested response formats (e.g., tabular intent)
        - Return only the expanded query text, without explanation

        Args:
            query: Original user query

        Returns:
            LLM response object containing the expanded query
        """
        prompt = f"""
You are a medical domain expert assisting a search system.

Expand the following user query with relevant medical terminology,
synonyms, and closely related concepts ONLY if it improves retrieval quality.

User Query:
{query}

Rules:
- If expansion is unnecessary, keep the query unchanged.
- Stay strictly within the domain implied by the query.
- Do NOT introduce unrelated medical fields.
- If the user specifies a response format (e.g., tabular), preserve that intent.
- Do NOT answer the query.
- Output ONLY the expanded query text.

Expanded Query:
"""
        expansion = self.model.invoke(prompt)
        return expansion
