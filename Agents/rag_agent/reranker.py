import os
import re
import logging
from pathlib import Path
from typing import List, Dict, Any, Union

from sentence_transformers import CrossEncoder


class Reranker:
    """
    Reranker reorders retrieved documents using a cross-encoder model.

    Purpose:
    - Improve ranking quality after initial vector search
    - Use query–document joint encoding for higher relevance accuracy
    - Combine original retrieval score with cross-encoder relevance score

    This component is typically used in a RAG pipeline after
    dense retrieval and before final answer generation.
    """

    def __init__(self, config):
        """
        Initialize the reranker with configuration parameters.

        Args:
            config: Application configuration object providing:
                - config.rag.reranker_model: Cross-encoder model name
                - config.rag.reranker_top_k: Maximum number of documents to keep
        """
        self.logger = logging.getLogger(__name__)

        try:
            # Load cross-encoder model for query–document relevance scoring
            self.model_name = config.rag.reranker_model
            self.logger.info(f"Loading reranker model: {self.model_name}")

            self.model = CrossEncoder(self.model_name)
            self.top_k = config.rag.reranker_top_k

        except Exception as e:
            self.logger.error(f"Failed to initialize reranker model: {e}")
            raise

    def rerank(
        self,
        query: str,
        documents: Union[List[Dict[str, Any]], List[str]],
        parsed_content_dir: str
    ) -> tuple[List[Dict[str, Any]], List[str]]:
        """
        Rerank retrieved documents using cross-encoder relevance scoring.

        The method:
        1) Normalizes document format
        2) Scores each (query, document) pair
        3) Combines original and rerank scores
        4) Sorts documents by combined score
        5) Extracts referenced image paths (if present)

        Args:
            query: User query string
            documents: Retrieved documents (list of dicts or list of strings)
            parsed_content_dir: Directory containing extracted images

        Returns:
            A tuple of:
            - reranked_docs: Top-K reranked document dictionaries
            - picture_reference_paths: List of image URLs referenced in content
        """
        try:
            if not documents:
                return [], []

            # --------------------------------------------------------------
            # Normalize document structure
            # --------------------------------------------------------------
            if isinstance(documents[0], str):
                # Convert raw text documents into structured dictionaries
                normalized_docs = []
                for i, doc_text in enumerate(documents):
                    normalized_docs.append({
                        "id": i,
                        "content": doc_text,
                        "score": 1.0  # Default retrieval score
                    })
                documents = normalized_docs

            elif isinstance(documents[0], dict):
                # Ensure required fields exist
                for i, doc in enumerate(documents):
                    doc.setdefault("id", i)
                    doc.setdefault("score", 1.0)

                    if "content" not in doc:
                        if "text" in doc:
                            doc["content"] = doc["text"]
                        else:
                            doc["content"] = f"Document {i}"

            # --------------------------------------------------------------
            # Cross-encoder scoring
            # --------------------------------------------------------------
            # Prepare (query, document) pairs
            pairs = [(query, doc["content"]) for doc in documents]

            # Predict relevance scores
            scores = self.model.predict(pairs)

            # Attach scores to documents
            for i, score in enumerate(scores):
                rerank_score = float(score)
                documents[i]["rerank_score"] = rerank_score

                # Combine original retrieval score with rerank score
                original_score = documents[i].get("score", 1.0)
                documents[i]["combined_score"] = (original_score + rerank_score) / 2

            # --------------------------------------------------------------
            # Sort and trim results
            # --------------------------------------------------------------
            reranked_docs = sorted(
                documents,
                key=lambda x: x["combined_score"],
                reverse=True
            )

            if self.top_k and len(reranked_docs) > self.top_k:
                reranked_docs = reranked_docs[:self.top_k]

            # --------------------------------------------------------------
            # Extract referenced image paths from content
            # --------------------------------------------------------------
            picture_reference_paths: List[str] = []

            for doc in reranked_docs:
                matches = re.finditer(r"picture_counter_(\d+)", doc["content"])
                for match in matches:
                    counter_value = int(match.group(1))

                    # Derive image path from document source and counter
                    doc_basename = os.path.splitext(doc.get("source", "doc"))[0]

                    picture_path = os.path.join(
                        "http://localhost:8000/",
                        parsed_content_dir,
                        f"{doc_basename}-picture-{counter_value}.png"
                    )

                    picture_reference_paths.append(picture_path)

            return reranked_docs, picture_reference_paths

        except Exception as e:
            self.logger.error(f"Error during reranking: {e}")
            self.logger.warning("Falling back to original ranking")

            # Fallback: return documents without reranking
            return documents, []
