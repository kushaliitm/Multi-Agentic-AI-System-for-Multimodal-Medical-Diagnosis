import os
import logging
from uuid import uuid4
from typing import List, Dict, Any, Tuple

from langchain_core.documents import Document
from langchain.storage import LocalFileStore
from langchain_qdrant import FastEmbedSparse, QdrantVectorStore, RetrievalMode
from qdrant_client import QdrantClient, models
from qdrant_client.http.models import Distance, SparseVectorParams, VectorParams


class VectorStore:
    """
    VectorStore manages document ingestion and retrieval using Qdrant + LangChain.

    Responsibilities:
    - Create and configure a Qdrant collection (dense + sparse vectors for hybrid retrieval)
    - Ingest chunked documents into:
        1) QdrantVectorStore (dense/sparse embeddings for retrieval)
        2) LocalFileStore (raw chunk content, stored by doc_id)
    - Retrieve relevant chunks for a query using similarity search

    Design notes:
    - Uses HYBRID retrieval (dense embeddings + sparse BM25) to improve recall and relevance.
    - Uses LocalFileStore as the source of truth for chunk text, while Qdrant stores embeddings.
    - Assumes each chunk is stored with a stable doc_id in both systems.
    """

    def __init__(self, config):
        """
        Initialize VectorStore with retrieval and storage configuration.

        Args:
            config: Configuration object providing:
                - config.rag.collection_name: Qdrant collection name
                - config.rag.embedding_dim: dense embedding vector dimension
                - config.rag.distance_metric: similarity metric (e.g., cosine)
                - config.rag.embedding_model: embedding model instance
                - config.rag.top_k: retrieval top-k
                - config.rag.vector_search_type: retrieval strategy (similarity/mmr)
                - config.rag.vector_local_path: qdrant local path
                - config.rag.doc_local_path: local docstore path
        """
        self.logger = logging.getLogger(__name__)

        self.collection_name = config.rag.collection_name
        self.embedding_dim = config.rag.embedding_dim
        self.distance_metric = config.rag.distance_metric

        self.embedding_model = config.rag.embedding_model
        self.retrieval_top_k = config.rag.top_k
        self.vector_search_type = config.rag.vector_search_type

        self.vectorstore_local_path = config.rag.vector_local_path
        self.docstore_local_path = config.rag.doc_local_path

        # Local Qdrant client (embedded/local mode)
        self.client = QdrantClient(path=self.vectorstore_local_path)

    def _does_collection_exist(self) -> bool:
        """
        Check whether the configured Qdrant collection exists.

        Returns:
            True if the collection exists, otherwise False.
        """
        try:
            collection_info = self.client.get_collections()
            collection_names = [c.name for c in collection_info.collections]
            return self.collection_name in collection_names
        except Exception as e:
            self.logger.error(f"Error checking collection existence: {e}")
            return False

    def _create_collection(self) -> None:
        """
        Create a new Qdrant collection configured for hybrid retrieval.

        Configuration:
        - Dense vectors: cosine similarity, size=embedding_dim
        - Sparse vectors: on-disk indexing disabled (can be tuned later)

        Raises:
            Exception: If Qdrant fails to create the collection.
        """
        try:
            self.client.create_collection(
                collection_name=self.collection_name,
                vectors_config={
                    "dense": VectorParams(size=self.embedding_dim, distance=Distance.COSINE)
                },
                sparse_vectors_config={
                    "sparse": SparseVectorParams(index=models.SparseIndexParams(on_disk=False))
                },
            )
            self.logger.info(f"Created Qdrant collection: {self.collection_name}")
        except Exception as e:
            self.logger.error(f"Error creating collection: {e}")
            raise

    def load_vectorstore(self) -> Tuple[QdrantVectorStore, LocalFileStore]:
        """
        Load an existing vectorstore + docstore without ingesting new content.

        Use this when the Qdrant collection already exists and you only need retrieval.

        Returns:
            (qdrant_vectorstore, docstore)

        Raises:
            ValueError: If the collection does not exist.
        """
        if not self._does_collection_exist():
            msg = f"Collection '{self.collection_name}' does not exist. Ingest documents first."
            self.logger.error(msg)
            raise ValueError(msg)

        # Sparse embedding model for hybrid retrieval (BM25)
        sparse_embeddings = FastEmbedSparse(model_name="Qdrant/bm25")

        # Qdrant VectorStore wrapper configured for hybrid retrieval
        qdrant_vectorstore = QdrantVectorStore(
            client=self.client,
            collection_name=self.collection_name,
            embedding=self.embedding_model,
            sparse_embedding=sparse_embeddings,
            retrieval_mode=RetrievalMode.HYBRID,
            vector_name="dense",
            sparse_vector_name="sparse",
        )

        # Local docstore holding raw chunk text keyed by doc_id
        docstore = LocalFileStore(self.docstore_local_path)

        self.logger.info("Loaded existing vectorstore and docstore successfully.")
        return qdrant_vectorstore, docstore

    def create_vectorstore(
        self,
        document_chunks: List[str],
        document_path: str,
    ) -> Tuple[QdrantVectorStore, LocalFileStore, List[str]]:
        """
        Create a vectorstore (and docstore) for document chunks, or upsert into an existing collection.

        Workflow:
        1) Generate a UUID for each chunk
        2) Wrap each chunk as a LangChain Document with metadata
        3) Create the collection if missing
        4) Add documents to Qdrant (embeddings stored there)
        5) Store raw chunk text in LocalFileStore (bytes)

        Args:
            document_chunks: List of chunked text sections.
            document_path: Source document path for metadata.

        Returns:
            (qdrant_vectorstore, docstore, doc_ids)
        """
        # Generate stable IDs for chunks to link Qdrant + docstore
        doc_ids = [str(uuid4()) for _ in range(len(document_chunks))]

        # Build LangChain Document objects (stores metadata in Qdrant payload)
        langchain_documents: List[Document] = []
        for idx, chunk in enumerate(document_chunks):
            langchain_documents.append(
                Document(
                    page_content=chunk,
                    metadata={
                        "source": os.path.basename(document_path),
                        "doc_id": doc_ids[idx],
                        "source_path": os.path.join("http://localhost:8000/", document_path),
                    },
                )
            )

        # Sparse embeddings for hybrid retrieval (BM25)
        sparse_embeddings = FastEmbedSparse(model_name="Qdrant/bm25")

        # Ensure collection exists
        if not self._does_collection_exist():
            self._create_collection()
            self.logger.info(f"Created new collection '{self.collection_name}'")
        else:
            self.logger.info(f"Collection '{self.collection_name}' exists. Upserting documents.")

        # Initialize vectorstore wrapper
        qdrant_vectorstore = QdrantVectorStore(
            client=self.client,
            collection_name=self.collection_name,
            embedding=self.embedding_model,
            sparse_embedding=sparse_embeddings,
            retrieval_mode=RetrievalMode.HYBRID,
            vector_name="dense",
            sparse_vector_name="sparse",
        )

        # Local docstore (raw chunk text) for reconstruction and display
        docstore = LocalFileStore(self.docstore_local_path)

        # Upsert documents into Qdrant
        qdrant_vectorstore.add_documents(documents=langchain_documents, ids=doc_ids)

        # Store raw chunk text in docstore (must be bytes)
        encoded_chunks = [chunk.encode("utf-8") for chunk in document_chunks]
        docstore.mset(list(zip(doc_ids, encoded_chunks)))

        return qdrant_vectorstore, docstore, doc_ids

    def retrieve_relevant_chunks(
        self,
        query: str,
        vectorstore: QdrantVectorStore,
        docstore: LocalFileStore,
    ) -> List[Dict[str, Any]]:
        """
        Retrieve relevant chunks for a user query.

        Retrieval:
        - Uses similarity_search_with_score to return (Document, score) pairs.
        - Uses docstore (LocalFileStore) to retrieve the authoritative chunk text.
        - Returns documents in a normalized dictionary format for downstream reranking.

        Args:
            query: User query.
            vectorstore: Initialized QdrantVectorStore instance.
            docstore: LocalFileStore containing raw chunk content.

        Returns:
            List of dictionaries with fields:
                - id: chunk doc_id
                - content: chunk text
                - score: retrieval similarity score
                - source: source filename
                - source_path: link/path to source document
        """
        results = vectorstore.similarity_search_with_score(
            query=query,
            k=self.retrieval_top_k
        )

        retrieved_docs: List[Dict[str, Any]] = []

        for chunk, score in results:
            # Fetch authoritative text from docstore
            doc_content_bytes = docstore.mget([chunk.metadata["doc_id"]])[0]
            doc_content = doc_content_bytes.decode("utf-8")

            retrieved_docs.append({
                "id": chunk.metadata["doc_id"],
                "content": doc_content,
                "score": score,
                "source": chunk.metadata.get("source"),
                "source_path": chunk.metadata.get("source_path"),
            })

        return retrieved_docs
