import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI, OpenAIEmbeddings

# Load environment variables from .env file
# override=True ensures existing environment variables are overwritten
load_dotenv(override=True)


class AgentDecisoinConfig:
    """
    Configuration for the agent decision controller.
    This LLM is responsible for high-level routing and decision-making
    between multiple agents in the system.
    """
    def __init__(self):
        self.llm = ChatOpenAI(
            model="gpt-4o",
            api_key=os.getenv("OPENAI_API_KEY"),
            temperature=0.1  # Low temperature for deterministic decisions
        )


class ConversationConfig:
    """
    Configuration for the conversational agent.
    Used for general user interaction and dialogue management.
    """
    def __init__(self):
        self.llm = ChatOpenAI(
            model=os.getenv("OPENAI_MODEL", "gpt-4o"),
            api_key=os.getenv("OPENAI_API_KEY"),
            temperature=0.7  # Higher temperature for natural conversation
        )


class WebSearchConfig:
    """
    Configuration for the web search agent.
    Used when external or real-time information retrieval is required.
    """
    def __init__(self):
        self.llm = ChatOpenAI(
            model=os.getenv("OPENAI_MODEL", "gpt-4o"),
            api_key=os.getenv("OPENAI_API_KEY"),
            temperature=0.3  # Balanced creativity and factuality
        )
        self.context_limit = 20  # Max number of web results to consider


class RAGConfig:
    """
    Configuration for the Retrieval-Augmented Generation (RAG) system.
    Handles document ingestion, embedding, vector search, reranking,
    and response generation.
    """
    def __init__(self):
        # Vector database configuration
        self.vector_db_type = "qdrant"
        self.embedding_dim = 1536
        self.distance_metric = "Cosine"
        self.use_local = True

        # Local storage paths
        self.vector_local_path = "./data/qdrant_db"
        self.doc_local_path = "./data/docs_db"
        self.parsed_content_dir = "./data/parsed_docs"

        # Remote Qdrant configuration (optional)
        self.url = os.getenv("QDRANT_URL")
        self.api_key = os.getenv("QDRANT_API_KEY")

        # Vector collection name
        self.collection_name = "medical_assistance_rag"

        # Text chunking parameters
        self.chunk_size = 512
        self.chunk_overlap = 50

        # Embedding model configuration
        self.embedding_model = OpenAIEmbeddings(
            model=os.getenv("OPENAI_EMBEDDING_MODEL", "text-embedding-3-large"),
            api_key=os.getenv("OPENAI_API_KEY")
        )

        # Core LLM used for RAG reasoning
        self.llm = ChatOpenAI(
            model=os.getenv("OPENAI_MODEL", "gpt-4o"),
            api_key=os.getenv("OPENAI_API_KEY"),
            temperature=0.3
        )

        # LLM used for summarizing retrieved content
        self.summarizer_model = ChatOpenAI(
            model=os.getenv("OPENAI_MODEL", "gpt-4o"),
            api_key=os.getenv("OPENAI_API_KEY"),
            temperature=0.5
        )

        # LLM used for chunking and preprocessing documents
        self.chunker_model = ChatOpenAI(
            model=os.getenv("OPENAI_MODEL", "gpt-4o"),
            api_key=os.getenv("OPENAI_API_KEY"),
            temperature=0.0  # Deterministic chunking
        )

        # LLM used for final response generation
        self.response_generator_model = ChatOpenAI(
            model=os.getenv("OPENAI_MODEL", "gpt-4o"),
            api_key=os.getenv("OPENAI_API_KEY"),
            temperature=0.3
        )

        # Retrieval and reranking settings
        self.top_k = 5
        self.vector_search_type = "similarity"  # or 'mmr'
        self.huggingface_token = os.getenv("HUGGINGFACE_TOKEN")
        self.reranker_model = "cross-encoder/ms-marco-TinyBERT-L-6"
        self.reranker_top_k = 3

        # Context and confidence thresholds
        self.max_context_length = 8192
        self.include_sources = True
        self.min_retrieval_confidence = 0.40
        self.context_limit = 20


class MedicalCVConfig:
    """
    Configuration for medical computer vision agents.
    Contains paths to trained deep learning models
    and LLM for medical image interpretation.
    """
    def __init__(self):
        self.chest_xray_model_path = (
            "./agents/image_analysis_agent/chest_xray_agent/models/covid_chest_xray_model.pth"
        )

        # LLM used for explaining and contextualizing CV outputs
        self.llm = ChatOpenAI(
            model=os.getenv("OPENAI_MODEL", "gpt-4o"),
            api_key=os.getenv("OPENAI_API_KEY"),
            temperature=0.1
        )


class SpeechConfig:
    """
    Configuration for speech synthesis (text-to-speech).
    Uses ElevenLabs for voice output.
    """
    def __init__(self):
        self.eleven_labs_api_key = os.getenv("ELEVEN_LABS_API_KEY")
        self.eleven_labs_voice_id = "21m00Tcm4TlvDq8ikWAM"


class ValidationConfig:
    """
    Configuration for agent output validation.
    Enables manual or automated validation for high-risk agents.
    """
    def __init__(self):
        self.require_validation = {
            "CONVERSATION_AGENT": False,
            "RAG_AGENT": False,
            "WEB_SEARCH_AGENT": False,
            "CHEST_XRAY_AGENT": True,
        }
        self.validation_timeout = 300  # seconds
        self.default_action = "reject"  # fallback behavior


class APIConfig:
    """
    Configuration for the backend API server.
    """
    def __init__(self):
        self.host = "0.0.0.0"
        self.port = 8000
        self.debug = True
        self.rate_limit = 10
        self.max_image_upload_size = 5  # MB


class UIConfig:
    """
    Configuration for the user interface.
    """
    def __init__(self):
        self.theme = "light"
        self.enable_speech = True
        self.enable_image_upload = True


class Config:
    """
    Central configuration aggregator.
    Initializes and exposes all subsystem configurations.
    """
    def __init__(self):
        self.agent_decision = AgentDecisoinConfig()
        self.conversation = ConversationConfig()
        self.rag = RAGConfig()
        self.medical_cv = MedicalCVConfig()
        self.web_search = WebSearchConfig()
        self.api = APIConfig()
        self.speech = SpeechConfig()
        self.validation = ValidationConfig()
        self.ui = UIConfig()

        # API keys
        self.eleven_labs_api_key = os.getenv("ELEVEN_LABS_API_KEY")
        self.tavily_api_key = os.getenv("TAVILY_API_KEY")

        # Conversation memory limit
        self.max_conversation_history = 20
