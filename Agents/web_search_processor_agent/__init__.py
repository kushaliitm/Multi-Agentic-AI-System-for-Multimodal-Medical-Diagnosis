from typing import List, Dict, Optional

from .web_search_processor import WebSearchProcessor


class WebSearchProcessorAgent:
    """
    WebSearchProcessorAgent acts as a thin orchestration layer for web-based information retrieval.

    Responsibilities:
    - Delegate web search result processing to the WebSearchProcessor component
    - Provide a clean agent-style interface consistent with the multi-agent architecture
    - Support optional conversation history for contextual response generation

    This agent is typically invoked when:
    - The user query requires up-to-date or time-sensitive information
    - Retrieved knowledge is not available in the local RAG knowledge base
    """

    def __init__(self, config):
        """
        Initialize the web search processor agent.

        Args:
            config: Application configuration object used to initialize
                    the underlying WebSearchProcessor (API keys, LLM, etc.)
        """
        self.web_search_processor = WebSearchProcessor(config)

    def process_web_search_results(
        self,
        query: str,
        chat_history: Optional[List[Dict[str, str]]] = None
    ) -> str:
        """
        Process web search results and generate a user-facing response.

        This method forwards the query and optional conversation history
        to the underlying WebSearchProcessor, which handles:
        - Web search execution
        - Result summarization
        - LLM-based response generation

        Args:
            query: User query requiring web-based information.
            chat_history: Optional list of prior conversation messages
                          to preserve context.

        Returns:
            A user-friendly, natural language response generated from
            processed web search results.
        """
        return self.web_search_processor.process_web_results(query, chat_history)
