import requests
from typing import Dict

from .pubmed_search import PubmedSearchAgent
from .tavily_search import TavilySearchAgent


class WebSearchAgent:
    """
    WebSearchAgent retrieves real-time information from external web sources.

    Purpose:
    - Fetch up-to-date information when local RAG knowledge is insufficient
    - Support medical and general queries requiring current data
    - Serve as a fallback or augmentation layer in an agentic RAG pipeline

    Current Implementation:
    - Uses Tavily for general web search
    - PubMed integration is scaffolded but currently disabled
    """

    def __init__(self, config):
        """
        Initialize the web search agent.

        Args:
            config: Application configuration object.
                    Used for future integrations such as PubMed API access.
        """
        self.tavily_search_agent = TavilySearchAgent()

        # PubMed support can be enabled when required
        # self.pubmed_search_agent = PubmedSearchAgent()
        # self.pubmed_api_url = config.pubmed_api_url

    def search(self, query: str) -> str:
        """
        Execute web-based searches for the given query.

        Currently performs:
        - General web search via Tavily

        Future extensions may include:
        - PubMed literature search
        - Multiple source aggregation
        - Source ranking and deduplication

        Args:
            query: User query requiring real-time or external information

        Returns:
            A formatted string containing aggregated web search results
        """

        # Execute Tavily web search
        tavily_results = self.tavily_search_agent.search_tavily(query=query)

        # Future PubMed integration
        # pubmed_results = self.pubmed_search_agent.search_pubmed(self.pubmed_api_url, query)

        return f"Tavily Results:\n{tavily_results}\n"
        # To include PubMed later:
        # return f"Tavily Results:\n{tavily_results}\n\nPubMed Results:\n{pubmed_results}"
