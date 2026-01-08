import requests
from langchain_community.tools.tavily_search import TavilySearchResults


class TavilySearchAgent:
    """
    TavilySearchAgent provides a lightweight interface for performing
    general-purpose web searches using the Tavily API.

    Purpose:
    - Retrieve up-to-date, real-world information from the web
    - Support queries that cannot be answered by local RAG knowledge
    - Return structured search results suitable for downstream LLM processing

    This agent is typically used as:
    - A fallback when RAG confidence is low
    - A primary source for time-sensitive or evolving topics
    """

    def __init__(self):
        """
        Initialize the Tavily search agent.

        This agent is stateless and relies on TavilySearchResults
        for executing web searches.
        """
        pass

    def search_tavily(self, query: str) -> str:
        """
        Perform a general web search using the Tavily API.

        This method:
        - Cleans the query string
        - Executes a web search using Tavily
        - Formats results into a readable, structured string

        Args:
            query: User query requiring web-based information

        Returns:
            A formatted string containing search results including:
            - Title
            - URL
            - Content summary
            - Relevance score

            Returns a descriptive message if no results or an error occurs.
        """

        # Initialize Tavily search tool with result limit
        tavily_search = TavilySearchResults(max_results=5)

        try:
            # Clean query by removing surrounding quotes if present
            query = query.strip('"\'')
            
            # Execute web search
            search_docs = tavily_search.invoke(query)

            if search_docs:
                # Format search results for downstream processing
                return "\n".join([
                    "title: " + str(res.get("title")) + " - "
                    "url: " + str(res.get("url")) + " - "
                    "content: " + str(res.get("content")) + " - "
                    "score: " + str(res.get("score"))
                    for res in search_docs
                ])

            return "No relevant results found."

        except Exception as e:
            return f"Error retrieving web search results: {e}"
