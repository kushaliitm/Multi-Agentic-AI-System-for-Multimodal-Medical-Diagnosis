import requests


class PubmedSearchAgent:
    """
    PubmedSearchAgent provides a lightweight interface for querying PubMed
    to retrieve links to relevant medical research articles.

    Purpose:
    - Search PubMed using the NCBI E-utilities API
    - Retrieve a small set of relevant article identifiers
    - Return human-readable PubMed article links
    - Serve as an external medical literature lookup component
      within a broader RAG or web-search pipeline
    """

    def __init__(self):
        """
        Initialize the PubMed search agent.

        This agent is stateless and does not require configuration at initialization.
        """
        pass

    def search_pubmed(self, pubmed_api_url: str, query: str) -> str:
        """
        Search PubMed for relevant medical articles based on a user query.

        The method uses PubMed's ESearch API endpoint to retrieve
        a limited number of article IDs and converts them into
        clickable PubMed URLs.

        Args:
            pubmed_api_url: Base URL for the PubMed ESearch API endpoint
            query: User query string representing a medical topic or question

        Returns:
            A newline-separated string of PubMed article URLs if results are found,
            or a descriptive message if no results or an error occurs.
        """
        params = {
            "db": "pubmed",
            "term": query,
            "retmode": "json",
            "retmax": 5  # Limit results to top 5 articles
        }

        try:
            response = requests.get(pubmed_api_url, params=params)
            response.raise_for_status()

            data = response.json()
            article_ids = data.get("esearchresult", {}).get("idlist", [])

            if not article_ids:
                return "No relevant PubMed articles found."

            # Construct PubMed article URLs
            article_links = [
                f"https://pubmed.ncbi.nlm.nih.gov/{article_id}/"
                for article_id in article_ids
            ]

            return "\n".join(article_links)

        except Exception as e:
            return f"Error retrieving PubMed articles: {e}"
