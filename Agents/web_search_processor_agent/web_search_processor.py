import os
from typing import Dict, List, Optional
from dotenv import load_dotenv

from .web_search_agent import WebSearchAgent

load_dotenv(override=True)


class WebSearchProcessor:
    """
    WebSearchProcessor orchestrates query refinement, web search execution,
    and LLM-based summarization of real-time web results.

    Responsibilities:
    - Refine the user query using conversation context (if relevant)
    - Fetch real-time web search results via WebSearchAgent
    - Summarize and contextualize results using an LLM
    - Produce a concise, medically accurate response for the user

    This component is typically invoked when:
    - Local RAG knowledge is insufficient or outdated
    - The query involves recent developments or real-time information
    """

    def __init__(self, config):
        """
        Initialize the WebSearchProcessor.

        Args:
            config: Application configuration object containing:
                    - web search LLM
                    - external API credentials (if applicable)
        """
        self.web_search_agent = WebSearchAgent(config)

        # LLM used to refine queries and summarize web search results
        self.llm = config.web_search.llm

    def _build_prompt_for_web_search(
        self,
        query: str,
        chat_history: Optional[List[Dict[str, str]]] = None
    ) -> str:
        """
        Build a prompt to refine the user's query for effective web searching.

        The LLM determines whether prior conversation context is relevant and,
        if so, synthesizes it into a single, well-formed search query.

        Args:
            query: Current user query
            chat_history: Optional conversation history for contextual refinement

        Returns:
            A prompt string for query refinement
        """

        prompt = f"""
        Here are the last few messages from our conversation:
        {chat_history}

        The user asked the following question:
        {query}

        Task:
        - Summarize the conversation into a single, well-formed search query
          ONLY if the prior context is relevant.
        - Preserve the original intent of the user's request.
        - Keep the query concise and suitable for a web search.
        """

        return prompt

    def process_web_results(
        self,
        query: str,
        chat_history: Optional[List[Dict[str, str]]] = None
    ):
        """
        Execute the complete web search processing pipeline.

        Steps:
        1. Refine the query using LLM and conversation context
        2. Fetch real-time web search results
        3. Summarize and contextualize results using LLM
        4. Return a user-friendly response

        Args:
            query: User query requiring real-time information
            chat_history: Optional conversation history

        Returns:
            LLM-generated response summarizing the web search results
        """

        # Step 1: Refine the query for web search
        web_search_query_prompt = self._build_prompt_for_web_search(
            query=query,
            chat_history=chat_history
        )

        refined_query = self.llm.invoke(web_search_query_prompt)

        # Step 2: Retrieve web search results
        web_results = self.web_search_agent.search(refined_query.content)

        # Step 3: Build summarization prompt
        llm_prompt = (
            "You are an AI assistant specialized in medical information.\n"
            "Below are web search results retrieved for a user query.\n\n"
            "Instructions:\n"
            "- Summarize the information clearly and concisely\n"
            "- Ensure medical accuracy and clarity\n"
            "- Avoid speculation or unverified claims\n"
            "- Do not provide diagnoses or prescriptions\n\n"
            f"User Query:\n{query}\n\n"
            f"Web Search Results:\n{web_results}\n\n"
            "Response:"
        )

        # Step 4: Generate final response
        response = self.llm.invoke(llm_prompt)

        return response
