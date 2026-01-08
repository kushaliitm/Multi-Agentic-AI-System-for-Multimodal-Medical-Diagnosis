import json
import os
import getpass
from typing import Dict, List, Optional, Any, Literal, TypedDict, Union, Annotated
import cv2
import numpy as np
from dotenv import load_dotenv
from langchain_core.messages import (
    HumanMessage,
    AIMessage,
    SystemMessage,
    BaseMessage
)
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import JsonOutputParser
from langchain_core.runnables import RunnablePassthrough
from langgraph.graph import MessagesState, StateGraph, END
from langgraph.checkpoint.memory import MemorySaver
from agents.rag_agent import MedicalRAG
from agents.web_search_processor_agent import WebSearchProcessorAgent
from agents.image_analysis_agent import ImageAnalysisAgent
from agents.guardrails.local_guardrails import LocalGuardrails
from config import Config


# Load environment variables from .env file
# override=True ensures environment variables are refreshed
load_dotenv(override=True)

# Initialize global configuration object
config = Config()

# Memory saver for LangGraph to persist conversation state
memory = MemorySaver()

# Thread configuration for multi-session or concurrent execution
thread_config = {
    "configurable": {
        "thread_id": "1"
    }
}


class AgentConfig:
    """
    Central configuration class for agent routing and decision-making logic.

    This class defines:
    - Model selection for decision and vision agents
    - Confidence thresholds for routing reliability
    - System prompt used by the decision-making LLM
    - Shared agent instances used across the graph
    """

    # Model used for agent routing and decision logic
    DECISION_MODEL = os.getenv("model_name")

    # Model used for medical vision-based analysis
    VISION_MODEL = os.getenv("model_name")

    # Minimum confidence required to accept routing decision
    CONFIDENCE_THRESHOLD = 0.85

    # System prompt that governs the agent routing behavior
    DECISION_SYSTEM_PROMPT = """

    You are an intelligent medical triage and routing system operating within a multi-agent clinical AI framework. 
    Your responsibility is to analyze each user interaction and determine the most appropriate specialized agent 
    to handle the request based on the user’s intent, uploaded inputs, medical relevance, and temporal sensitivity.

    Your decision must prioritize patient safety, clarity of intent, and correct agent specialization.

    ────────────────────────────────────
    AVAILABLE AGENTS
    ────────────────────────────────────

    1. CONVERSATION_AGENT  
    Handles general conversation, greetings, clarifications, follow-up questions, and non-medical topics.

    2. RAG_AGENT  
    Handles factual, non-urgent medical knowledge questions that can be answered using established medical literature.  
    Current knowledge base includes:
    - Deep learning methods for COVID / COVID-19 detection from chest X-ray  

    3. WEB_SEARCH_PROCESSOR_AGENT  
    Handles questions requiring up-to-date, time-sensitive, or real-world information, such as:
    - Recent medical discoveries or guidelines  
    - Current outbreaks or epidemiological updates  
    - Newly released treatments or diagnostics  

    4. CHEST_XRAY_AGENT  
    Handles analysis and interpretation of uploaded chest X-ray images, including detection of abnormalities 
    and medically relevant patterns.

    ────────────────────────────────────
    ROUTING RULES (STRICT PRIORITY ORDER)
    ────────────────────────────────────

    1. IMAGE-FIRST RULE  
    If the user uploads any medical image, ALWAYS route to the appropriate medical vision agent, 
    regardless of text content.  
    - If the image is a chest X-ray, route to CHEST_XRAY_AGENT.
    - If an image is uploaded without an accompanying question, still route to the correct vision agent.

    2. NO-IMAGE RULE  
    If no image is uploaded:
    - Route to CONVERSATION_AGENT for general chat or non-medical questions.
    - Route to RAG_AGENT for medical knowledge questions based on established literature.
    - Route to WEB_SEARCH_PROCESSOR_AGENT for recent, evolving, or time-sensitive medical topics.

    3. TEMPORAL SENSITIVITY RULE  
    If the query depends on current events, recent publications, or real-time health situations, 
    prefer WEB_SEARCH_PROCESSOR_AGENT over RAG_AGENT.

    4. DEFAULT SAFETY RULE  
    When uncertainty exists and no image is present, prefer CONVERSATION_AGENT unless the medical intent 
    is explicit and well-defined.

    ────────────────────────────────────
    OUTPUT FORMAT (MANDATORY)
    ────────────────────────────────────
    You must return your decision in valid JSON format only, using the structure below:

    {{
    "agent": "AGENT_NAME",
    "reasoning": "Your step-by-step reasoning for selecting this agent",
    "confidence": 0.95  // Value between 0.0 and 1.0 indicating your confidence in this decision
    }}
    
    - The agent field must exactly match one of the available agent names.
    - The reasoning must explicitly reference the applied routing rules.
    - Confidence must be a float between 0.0 and 1.0.

    Do not include any additional text outside the JSON response.
    """

    # Shared image analysis agent instance
    # Responsible for handling all medical vision tasks
    image_analyzer = ImageAnalysisAgent(config=config)
