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

class AgentState(MessagesState):
    """
    State container for the LangGraph workflow.

    Inherits LangGraph's MessagesState to keep a running list of chat messages
    and extends it with routing and agent execution metadata.
    """
    agent_name: Optional[str]  # Name of the selected/active agent for current turn
    current_input: Optional[Union[str, Dict]]  # Raw user input (text or structured dict with image/text)
    has_image: bool  # True if the current_input includes an image payload
    image_type: Optional[str]  # Detected image type (e.g., "chest_xray") if present
    output: Optional[str]  # Final response (string or AIMessage)
    needs_human_validation: bool  # Flag indicating whether human validation is required
    retrieval_confidence: float  # RAG confidence score for retrieval quality
    bypass_routing: bool  # If True, skip routing and go directly to guardrails/output handling
    insufficient_info: bool  # True if RAG response indicates missing context to answer reliably


class AgentDecision(TypedDict):
    """
    Expected JSON schema returned by the decision routing LLM.
    """
    agent: str
    reasoning: str
    confidence: float

def create_agent_graph():
    """
    Build and compile the LangGraph workflow that orchestrates:
    1) Input analysis (guardrails + image detection)
    2) Agent routing via an LLM decision step
    3) Agent execution (Conversation / RAG / Web Search / Chest X-ray)
    4) Optional human validation for high-risk outputs
    5) Output guardrails and finalization

    Returns:
        A compiled LangGraph runnable with memory checkpointing enabled.
    """

    # Initialize local guardrails (input/output safety filters)
    guardrails = LocalGuardrails(config.rag.llm)

    # Decision-making model (router LLM)
    decision_model = config.agent_decision.llm

    # Parse router output into the AgentDecision JSON schema
    json_parser = JsonOutputParser(pydantic_object=AgentDecision)

    # Router prompt: system-level routing policy + user input payload
    decision_prompt = ChatPromptTemplate.from_messages([
        ("system", AgentConfig.DECISION_SYSTEM_PROMPT),
        ("human", "{input}")
    ])

    # Decision chain: prompt -> router LLM -> JSON parsing
    decision_chain = decision_prompt | decision_model | json_parser

    # ---------------------------------------------------------------------
    # Node 1: Analyze input (guardrails + image detection)
    # ---------------------------------------------------------------------
    def analyze_input(state: AgentState) -> AgentState:
        """
        Inspect current_input to:
        - Extract text for input guardrails
        - Detect whether an image is present
        - If image exists, classify its type via image analyzer

        If input guardrails block the request, routing is bypassed and the flow
        is redirected to output guardrails for a safe response.
        """
        current_input = state["current_input"]
        has_image = False
        image_type = None

        # Normalize user text from either raw string input or dict payload
        input_text = ""
        if isinstance(current_input, str):
            input_text = current_input
        elif isinstance(current_input, dict):
            input_text = current_input.get("text", "")

        # Apply input guardrails when text exists
        if input_text:
            is_allowed, message = guardrails.check_input(input_text)
            if not is_allowed:
                # Blocked input: bypass routing and return guardrail message
                print("Selected agent: INPUT_GUARDRAILS, Message:", message)
                return {
                    **state,
                    "messages": message,
                    "agent_name": "INPUT_GUARDRAILS",
                    "has_image": False,
                    "image_type": None,
                    "bypass_routing": True
                }

        # Detect image in structured input payload
        if isinstance(current_input, dict) and "image" in current_input:
            has_image = True
            image_path = current_input.get("image")
            image_type_response = AgentConfig.image_analyzer.analyze_image(image_path)
            image_type = image_type_response.get("image_type")
            print("ANALYZED IMAGE TYPE:", image_type)

        return {
            **state,
            "has_image": has_image,
            "image_type": image_type,
            "bypass_routing": False
        }
    
    # ---------------------------------------------------------------------
    # Routing gate: if bypass_routing is True, skip router and apply guardrails
    # ---------------------------------------------------------------------
    def check_if_bypassing(state: AgentState) -> str:
        """
        Determines whether routing should be skipped.
        - If bypass_routing: go directly to apply_guardrails.
        - Else: proceed to route_to_agent.
        """
        return "apply_guardrails" if state.get("bypass_routing", False) else "route_to_agent"

    # ---------------------------------------------------------------------
    # Node 2: Decide which agent should handle the request
    # ---------------------------------------------------------------------
    def route_to_agent(state: AgentState) -> Dict:
        """
        Uses the router LLM to select an agent based on:
        - current user query
        - recent conversation context
        - presence and type of image

        Returns a dict containing:
        - agent_state: updated workflow state
        - next: node name to transition to
        """
        messages = state["messages"]
        current_input = state["current_input"]
        has_image = state["has_image"]
        image_type = state["image_type"]

        # Normalize user text
        input_text = ""
        if isinstance(current_input, str):
            input_text = current_input
        elif isinstance(current_input, dict):
            input_text = current_input.get("text", "")

        # Include recent context (last 3 exchanges = 6 messages)
        recent_context = ""
        for msg in messages[-6:]:
            if isinstance(msg, HumanMessage):
                recent_context += f"User: {msg.content}\n"
            elif isinstance(msg, AIMessage):
                recent_context += f"Assistant: {msg.content}\n"

        decision_input = f"""
        User query: {input_text}

        Recent conversation context:
        {recent_context}

        Has image: {has_image}
        Image type: {image_type if has_image else 'None'}

        Based on this information, which agent should handle this query?
        """.strip()

        # Invoke routing chain and capture decision
        decision = decision_chain.invoke({"input": decision_input})
        print(f"Decision: {decision['agent']}")

        updated_state = {**state, "agent_name": decision["agent"]}

        # If router confidence is low, route to validation path
        if decision["confidence"] < AgentConfig.CONFIDENCE_THRESHOLD:
            return {"agent_state": updated_state, "next": "needs_validation"}

        # Otherwise route directly to the chosen agent node
        return {"agent_state": updated_state, "next": decision["agent"]}

    # ---------------------------------------------------------------------
    # Agent Node: Conversation Agent
    # ---------------------------------------------------------------------
    def run_conversation_agent(state: AgentState) -> AgentState:
        """
        Handles general dialogue and low-risk medical conversations.
        Uses full conversation history (currently) to keep context.
        """
        print("Selected agent: CONVERSATION_AGENT")

        messages = state["messages"]
        current_input = state["current_input"]

        # Normalize user text
        input_text = current_input if isinstance(current_input, str) else current_input.get("text", "")

        # Build conversation context
        recent_context = ""
        for msg in messages:
            if isinstance(msg, HumanMessage):
                recent_context += f"User: {msg.content}\n"
            elif isinstance(msg, AIMessage):
                recent_context += f"Assistant: {msg.content}\n"

        # Instruction-heavy prompt to keep assistant aligned for conversation mode
        conversation_prompt = f"""User query: {input_text}

        Recent conversation context: {recent_context}

        You are an AI-powered Medical Conversation Assistant operating within a multi-agent clinical AI system.

        Your primary role is to engage users in clear, safe, and informative conversations while supporting medical understanding. 
        You must respond naturally and professionally, prioritize clarity and user safety, and strictly respect medical and ethical boundaries.

        You are NOT a diagnostic authority and must never replace a licensed healthcare professional.

        ────────────────────────────────────
        CORE RESPONSIBILITIES
        ────────────────────────────────────

        • Handle general conversation, greetings, and casual interactions in a polite and engaging manner  
        • Answer general medical questions using established, non-speculative medical knowledge  
        • Maintain conversational context across multiple turns to support follow-up questions  
        • Identify when a query exceeds conversational scope and should be handled by:
        – Retrieval-Augmented Generation (RAG)
        – Web Search for recent or time-sensitive information
        – Medical computer vision agents for image-based analysis  
        • Support users in understanding outputs generated by other agents without adding new medical conclusions  

        ────────────────────────────────────
        STRICT OPERATIONAL RULES
        ────────────────────────────────────

        1. MEDICAL SAFETY FIRST  
        • Do NOT provide medical diagnoses, treatment plans, or prescriptions  
        • Do NOT speculate or infer beyond the provided information  
        • When symptoms are serious, persistent, or unclear, advise consulting a licensed healthcare professional  

        2. IMAGE HANDLING  
        • Never analyze, interpret, or diagnose medical images yourself  
        • If a user asks about detecting, classifying, segmenting, or diagnosing disease from an image:
            – Ask the user to upload the image
            – Explain that a specialized medical vision agent will handle the analysis  
        • If an image has already been analyzed by a vision agent:
            – Read prior messages
            – Help the user understand the reported findings without modifying or extending them  

        3. RESPONSE QUALITY  
        • Be clear, concise, and factual  
        • Ask clarifying questions if the user’s intent is ambiguous  
        • Avoid alarmist language or definitive claims  
        • Clearly distinguish between general information and professional medical advice  

        ────────────────────────────────────
        RESPONSE GUIDELINES
        ────────────────────────────────────

        GENERAL CONVERSATION  
        • Respond naturally and politely to greetings and small talk  
        • Keep answers brief unless additional detail is requested  

        MEDICAL QUESTIONS  
        • Answer only when confidence is high and information is well-established  
        • Use neutral, educational language  
        • When applicable, reference reputable sources (e.g., CDC, WHO, Mayo Clinic)  
        • If unsure, explicitly state uncertainty  

        FOLLOW-UPS & CONTEXT  
        • Use prior conversation context to maintain continuity  
        • Ask focused follow-up questions when necessary before responding  

        ────────────────────────────────────
        OUTPUT STYLE
        ────────────────────────────────────

        • Conversational yet professional tone  
        • Use bullet points or numbered lists for clarity when helpful  
        • Cite sources when information comes from RAG or Web Search  
        • Never overstate certainty  

        ────────────────────────────────────
        EXAMPLES
        ────────────────────────────────────

        User: "Hey, how’s your day going?"  
        Assistant: "I’m here and ready to help. How can I assist you today?"

        User: "I have a headache and fever. What should I do?"  
        Assistant: "Headache and fever can have many causes, such as infections or dehydration. I’m not a doctor, but if your symptoms persist or worsen, it’s important to consult a healthcare professional."

        User: "Can you detect COVID from this X-ray?"  
        Assistant: "I can’t analyze medical images directly. If you upload the X-ray, a specialized medical imaging agent can analyze it for you."

        ────────────────────────────────────
        FINAL INSTRUCTION
        ────────────────────────────────────

        Respond only as a conversational medical assistant.
        Do not perform routing logic explicitly.
        Do not analyze images.
        Do not provide diagnoses.
        Prioritize safety, clarity, and professionalism in every response.

        Conversational LLM Response:"""


        response = config.conversation.llm.invoke(conversation_prompt)

        return {
            **state,
            "output": response,
            "agent_name": "CONVERSATION_AGENT"
        }
    
    # ---------------------------------------------------------------------
    # Agent Node: RAG Agent
    # ---------------------------------------------------------------------
    def run_rag_agent(state: AgentState) -> AgentState:
        """
        Handles medical knowledge queries using Retrieval-Augmented Generation.
        Tracks retrieval confidence and flags insufficient-information responses.
        """
        print("Selected agent: RAG_AGENT")

        rag_agent = MedicalRAG(config)

        messages = state["messages"]
        query = state["current_input"]
        rag_context_limit = config.rag.context_limit

        # Build limited chat history context for retrieval grounding
        recent_context = ""
        for msg in messages[-rag_context_limit:]:
            if isinstance(msg, HumanMessage):
                recent_context += f"User: {msg.content}\n"
            elif isinstance(msg, AIMessage):
                recent_context += f"Assistant: {msg.content}\n"

        response = rag_agent.process_query(query, chat_history=recent_context)
        retrieval_confidence = response.get("confidence", 0.0)

        print(f"Retrieval Confidence: {retrieval_confidence}")
        print(f"Sources: {len(response.get('sources', []))}")

        # Normalize response text
        response_content = response.get("response", "")
        response_text = response_content.content if hasattr(response_content, "content") else response_content

        # Heuristic detection of insufficient context responses
        insufficient_info = False
        if isinstance(response_text, str):
            low_info_markers = [
                "i don't have enough information",
                "not enough information",
                "insufficient information",
                "cannot answer",
                "unable to answer",
            ]
            if any(m in response_text.lower() for m in low_info_markers):
                insufficient_info = True
                print("RAG response indicates insufficient information")

        # Only return content if confidence meets minimum threshold
        if retrieval_confidence >= config.rag.min_retrieval_confidence:
            response_output = AIMessage(content=response_text)
        else:
            response_output = AIMessage(content="")

        return {
            **state,
            "output": response_output,
            "needs_human_validation": False,
            "retrieval_confidence": retrieval_confidence,
            "agent_name": "RAG_AGENT",
            "insufficient_info": insufficient_info
        }

    # ---------------------------------------------------------------------
    # Agent Node: Web Search Processor
    # ---------------------------------------------------------------------
    def run_web_search_processor_agent(state: AgentState) -> AgentState:
        """
        Handles time-sensitive queries by:
        - retrieving web results via WebSearchProcessorAgent
        - synthesizing and refining answers using an LLM
        """
        print("Selected agent: WEB_SEARCH_PROCESSOR_AGENT")

        messages = state["messages"]
        web_search_context_limit = config.web_search.context_limit

        # Build limited chat history context
        recent_context = ""
        for msg in messages[-web_search_context_limit:]:
            if isinstance(msg, HumanMessage):
                recent_context += f"User: {msg.content}\n"
            elif isinstance(msg, AIMessage):
                recent_context += f"Assistant: {msg.content}\n"

        web_search_processor = WebSearchProcessorAgent(config)
        processed_response = web_search_processor.process_web_search_results(
            query=state["current_input"],
            chat_history=recent_context
        )

        # Track involved agents for observability/debugging
        involved_agents = (
            f"{state['agent_name']}, WEB_SEARCH_PROCESSOR_AGENT"
            if state.get("agent_name") else "WEB_SEARCH_PROCESSOR_AGENT"
        )

        return {
            **state,
            "output": processed_response,
            "agent_name": involved_agents
        }

    # ---------------------------------------------------------------------
    # Conditional routing after RAG: if low confidence or insufficient info, fall back to Web Search
    # ---------------------------------------------------------------------
    def confidence_based_routing(state: AgentState) -> str:
        """
        If RAG confidence is below threshold OR response indicates insufficient information,
        route to WEB_SEARCH_PROCESSOR_AGENT. Otherwise, proceed to validation stage.
        """
        print("Routing check - Retrieval confidence:", state.get("retrieval_confidence", 0.0))
        print("Routing check - Insufficient info flag:", state.get("insufficient_info", False))

        if (
            state.get("retrieval_confidence", 0.0) < config.rag.min_retrieval_confidence
            or state.get("insufficient_info", False)
        ):
            print("Re-routed to Web Search Agent due to low confidence or insufficient information...")
            return "WEB_SEARCH_PROCESSOR_AGENT"

        return "check_validation"

    # ---------------------------------------------------------------------
    # Agent Node: Chest X-ray Analysis
    # ---------------------------------------------------------------------
    def run_chest_xray_agent(state: AgentState) -> AgentState:
        """
        Performs chest X-ray classification using the vision pipeline.
        Always requires human validation because it resembles a diagnostic output.
        """
        print("Selected agent: CHEST_XRAY_AGENT")

        current_input = state["current_input"]
        image_path = current_input.get("image")

        predicted_class = AgentConfig.image_analyzer.classify_chest_xray(image_path)

        if predicted_class == "covid19":
            response = AIMessage(content="The uploaded chest X-ray analysis indicates a POSITIVE result for COVID-19.")
        elif predicted_class == "normal":
            response = AIMessage(content="The uploaded chest X-ray analysis indicates a NEGATIVE result for COVID-19, consistent with NORMAL.")
        else:
            response = AIMessage(content="The uploaded image is unclear or not recognized as a valid medical image for analysis.")

        return {
            **state,
            "output": response,
            "needs_human_validation": True,
            "agent_name": "CHEST_XRAY_AGENT"
        }

    # ---------------------------------------------------------------------
    # Validation routing node
    # ---------------------------------------------------------------------
    def handle_human_validation(state: AgentState) -> Dict:
        """
        Routes to human validation UI step when required.
        Otherwise, terminates the agent phase and proceeds to output guardrails.
        """
        if state.get("needs_human_validation", False):
            return {"agent_state": state, "next": "human_validation", "agent": "HUMAN_VALIDATION"}
        return {"agent_state": state, "next": END}

    # ---------------------------------------------------------------------
    # Human validation node
    # ---------------------------------------------------------------------
    def perform_human_validation(state: AgentState) -> AgentState:
        """
        Adds a human validation instruction block to the response.
        This is intended for UI workflows where a user can confirm or reject.
        """
        print("Selected agent: HUMAN_VALIDATION")

        validation_prompt = (
            f"{state['output'].content}\n\nHuman Validation Required:\n"
            "- If you are a healthcare professional: validate the output. Reply Yes or No. If No, add comments.\n"
            "- If you are a patient: reply Yes to confirm."
        )

        return {
            **state,
            "output": AIMessage(content=validation_prompt),
            "agent_name": f"{state['agent_name']}, HUMAN_VALIDATION"
        }

    # ---------------------------------------------------------------------
    # Output guardrails node
    # ---------------------------------------------------------------------
    def apply_output_guardrails(state: AgentState) -> AgentState:
        """
        Applies output guardrails for safety, formatting, and policy alignment.
        Also supports post-validation handling if the previous step requested validation.
        """
        output = state.get("output")
        current_input = state.get("current_input")

        # Defensive checks: only proceed if output is string-like
        if not output or not isinstance(output, (str, AIMessage)):
            return state

        output_text = output if isinstance(output, str) else output.content

        # Handle human validation responses: if user replied Yes/No, append to history
        if "Human Validation Required" in output_text:
            validation_input = current_input if isinstance(current_input, str) else current_input.get("text", "")
            if validation_input and validation_input.lower().startswith(("yes", "no")):
                validation_response = HumanMessage(content=f"Validation Result: {validation_input}")

                # If validator rejected, override with a safe fallback message
                if validation_input.lower().startswith("no"):
                    fallback_message = AIMessage(
                        content="The previous medical analysis requires further review. A healthcare professional flagged potential inaccuracies."
                    )
                    return {
                        **state,
                        "messages": [validation_response, fallback_message],
                        "output": fallback_message
                    }

                return {**state, "messages": validation_response}

        # Normalize original input text for guardrails context
        input_text = current_input if isinstance(current_input, str) else current_input.get("text", "")

        # Sanitize output using guardrails policies
        sanitized_output = guardrails.check_output(output_text, input_text)
        sanitized_message = AIMessage(content=sanitized_output) if isinstance(output, AIMessage) else sanitized_output

        return {
            **state,
            "messages": sanitized_message,
            "output": sanitized_message
        }

    # ---------------------------------------------------------------------
    # Build workflow graph
    # ---------------------------------------------------------------------
    workflow = StateGraph(AgentState)

    # Register nodes
    workflow.add_node("analyze_input", analyze_input)
    workflow.add_node("route_to_agent", route_to_agent)
    workflow.add_node("CONVERSATION_AGENT", run_conversation_agent)
    workflow.add_node("RAG_AGENT", run_rag_agent)
    workflow.add_node("WEB_SEARCH_PROCESSOR_AGENT", run_web_search_processor_agent)
    workflow.add_node("CHEST_XRAY_AGENT", run_chest_xray_agent)
    workflow.add_node("check_validation", handle_human_validation)
    workflow.add_node("human_validation", perform_human_validation)
    workflow.add_node("apply_guardrails", apply_output_guardrails)

    # Entry point
    workflow.set_entry_point("analyze_input")

    # Conditional route after input analysis:
    # - If blocked by guardrails: apply_guardrails
    # - Else: proceed to agent routing
    workflow.add_conditional_edges(
        "analyze_input",
        check_if_bypassing,
        {
            "apply_guardrails": "apply_guardrails",
            "route_to_agent": "route_to_agent",
        }
    )

    # Conditional route from router to agent node
    workflow.add_conditional_edges(
        "route_to_agent",
        lambda x: x["next"],
        {
            "CONVERSATION_AGENT": "CONVERSATION_AGENT",
            "RAG_AGENT": "RAG_AGENT",
            "WEB_SEARCH_PROCESSOR_AGENT": "WEB_SEARCH_PROCESSOR_AGENT",
            "CHEST_XRAY_AGENT": "CHEST_XRAY_AGENT",
            "needs_validation": "RAG_AGENT",  # Fallback path for low-confidence router results
        }
    )

    # Connect agent outputs into validation
    workflow.add_edge("CONVERSATION_AGENT", "check_validation")
    workflow.add_edge("WEB_SEARCH_PROCESSOR_AGENT", "check_validation")
    workflow.add_edge("CHEST_XRAY_AGENT", "check_validation")

    # After RAG: either go to Web Search fallback or to validation
    workflow.add_conditional_edges("RAG_AGENT", confidence_based_routing)

    # Validation and guardrails pipeline
    workflow.add_edge("human_validation", "apply_guardrails")
    workflow.add_edge("apply_guardrails", END)

    # Conditional branch from validation check:
    # - If needs human validation: human_validation
    # - Else: apply output guardrails and end
    workflow.add_conditional_edges(
        "check_validation",
        lambda x: x["next"],
        {
            "human_validation": "human_validation",
            END: "apply_guardrails",
        }
    )

    # Compile graph with checkpointing for persistent state
    return workflow.compile(checkpointer=memory)

def init_agent_state() -> AgentState:
    """
    Create a fresh AgentState object with safe defaults.

    This state is the single source of truth for the LangGraph workflow and includes:
    - message history (chat memory)
    - routing metadata (selected agent, image flags)
    - output payload (final response)
    - confidence and safety flags (RAG confidence, validation requirements)
    """
    return {
        "messages": [],                    # Conversation history (LangChain BaseMessage list)
        "agent_name": None,                # Active agent selected for the current turn
        "current_input": None,             # Raw input payload (str or dict containing text/image)
        "has_image": False,                # True if current_input includes an image
        "image_type": None,                # Detected medical image type (e.g., "chest_xray")
        "output": None,                    # Final output message returned by the selected agent
        "needs_human_validation": False,   # True when high-risk output requires human confirmation
        "retrieval_confidence": 0.0,       # RAG confidence score for the current response
        "bypass_routing": False,           # Used to skip routing when guardrails block input
        "insufficient_info": False         # True when RAG signals insufficient context to answer
    }


def process_query(query: Union[str, Dict], conversation_history: List[BaseMessage] = None) -> str:
    """
    Process a user query through the multi-agent orchestration graph.

    The function:
    1) Builds the agent LangGraph workflow (router + agent nodes)
    2) Initializes a new AgentState and injects the user's current input
    3) Stores the current user message into state["messages"] for context
    4) Executes the graph to produce a final output
    5) Trims message history to a configurable limit for memory control
    6) Prints the resulting conversation history for debugging/observability

    Args:
        query:
            User input payload. Can be:
            - str: plain text
            - dict: structured payload that may include:
              - "text": user text
              - "image": image file path or identifier
        conversation_history:
            Legacy parameter. Not required if you rely on LangGraph state persistence.
            Kept for backward compatibility.

    Returns:
        The full graph result (AgentState-like dict) containing:
        - messages: updated conversation history
        - output: final assistant message
        - agent_name: agent(s) involved in producing the response
        - other routing and confidence metadata
    """
    # Build and compile the agent orchestration graph
    graph = create_agent_graph()

    # Optional: export a visual diagram of the graph for debugging and documentation
    # NOTE: This runs on every call. For production, consider gating behind a flag.
    image_bytes = graph.get_graph().draw_mermaid_png()
    decoded = cv2.imdecode(np.frombuffer(image_bytes, np.uint8), -1)
    cv2.imwrite("./assets/graph.png", decoded)
    print("Graph flowchart saved in assets.")

    # Initialize a fresh workflow state
    state = init_agent_state()

    # If you want to seed with external history, uncomment below.
    # Typically unnecessary if you are using persistent checkpointers.
    # if conversation_history:
    #     state["messages"] = conversation_history

    # Persist raw current input (keeps image payload intact if present)
    state["current_input"] = query

    # Normalize the visible user message for message history.
    # If an image payload is present, append a short diagnostic hint for routing context.
    if isinstance(query, dict):
        visible_user_text = query.get("text", "")
        visible_user_text = f"{visible_user_text}, user uploaded an image for diagnosis."
    else:
        visible_user_text = query

    # Store the user message as the latest turn in the message history
    state["messages"] = [HumanMessage(content=visible_user_text)]

    # Execute the graph. thread_config enables state persistence by thread_id.
    result = graph.invoke(state, thread_config)

    # Enforce a maximum conversation history to prevent unbounded memory growth
    # Keeps only the most recent messages.
    if len(result["messages"]) > config.max_conversation_history:
        result["messages"] = result["messages"][-config.max_conversation_history:]

    # Debugging: print conversation history in the console
    for m in result["messages"]:
        m.pretty_print()

    # Return the full structured result (includes output + metadata)
    return result
