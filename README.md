# Multi-Agentic AI System for Multimodal Medical Diagnosis

## Overview

A comprehensive multi-agent AI system designed for multimodal medical diagnosis and consultation. This system intelligently routes medical queries and image uploads to specialized agents, providing evidence-based responses with human-in-the-loop validation for critical outputs.

**Key Features:**
- 🤖 Multi-agent orchestration with LangGraph
- 🏥 Specialized medical domain agents
- 🖼️ Medical image analysis (Chest X-ray COVID-19 classification)
- 📚 Retrieval-Augmented Generation (RAG) with Qdrant vector database
- 🌐 Web search integration for real-time medical information
- 🛡️ Comprehensive input/output guardrails
- ✅ Human-in-the-loop validation for high-risk outputs
- 🔒 Session management and conversation memory

---

## 📁 Project Structure

```
Multi-Agentic-AI-System-for-Multimodal-Medical-Diagnosis/
├── agents/                          # Core agent implementations
│   ├── agent_decision.py           # Agent routing logic & graph orchestration
│   ├── guardrails/
│   │   └── local_guardrails.py    # Input/output safety filters
│   ├── image_analysis_agent/
│   │   ├── __init__.py
│   │   ├── image_classifier.py    # Medical image classification
│   │   └── chest_xray_agent/      # Chest X-ray specific models
│   ├── rag_agent/
│   │   ├── __init__.py            # MedicalRAG system coordinator
│   │   ├── doc_parser.py          # PDF parsing with Docling
│   │   ├── content_processor.py   # Document formatting & chunking
│   │   ├── vectorstore_qdrant.py  # Hybrid vector store (dense + sparse)
│   │   ├── reranker.py            # Cross-encoder reranking
│   │   ├── query_expander.py      # Query expansion with medical terminology
│   │   └── response_generator.py  # RAG response generation with sources
│   └── web_search_processor_agent/
│       ├── web_search_agent.py    # Web search orchestration
│       ├── web_search_processor.py # Result processing
│       └── pubmed_search.py       # PubMed literature search (future)
├── data/
│   ├── docs_db/                   # Local document store
│   ├── qdrant_db/                 # Qdrant vector database
│   ├── parsed_docs/               # Extracted document content
│   └── raw/                       # Raw ingestion files
├── sample_images/
│   └── chest_x-ray_covid_and_normal/  # Test images
├── uploads/
│   ├── backend/                   # Temporary backend uploads
│   ├── frontend/                  # Frontend-accessible uploads
│   └── speech/                    # Generated speech files
├── templates/
│   └── index.html                 # Web UI (Bootstrap + Vanilla JS)
├── app.py                         # FastAPI backend server
├── config.py                      # Centralized configuration
├── rag_data_ingest.py            # Document ingestion CLI tool
├── requirements.txt               # Python dependencies
├── .env                          # Environment variables (create this)
├── .gitignore
└── README.md
```

---

## 🏗️ Architecture

### System Components

#### 1. **Agent Decision Router** ([`agents/agent_decision.py`](agents/agent_decision.py))
   - Routes user queries to the most appropriate agent
   - Implements LangGraph-based workflow orchestration
   - Manages conversation state and message history
   - Applies input/output guardrails
   - Orchestrates human validation when needed

   **Routing Logic:**
   - **IMAGE UPLOADED** → Medical vision agent (Chest X-ray, etc.)
   - **Medical Knowledge Query** → RAG Agent
   - **Time-Sensitive/Recent Info** → Web Search Agent
   - **General Chat** → Conversation Agent

#### 2. **Conversation Agent**
   - General medical conversation and clarification
   - Non-diagnostic Q&A
   - Handles follow-ups and context awareness
   - Routes complex queries to specialized agents

#### 3. **RAG Agent** ([`agents/rag_agent/`](agents/rag_agent/))
   - **Pipeline:**
     1. **Doc Parser** ([`doc_parser.py`](agents/rag_agent/doc_parser.py)): Extracts text, tables, images from PDFs using Docling
     2. **Content Processor** ([`content_processor.py`](agents/rag_agent/content_processor.py)): Summarizes images, formats documents, semantic chunking
     3. **Vector Store** ([`vectorstore_qdrant.py`](agents/rag_agent/vectorstore_qdrant.py)): Hybrid retrieval (dense embeddings + sparse BM25)
     4. **Query Expander** ([`query_expander.py`](agents/rag_agent/query_expander.py)): Enriches queries with medical terminology
     5. **Reranker** ([`reranker.py`](agents/rag_agent/reranker.py)): Cross-encoder reranking of results
     6. **Response Generator** ([`response_generator.py`](agents/rag_agent/response_generator.py)): Context-aware response generation with sources

   - **Key Features:**
     - LLM-based semantic chunking
     - Medical image summarization
     - Hybrid vector search (Qdrant)
     - Confidence scoring
     - Source attribution

#### 4. **Web Search Agent** ([`agents/web_search_processor_agent/`](agents/web_search_processor_agent/))
   - Real-time web search via Tavily
   - Medical literature search (PubMed support planned)
   - Result summarization
   - Fallback from RAG when confidence is low

#### 5. **Medical Vision Agent** ([`agents/image_analysis_agent/`](agents/image_analysis_agent/))
   - **Chest X-ray COVID-19 Classification**
   - Medical image classification pipeline
   - Outputs: POSITIVE (COVID-19), NEGATIVE (Normal), UNCLEAR
   - Requires human validation before display

#### 6. **Guardrails System** ([`agents/guardrails/local_guardrails.py`](agents/guardrails/local_guardrails.py))
   - **Input Guardrails:**
     - Blocks harmful, illegal, or unsafe requests
     - Prevents prompt injection attacks
     - Validates medical safety
   
   - **Output Guardrails:**
     - Reviews all agent responses for safety
     - Detects harmful medical advice
     - Revises unsafe content automatically
     - Adds disclaimers when needed

#### 7. **Human Validation** ([`agents/agent_decision.py`](agents/agent_decision.py))
   - Required for high-risk outputs (medical vision, certain RAG responses)
   - UI-based approval/rejection workflow
   - Feedback collection for output refinement

---

### Key Settings:

| Setting | Value | Purpose |
|---------|-------|---------|
| `RAG_TOP_K` | 5 | Number of documents to retrieve |
| `RERANKER_TOP_K` | 3 | Reranked results to use |
| `MIN_RETRIEVAL_CONFIDENCE` | 0.40 | Confidence threshold for RAG answers |
| `CHUNK_SIZE` | 512 | Document chunk size (tokens) |
| `CHUNK_OVERLAP` | 50 | Overlap between chunks |
| `COLLECTION_NAME` | `medical_assistance_rag` | Qdrant collection |
| `MAX_IMAGE_UPLOAD_SIZE` | 5 MB | Image upload limit |
| `MAX_CONVERSATION_HISTORY` | 20 | Messages to maintain |

---

## 🚀 Getting Started

### Prerequisites

- Python 3.10+
- API Keys:
  - OpenAI (GPT-4o)
  - ElevenLabs (speech synthesis)
  - Tavily (web search)

### Installation

1. **Clone the repository:**
   ```bash
   git clone <repository-url>
   cd Multi-Agentic-AI-System-for-Multimodal-Medical-Diagnosis
   ```

2. **Create virtual environment:**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

4. **Set up environment variables:**
   ```bash
   cp .env.example .env
   ```
   
   Edit `.env` with your API keys:
   ```env
   OPENAI_API_KEY=sk-...
   OPENAI_MODEL=gpt-4o
   OPENAI_EMBEDDING_MODEL=text-embedding-3-large
   ELEVEN_LABS_API_KEY=sk_...
   TAVILY_API_KEY=...
   HUGGINGFACE_TOKEN=hf_...
   QDRANT_URL=http://localhost:6333  # If using remote Qdrant
   QDRANT_API_KEY=...                 # If using remote Qdrant
   ```

5. **Ingest medical documents (optional):**
   ```bash
   # Single file
   python rag_data_ingest.py --file data/raw/medical_paper.pdf
   
   # Directory of documents
   python rag_data_ingest.py --dir data/raw/
   ```

6. **Start the backend server:**
   ```bash
   python app.py
   ```
   
   Server runs at: `http://localhost:8000`

7. **Access the web UI:**
   ```
   http://localhost:8000
   ```

---

## 📖 RAG Pipeline Details

### Document Ingestion Workflow

1. **Parsing** → Docling extracts text, tables, images from PDFs
2. **Image Summarization** → Vision LLM summarizes figures
3. **Content Formatting** → Markdown formatting with embedded summaries
4. **Semantic Chunking** → LLM-guided chunk merging/splitting
5. **Vectorization** → OpenAI embeddings (text-embedding-3-large)
6. **Storage** → Qdrant (dense + sparse vectors)

### Query Processing Workflow

1. **Input Guardrails** → Safety validation
2. **Query Expansion** → Medical terminology enrichment
3. **Retrieval** → Hybrid search (dense + BM25)
4. **Reranking** → Cross-encoder reranking
5. **Response Generation** → Context-aware LLM response
6. **Source Attribution** → Include document sources
7. **Confidence Scoring** → Retrieval-based confidence
8. **Output Guardrails** → Safety review & revision

### Vector Store Configuration

- **Database:** Qdrant
- **Dense Embeddings:** OpenAI `text-embedding-3-large` (3072-dim)
- **Sparse Embeddings:** FastEmbed BM25 (keyword matching)
- **Retrieval Mode:** Hybrid (combines both)
- **Distance Metric:** Cosine
- **Collection:** `medical_assistance_rag`


---

## 🛡️ Safety & Validation

### Input Guardrails
Blocks:
- Harmful/illegal requests
- Self-harm content
- Prompt injection attacks
- Non-medical requests
- Copyright violations
- System prompt disclosure

### Output Guardrails
Reviews for:
- Medical misinformation
- Unsafe medical advice
- Overconfident claims
- Self-harm content
- Unauthorized medical practice
- Ethical violations

### Human Validation
**Required for:**
- ✅ Chest X-ray analysis (always)
- ✅ High-risk RAG outputs (insufficient confidence)
- ⚠️ Sensitive medical responses

**Validation Workflow:**
1. System generates response
2. UI presents validation form (Yes/No)
3. Healthcare professional confirms/rejects
4. Feedback stored for monitoring

---

## 📦 Key Dependencies

| Package | Purpose |
|---------|---------|
| `fastapi` | Web server framework |
| `langchain` | LLM orchestration |
| `langgraph` | Agent graph workflow |
| `qdrant-client` | Vector database client |
| `docling` | PDF parsing & extraction |
| `openai` | GPT-4o, embeddings |
| `elevenlabs` | Speech synthesis/recognition |
| `sentence-transformers` | Cross-encoder reranking |
| `tavily-python` | Web search |
| `torch` | Medical vision models |
| `pydub` | Audio processing |

See [`requirements.txt`](requirements.txt) for complete list.

---

## 🧪 Testing & Validation

### Manual Testing

1. **Basic Chat:**
   ```
   User: "What are symptoms of COVID-19?"
   Expected: RAG_AGENT response with sources
   ```

2. **Image Upload:**
   ```
   User: [Upload chest X-ray] + "Is this COVID positive?"
   Expected: CHEST_XRAY_AGENT analysis + human validation UI
   ```

3. **Web Search:**
   ```
   User: "Latest COVID-19 treatment guidelines"
   Expected: WEB_SEARCH_PROCESSOR_AGENT response with recent data
   ```

4. **Voice:**
   ```
   User: [Speak medical question via microphone]
   Expected: Transcribed to text, processed, response played aloud
   ```

---

## 🚨 Limitations & Future Work

### Current Limitations
- ⚠️ Medical image analysis limited to chest X-rays
- ⚠️ Knowledge base limited to ingested documents
- ⚠️ Limited to English language

### Planned Enhancements
- 🔜 PubMed literature integration
- 🔜 Multiple medical imaging modalities (CT, MRI, ultrasound)
- 🔜 Multilingual support
- 🔜 Persistent conversation logging
- 🔜 Performance monitoring & observability
- 🔜 Advanced reranking models
- 🔜 Fine-tuned medical LLMs

---

## 📝 Document Ingestion Examples

### Single File
```bash
python rag_data_ingest.py --file data/raw/covid_detection_paper.pdf
```

### Directory
```bash
python rag_data_ingest.py --dir data/raw/medical_papers/
```

### Output
```
Starting document ingestion process...

Ingestion result:
{
  "success": true,
  "documents_ingested": 1,
  "chunks_processed": 125,
  "processing_time": 45.32
}
```

---


## 📞 Support & Troubleshooting

### Common Issues

**Issue:** OpenAI API key invalid
```
Solution: Check .env file has correct key
export OPENAI_API_KEY=sk-...
```

**Issue:** Document ingestion slow
```
Solution: This is expected for large PDFs
- Docling processes each page
- Vision LLM summarizes images
- Typical: 5-10 min per 50-page document
```

**Issue:** Low retrieval confidence
```
Solution: 
- Add more documents to knowledge base
- Adjust RAG_CONFIG.min_retrieval_confidence
- Check query expansion is working (logs)
```

---

## 📖 References & Resources

### Medical AI Papers
- COVID-19 Detection from Chest X-rays (training data included)
- Retrieval-Augmented Generation for Medical QA
- Medical NLP and Biomedical BERT models

### Libraries & Tools
- [LangChain Docs](https://python.langchain.com/)
- [LangGraph Docs](https://python.langchain.com/docs/langgraph/)
- [Qdrant Docs](https://qdrant.tech/documentation/)
- [Docling](https://ds4sd.github.io/docling/)
- [OpenAI API](https://platform.openai.com/docs/)

---

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

## 📧 Contact & Questions

For questions, issues, or suggestions:
- Open a GitHub issue
- Contact: [kushalpatelds@gmail.com]

---

## 🎯 Quick Start Summary

```bash
# 1. Setup
git clone <repo>
cd Multi-Agentic-AI-System-for-Multimodal-Medical-Diagnosis
python -m venv venv && source venv/bin/activate
pip install -r requirements.txt
cp .env.example .env  # Add your API keys

# 2. Ingest documents (optional)
python rag_data_ingest.py --dir data/raw/

# 3. Run server
python app.py

# 4. Access UI
# Open http://localhost:8000 in browser
```

---


