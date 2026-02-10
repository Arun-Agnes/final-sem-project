# 📚 AI Research Literature Assistant

An intelligent research paper analysis system built with RAG (Retrieval-Augmented Generation) that allows users to upload, index, and query academic PDF papers using natural language.

## 🎯 Overview

This application combines document processing, vector embeddings, semantic search, and LLM-powered response generation to help researchers quickly find answers from their research paper collections. It features a user-friendly Streamlit web interface and a powerful CLI for advanced queries.

## ✨ Features

- **📄 PDF Processing**: Extract and parse text from academic PDFs
- **🔍 Smart Chunking**: Intelligent document segmentation that preserves paper structure (title, abstract, sections)
- **🧠 Semantic Search**: Vector-based retrieval using sentence embeddings
- **💬 Natural Language Queries**: Ask questions in plain English and get contextual answers
- **🤖 AI-Powered Responses**: GPT-4o-mini generates comprehensive answers based on retrieved context
- **📊 Metadata Extraction**: Automatically extracts titles, authors, abstracts, and sections
- **🎨 Web Interface**: Beautiful Streamlit UI with real-time progress tracking
- **⚡ CLI Mode**: Command-line interface for batch processing and scripting

## 🏗️ Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                     User Interface Layer                     │
│  ┌──────────────────┐          ┌───────────────────────┐   │
│  │  Streamlit UI    │          │   CLI (query.py)      │   │
│  │  (app.py)        │          │   Interactive Mode    │   │
│  └──────────────────┘          └───────────────────────┘   │
└─────────────────────────────────────────────────────────────┘
                              │
┌─────────────────────────────────────────────────────────────┐
│                      RAG Pipeline Layer                      │
│  ┌──────────────────────────────────────────────────────┐  │
│  │              RAGPipeline (rag/pipeline.py)           │  │
│  └──────────────────────────────────────────────────────┘  │
│         │                                      │             │
│  ┌──────────────┐                   ┌─────────────────┐    │
│  │  Retriever   │                   │  Response Gen.  │    │
│  │ (retriever.py)│                   │  (response_gen) │    │
│  └──────────────┘                   └─────────────────┘    │
└─────────────────────────────────────────────────────────────┘
                              │
┌─────────────────────────────────────────────────────────────┐
│                   Document Processing Layer                  │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────────┐ │
│  │ PDF Loader   │→ │ Preprocessor │→ │ Embed & Store   │ │
│  │ (pdf_loader) │  │ (preprocess) │  │ (embed_store)   │ │
│  └──────────────┘  └──────────────┘  └──────────────────┘ │
└─────────────────────────────────────────────────────────────┘
                              │
┌─────────────────────────────────────────────────────────────┐
│                        Storage Layer                         │
│  ┌──────────────────┐          ┌───────────────────────┐   │
│  │   ChromaDB       │          │  Sentence Transformer │   │
│  │  Vector Store    │          │  (all-mpnet-base-v2)  │   │
│  └──────────────────┘          └───────────────────────┘   │
└─────────────────────────────────────────────────────────────┘
```

## 📦 Installation

### Prerequisites

- Python 3.8 or higher
- OpenAI API key (for GPT-powered responses)

### Setup Steps

1. **Clone the repository**:
   ```bash
   cd /path/to/ai_literature_assistant
   ```

2. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

3. **Set up environment variables**:
   Copy the example environment file:
   ```bash
   cp .env.example .env
   ```
   Open `.env` and add your OpenAI API key:
   ```
   OPENAI_API_KEY=your-api-key-here
   ```

4. **Verify directory structure**:
   The setup script automatically creates necessary directories:
   - `data/` - Main data directory
   - `data/chroma_db/` - Vector database storage
   - `data/pdfs/` - PDF storage location

## 🚀 Usage

### Web Interface (Streamlit)

Launch the interactive web application:

```bash
streamlit run app.py
```

**Features**:
- Drag-and-drop PDF upload with progress tracking
- Real-time document processing with status updates
- Chat interface for asking questions
- Source citation viewer
- Conversation history
- Database management (clear and re-index)

### Command-Line Interface

#### Single Query Mode

Ask a one-time question:
```bash
python query.py "What are the main findings of the research?"
```

#### Interactive Mode

Start an interactive session:
```bash
python query.py --interactive
```

**Available commands**:
- `/filter <key> <value>` - Apply metadata filters
- `/clear` - Clear active filters
- `/stats` - Show database statistics
- `/save` - Save last response to JSON
- `/quit` or `/exit` - Exit interactive mode

#### Advanced Options

```bash
# Retrieve more documents
python query.py "Your question" -r 10

# Apply metadata filter
python query.py "Your question" --filter "section=abstract"

# Use different model
python query.py "Your question" --model "gpt-4"

# Save results to file
python query.py "Your question" --save

# Retrieval only (no LLM)
python query.py "Your question" --no-chatgpt
```

### Listing Papers and Structure

Use the `list_papers.py` utility to view indexed papers:

```bash
# List all paper titles
python list_papers.py --titles

# Show structure (sections/headings) of all papers
python list_papers.py --structure

# Show detailed structure for a specific paper
python list_papers.py --paper "2601.11516v1"

# Show both titles and structure (default)
python list_papers.py
```

**Example Queries for Paper Structure:**
- "List all papers in the database" → Use `list_papers.py --titles`
- "What sections does this paper have?" → Use `list_papers.py --structure`
- "Show me the paper structure" → Use `list_papers.py --paper <paper_id>`
- "What are the main headings?" → Query: "What are the main sections of this paper?"

### Batch Ingestion

Process all PDFs in the `data/pdfs/` directory:

```bash
python -m ingestion.ingest_all
```

This will:
1. Extract text from all PDFs
2. Extract metadata (title, authors, abstract, keywords)
3. Chunk documents intelligently by sections
4. Generate embeddings
5. Store in ChromaDB with metadata

## 📂 Project Structure

```
ai_literature_assistant/
├── app.py                      # Streamlit web interface
├── config.py                   # Configuration settings
├── query.py                    # CLI query interface
├── requirements.txt            # Python dependencies
├── setup_env.py               # Environment setup script
│
├── data/                      # Data directory
│   ├── chroma_db/            # Vector database storage
│   ├── pdfs/                 # Place PDF files here
│   └── README.txt
│
├── ingestion/                 # Document processing pipeline
│   ├── __init__.py
│   ├── pdf_loader.py         # PDF text extraction
│   ├── preprocess.py         # Text chunking & metadata extraction
│   ├── embed_store.py        # Embedding generation & storage
│   └── ingest_all.py         # Batch ingestion script
│
└── rag/                       # RAG implementation
    ├── __init__.py
    ├── retriever.py          # Semantic search & retrieval
    ├── response_generator.py # LLM response generation
    └── pipeline.py           # End-to-end RAG pipeline
```

## 🔧 Configuration

Edit `config.py` to customize:

```python
# Embedding model
EMBEDDING_MODEL = "all-mpnet-base-v2"

# OpenAI settings
OPENAI_MODEL = "gpt-4o-mini"
OPENAI_TEMPERATURE = 0.3
OPENAI_MAX_TOKENS = 1500

# RAG settings
MAX_RETRIEVAL_RESULTS = 5
MAX_CONTEXT_LENGTH = 4000
MIN_SIMILARITY_THRESHOLD = 0.3

# Chunking parameters (in preprocess.py)
chunk_size = 1000
overlap = 150
```

## 🧩 Key Components

### 1. PDF Loader (`ingestion/pdf_loader.py`)
- Extracts text from PDFs using PyPDF
- Handles both file paths and byte streams
- Supports multiple extraction modes with fallback
- Robust error handling for corrupted pages

### 2. Preprocessor (`ingestion/preprocess.py`)
- **ResearchPaperChunker**: Intelligent chunking that preserves document structure
- **Metadata Extraction**: Extracts titles, authors, affiliations, abstracts, keywords, publication year
- **Section Detection**: Identifies Introduction, Methodology, Results, Discussion, Conclusion, References
- **Smart Chunking**: Uses LangChain's RecursiveCharacterTextSplitter with custom separators

### 3. Embedding & Storage (`ingestion/embed_store.py`)
- Generates embeddings using Sentence Transformers (all-mpnet-base-v2)
- Stores documents in ChromaDB with metadata
- Supports both dictionary and dataclass inputs
- Tracks metadata distribution

### 4. Retriever (`rag/retriever.py`)
- **RAGRetriever**: Core retrieval engine
- Semantic search using cosine similarity
- Metadata filtering support
- Context building with source citations
- Collection statistics

### 5. Response Generator (`rag/response_generator.py`)
- **ResponseGenerator**: LLM-powered answer generation
- Uses OpenAI GPT models
- Context-aware prompting
- Token usage tracking
- Graceful error handling

### 6. RAG Pipeline (`rag/pipeline.py`)
- **RAGPipeline**: Orchestrates entire workflow
- Query → Retrieve → Generate → Format
- Support for filtered queries
- Fallback to retrieval-only mode
- Comprehensive response formatting with sources

## 🎯 Use Cases

1. **Literature Review**: Quickly find relevant findings across multiple papers
2. **Citation Finding**: Locate specific claims or methodologies
3. **Summary Generation**: Get concise summaries of paper sections
4. **Comparative Analysis**: Compare approaches across different papers
5. **Fact Checking**: Verify claims against source documents

## 🛠️ Dependencies

```
chromadb>=0.4.0           # Vector database
sentence-transformers>=2.2.0  # Embedding model
openai>=1.0.0             # LLM API
python-dotenv>=1.0.0      # Environment management
numpy>=1.24.0             # Numerical operations
pypdf>=3.17.0             # PDF processing
streamlit>=1.28.0         # Web interface
langchain-text-splitters>=0.0.1  # Text chunking
```

## 📊 Performance

- **Embedding Model**: all-mpnet-base-v2 (768 dimensions)
- **Average Chunks per Paper**: ~50-150 (depending on paper length)
- **Retrieval Speed**: ~100-200ms per query
- **Embedding Generation**: ~1-2 seconds per paper
- **Typical Query Response Time**: 2-5 seconds (including LLM)

## 🔒 Security

- API keys stored in `.env` file (excluded from git)
- Environment variables loaded via python-dotenv
- No sensitive data in configuration files
- Local ChromaDB storage (no external data transmission except OpenAI API)

## 🤝 Contributing

This is an academic project. For improvements:

1. Test on diverse paper formats
2. Optimize chunk sizes for your domain
3. Experiment with different embedding models
4. Fine-tune prompts for specific research areas
5. Add support for other document formats (DOCX, LaTeX)

## 📝 Future Enhancements

- [ ] Support for multiple LLM providers (Anthropic, Google, local models)
- [ ] Citation graph visualization
- [ ] PDF image/figure extraction and analysis
- [ ] Multi-language support
- [ ] Export conversations to PDF/Word
- [ ] Custom metadata fields
- [ ] Paper recommendation system
- [ ] Automated literature review generation

## 🐛 Troubleshooting

### Common Issues

1. **"ChromaDB directory not found"**
   - Run `python setup_env.py` to create directories
   - Or manually create `data/chroma_db/` folder

2. **"No PDF files found"**
   - Place PDF files in `data/pdfs/` directory
   - Run ingestion: `python -m ingestion.ingest_all`

3. **"OPENAI_API_KEY not found"**
   - Add your API key to `.env` file
   - Or set environment variable: `export OPENAI_API_KEY='your-key'`

4. **"Collection 'research_papers' not found"**
   - No documents have been ingested yet
   - Upload PDFs via Streamlit UI or run batch ingestion

5. **Poor retrieval results**
   - Try different chunk sizes in config.py
   - Ensure PDFs contain extractable text (not scanned images)
   - Increase `n_retrieve` parameter for more context

## 📄 License

This project is developed for academic purposes.

## 👨‍💻 Author

Final Semester Project - AI Literature Assistant

## 🙏 Acknowledgments

- OpenAI for GPT models
- ChromaDB for vector storage
- Sentence Transformers for embeddings
- Streamlit for the web interface
- LangChain for text splitting utilities

---

**Note**: This system requires an active internet connection for OpenAI API calls. For offline operation, consider using local LLM alternatives like Ollama or LlamaCPP.
