# Solution Architecture

## Overview

RAG-based Q&A system for Hebrew insurance document processing. Consists of a console application (Part A) and REST API (Part B).

## Part A Architecture (Console Application)

### Components

**1. Document Loader** (`document_loader.py`)
- PDF loading via `pdfplumber`
- Text extraction with RTL support (Hebrew)
- Text cleaning and normalization

**2. Text Processor** (`text_processor.py`)
- Text chunking (600 chars, 120 overlap)
- Semantic splitting by paragraphs
- Metadata preservation (page number, chunk index)

**3. Embeddings** (`embeddings.py`)
- OpenAI: `text-embedding-3-small` (1536 dimensions)
- Local: `paraphrase-multilingual-mpnet-base-v2` (768 dimensions)
- Factory pattern for model switching

**4. Vector Store** (`retriever.py`)
- ChromaDB for embedding storage
- Cosine similarity search
- Persistent storage in `./vector_store/`

**5. RAG Chain** (`rag_chain.py`)
- Retrieve: top-K relevant chunks (default K=4)
- Augment: context building from chunks
- Generate: OpenAI GPT-4o-mini for answer generation

**6. Main Application** (`main.py`)
- System initialization
- Interactive console interface
- Logging to file and console

### Data Flow

```
PDF → Document Loader → Text Processor → Chunks
                                            ↓
Question → Vector Search ← Embeddings ← ChromaDB
              ↓
    Top-K Chunks → Context → LLM → Answer
```

## Part B Architecture (Web Service API)

### Components

**1. FastAPI App** (`main.py`)
- 3 endpoints: upload, question, answer
- CORS middleware
- Error handlers for all HTTP codes

**2. Service Layer** (`service.py`)
- Integration with Part A (RAG chain)
- Business logic for request processing
- RAG state management

**3. Storage** (`storage.py`)
- SQLAlchemy ORM
- SQLite for Q&A history
- Model: id, question, answer, timestamp, status

**4. Models** (`models.py`)
- Pydantic schemas for request/response
- Input validation

**5. Error Handling** (`errors.py`)
- Custom exceptions for each HTTP code
- Centralized error handling
- Error documentation

### Data Flow

```
Client → FastAPI → Service → Part A (RAG) → LLM
                       ↓
                   SQLite Storage
                       ↓
                   Response → Client
```

## Technology Stack

### Core Libraries

- **LangChain** (0.1.0) - RAG framework
- **ChromaDB** (0.4.22) - vector database
- **OpenAI** (1.12.0) - embeddings + LLM
- **pdfplumber** (0.10.3) - PDF parsing
- **FastAPI** (0.109.0) - REST API
- **SQLAlchemy** (2.0.25) - ORM

### Alternatives

- Local embeddings: `sentence-transformers` instead of OpenAI
- LLM: any OpenAI-compatible model (gpt-4, gpt-3.5-turbo)

## Configuration

Managed via `.env`:
- `CHUNK_SIZE=600` - chunk size
- `CHUNK_OVERLAP=120` - overlap between chunks
- `RETRIEVAL_K=4` - number of chunks for context
- `LLM_MODEL=gpt-4o-mini` - generation model
- `USE_LOCAL_EMBEDDINGS=false` - use local embeddings

## Scalability

### Current Implementation
- Single document in memory
- Synchronous request processing
- SQLite for simplicity

### Possible Improvements
- Multiple documents in ChromaDB
- Async processing (FastAPI async)
- PostgreSQL for production
- Query caching
- Rate limiting
