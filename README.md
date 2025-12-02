# RAG Service - Insurance Q&A System

Professional RAG (Retrieval-Augmented Generation) system for Hebrew insurance document question-answering.

## Features

- **Hebrew PDF Processing**: Specialized extraction for RTL text
- **Intelligent Chunking**: Semantic text splitting optimized for Hebrew
- **Vector Search**: ChromaDB-powered similarity search
- **Flexible Embeddings**: OpenAI or local multilingual models
- **Interactive Console**: User-friendly CLI interface
- **Professional Architecture**: Clean, maintainable, production-ready code

## Project Structure

```
rag_service/
├── part_a/                    # Console application (Part A)
│   ├── __init__.py
│   ├── main.py               # Entry point
│   ├── config.py             # Configuration management
│   ├── document_loader.py    # PDF loading & parsing
│   ├── text_processor.py     # Chunking strategies
│   ├── embeddings.py         # Embedding models
│   ├── retriever.py          # Vector store & retrieval
│   └── rag_chain.py          # RAG pipeline
├── part_b/                    # Web Service API (Part B)
│   ├── __init__.py
│   ├── main.py               # FastAPI application
│   ├── models.py             # Pydantic models
│   ├── service.py            # Business logic
│   ├── storage.py            # SQLite Q&A storage
│   └── errors.py             # HTTP error handlers
├── data/                     # PDF documents directory
├── db/                       # SQLite database
├── vector_store/             # ChromaDB persistence
├── tests/                    # Test files
│   ├── test_part_a.py        # Part A tests
│   └── test_part_b.py        # Part B tests
├── requirements.txt          # Dependencies
├── .env.example             # Environment variables template
└── README.md                # This file
```

## Installation

### 1. Create Virtual Environment

```bash
cd rag_service
python -m venv venv

# Activate (Windows)
venv\Scripts\activate

# Activate (Linux/Mac)
source venv/bin/activate
```

### 2. Install Dependencies

```bash
pip install -r requirements.txt
```

### 3. Configure Environment

```bash
# Copy example config
cp .env.example .env

# Edit .env and add your OpenAI API key
# For Windows: notepad .env
# For Linux/Mac: nano .env
```

### 4. Add PDF Document

Place your insurance PDF document in the `data/` directory:

```bash
cp path/to/your/insurance_document.pdf data/
```

## Usage

### Console Application (Part A)

Run the interactive Q&A system:

```bash
python -m part_a.main
```

The system will:
1. Load and process the PDF document
2. Create embeddings and index chunks
3. Start an interactive console

### Example Session

```
======================================================================
Initializing RAG System
======================================================================
Step 1/5: Loading document...
✓ Document loaded: 5 pages, 12543 characters
Step 2/5: Processing and chunking text...
✓ Created 23 chunks (avg size: 587 chars)
Step 3/5: Initializing embedding model...
✓ Embedding model ready (dimension: 1536)
Step 4/5: Creating vector store and indexing chunks...
✓ Indexed 23 chunks in vector store
Step 5/5: Initializing RAG chain...
✓ RAG chain initialized
======================================================================
System ready!
======================================================================

======================================================================
מערכת שאלות ותשובות - רפואה משלימה
Insurance Q&A System - Complementary Medicine
======================================================================

💬 שאלה / Question: כמה יעלה לי ביטוח טיפולי אקופונקטורה?

🔍 מעבד שאלה... / Processing question...

✅ תשובה / Answer:
עבור 20 הטיפולים הראשונים בשנה, התעריף הוא 8.22 ₪ לטיפול.
החל מהטיפול ה-21 ואילך, התעריף הוא 21.86 ₪ לטיפול.
```

## Configuration

Edit `.env` file to customize:

| Variable | Description | Default |
|----------|-------------|---------|
| `OPENAI_API_KEY` | OpenAI API key | Required |
| `CHUNK_SIZE` | Characters per chunk | 600 |
| `CHUNK_OVERLAP` | Overlapping characters | 120 |
| `RETRIEVAL_K` | Number of chunks to retrieve | 4 |
| `LLM_MODEL` | OpenAI model name | gpt-4o-mini |
| `USE_LOCAL_EMBEDDINGS` | Use local instead of OpenAI | false |

## Advanced Usage

### Using Local Embeddings

To avoid OpenAI API costs for embeddings:

```bash
# In .env file
USE_LOCAL_EMBEDDINGS=true
```

This uses `sentence-transformers/paraphrase-multilingual-mpnet-base-v2` model.

### Programmatic Usage

```python
from pathlib import Path
from part_a.main import RAGApplication

# Initialize
app = RAGApplication()
app.initialize_system(Path("data/your_document.pdf"))

# Ask questions
answer = app.answer_question("כמה טיפולים מכוסים בשנה?")
print(answer)
```

## Architecture

### Document Processing Pipeline

1. **PDF Loading** (`document_loader.py`)
   - Extracts text using `pdfplumber`
   - Handles Hebrew RTL properly
   - Validates content

2. **Text Processing** (`text_processor.py`)
   - Semantic chunking by paragraphs/sections
   - Recursive splitting with overlap
   - Metadata preservation

3. **Embeddings** (`embeddings.py`)
   - OpenAI: `text-embedding-3-small`
   - Local: `paraphrase-multilingual-mpnet-base-v2`
   - Factory pattern for flexibility

4. **Vector Store** (`retriever.py`)
   - ChromaDB for persistence
   - Cosine similarity search
   - Configurable retrieval parameters

5. **RAG Chain** (`rag_chain.py`)
   - Retrieve relevant chunks
   - Build context-aware prompts
   - Generate answers with LLM

---

## Part B: Web Service API

### Overview

FastAPI-based REST API that wraps Part A RAG system with HTTP endpoints. Includes:
- Document upload endpoint
- Question submission endpoint (synchronous)
- Answer retrieval by ID
- SQLite storage for Q&A history
- Comprehensive error handling

### Running the API

```bash
# Install dependencies (if not done yet)
pip install -r requirements.txt

# Run the API server
python -m part_b.main

# Or with uvicorn directly
uvicorn part_b.main:app --host 0.0.0.0 --port 8000 --reload
```

The API will be available at:
- **API**: http://localhost:8000
- **Swagger Docs**: http://localhost:8000/docs
- **ReDoc**: http://localhost:8000/redoc

### API Endpoints

#### 1. Health Check
```bash
GET /health
```

Response:
```json
{
  "status": "healthy",
  "version": "1.0.0",
  "rag_initialized": true
}
```

#### 2. Upload Document
```bash
POST /api/upload
Content-Type: multipart/form-data

# Using curl
curl -X POST http://localhost:8000/api/upload \
  -F "file=@path/to/document.pdf"
```

Response:
```json
{
  "status": "success",
  "document_id": "doc-550e8400-e29b-41d4-a716-446655440000",
  "pages": 11,
  "chunks": 45
}
```

#### 3. Ask Question
```bash
POST /api/question
Content-Type: application/json

{
  "input": "כמה יעלה לי ביטוח טיפולי אקופונקטורה?",
  "id": "my-custom-id"  // Optional, UUID generated if omitted
}
```

Response:
```json
{
  "id": "my-custom-id",
  "question": "כמה יעלה לי ביטוח טיפולי אקופונקטורה?",
  "answer": "עלות הביטוח תלויה בגיל: עבור גילאי 0-20 התעריף הוא 8.22 ₪ לחודש...",
  "status": "completed",
  "timestamp": "2024-12-02T10:30:00"
}
```

#### 4. Get Answer by ID
```bash
GET /api/answer/{query_id}

# Using curl
curl http://localhost:8000/api/answer/my-custom-id
```

Response:
```json
{
  "id": "my-custom-id",
  "question": "כמה יעלה לי ביטוח טיפולי אקופונקטורה?",
  "answer": "עלות הביטוח תלויה בגיל...",
  "status": "completed",
  "timestamp": "2024-12-02T10:30:00"
}
```

### HTTP Error Codes

The API returns standard HTTP error codes with detailed error responses:

| Code | Error | Description |
|------|-------|-------------|
| **400** | Bad Request | Invalid JSON, missing fields, RAG not initialized |
| **404** | Not Found | Query ID does not exist |
| **413** | Payload Too Large | PDF file exceeds 10MB limit |
| **422** | Unprocessable Entity | Request validation failed (Pydantic) |
| **500** | Internal Server Error | RAG processing error, database error |

Error response format:
```json
{
  "error": "NotFoundError",
  "message": "Query ID not found",
  "detail": "No record exists with ID: abc-123"
}
```

### Configuration (Part B)

Add these to your `.env` file:

| Variable | Description | Default |
|----------|-------------|---------|
| `API_HOST` | API host address | 0.0.0.0 |
| `API_PORT` | API port | 8000 |
| `MAX_UPLOAD_SIZE_MB` | Max PDF size in MB | 10 |
| `DATABASE_URL` | SQLite database URL | sqlite:///./db/qa_history.db |

### Testing the API

#### Manual Testing with curl

```bash
# 1. Upload a document
curl -X POST http://localhost:8000/api/upload \
  -F "file=@data/insurance_document.pdf"

# 2. Ask a question
curl -X POST http://localhost:8000/api/question \
  -H "Content-Type: application/json" \
  -d '{"input": "כמה טיפולים מכוסים?"}'

# 3. Get answer by ID (use ID from step 2)
curl http://localhost:8000/api/answer/{query-id}
```

#### Automated Tests

```bash
# Run Part B unit tests
pytest tests/test_part_b.py -v

# Run all tests
pytest tests/ -v
```

### Part B Architecture

```
Request → FastAPI → Service Layer → Part A (RAG) → Storage (SQLite)
                                    ↓
                               Response
```

Components:
- **main.py**: FastAPI app with endpoint definitions
- **models.py**: Pydantic request/response schemas
- **service.py**: Business logic, integrates Part A RAG system
- **storage.py**: SQLAlchemy models for SQLite Q&A history
- **errors.py**: Custom exceptions and HTTP error handlers

---

## Testing

Test the system with these questions:

1. כמה יעלה לי ביטוח טיפולי אקופונקטורה?
2. ממתי ניתן לקבל החזר על הטיפול באקופונקטורה?
3. האם מקבלים החזר מלא על הטיפולים?
4. כמה טיפולים מכוסים?

## Development

### Code Quality

```bash
# Format code
black part_a/

# Lint
flake8 part_a/

# Type checking (optional)
mypy part_a/
```

### Logging

Logs are written to:
- Console (INFO level)
- `rag_service.log` file (detailed)

## Troubleshooting

### No text extracted from PDF
- Ensure PDF has a text layer (not just scanned images)
- Try different PDF processing tools

### API errors
- Verify `OPENAI_API_KEY` in `.env`
- Check API quota and rate limits

### Poor answer quality
- Adjust `CHUNK_SIZE` and `CHUNK_OVERLAP`
- Increase `RETRIEVAL_K` for more context
- Lower `similarity_threshold` in config

### Memory issues
- Use local embeddings instead of storing all in memory
- Process document in batches

## License

MIT License - See LICENSE file for details

## Support

For issues and questions:
- Check logs in `rag_service.log`
- Review configuration in `.env`
- Ensure PDF document is valid
