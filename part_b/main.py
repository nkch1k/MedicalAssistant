"""
FastAPI application for Part B - Web Service API.
Provides REST endpoints for RAG-based Q&A system.
"""

import logging
import os
from pathlib import Path
from typing import Optional

from fastapi import FastAPI, File, UploadFile, HTTPException, status
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware

from .models import (
    QuestionRequest,
    QuestionResponse,
    AnswerResponse,
    UploadResponse,
    HealthResponse,
    ErrorResponse
)
from .service import RAGService
from .errors import (
    APIError,
    NotFoundError,
    BadRequestError,
    PayloadTooLargeError,
    InternalServerError,
    api_error_handler,
    validation_error_handler,
    general_exception_handler,
    ERROR_DOCS
)


# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


# Initialize FastAPI app
app = FastAPI(
    title="RAG Insurance Q&A API",
    description="REST API for Hebrew insurance document question-answering using RAG",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)


# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# Register error handlers
app.add_exception_handler(APIError, api_error_handler)
app.add_exception_handler(RequestValidationError, validation_error_handler)
app.add_exception_handler(Exception, general_exception_handler)


# Initialize service
DATABASE_URL = os.getenv("DATABASE_URL", "sqlite:///./db/qa_history.db")
rag_service = RAGService(database_url=DATABASE_URL)


# Configuration
MAX_UPLOAD_SIZE_MB = int(os.getenv("MAX_UPLOAD_SIZE_MB", "10"))
MAX_UPLOAD_SIZE_BYTES = MAX_UPLOAD_SIZE_MB * 1024 * 1024


@app.get("/health", response_model=HealthResponse, tags=["Health"])
async def health_check():
    """
    Health check endpoint.

    Returns service status and RAG system initialization state.
    """
    stats = rag_service.get_stats()

    return HealthResponse(
        status="healthy",
        version="1.0.0",
        rag_initialized=stats["rag_initialized"]
    )


@app.post(
    "/api/upload",
    response_model=UploadResponse,
    status_code=status.HTTP_200_OK,
    tags=["Document"],
    responses={
        413: {"model": ErrorResponse, "description": "File too large"},
        422: {"model": ErrorResponse, "description": "Invalid file format"},
        500: {"model": ErrorResponse, "description": "Processing error"}
    }
)
async def upload_document(file: UploadFile = File(...)):
    """
    Upload and process a PDF document.

    Initializes the RAG system with the uploaded document.

    **Error codes:**
    - **413**: File exceeds maximum size limit (10MB)
    - **422**: File is not a valid PDF
    - **500**: Document processing failed
    """
    logger.info(f"Received file upload: {file.filename}")

    # Validate file type
    if not file.filename.endswith('.pdf'):
        raise BadRequestError(
            message="Invalid file format",
            detail="Only PDF files are supported"
        )

    # Check file size
    file_content = await file.read()
    file_size = len(file_content)

    if file_size > MAX_UPLOAD_SIZE_BYTES:
        raise PayloadTooLargeError(
            message="File too large",
            detail=f"Maximum file size is {MAX_UPLOAD_SIZE_MB}MB. Uploaded: {file_size / 1024 / 1024:.2f}MB"
        )

    # Save temporary file
    temp_dir = Path("./data/temp")
    temp_dir.mkdir(parents=True, exist_ok=True)
    temp_path = temp_dir / file.filename

    try:
        with open(temp_path, "wb") as f:
            f.write(file_content)

        # Initialize RAG system
        pages, chunks = rag_service.initialize_rag(temp_path)

        logger.info(f"Document processed successfully: {pages} pages, {chunks} chunks")

        return UploadResponse(
            status="success",
            document_id=rag_service.current_document_id,
            pages=pages,
            chunks=chunks
        )

    except ValueError as e:
        logger.error(f"Document processing error: {e}")
        raise InternalServerError(
            message="Failed to process document",
            detail=str(e)
        )
    except Exception as e:
        logger.error(f"Unexpected error during upload: {e}")
        raise InternalServerError(
            message="An unexpected error occurred",
            detail=str(e)
        )
    finally:
        # Clean up temporary file
        if temp_path.exists():
            temp_path.unlink()


@app.post(
    "/api/question",
    response_model=QuestionResponse,
    status_code=status.HTTP_200_OK,
    tags=["Q&A"],
    responses={
        400: {"model": ErrorResponse, "description": "Invalid request"},
        422: {"model": ErrorResponse, "description": "Validation error"},
        500: {"model": ErrorResponse, "description": "Processing error"}
    }
)
async def ask_question(request: QuestionRequest):
    """
    Submit a question and get an answer (synchronous).

    The system processes the question and returns the answer immediately.
    The Q&A pair is stored in the database with the provided or generated ID.

    **Parameters:**
    - **input**: Question text in Hebrew (required)
    - **id**: Optional custom query ID. If not provided, a UUID will be generated.

    **Error codes:**
    - **400**: RAG system not initialized (upload document first)
    - **422**: Request validation failed
    - **500**: Question processing failed
    """
    logger.info(f"Received question: '{request.input[:50]}...'")

    # Check if RAG is initialized
    if not rag_service.is_initialized():
        raise BadRequestError(
            message="RAG system not initialized",
            detail="Please upload a document first using POST /api/upload"
        )

    try:
        # Process question
        record = rag_service.process_question(
            question=request.input,
            query_id=request.id
        )

        logger.info(f"Question processed successfully: {record.id}")

        return QuestionResponse(
            id=record.id,
            question=record.question,
            answer=record.answer,
            status=record.status,
            timestamp=record.timestamp
        )

    except ValueError as e:
        # Query ID already exists
        raise BadRequestError(
            message="Invalid query ID",
            detail=str(e)
        )
    except RuntimeError as e:
        # RAG processing error
        raise InternalServerError(
            message="Failed to process question",
            detail=str(e)
        )
    except Exception as e:
        logger.error(f"Unexpected error processing question: {e}")
        raise InternalServerError(
            message="An unexpected error occurred",
            detail=str(e)
        )


@app.get(
    "/api/answer/{query_id}",
    response_model=AnswerResponse,
    status_code=status.HTTP_200_OK,
    tags=["Q&A"],
    responses={
        404: {"model": ErrorResponse, "description": "Query ID not found"},
        500: {"model": ErrorResponse, "description": "Retrieval error"}
    }
)
async def get_answer(query_id: str):
    """
    Retrieve a previously processed answer by query ID.

    **Parameters:**
    - **query_id**: The unique identifier from a previous question submission

    **Error codes:**
    - **404**: No record found with the provided query ID
    - **500**: Database retrieval error
    """
    logger.info(f"Retrieving answer for ID: {query_id}")

    try:
        record = rag_service.get_answer(query_id)

        if record is None:
            raise NotFoundError(
                message="Query ID not found",
                detail=f"No record exists with ID: {query_id}"
            )

        logger.info(f"Answer retrieved successfully: {query_id}")

        return AnswerResponse(
            id=record.id,
            question=record.question,
            answer=record.answer,
            status=record.status,
            timestamp=record.timestamp
        )

    except NotFoundError:
        raise
    except Exception as e:
        logger.error(f"Error retrieving answer: {e}")
        raise InternalServerError(
            message="Failed to retrieve answer",
            detail=str(e)
        )


@app.get("/api/errors/docs", tags=["Documentation"])
async def error_documentation():
    """
    Get documentation for all possible HTTP errors.

    Returns detailed information about each error code including:
    - Description
    - Common situations when the error occurs
    - Example error response
    """
    return ERROR_DOCS


# Startup event
@app.on_event("startup")
async def startup_event():
    """Initialize system on startup."""
    logger.info("Starting RAG Insurance Q&A API...")
    logger.info(f"Database: {DATABASE_URL}")
    logger.info(f"Max upload size: {MAX_UPLOAD_SIZE_MB}MB")
    logger.info("API ready!")


# Shutdown event
@app.on_event("shutdown")
async def shutdown_event():
    """Clean up on shutdown."""
    logger.info("Shutting down RAG Insurance Q&A API...")


if __name__ == "__main__":
    import uvicorn

    # Run with: python -m part_b.main
    uvicorn.run(
        app,
        host=os.getenv("API_HOST", "0.0.0.0"),
        port=int(os.getenv("API_PORT", "8000")),
        log_level="info"
    )
