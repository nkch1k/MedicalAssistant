"""
Pydantic models for Part B API.
Defines request and response schemas.
"""

from datetime import datetime
from typing import Optional
from pydantic import BaseModel, Field


class QuestionRequest(BaseModel):
    """Request model for asking a question."""

    input: str = Field(..., description="Question text in Hebrew", min_length=1)
    id: Optional[str] = Field(None, description="Optional custom query ID")

    class Config:
        json_schema_extra = {
            "example": {
                "input": "лод йтмд мй бйиез ийфемй ачеферчиешд?",
                "id": "query-123"
            }
        }


class QuestionResponse(BaseModel):
    """Response model for question submission."""

    id: str = Field(..., description="Query ID")
    question: str = Field(..., description="Question text")
    answer: str = Field(..., description="Generated answer")
    status: str = Field("completed", description="Query status")
    timestamp: datetime = Field(..., description="Processing timestamp")

    class Config:
        json_schema_extra = {
            "example": {
                "id": "550e8400-e29b-41d4-a716-446655440000",
                "question": "лод йтмд мй бйиез ийфемй ачеферчиешд?",
                "answer": "тмеъ дбйиез ъмейд бвйм...",
                "status": "completed",
                "timestamp": "2024-12-02T10:30:00"
            }
        }


class AnswerResponse(BaseModel):
    """Response model for retrieving an answer by ID."""

    id: str = Field(..., description="Query ID")
    question: str = Field(..., description="Original question")
    answer: str = Field(..., description="Generated answer")
    status: str = Field(..., description="Query status")
    timestamp: datetime = Field(..., description="Processing timestamp")

    class Config:
        json_schema_extra = {
            "example": {
                "id": "550e8400-e29b-41d4-a716-446655440000",
                "question": "лод йтмд мй бйиез ийфемй ачеферчиешд?",
                "answer": "тмеъ дбйиез ъмейд бвйм...",
                "status": "completed",
                "timestamp": "2024-12-02T10:30:00"
            }
        }


class UploadResponse(BaseModel):
    """Response model for document upload."""

    status: str = Field(..., description="Upload status")
    document_id: str = Field(..., description="Unique document identifier")
    pages: int = Field(..., description="Number of pages processed")
    chunks: int = Field(..., description="Number of text chunks created")

    class Config:
        json_schema_extra = {
            "example": {
                "status": "success",
                "document_id": "doc-550e8400-e29b-41d4-a716-446655440000",
                "pages": 11,
                "chunks": 45
            }
        }


class HealthResponse(BaseModel):
    """Response model for health check."""

    status: str = Field(..., description="Service status")
    version: str = Field(..., description="API version")
    rag_initialized: bool = Field(..., description="RAG system status")

    class Config:
        json_schema_extra = {
            "example": {
                "status": "healthy",
                "version": "1.0.0",
                "rag_initialized": True
            }
        }


class ErrorResponse(BaseModel):
    """Standard error response model."""

    error: str = Field(..., description="Error type")
    message: str = Field(..., description="Error message")
    detail: Optional[str] = Field(None, description="Additional error details")

    class Config:
        json_schema_extra = {
            "example": {
                "error": "NotFoundError",
                "message": "Query ID not found",
                "detail": "No record exists with ID: query-123"
            }
        }
