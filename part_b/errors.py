"""
HTTP error handlers and custom exceptions for Part B API.
Defines standard error responses per technical specification.
"""

from fastapi import Request, status
from fastapi.responses import JSONResponse
from fastapi.exceptions import RequestValidationError
from pydantic import ValidationError


class APIError(Exception):
    """Base API error class."""

    def __init__(self, message: str, status_code: int, detail: str = None):
        self.message = message
        self.status_code = status_code
        self.detail = detail
        super().__init__(self.message)


class NotFoundError(APIError):
    """404 Not Found error."""

    def __init__(self, message: str = "Resource not found", detail: str = None):
        super().__init__(message, status.HTTP_404_NOT_FOUND, detail)


class BadRequestError(APIError):
    """400 Bad Request error."""

    def __init__(self, message: str = "Bad request", detail: str = None):
        super().__init__(message, status.HTTP_400_BAD_REQUEST, detail)


class PayloadTooLargeError(APIError):
    """413 Payload Too Large error."""

    def __init__(self, message: str = "Payload too large", detail: str = None):
        super().__init__(message, status.HTTP_413_REQUEST_ENTITY_TOO_LARGE, detail)


class UnprocessableEntityError(APIError):
    """422 Unprocessable Entity error."""

    def __init__(self, message: str = "Unprocessable entity", detail: str = None):
        super().__init__(message, status.HTTP_422_UNPROCESSABLE_ENTITY, detail)


class InternalServerError(APIError):
    """500 Internal Server Error."""

    def __init__(self, message: str = "Internal server error", detail: str = None):
        super().__init__(message, status.HTTP_500_INTERNAL_SERVER_ERROR, detail)


# Error handlers

async def api_error_handler(request: Request, exc: APIError) -> JSONResponse:
    """
    Handle custom API errors.

    Used for: 400, 404, 413, 422, 500 errors as per technical specification.
    """
    return JSONResponse(
        status_code=exc.status_code,
        content={
            "error": exc.__class__.__name__,
            "message": exc.message,
            "detail": exc.detail
        }
    )


async def validation_error_handler(request: Request, exc: RequestValidationError) -> JSONResponse:
    """
    Handle Pydantic validation errors (422).

    Situation: Request body doesn't match expected schema
    Example: Missing required field, wrong type, invalid format
    """
    return JSONResponse(
        status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
        content={
            "error": "ValidationError",
            "message": "Request validation failed",
            "detail": str(exc.errors())
        }
    )


async def general_exception_handler(request: Request, exc: Exception) -> JSONResponse:
    """
    Handle unexpected exceptions (500).

    Situation: Unhandled errors in application logic
    Example: Database connection error, RAG system failure, unexpected bugs
    """
    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content={
            "error": "InternalServerError",
            "message": "An unexpected error occurred",
            "detail": str(exc) if hasattr(exc, '__str__') else "Unknown error"
        }
    )


# Error documentation for API

ERROR_DOCS = {
    "400": {
        "description": "Bad Request",
        "situations": [
            "Invalid JSON format in request body",
            "Missing required parameters",
            "Malformed request data"
        ],
        "example": {
            "error": "BadRequestError",
            "message": "Invalid request format",
            "detail": "Request body must be valid JSON"
        }
    },
    "404": {
        "description": "Not Found",
        "situations": [
            "Query ID does not exist in database",
            "Requested resource not found"
        ],
        "example": {
            "error": "NotFoundError",
            "message": "Query ID not found",
            "detail": "No record exists with ID: abc-123"
        }
    },
    "413": {
        "description": "Payload Too Large",
        "situations": [
            "Uploaded PDF file exceeds maximum size limit",
            "Request body too large"
        ],
        "example": {
            "error": "PayloadTooLargeError",
            "message": "File too large",
            "detail": "Maximum file size is 10MB"
        }
    },
    "422": {
        "description": "Unprocessable Entity",
        "situations": [
            "Request validation failed (Pydantic)",
            "Invalid field types or values",
            "Schema mismatch"
        ],
        "example": {
            "error": "ValidationError",
            "message": "Request validation failed",
            "detail": "Field 'input' is required"
        }
    },
    "500": {
        "description": "Internal Server Error",
        "situations": [
            "RAG system error or failure",
            "Database connection error",
            "LLM API error",
            "Unexpected system failure"
        ],
        "example": {
            "error": "InternalServerError",
            "message": "An unexpected error occurred",
            "detail": "Failed to process document"
        }
    }
}
