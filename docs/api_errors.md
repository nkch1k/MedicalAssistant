# API Error Documentation (Part B)

## Overview

Part B REST API implements comprehensive HTTP error handling with detailed error responses. All errors follow a consistent JSON format.

## Error Response Format

```json
{
  "error": "ErrorClassName",
  "message": "Human-readable error description",
  "detail": "Specific information about what went wrong"
}
```

## HTTP Error Codes

### 400 Bad Request

**Description:** Invalid request parameters or system state

**Common Situations:**
- Invalid JSON in request body
- RAG system not initialized (no document uploaded)
- Query ID already exists
- Missing required fields

**Example:**
```json
{
  "error": "BadRequestError",
  "message": "RAG system not initialized",
  "detail": "Please upload a document first using POST /api/upload"
}
```

**How to Fix:**
- Validate JSON syntax before sending
- Upload a document before asking questions
- Use unique query IDs

### 404 Not Found

**Description:** Requested resource doesn't exist

**Common Situations:**
- Query ID not found in database
- Trying to retrieve answer before question was asked

**Example:**
```json
{
  "error": "NotFoundError",
  "message": "Query ID not found",
  "detail": "No record exists with ID: abc-123"
}
```

**How to Fix:**
- Verify query ID was returned from POST /api/question
- Check for typos in query ID
- Ensure question was successfully processed

### 413 Payload Too Large

**Description:** Uploaded file exceeds size limit

**Common Situations:**
- PDF file larger than 10MB
- Very large request body

**Example:**
```json
{
  "error": "PayloadTooLargeError",
  "message": "File too large",
  "detail": "Maximum file size is 10MB. Uploaded: 15.3MB"
}
```

**How to Fix:**
- Compress PDF file
- Split large documents
- Contact admin to increase limit

### 422 Unprocessable Entity

**Description:** Request validation failed (Pydantic schema mismatch)

**Common Situations:**
- Missing required field (`input`)
- Wrong data type (e.g., number instead of string)
- Invalid field format

**Example:**
```json
{
  "error": "ValidationError",
  "message": "Request validation failed",
  "detail": "[{'loc': ['body', 'input'], 'msg': 'field required', 'type': 'value_error.missing'}]"
}
```

**How to Fix:**
- Check API documentation for required fields
- Verify data types match schema
- Use proper JSON encoding

### 500 Internal Server Error

**Description:** Unexpected server-side error

**Common Situations:**
- RAG system processing error
- LLM API failure
- Database connection error
- PDF processing failure
- Unexpected bugs

**Example:**
```json
{
  "error": "InternalServerError",
  "message": "Failed to process document",
  "detail": "Cannot extract text from PDF: corrupted file"
}
```

**How to Fix:**
- Check server logs for details
- Verify OpenAI API key is valid
- Ensure PDF file is not corrupted
- Retry the request
- Contact support if persistent

## API Endpoints & Possible Errors

### POST /api/upload

**Possible Errors:**
- 413: File too large
- 422: Not a PDF file
- 500: PDF processing failed

### POST /api/question

**Possible Errors:**
- 400: RAG not initialized
- 422: Invalid request body
- 500: Answer generation failed

### GET /api/answer/{query_id}

**Possible Errors:**
- 404: Query ID not found
- 500: Database retrieval error

## Testing Errors

### Using curl

```bash
# 400: RAG not initialized
curl -X POST http://localhost:8000/api/question \
  -H "Content-Type: application/json" \
  -d '{"input": "test question"}'

# 404: Query not found
curl http://localhost:8000/api/answer/nonexistent-id

# 413: File too large
curl -X POST http://localhost:8000/api/upload \
  -F "file=@large_file.pdf"

# 422: Missing field
curl -X POST http://localhost:8000/api/question \
  -H "Content-Type: application/json" \
  -d '{}'
```

## Error Documentation Endpoint

Get programmatic error docs:

```bash
GET /api/errors/docs
```

Returns JSON with all error codes, situations, and examples.

## Best Practices

1. **Always check status code** before parsing response
2. **Log error details** for debugging
3. **Show user-friendly messages** based on error type
4. **Implement retry logic** for 500 errors
5. **Validate inputs client-side** to avoid 422 errors
6. **Handle timeouts** for long-running operations
