"""
Unit tests for Part B - Web Service API.
Tests for the three main endpoints: upload, question, get_answer.
"""

import pytest
from fastapi.testclient import TestClient
from unittest.mock import Mock, patch, MagicMock
import io

from part_b.main import app
from part_b.storage import QARecord
from datetime import datetime


# Test client
client = TestClient(app)


class TestHealthEndpoint:
    """Tests for /health endpoint."""

    def test_health_check_returns_200(self):
        """Test that health endpoint returns 200 OK."""
        response = client.get("/health")

        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "healthy"
        assert "version" in data
        assert "rag_initialized" in data


class TestUploadEndpoint:
    """Tests for POST /api/upload endpoint."""

    @patch("part_b.main.rag_service.initialize_rag")
    def test_upload_pdf_success(self, mock_initialize):
        """Test successful PDF upload."""
        mock_initialize.return_value = (11, 45)  # pages, chunks

        # Create a fake PDF file
        pdf_content = b"%PDF-1.4 fake pdf content"
        files = {"file": ("test_document.pdf", io.BytesIO(pdf_content), "application/pdf")}

        response = client.post("/api/upload", files=files)

        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "success"
        assert "document_id" in data
        assert data["pages"] == 11
        assert data["chunks"] == 45

    def test_upload_non_pdf_fails(self):
        """Test that uploading non-PDF file fails with 400."""
        txt_content = b"This is not a PDF"
        files = {"file": ("document.txt", io.BytesIO(txt_content), "text/plain")}

        response = client.post("/api/upload", files=files)

        assert response.status_code == 400
        data = response.json()
        assert "error" in data
        assert data["message"] == "Invalid file format"

    @patch("part_b.main.rag_service.initialize_rag")
    def test_upload_too_large_file_fails(self, mock_initialize):
        """Test that uploading large file fails with 413."""
        # Create a file larger than MAX_UPLOAD_SIZE_MB (10MB)
        large_content = b"x" * (11 * 1024 * 1024)  # 11MB
        files = {"file": ("large.pdf", io.BytesIO(large_content), "application/pdf")}

        response = client.post("/api/upload", files=files)

        assert response.status_code == 413
        data = response.json()
        assert "error" in data
        assert "too large" in data["message"].lower()

    @patch("part_b.main.rag_service.initialize_rag")
    def test_upload_processing_error_returns_500(self, mock_initialize):
        """Test that processing error returns 500."""
        mock_initialize.side_effect = ValueError("Failed to process document")

        pdf_content = b"%PDF-1.4 fake pdf"
        files = {"file": ("test.pdf", io.BytesIO(pdf_content), "application/pdf")}

        response = client.post("/api/upload", files=files)

        assert response.status_code == 500
        data = response.json()
        assert "error" in data


class TestQuestionEndpoint:
    """Tests for POST /api/question endpoint."""

    @patch("part_b.main.rag_service.process_question")
    @patch("part_b.main.rag_service.is_initialized")
    def test_ask_question_success(self, mock_is_init, mock_process):
        """Test successful question submission."""
        mock_is_init.return_value = True

        # Mock Q&A record
        mock_record = Mock(spec=QARecord)
        mock_record.id = "test-query-123"
        mock_record.question = "ÛÞÔ âÕÜÔ ÑÙØÕ×?"
        mock_record.answer = "âÜÕê ÔÑÙØÕ× ÔÙÐ 8.22 ª"
        mock_record.status = "completed"
        mock_record.timestamp = datetime.now()

        mock_process.return_value = mock_record

        # Send request
        response = client.post(
            "/api/question",
            json={"input": "ÛÞÔ âÕÜÔ ÑÙØÕ×?", "id": "test-query-123"}
        )

        assert response.status_code == 200
        data = response.json()
        assert data["id"] == "test-query-123"
        assert data["question"] == "ÛÞÔ âÕÜÔ ÑÙØÕ×?"
        assert data["answer"] == "âÜÕê ÔÑÙØÕ× ÔÙÐ 8.22 ª"
        assert data["status"] == "completed"

    @patch("part_b.main.rag_service.process_question")
    @patch("part_b.main.rag_service.is_initialized")
    def test_ask_question_without_id_generates_uuid(self, mock_is_init, mock_process):
        """Test that question without ID gets auto-generated UUID."""
        mock_is_init.return_value = True

        mock_record = Mock(spec=QARecord)
        mock_record.id = "auto-generated-uuid"
        mock_record.question = "éÐÜÔ"
        mock_record.answer = "êéÕÑÔ"
        mock_record.status = "completed"
        mock_record.timestamp = datetime.now()

        mock_process.return_value = mock_record

        response = client.post(
            "/api/question",
            json={"input": "éÐÜÔ"}  # No ID provided
        )

        assert response.status_code == 200
        data = response.json()
        assert "id" in data

    @patch("part_b.main.rag_service.is_initialized")
    def test_ask_question_without_rag_init_fails(self, mock_is_init):
        """Test that question fails if RAG not initialized (400)."""
        mock_is_init.return_value = False

        response = client.post(
            "/api/question",
            json={"input": "ÛÞÔ âÕÜÔ ÑÙØÕ×?"}
        )

        assert response.status_code == 400
        data = response.json()
        assert "RAG system not initialized" in data["message"]

    def test_ask_question_empty_input_fails(self):
        """Test that empty question fails validation (422)."""
        response = client.post(
            "/api/question",
            json={"input": ""}  # Empty string
        )

        assert response.status_code == 422
        data = response.json()
        assert "error" in data

    def test_ask_question_invalid_json_fails(self):
        """Test that invalid JSON fails (422)."""
        response = client.post(
            "/api/question",
            data="not valid json",
            headers={"Content-Type": "application/json"}
        )

        assert response.status_code == 422


class TestGetAnswerEndpoint:
    """Tests for GET /api/answer/{id} endpoint."""

    @patch("part_b.main.rag_service.get_answer")
    def test_get_answer_success(self, mock_get_answer):
        """Test successful answer retrieval."""
        mock_record = Mock(spec=QARecord)
        mock_record.id = "test-query-456"
        mock_record.question = "ÛÞÔ ØÙäÕÜÙÝ ÞÛÕáÙÝ?"
        mock_record.answer = "20 ØÙäÕÜÙÝ ÑéàÔ"
        mock_record.status = "completed"
        mock_record.timestamp = datetime.now()

        mock_get_answer.return_value = mock_record

        response = client.get("/api/answer/test-query-456")

        assert response.status_code == 200
        data = response.json()
        assert data["id"] == "test-query-456"
        assert data["question"] == "ÛÞÔ ØÙäÕÜÙÝ ÞÛÕáÙÝ?"
        assert data["answer"] == "20 ØÙäÕÜÙÝ ÑéàÔ"

    @patch("part_b.main.rag_service.get_answer")
    def test_get_answer_not_found(self, mock_get_answer):
        """Test that non-existent ID returns 404."""
        mock_get_answer.return_value = None

        response = client.get("/api/answer/non-existent-id")

        assert response.status_code == 404
        data = response.json()
        assert "error" in data
        assert "not found" in data["message"].lower()

    @patch("part_b.main.rag_service.get_answer")
    def test_get_answer_database_error_returns_500(self, mock_get_answer):
        """Test that database error returns 500."""
        mock_get_answer.side_effect = Exception("Database connection error")

        response = client.get("/api/answer/test-id")

        assert response.status_code == 500
        data = response.json()
        assert "error" in data


class TestIntegrationScenarios:
    """Integration tests for common user scenarios."""

    @patch("part_b.main.rag_service")
    def test_full_workflow(self, mock_service):
        """Test complete workflow: upload -> question -> get_answer."""
        # Setup mocks
        mock_service.initialize_rag.return_value = (11, 45)
        mock_service.is_initialized.return_value = True

        # Mock question processing
        mock_record = Mock(spec=QARecord)
        mock_record.id = "workflow-test-id"
        mock_record.question = "Test question"
        mock_record.answer = "Test answer"
        mock_record.status = "completed"
        mock_record.timestamp = datetime.now()

        mock_service.process_question.return_value = mock_record
        mock_service.get_answer.return_value = mock_record

        # Step 1: Upload document
        pdf_content = b"%PDF-1.4 test"
        files = {"file": ("test.pdf", io.BytesIO(pdf_content), "application/pdf")}
        upload_response = client.post("/api/upload", files=files)
        assert upload_response.status_code == 200

        # Step 2: Ask question
        question_response = client.post(
            "/api/question",
            json={"input": "Test question", "id": "workflow-test-id"}
        )
        assert question_response.status_code == 200
        query_id = question_response.json()["id"]

        # Step 3: Retrieve answer
        answer_response = client.get(f"/api/answer/{query_id}")
        assert answer_response.status_code == 200
        assert answer_response.json()["id"] == query_id


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
