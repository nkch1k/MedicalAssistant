"""
Document loading and PDF processing.
Specialized for Hebrew RTL text extraction.
"""

import logging
from pathlib import Path
import pdfplumber
import unicodedata
import re
from bidi.algorithm import get_display


logger = logging.getLogger(__name__)


class DocumentLoader:
    """Loads and extracts text from PDF documents with Hebrew support."""

    def load_pdf(self, file_path: Path) -> tuple[str, dict]:
        """
        Load and extract text from a PDF file.

        Args:
            file_path: Path to the PDF file

        Returns:
            Tuple of (extracted_text, metadata)

        Raises:
            FileNotFoundError: If the PDF file doesn't exist
            ValueError: If the file is not a valid PDF or text extraction fails
        """
        if not file_path.exists():
            raise FileNotFoundError(f"PDF file not found: {file_path}")

        if file_path.suffix.lower() != ".pdf":
            raise ValueError(f"File must be a PDF, got: {file_path.suffix}")

        try:
            pages = []
            page_count = 0

            with pdfplumber.open(file_path) as pdf:
                page_count = len(pdf.pages)
                logger.info(f"Processing PDF: {file_path.name} ({page_count} pages)")

                for page_num, page in enumerate(pdf.pages, start=1):
                    text = page.extract_text(layout=True)

                    if text:
                        pages.append(self._clean_text(text))
                    else:
                        logger.warning(f"No text extracted from page {page_num}")

            full_text = "\n\n".join(pages)

            if not full_text.strip():
                raise ValueError("No text could be extracted from the PDF. It might be a scanned image.")

            # Check for Hebrew content
            if not re.search(r"[\u0590-\u05FF]", full_text):
                logger.warning("Warning: No Hebrew characters detected in document")

            metadata = {
                "file_name": file_path.name,
                "total_pages": page_count,
                "total_characters": len(full_text)
            }

            logger.info(f"Successfully extracted {len(full_text)} characters from {page_count} pages")

            return full_text, metadata

        except Exception as e:
            logger.error(f"Error processing PDF: {e}")
            raise ValueError(f"Failed to process PDF: {e}")

    def _has_hebrew(self, text: str) -> bool:
        """Check if text contains Hebrew characters."""
        return bool(re.search(r'[\u0590-\u05FF]', text))

    def _fix_bidi_text(self, text: str) -> str:
        """
        Fix bidirectional text using python-bidi library.
        Handles RTL (Hebrew) text while preserving LTR elements (numbers, Latin).
        """
        lines = text.split('\n')
        fixed_lines = []

        for line in lines:
            if not line.strip():
                fixed_lines.append(line)
                continue

            # If line contains Hebrew, use bidi algorithm
            if self._has_hebrew(line):
                try:
                    # get_display handles bidirectional text properly
                    # It keeps numbers and Latin text in correct order
                    fixed_line = get_display(line)
                    fixed_lines.append(fixed_line)
                except Exception as e:
                    logger.warning(f"Failed to process bidi text: {e}")
                    fixed_lines.append(line)
            else:
                fixed_lines.append(line)

        return '\n'.join(fixed_lines)

    def _clean_text(self, text: str) -> str:
        """Clean and normalize extracted text."""
        # Normalize unicode (important for Hebrew)
        text = unicodedata.normalize("NFC", text)

        # Fix bidirectional text (RTL/LTR) using python-bidi
        text = self._fix_bidi_text(text)

        # Remove excessive whitespace while preserving paragraph structure
        lines = text.split("\n")
        cleaned_lines = [line.strip() for line in lines if line.strip()]

        # Join lines with single newline
        cleaned_text = "\n".join(cleaned_lines)

        # Replace multiple newlines with double newline (paragraph separator)
        cleaned_text = re.sub(r"\n{3,}", "\n\n", cleaned_text)

        return cleaned_text
