"""
RAG Chain implementation.
Orchestrates retrieval and generation for question-answering.
"""

import logging
from typing import List, Dict, Any, Optional
from dataclasses import dataclass

from .retriever import VectorStoreRetriever, RetrievedChunk


logger = logging.getLogger(__name__)


@dataclass
class RAGResponse:
    """Response from RAG pipeline."""

    question: str
    answer: str
    retrieved_chunks: List[RetrievedChunk]
    metadata: Dict[str, Any]


class LLMInterface:
    """Interface for Language Model."""

    def __init__(self, api_key: str, model: str = "gpt-4o-mini", temperature: float = 0.0, max_tokens: int = 500):
        """
        Initialize LLM interface.

        Args:
            api_key: OpenAI API key
            model: Model name
            temperature: Sampling temperature
            max_tokens: Maximum tokens in response
        """
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens

        try:
            from openai import OpenAI
            self.client = OpenAI(api_key=api_key)
        except ImportError:
            raise ImportError(
                "OpenAI package not installed. Install with: pip install openai"
            )

        logger.info(f"Initialized LLM: {model}")

    def generate(self, prompt: str, system_prompt: Optional[str] = None) -> str:
        """
        Generate response from LLM.

        Args:
            prompt: User prompt
            system_prompt: System prompt (optional)

        Returns:
            Generated text
        """
        messages = []

        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})

        messages.append({"role": "user", "content": prompt})

        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                temperature=self.temperature,
                max_tokens=self.max_tokens
            )

            answer = response.choices[0].message.content
            logger.debug(f"Generated response: {answer[:100]}...")
            return answer

        except Exception as e:
            logger.error(f"Error generating response: {e}")
            raise


class RAGChain:
    """RAG pipeline for question-answering."""

    def __init__(
        self,
        retriever: VectorStoreRetriever,
        llm: LLMInterface,
        retrieval_k: int = 4,
        similarity_threshold: float = 0.7
    ):
        """
        Initialize RAG chain.

        Args:
            retriever: Vector store retriever
            llm: Language model interface
            retrieval_k: Number of chunks to retrieve
            similarity_threshold: Minimum similarity for retrieval
        """
        self.retriever = retriever
        self.llm = llm
        self.retrieval_k = retrieval_k
        self.similarity_threshold = similarity_threshold

        logger.info(f"Initialized RAG chain with k={retrieval_k}, threshold={similarity_threshold}")

    def answer(self, question: str) -> RAGResponse:
        """
        Answer a question using RAG.

        Args:
            question: User question

        Returns:
            RAGResponse with answer and metadata
        """
        if not question or not question.strip():
            raise ValueError("Question cannot be empty")

        logger.info(f"Processing question: '{question[:50]}...'")

        # Step 1: Retrieve relevant chunks
        retrieved_chunks = self.retriever.retrieve(
            query=question,
            k=self.retrieval_k,
            similarity_threshold=self.similarity_threshold
        )

        if not retrieved_chunks:
            logger.warning("No relevant chunks found")
            return RAGResponse(
                question=question,
                answer="לא נמצא מידע רלוונטי במסמך לשאלה זו.",
                retrieved_chunks=[],
                metadata={"chunks_retrieved": 0, "answer_generated": False}
            )

        # Step 2: Build context from retrieved chunks
        context = self._build_context(retrieved_chunks)

        # Step 3: Generate answer
        prompt = self._build_prompt(question, context)
        system_prompt = self._get_system_prompt()

        answer = self.llm.generate(prompt, system_prompt)

        # Step 4: Create response
        response = RAGResponse(
            question=question,
            answer=answer,
            retrieved_chunks=retrieved_chunks,
            metadata={
                "chunks_retrieved": len(retrieved_chunks),
                "answer_generated": True,
                "avg_similarity": sum(c.similarity_score for c in retrieved_chunks) / len(retrieved_chunks)
            }
        )

        logger.info(f"Generated answer with {len(retrieved_chunks)} chunks")
        return response

    def _build_context(self, chunks: List[RetrievedChunk]) -> str:
        """
        Build context string from retrieved chunks.

        Args:
            chunks: Retrieved chunks

        Returns:
            Formatted context string
        """
        context_parts = []

        for i, chunk in enumerate(chunks, 1):
            section = chunk.metadata.get("section", "")
            section_info = f" ({section})" if section else ""

            context_parts.append(
                f"קטע {i}{section_info}:\n{chunk.content}\n"
            )

        return "\n".join(context_parts)

    def _get_system_prompt(self) -> str:
        """Get the system prompt for the LLM."""
        return """אתה עוזר שירות לקוחות מקצועי של חברת ביטוח.
המשימה שלך היא לענות על שאלות לקוחות בנוגע לפוליסת ביטוח רפואה משלימה.

הנחיות חשובות:
1. ענה רק על בסיס המידע שניתן לך בקטעים הרלוונטיים
2. היה מדויק במיוחד עם מספרים, תאריכים וסכומי כסף
3. אם המידע לא מופיע בקטעים שניתנו, אמור "המידע אינו מופיע במסמך"
4. השתמש בעברית פשוטה וברורה
5. אם השאלה מתייחסת למספר היבטים, ענה על כולם
6. לעולם אל תמציא מידע שאינו מופיע בקטעים

התשובה שלך צריכה להיות תמציתית אך מלאה."""

    def _build_prompt(self, question: str, context: str) -> str:
        """
        Build the user prompt.

        Args:
            question: User question
            context: Retrieved context

        Returns:
            Formatted prompt
        """
        return f"""להלן קטעים רלוונטיים מתוך פוליסת הביטוח:

{context}

---

שאלת הלקוח: {question}

תשובה:"""
