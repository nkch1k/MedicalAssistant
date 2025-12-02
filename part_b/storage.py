"""
SQLite storage for Q&A history.
Manages persistent storage of questions and answers.
"""

import logging
from datetime import datetime
from pathlib import Path
from typing import Optional, Dict, Any

from sqlalchemy import create_engine, Column, String, DateTime, Text
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker, Session


logger = logging.getLogger(__name__)

Base = declarative_base()


class QARecord(Base):
    """Database model for Q&A history."""

    __tablename__ = "qa_history"

    id = Column(String, primary_key=True, index=True)
    question = Column(Text, nullable=False)
    answer = Column(Text, nullable=False)
    status = Column(String, default="completed")
    timestamp = Column(DateTime, default=datetime.utcnow)

    def to_dict(self) -> Dict[str, Any]:
        """Convert record to dictionary."""
        return {
            "id": self.id,
            "question": self.question,
            "answer": self.answer,
            "status": self.status,
            "timestamp": self.timestamp
        }


class QAStorage:
    """Manages Q&A history storage using SQLite."""

    def __init__(self, database_url: str = "sqlite:///./db/qa_history.db"):
        """
        Initialize storage.

        Args:
            database_url: SQLAlchemy database URL
        """
        self.database_url = database_url

        # Create db directory if needed
        if database_url.startswith("sqlite"):
            db_path = database_url.replace("sqlite:///", "")
            Path(db_path).parent.mkdir(parents=True, exist_ok=True)

        # Create engine and session
        self.engine = create_engine(
            database_url,
            connect_args={"check_same_thread": False} if "sqlite" in database_url else {}
        )
        self.SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=self.engine)

        # Create tables
        Base.metadata.create_all(bind=self.engine)
        logger.info(f"Initialized QAStorage with database: {database_url}")

    def get_session(self) -> Session:
        """Get a new database session."""
        return self.SessionLocal()

    def save_qa(
        self,
        query_id: str,
        question: str,
        answer: str,
        status: str = "completed"
    ) -> QARecord:
        """
        Save a Q&A record.

        Args:
            query_id: Unique query identifier
            question: Question text
            answer: Answer text
            status: Query status

        Returns:
            Created QARecord
        """
        session = self.get_session()
        try:
            record = QARecord(
                id=query_id,
                question=question,
                answer=answer,
                status=status,
                timestamp=datetime.utcnow()
            )
            session.add(record)
            session.commit()
            session.refresh(record)

            logger.info(f"Saved Q&A record with ID: {query_id}")
            return record

        except Exception as e:
            session.rollback()
            logger.error(f"Error saving Q&A record: {e}")
            raise
        finally:
            session.close()

    def get_qa(self, query_id: str) -> Optional[QARecord]:
        """
        Retrieve a Q&A record by ID.

        Args:
            query_id: Query identifier

        Returns:
            QARecord if found, None otherwise
        """
        session = self.get_session()
        try:
            record = session.query(QARecord).filter(QARecord.id == query_id).first()

            if record:
                logger.info(f"Retrieved Q&A record with ID: {query_id}")
            else:
                logger.warning(f"Q&A record not found with ID: {query_id}")

            return record

        except Exception as e:
            logger.error(f"Error retrieving Q&A record: {e}")
            raise
        finally:
            session.close()

    def exists(self, query_id: str) -> bool:
        """
        Check if a query ID already exists.

        Args:
            query_id: Query identifier

        Returns:
            True if exists, False otherwise
        """
        session = self.get_session()
        try:
            count = session.query(QARecord).filter(QARecord.id == query_id).count()
            return count > 0
        except Exception as e:
            logger.error(f"Error checking Q&A existence: {e}")
            raise
        finally:
            session.close()

    def count_records(self) -> int:
        """Get total number of Q&A records."""
        session = self.get_session()
        try:
            return session.query(QARecord).count()
        except Exception as e:
            logger.error(f"Error counting records: {e}")
            raise
        finally:
            session.close()

    def clear_all(self) -> None:
        """Clear all Q&A records (for testing)."""
        session = self.get_session()
        try:
            session.query(QARecord).delete()
            session.commit()
            logger.warning("Cleared all Q&A records")
        except Exception as e:
            session.rollback()
            logger.error(f"Error clearing records: {e}")
            raise
        finally:
            session.close()
