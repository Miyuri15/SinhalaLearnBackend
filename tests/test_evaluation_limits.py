# tests/test_evaluation_limits.py
import os
import sys
from datetime import datetime, timedelta
from pathlib import Path
from uuid import uuid4

from sqlalchemy import Column, DateTime, ForeignKey, String, create_engine
from sqlalchemy.orm import declarative_base, sessionmaker

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(BASE_DIR)

Base = declarative_base()


class User(Base):
    __tablename__ = "users"

    id = Column(String, primary_key=True)
    email = Column(String)
    tier = Column(String, default="basic")


class ChatSession(Base):
    __tablename__ = "chat_sessions"

    id = Column(String, primary_key=True)
    user_id = Column(String, ForeignKey("users.id"))
    mode = Column(String, default="learning")


class Message(Base):
    __tablename__ = "messages"

    id = Column(String, primary_key=True)
    session_id = Column(String, ForeignKey("chat_sessions.id"))
    role = Column(String)
    created_at = Column(DateTime, default=datetime.utcnow)


class EvaluationSession(Base):
    __tablename__ = "evaluation_sessions"

    id = Column(String, primary_key=True)
    session_id = Column(String, ForeignKey("chat_sessions.id"))
    created_at = Column(DateTime, default=datetime.utcnow)


from app.services.usage_service import UsageService

import app.services.usage_service

app.services.usage_service.User = User
app.services.usage_service.ChatSession = ChatSession
app.services.usage_service.Message = Message
app.services.usage_service.EvaluationSession = EvaluationSession


def setup_db():
    engine = create_engine("sqlite:///:memory:")
    Base.metadata.create_all(engine)
    Session = sessionmaker(bind=engine)
    return Session()


def test_tier_aliases_return_daily_evaluation_session_limits():
    db = setup_db()
    service = UsageService(db)

    assert service.get_user_tier_limit("normal") == 1
    assert service.get_user_tier_limit("basic") == 1
    assert service.get_user_tier_limit("classroom") == 5
    assert service.get_user_tier_limit("intermediate") == 5
    assert service.get_user_tier_limit("institution") == 10
    assert service.get_user_tier_limit("enterprise") == 10


def test_user_tier_migration_maps_old_tiers():
    migration_path = (
        Path(BASE_DIR)
        / "migrations"
        / "versions"
        / "c9d2f1e8a4b3_update_user_tier_defaults.py"
    )
    migration_text = migration_path.read_text(encoding="utf-8")

    assert "normal' THEN 'basic" in migration_text
    assert "classroom' THEN 'intermediate" in migration_text
    assert "institution' THEN 'enterprise" in migration_text


def test_basic_evaluation_session_limit_blocks_second_session_today():
    db = setup_db()
    service = UsageService(db)
    user_id = str(uuid4())
    chat_id = str(uuid4())

    db.add(User(id=user_id, email="basic@test.com", tier="basic"))
    db.add(ChatSession(id=chat_id, user_id=user_id, mode="evaluation"))
    db.add(EvaluationSession(id=str(uuid4()), session_id=chat_id, created_at=datetime.now()))
    db.commit()

    try:
        service.check_evaluation_session_limit(user_id)
    except Exception as exc:
        assert getattr(exc, "status_code", None) == 403
    else:
        raise AssertionError("Expected evaluation session limit to block the request")


def test_basic_evaluation_session_limit_resets_next_day():
    db = setup_db()
    service = UsageService(db)
    user_id = str(uuid4())
    chat_id = str(uuid4())

    db.add(User(id=user_id, email="basic@test.com", tier="basic"))
    db.add(ChatSession(id=chat_id, user_id=user_id, mode="evaluation"))
    db.add(
        EvaluationSession(
            id=str(uuid4()),
            session_id=chat_id,
            created_at=datetime.now() - timedelta(days=1),
        )
    )
    db.commit()

    assert service.check_evaluation_session_limit(user_id) is True


def test_basic_learning_request_limit_blocks_sixth_request():
    db = setup_db()
    service = UsageService(db)
    user_id = str(uuid4())
    chat_id = str(uuid4())

    db.add(User(id=user_id, email="basic@test.com", tier="basic"))
    db.add(ChatSession(id=chat_id, user_id=user_id, mode="learning"))
    for _ in range(5):
        db.add(
            Message(
                id=str(uuid4()),
                session_id=chat_id,
                role="user",
                created_at=datetime.now(),
            )
        )
    db.commit()

    try:
        service.check_learning_request_limit(user_id)
    except Exception as exc:
        assert getattr(exc, "status_code", None) == 403
    else:
        raise AssertionError("Expected learning request limit to block the request")


def test_intermediate_learning_request_limit_blocks_twenty_first_request():
    db = setup_db()
    service = UsageService(db)
    user_id = str(uuid4())
    chat_id = str(uuid4())

    db.add(User(id=user_id, email="intermediate@test.com", tier="intermediate"))
    db.add(ChatSession(id=chat_id, user_id=user_id, mode="learning"))
    for _ in range(20):
        db.add(
            Message(
                id=str(uuid4()),
                session_id=chat_id,
                role="user",
                created_at=datetime.now(),
            )
        )
    db.commit()

    try:
        service.check_learning_request_limit(user_id)
    except Exception as exc:
        assert getattr(exc, "status_code", None) == 403
    else:
        raise AssertionError("Expected intermediate learning limit to block the request")


def test_enterprise_learning_request_limit_blocks_fifty_first_request():
    db = setup_db()
    service = UsageService(db)
    user_id = str(uuid4())
    chat_id = str(uuid4())

    db.add(User(id=user_id, email="enterprise@test.com", tier="enterprise"))
    db.add(ChatSession(id=chat_id, user_id=user_id, mode="learning"))
    for _ in range(50):
        db.add(
            Message(
                id=str(uuid4()),
                session_id=chat_id,
                role="user",
                created_at=datetime.now(),
            )
        )
    db.commit()

    try:
        service.check_learning_request_limit(user_id)
    except Exception as exc:
        assert getattr(exc, "status_code", None) == 403
    else:
        raise AssertionError("Expected enterprise learning limit to block the request")


def test_intermediate_evaluation_session_limit_blocks_sixth_session_today():
    db = setup_db()
    service = UsageService(db)
    user_id = str(uuid4())
    chat_id = str(uuid4())

    db.add(User(id=user_id, email="intermediate@test.com", tier="intermediate"))
    db.add(ChatSession(id=chat_id, user_id=user_id, mode="evaluation"))
    for _ in range(5):
        db.add(EvaluationSession(id=str(uuid4()), session_id=chat_id, created_at=datetime.now()))
    db.commit()

    try:
        service.check_evaluation_session_limit(user_id)
    except Exception as exc:
        assert getattr(exc, "status_code", None) == 403
    else:
        raise AssertionError("Expected intermediate evaluation limit to block the request")


def test_enterprise_evaluation_session_limit_blocks_eleventh_session_today():
    db = setup_db()
    service = UsageService(db)
    user_id = str(uuid4())
    chat_id = str(uuid4())

    db.add(User(id=user_id, email="enterprise@test.com", tier="enterprise"))
    db.add(ChatSession(id=chat_id, user_id=user_id, mode="evaluation"))
    for _ in range(10):
        db.add(EvaluationSession(id=str(uuid4()), session_id=chat_id, created_at=datetime.now()))
    db.commit()

    try:
        service.check_evaluation_session_limit(user_id)
    except Exception as exc:
        assert getattr(exc, "status_code", None) == 403
    else:
        raise AssertionError("Expected enterprise evaluation limit to block the request")


def test_basic_evaluations_per_session_limit_blocks_over_ten():
    db = setup_db()
    service = UsageService(db)
    user_id = str(uuid4())

    db.add(User(id=user_id, email="basic@test.com", tier="basic"))
    db.commit()

    answer_resource_ids = [uuid4() for _ in range(11)]

    try:
        service.check_evaluations_per_session_limit(user_id, answer_resource_ids)
    except Exception as exc:
        assert getattr(exc, "status_code", None) == 403
    else:
        raise AssertionError("Expected per-session evaluation limit to block the request")
