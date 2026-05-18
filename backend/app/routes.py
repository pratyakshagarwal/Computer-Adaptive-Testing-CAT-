from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, Field
from typing import List, Optional, Dict
import uuid, copy

from app.llm_questions import build_graph
from app.llm_insight import generate_plan
from app.irt_lite import IRTEngine
from app.db_models import get_db, SessionModel, QuestionModel, AttemptModel, ModelInsight, Session
from app.schemas import SessionCreate, GenerateInsightRequest, GenerateQuestionRequest, QuestionResponse, SubmitAnswerRequest

# Router
router = APIRouter()
graph = build_graph()

# ── /start-session ────────────────────────────────────────────────────────────
@router.post("/start-session")
def start_session(data: SessionCreate, db: Session = Depends(get_db)):
    user_hash  = str(uuid.uuid4())
    session_id = str(uuid.uuid4())

    new_session = SessionModel(
        id=session_id,
        user_hash=user_hash,
        subjects=data.subjects,
        topics=data.topics,
        exam=data.exam,
        difficulty=0.5,
    )

    db.add(new_session)
    db.commit()

    return {
        "message":    "Session created",
        "user_hash":  user_hash,
        "session_id": session_id
    }


# ── /generate_questions ───────────────────────────────────────────────────────
@router.post("/generate_questions", response_model=QuestionResponse)
def generate_question(data: GenerateQuestionRequest, db: Session = Depends(get_db)):

    session = db.query(SessionModel).filter(SessionModel.id == data.session_id).first()
    if not session:
        raise HTTPException(status_code=404, detail="Session not found")

    # ── Build LLM state ───────────────────────────────────────────────────────
    state = {
        "Subjects":   session.subjects,
        "Topics":     session.topics,
        "Exam":       session.exam,
        "Difficulty": session.difficulty,
        "History":    session.topic_distribution,
        "RetryCount": 0,
        "RetryQ":     False,
        "RetryOpt":   False,
        "RetryExp":   False,
    }
    import traceback
    try:
        final_state = graph.invoke(state)
    except Exception as e:
        traceback.print_exc()  # prints full error in terminal
        raise HTTPException(status_code=500, detail=f"LLM generation failed: {str(e)}")

    # ── Extract from new pipeline state ───────────────────────────────────────
    q   = final_state.get("Q")
    opt = final_state.get("Options")
    exp = final_state.get("Explanation")

    if not q or not opt or not exp:
        raise HTTPException(status_code=500, detail="Pipeline failed to generate a complete question")

    # ── Save to DB ────────────────────────────────────────────────────────────
    q_model = QuestionModel(
        id          = str(uuid.uuid4()),
        session_id  = session.id,

        q_text      = q.q,
        options     = opt.opt,
        solution    = opt.solution,
        explanation = exp.explanation,

        difficulty  = q.difficulty,
        subject     = q.tags.get("subject"),
        topic       = q.tags.get("topic"),
        sub_topic   = q.tags.get("sub_topic"),
    )

    # ── Update topic distribution ─────────────────────────────────────────────
    key  = (q_model.topic or "") + "_" + (q_model.sub_topic or "")
    dist = copy.copy(session.topic_distribution) or {}
    dist[key] = dist.get(key, 0) + 1
    session.topic_distribution = dist

    db.add(q_model)
    db.commit()
    db.refresh(q_model)

    return q_model


# ── /submit-answer ────────────────────────────────────────────────────────────
@router.post("/submit-answer")
def submit_answer(data: SubmitAnswerRequest, db: Session = Depends(get_db)):
    session = db.query(SessionModel).filter(
        SessionModel.id == data.session_id
    ).first()
    if not session:
        raise HTTPException(404, "Session not found")

    question = db.query(QuestionModel).filter(
        QuestionModel.id == data.question_id
    ).first()
    if not question:
        raise HTTPException(404, "Question not found")

    is_correct = data.user_answer == question.solution

    attempt = AttemptModel(
        id          = str(uuid.uuid4()),
        session_id  = data.session_id,
        question_id = data.question_id,
        user_answer = data.user_answer,
        is_correct  = is_correct
    )
    db.add(attempt)

    # ── IRT difficulty update ─────────────────────────────────────────────────
    irt       = IRTEngine(ability=0.5)
    irt.theta = session.theta
    irt.update(correct=is_correct, question_difficulty=question.difficulty)

    d_irt    = irt.update_difficulty()
    new_diff = irt.from_irt(d_irt)

    session.theta      = irt.theta
    session.difficulty = new_diff

    db.commit()

    return {
        "correct":        is_correct,
        "correct_answer": question.solution,
        "explanation":    question.explanation,
        "new_difficulty": new_diff
    }


# ── /get_insights ─────────────────────────────────────────────────────────────
@router.get("/get_insights/{session_id}")
def get_insights(session_id: str, db: Session = Depends(get_db)):
    from collections import defaultdict

    session = db.query(SessionModel).filter(
        SessionModel.id == session_id
    ).first()
    if not session:
        raise HTTPException(404, "Session not found")

    questions = db.query(QuestionModel).filter(
        QuestionModel.session_id == session_id
    ).all()

    attempts = db.query(AttemptModel).filter(
        AttemptModel.session_id == session_id
    ).all()

    if not attempts:
        return {"message": "Not enough data"}

    # ── Aggregation ───────────────────────────────────────────────────────────
    total    = len(attempts)
    correct  = sum(a.is_correct for a in attempts)
    accuracy = correct / total

    topic_stats = defaultdict(lambda: {"total": 0, "correct": 0})
    q_map       = {q.id: q for q in questions}

    for a in attempts:
        q = q_map.get(a.question_id)
        if not q:
            continue
        topic_stats[q.topic]["total"] += 1
        if a.is_correct:
            topic_stats[q.topic]["correct"] += 1

    weak_topics   = [t for t, v in topic_stats.items() if v["correct"] / v["total"] < 0.5]
    strong_topics = [t for t, v in topic_stats.items() if v["correct"] / v["total"] > 0.8]
    recent_accuracy = sum(a.is_correct for a in attempts[-5:]) / len(attempts[-5:])

    # ── LLM insight ───────────────────────────────────────────────────────────
    state = {
        "accuracy":         accuracy,
        "recent_accuracy":  recent_accuracy,
        "weak_topics":      weak_topics,
        "strong_topics":    strong_topics,
        "topic_stats":      dict(topic_stats),
        "theta":            session.theta
    }
    response = generate_plan(state)

    return {"insight": response.content}