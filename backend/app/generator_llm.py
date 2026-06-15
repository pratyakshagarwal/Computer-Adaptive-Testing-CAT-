import os 
from dotenv import load_dotenv
load_dotenv()

from langchain_core.prompts import ChatPromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI
from app.schemas import Question_Schema, _difficulty_label

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")
GEMINI_MODEL   = os.getenv("GEMINI_MODEL", "gemini-3.1-flash-lite")

QGEN_SYSTEM = """

Your task is to produce ONE multiple-choice question that will be used in a
Computer Adaptive Test. The question must be unambiguous, pedagogically sound,
and match the requested difficulty exactly

DIFFICULTY SCALE (target value supplied in the request):
  0.1–0.3  → recall / basic definition, one concept, no calculations
  0.4–0.6  → conceptual understanding, single-step reasoning or small calculation
  0.7–0.8  → multi-step reasoning, requires connecting two concepts
  0.9–1.0  → edge cases, common misconceptions as distractors, deep understanding

OUTPUT CONTRACT:
 - "q_text"  -> "question text"
 - "difficulty" -> float matching the requested value ±0.05
 -  tags -> {{"subject": ..., "topic":..., "sub_topic":...}}
No extra fields, no markdown, no preamble
"""

# ── Human prompt ─────────────────────────────────────────────────────────────
QGEN_HUMAN = """
Exam         : {exam}
Subject(s)   : {subjects}
Topic(s)     : {topics}
Difficulty   : {difficulty}  ({difficulty_label})

--- HISTORY (avoid repeating these sub_topics) ---
{history}

--- EVALUATOR FEEDBACK (from previous attempt, address all points) ---
{feedback}

Generate a question now.
"""

_llm   = ChatGoogleGenerativeAI(model=GEMINI_MODEL, temperature=0.3, api_key=GEMINI_API_KEY)
_q_llm = _llm.with_structured_output(Question_Schema)

_prompt = ChatPromptTemplate.from_messages([
    ("system", QGEN_SYSTEM),
    ("human",  QGEN_HUMAN),
])

def gen_qtext_node(state: dict) -> dict:
    difficulty = state.get("Difficulty", 0.5)
    feedback   = state.get("EvalFeedback") or "None — first attempt."

    response = (_prompt | _q_llm).invoke({
        "exam":             state.get("Exam") or "General Practice",
        "subjects":         ", ".join(state.get("Subjects", [])),  # ✅ plural
        "topics":           ", ".join(state.get("Topics", [])),    # ✅ plural
        "difficulty":       difficulty,
        "difficulty_label": _difficulty_label(difficulty),
        "history":          state.get("History") or "No history yet.",
        "feedback":         feedback,
    })

    return {
        "Q":          response,
        "RetryCount": state.get("RetryCount", 0),
    }

