import os 
from dotenv import load_dotenv
load_dotenv()

from langchain_core.prompts import ChatPromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI

from app.schemas import Option_Schema, Question_Schema, Explain_Schema, Eval_Schema, _difficulty_label

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")
GEMINI_MODEL   = os.getenv("GEMINI_MODEL", "gemini-3.1-flash-lite")
THRESHOLD = float(os.getenv("PASS_THRESHOLD", 0.7)) 

EVAL_SYSTEM = """
You are a strict quality evaluator for a Computer Adaptive Test question pipeline.
You will evaluate three components independently: the question, the options, and the explanation.

SCORING CRITERIA:

Question (q_score):
  1.0 → unambiguous, matches difficulty exactly, tests the right concept
  0.7 → minor clarity issues or slight difficulty mismatch
  0.4 → ambiguous, too easy/hard for stated difficulty, or tests wrong concept
  0.0 → completely unusable

Options (opt_score):
  1.0 → all distractors are plausible, no giveaways, correct answer is defensible
  0.7 → one distractor is weak or slightly obvious
  0.4 → multiple weak distractors or correct answer is debatable
  0.0 → completely unusable

Explanation (exp_score):
  1.0 → clearly explains correct answer and all wrong options, appropriate depth
  0.7 → explanation is correct but missing depth or skips a distractor
  0.4 → explanation is incomplete or partially incorrect
  0.0 → completely unusable

OUTPUT CONTRACT:
 - q_score, opt_score, exp_score -> float between 0.0 and 1.0
 - q_feedback, opt_feedback, exp_feedback -> one sentence each, actionable
No extra fields, no markdown, no preamble
"""

EVAL_HUMAN = """
Question     : {q_text}
Difficulty   : {difficulty} ({difficulty_label})
Subject      : {subject}
Topic        : {topic}

Options      : {options}
Correct      : {solution}

Explanation  : {explanation}

Evaluate all three components now.
"""
_eval_prompt = ChatPromptTemplate.from_messages([
    ("system", EVAL_SYSTEM),
    ("human",  EVAL_HUMAN),
])

def eval_node(state: dict) -> dict:
    q   = state.get("Q")
    opt = state.get("Options")
    exp = state.get("Explanation")

    options_str = " | ".join([f"{k}: {v}" for k, v in opt.opt.items()])
    
    _llm      =  ChatGoogleGenerativeAI(model=GEMINI_MODEL, temperature=0.3, api_key=GEMINI_API_KEY)
    _eval_llm = _llm.with_structured_output(Eval_Schema)

    response = (_eval_prompt | _eval_llm).invoke({
        "q_text":           q.q,
        "difficulty":       q.difficulty,
        "difficulty_label": _difficulty_label(q.difficulty),
        "subject":          q.tags.get("subject", "General"),
        "topic":            q.tags.get("topic", "General"),
        "options":          options_str,
        "solution":         f"{opt.solution}: {opt.opt[opt.solution]}",
        "explanation":      exp.explanation,
    })

    return {
        "Eval":         response,
        "RetryCount":   state.get("RetryCount", 0) + 1,  # ✅ increment here
        "RetryQ":       response.q_score   < THRESHOLD,
        "RetryOpt":     response.opt_score < THRESHOLD,
        "RetryExp":     response.exp_score < THRESHOLD,
        "EvalFeedback": (
            response.q_feedback   if response.q_score   < THRESHOLD else
            response.opt_feedback if response.opt_score < THRESHOLD else
            response.exp_feedback if response.exp_score < THRESHOLD else None
        ),
    }