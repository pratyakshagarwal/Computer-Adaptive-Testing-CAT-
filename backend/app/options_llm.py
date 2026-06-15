import os 
from dotenv import load_dotenv
load_dotenv()
from langchain_core.prompts import ChatPromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI

from app.schemas import Option_Schema, Question_Schema, _difficulty_label

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")
GEMINI_MODEL   = os.getenv("GEMINI_MODEL", "gemini-3.1-flash-lite")

_llm   = ChatGoogleGenerativeAI(model=GEMINI_MODEL, temperature=0.3, api_key=GEMINI_API_KEY)


OPTGEN_SYSTEM = """
You are an expert question designer for a Computer Adaptive Test.
Given a question, generate exactly 4 multiple choice options (A, B, C, D).

RULES:
 - Exactly ONE option must be correct
 - The 3 wrong options (distractors) must be plausible, not obviously wrong
 - Distractors should reflect common misconceptions or mistakes
 - Higher difficulty → distractors should be harder to distinguish from correct answer
 - Options should be similar in length and style (no giveaways)
 - No "All of the above" or "None of the above"

OUTPUT CONTRACT:
 - opt      -> {{"A": "...", "B": "...", "C": "...", "D": "..."}}
 - solution -> the key of the correct option (e.g., "A")
No extra fields, no markdown, no preamble
"""

OPTGEN_HUMAN = """
Question     : {q_text}
Subject      : {subject}
Topic        : {topic}
Difficulty   : {difficulty} ({difficulty_label})

Generate 4 options now.
"""

_opt_llm = _llm.with_structured_output(Option_Schema)

_opt_prompt = ChatPromptTemplate.from_messages([
    ("system", OPTGEN_SYSTEM),
    ("human",  OPTGEN_HUMAN),
])

def gen_options_node(state: dict) -> dict:
    """
    Reads from state:
        Q (Question_Schema object)
    Writes to state:
        Options (Option_Schema object)
    """
    q: Question_Schema = state.get("Q")

    response = (_opt_prompt | _opt_llm).invoke({
        "q_text":           q.q,
        "subject":          q.tags.get("subject", "General"),
        "topic":            q.tags.get("topic", "General"),
        "difficulty":       q.difficulty,
        "difficulty_label": _difficulty_label(q.difficulty),
    })

    return {"Options": response}