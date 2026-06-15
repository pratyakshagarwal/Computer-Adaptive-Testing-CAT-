import os 
from dotenv import load_dotenv
load_dotenv()

from langchain_core.prompts import ChatPromptTemplate
from langchain_groq import ChatGroq

from app.schemas import Option_Schema, Question_Schema, Explain_Schema, _difficulty_label

GROQ_API_KEY = os.getenv("GROQ_API_KEY", "")
GROQ_MODEL   = os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile")


EXPLAIN_SYSTEM = """
You are an expert tutor for a Computer Adaptive Test.
Given a question, its correct answer, and all options, write a clear concise explanation.

RULES:
 - Explain WHY the correct answer is right
 - Explain WHY each wrong option is wrong (one line each)
 - Keep it concise and technical — this is for serious exam prep
 - Adapt depth to difficulty level

OUTPUT CONTRACT:
 - explanation -> clear explanation of correct answer and why distractors are wrong
No extra fields, no markdown, no preamble
"""

EXPLAIN_HUMAN = """
Question   : {q_text}
Options    : {options}
Correct    : {solution}
Subject    : {subject}
Topic      : {topic}
Difficulty : {difficulty} ({difficulty_label})

Generate explanation now.
"""

_llm   = ChatGroq(model=GROQ_MODEL, temperature=0.3, api_key=GROQ_API_KEY)
_exp_llm    = _llm.with_structured_output(Explain_Schema)
_exp_prompt = ChatPromptTemplate.from_messages([
    ("system", EXPLAIN_SYSTEM),
    ("human",  EXPLAIN_HUMAN),
])

def gen_explanation_node(state: dict) -> dict:
    q:   Question_Schema = state.get("Q")
    opt: Option_Schema   = state.get("Options")

    options_str = " | ".join([f"{k}: {v}" for k, v in opt.opt.items()])

    response = (_exp_prompt | _exp_llm).invoke({
        "q_text":           q.q,
        "options":          options_str,
        "solution":         f"{opt.solution}: {opt.opt[opt.solution]}",
        "subject":          q.tags.get("subject", "General"),
        "topic":            q.tags.get("topic", "General"),
        "difficulty":       q.difficulty,
        "difficulty_label": _difficulty_label(q.difficulty),
    })

    return {"Explanation": response}