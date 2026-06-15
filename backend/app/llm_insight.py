import os
from dotenv import load_dotenv
from pydantic import BaseModel, Field
from typing import Dict

from langchain_core.prompts import ChatPromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI


load_dotenv()

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")
GEMINI_MODEL   = os.getenv("GEMINI_MODEL", "gemini-3.1-flash-lite")


system_prompt = """
You are a brutally honest performance analyst for a student.

Your job is to:
- Identify weak areas precisely
- Identify strong areas
- Detect patterns in mistakes
- Suggest a focused improvement plan

Do NOT give generic advice.
Be specific and actionable.
"""

human_prompt = """
Student Performance Summary:

Overall Accuracy: {accuracy}
Recent Accuracy (last 5): {recent_accuracy}

Weak Topics:
{weak_topics}

Strong Topics:
{strong_topics}

Topic Breakdown:
{topic_stats}

Current Ability (theta): {theta}

Generate:
1. Key weaknesses (max 3)
2. Key strengths (max 3)
3. Mistake patterns
4. A precise improvement plan (step-by-step)
"""

# 3. Create the Template
prompt_template = ChatPromptTemplate.from_messages([
    ("system", system_prompt),
    ("human", human_prompt),
])



I_LLM   = ChatGoogleGenerativeAI(model=GEMINI_MODEL, temperature=0.3, api_key=GEMINI_API_KEY)

def generate_plan(state):

    chain = prompt_template | I_LLM

    response = chain.invoke({
        "accuracy": state['accuracy'],
        "recent_accuracy": state['recent_accuracy'],
        "weak_topics": state['weak_topics'],
        "strong_topics": state['strong_topics'],
        "topic_stats": state['strong_topics'],
        "theta": state['theta']
    })

    return response