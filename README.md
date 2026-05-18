# CAT — Computer Adaptive Test Engine

An AI-powered adaptive testing platform that dynamically generates, evaluates, and serves questions tailored to each student's ability in real time — built on Item Response Theory and a multi-node LLM pipeline.

---

## What is CAT?

Most practice platforms give everyone the same questions. CAT doesn't.

CAT tracks your ability as you answer, and every question it generates is calibrated to your current level — not too easy, not too hard. Get something right and the next question gets harder. Get it wrong and it recalibrates. Over time, it builds a precise picture of where you're strong and where you need work, then generates a personalized study plan.

---

## How it works

### Adaptive Engine (IRT)
CAT uses a 2-Parameter Item Response Theory model to track student ability. Every answer updates your ability score using:

```
P(correct) = 1 / (1 + e ^ (−discrimination × (ability − difficulty)))
new_ability = ability + learning_rate × discrimination × (result − P(correct))
```

Questions are selected to maximize Fisher Information at your current ability — meaning every question is the one that will tell us the most about you.

### LLM Question Pipeline (LangGraph)
Questions aren't pulled from a static bank — they're generated fresh by a 4-node LangGraph pipeline:

```
gen_q → gen_opt → gen_exp → eval
          ↑          ↑        ↑
          └──────────┴────────┘
              retry if score < threshold
```

- **gen_q** — generates a question matched to your difficulty and topic
- **gen_opt** — generates 4 options with plausible distractors
- **gen_exp** — generates a clean explanation of the correct answer
- **eval** — scores all three components (0.0 → 1.0) and triggers selective regeneration if any fall below threshold

Only the failing component gets regenerated — no wasted LLM calls.

---

## Tech Stack

| Layer | Technology |
|---|---|
| Backend | FastAPI |
| Database | PostgreSQL + SQLAlchemy |
| LLM Orchestration | LangGraph + LangChain |
| LLM Provider | Groq (Llama 3.3 70B) |
| Frontend | HTML + CSS |
| Adaptive Algorithm | Custom IRT Engine |

---

## Features

- Real-time difficulty adaptation per student session
- AI-generated questions, options, and explanations — no static question bank
- Self-evaluating pipeline with automatic retry on quality failure
- Topic distribution tracking to avoid repetition
- Session insights with personalized AI study plan
- Clean REST API — easy to integrate with any frontend

---

## Project Structure

```
CAT/
├── backend/
│   ├── app/
│   │   ├── main.py              # FastAPI entry point
│   │   ├── routes.py            # API endpoints
│   │   ├── llm_questions.py     # LangGraph pipeline
│   │   ├── generator_llm.py     # Question generation node
│   │   ├── options_llm.py       # Options generation node
│   │   ├── explanation_llm.py   # Explanation generation node
│   │   ├── evaluator_llm.py     # Eval + retry routing node
│   │   ├── irt_lite.py          # IRT adaptive engine
│   │   ├── llm_insight.py       # Study plan generation
│   │   ├── db_models.py         # SQLAlchemy models
│   │   └── schemas.py           # Pydantic schemas
│   ├── requirements.txt
│   └── .env
├── frontend/
│   └── index.html
├── notebooks/
│   └── 0_workflow.ipynb
└── README.md
```

---

## Getting Started

### Prerequisites
- Python 3.11+
- PostgreSQL
- A free Groq API key from [console.groq.com](https://console.groq.com)

### Installation

```bash
git clone https://github.com/yourusername/CAT.git
cd CAT/backend
python -m venv venv
source venv/Scripts/activate  # Windows
pip install -r requirements.txt
```

### Environment Setup

Create a `.env` file inside `backend/`:

```env
DATABASE_URL=postgresql://user:password@localhost:5432/cat_db
GROQ_API_KEY=your_groq_api_key_here
GROQ_MODEL=llama-3.3-70b-versatile
MAX_RETRIES=3
PASS_THRESHOLD=0.7
```

### Run

```bash
cd backend
uvicorn app.main:app --reload
```

Open [http://localhost:8000/docs](http://localhost:8000/docs) for the interactive API docs.

---

## API Reference

### `POST /start-session`
Start a new test session.

```json
// Request
{ "subjects": ["Python", "Deep Learning"], "topics": ["CNN", "RNN"], "exam": "AI Engineer Interview" }

// Response
{ "session_id": "uuid", "user_hash": "uuid" }
```

### `POST /generate_questions`
Generate the next adaptive question for a session.

```json
// Request
{ "session_id": "uuid" }

// Response
{ "q_text": "...", "options": {"A": "...", "B": "...", "C": "...", "D": "..."}, "difficulty": 0.6, "topic": "CNN" }
```

### `POST /submit-answer`
Submit an answer and update ability score.

```json
// Request
{ "session_id": "uuid", "question_id": "uuid", "user_answer": "B" }

// Response
{ "correct": true, "correct_answer": "B", "explanation": "...", "new_difficulty": 0.65 }
```

### `GET /get_insights/{session_id}`
Get performance analytics and an AI-generated study plan.

```json
// Response
{ "insight": "Based on your session, you're strong in CNNs but struggling with LSTM gating mechanisms..." }
```

---

## License

MIT
