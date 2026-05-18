import os
from dotenv import load_dotenv
load_dotenv()
from langgraph.graph import StateGraph, START, END

from app.schemas import UserSession
from app.generator_llm import gen_qtext_node
from app.options_llm import gen_options_node
from app.explanation_llm import gen_explanation_node
from app.evaluator_llm import eval_node

load_dotenv()
MAX_RETRIES = int(os.getenv("MAX_RETRIES", 3))

# ── Conditional edge ──────────────────────────────────────────────────────────
def route_after_eval(state: dict) -> str:
    retries = state.get("RetryCount", 0)

    if retries >= MAX_RETRIES:
        return "discard"

    if state.get("RetryQ"):
        return "regen_q"

    if state.get("RetryOpt"):
        return "regen_opt"

    if state.get("RetryExp"):
        return "regen_exp"

    return "accept"


# ── Graph builder ─────────────────────────────────────────────────────────────
def build_graph():
    nodes = [
        ("gen_q",   gen_qtext_node),
        ("gen_opt", gen_options_node),
        ("gen_exp", gen_explanation_node),
        ("eval",    eval_node),
    ]

    edges = [
        (START,     "gen_q"),
        ("gen_q",   "gen_opt"),
        ("gen_opt", "gen_exp"),
        ("gen_exp", "eval"),
    ]

    conditional_edges = [
        ("eval", route_after_eval, {
            "accept":    END,
            "discard":   END,
            "regen_q":   "gen_q",    # q failed → full restart
            "regen_opt": "gen_opt",  # opt failed → skip gen_q
            "regen_exp": "gen_exp",  # exp failed → skip gen_q + gen_opt
        })
    ]

    builder = StateGraph(UserSession)

    for name, fn in nodes:
        builder.add_node(name, fn)

    for src, dst in edges:
        builder.add_edge(src, dst)

    for src, fn, mapping in conditional_edges:
        builder.add_conditional_edges(src, fn, mapping)

    return builder.compile()


if __name__ == "__main__":
    graph = build_graph()
    
    initial_state = {
        "Exam":       "Interview for AI Engineer at fxis.ai",
        "Subjects":   ["Python", "Deep Learning"],
        "Topics":     ["ANN", "RNN", "LSTM", "CNN"],
        "Difficulty": 0.5,
        "History":    None,
        "RetryCount": 0,
    }

    final_state = graph.invoke(initial_state)
    for k in final_state.keys():
        print(f"{k}: {final_state[k]}")