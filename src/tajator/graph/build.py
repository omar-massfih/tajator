"""StateGraph wiring.

START → fetch_data → compute_context
  position open → manage_position → {stop/runner → do_exit | target → llm_manage → do_scale_out/do_exit | hold → END}
  flat          → detect_setups → {none → END | llm_decide → risk_gate → {veto → END | do_entry}} → END
"""

from __future__ import annotations

from langgraph.graph import END, START, StateGraph

from .nodes import RuntimeContext, make_nodes
from .state import AgentState


def route_mode(state: AgentState) -> str:
    return "manage_position" if state.get("position") is not None else "detect_setups"


def route_manage(state: AgentState) -> str:
    kind = state["manage_action"].kind
    if kind in ("stop_exit", "runner_exit"):
        return "do_exit"
    if kind == "scale_candidate":
        return "llm_manage"
    return END


def route_manage_decision(state: AgentState) -> str:
    action = state["decision"].action
    if action == "scale_out":
        return "do_scale_out"
    if action == "exit":
        return "do_exit"
    return END  # hold one more bar


def route_setups(state: AgentState) -> str:
    if not state.get("candidates") or state.get("entry_blockers"):
        return END
    return "llm_decide"


def route_decision(state: AgentState) -> str:
    return "risk_gate" if state["decision"].action in ("enter_call", "enter_put") else END


def route_risk(state: AgentState) -> str:
    return "do_entry" if state["risk"].approved else END


def build_graph(ctx: RuntimeContext):
    nodes = make_nodes(ctx)
    g = StateGraph(AgentState)
    for name, fn in nodes.items():
        g.add_node(name, fn)

    g.add_edge(START, "fetch_data")
    g.add_edge("fetch_data", "compute_context")
    g.add_conditional_edges("compute_context", route_mode, ["manage_position", "detect_setups"])

    g.add_conditional_edges("manage_position", route_manage, ["do_exit", "llm_manage", END])
    g.add_conditional_edges("llm_manage", route_manage_decision, ["do_scale_out", "do_exit", END])
    g.add_edge("do_scale_out", END)
    g.add_edge("do_exit", END)

    g.add_conditional_edges("detect_setups", route_setups, ["llm_decide", END])
    g.add_conditional_edges("llm_decide", route_decision, ["risk_gate", END])
    g.add_conditional_edges("risk_gate", route_risk, ["do_entry", END])
    g.add_edge("do_entry", END)

    return g.compile()
