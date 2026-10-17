
from app.state import GraphState


ABSTAIN_MESSAGE = (
    "I couldn't find sufficient evidence in the provided documents."
)


def abstain_node(state: GraphState) -> GraphState:
    return {
        **state,
        "answer": ABSTAIN_MESSAGE,
        "final_outcome": "abstained_judge_rejected",
    }
