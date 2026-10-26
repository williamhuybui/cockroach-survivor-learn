import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # repo root, so api_view is importable
from api_view import chat_graph


def save_graph_image(graph, path="tmp/graph.png"):
    """Save a compiled LangGraph graph as a PNG image and return the path."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(graph.get_graph().draw_mermaid_png())
    return path

print(save_graph_image(chat_graph))
