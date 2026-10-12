from api_view import chat_graph
from pathlib import Path

def save_graph_image(graph, path="tmp/graph.png"):
    """Save a compiled LangGraph graph as a PNG image and return the path."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(graph.get_graph().draw_mermaid_png())
    return path

print(save_graph_image(chat_graph))
