from __future__ import annotations

import json
import os
import time
from typing import Any, TypedDict

from dotenv import load_dotenv

load_dotenv()

MODEL = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
QUESTION = os.getenv(
    "QUESTION",
    "What's the weather in Tokyo right now? Use the weather tool.",
)


# LangChain
from langchain_core.messages import HumanMessage, SystemMessage, ToolMessage
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI
from langgraph.graph import START, MessagesState, StateGraph
from langgraph.prebuilt import ToolNode, tools_condition


@tool
def get_weather(city: str) -> str:
    """Get the current weather for a city (demo data)."""
    return f"{city}: 82°F, sunny"


def build_langchain_flow():
    llm = ChatOpenAI(model=MODEL, temperature=0).bind_tools([get_weather])

    def agent(state: MessagesState) -> dict[str, list[Any]]:
        return {"messages": [llm.invoke(state["messages"])]}

    graph = StateGraph(MessagesState)
    graph.add_node("agent", agent)
    graph.add_node("tools", ToolNode([get_weather]))
    graph.add_edge(START, "agent")
    graph.add_conditional_edges("agent", tools_condition)
    graph.add_edge("tools", "agent")
    return graph.compile()


# No langchain
from openai import OpenAI
from langgraph.graph import END


def get_weather_without_langchain(city: str) -> str:
    """Get the current weather for a city (demo data)."""
    return f"{city}: 82°F, sunny"


TOOLS = [
    {
        "type": "function",
        "name": "get_weather",
        "description": "Get the current weather for a city (demo data).",
        "parameters": {
            "type": "object",
            "properties": {"city": {"type": "string"}},
            "required": ["city"],
            "additionalProperties": False,
        },
        "strict": True,
    }
]
REGISTRY = {"get_weather": get_weather_without_langchain}


class ManualState(TypedDict):
    messages: list[dict[str, Any]]
    pending_calls: list[dict[str, Any]]


def build_manual_flow():
    client = OpenAI()

    def agent(state: ManualState) -> dict[str, Any]:
        response = client.responses.create(
            model=MODEL,
            input=state["messages"],
            tools=TOOLS,
            instructions=(
                "You are a helpful assistant. For weather questions, you must call "
                "the get_weather tool before answering."
            ),
        )
        output_items = [item.model_dump(exclude_none=True) for item in response.output]
        calls = [item for item in output_items if item.get("type") == "function_call"]
        return {
            "messages": state["messages"] + output_items,
            "pending_calls": calls,
        }

    def execute_tools(state: ManualState) -> dict[str, Any]:
        outputs = []
        for call in state["pending_calls"]:
            name = call["name"]
            arguments = json.loads(call["arguments"])
            result = REGISTRY[name](**arguments)
            outputs.append(
                {
                    "type": "function_call_output",
                    "call_id": call["call_id"],
                    "output": result,
                }
            )
        return {
            "messages": state["messages"] + outputs,
            "pending_calls": [],
        }

    def route(state: ManualState) -> str:
        return "tools" if state["pending_calls"] else END

    graph = StateGraph(ManualState)
    graph.add_node("agent", agent)
    graph.add_node("tools", execute_tools)
    graph.add_edge(START, "agent")
    graph.add_conditional_edges("agent", route)
    graph.add_edge("tools", "agent")
    return graph.compile()


def langchain_answer(messages: list[Any]) -> str:
    for message in reversed(messages):
        if getattr(message, "type", None) == "ai":
            content = message.content
            if isinstance(content, str):
                return content
            return json.dumps(content, ensure_ascii=False)
    return "(No assistant response found.)"


def manual_answer(messages: list[dict[str, Any]]) -> str:
    for message in reversed(messages):
        if message.get("type") == "message" and message.get("role") == "assistant":
            text = [
                part.get("text", "")
                for part in message.get("content", [])
                if part.get("type") == "output_text"
            ]
            if text:
                return "\n".join(text)
    return "(No assistant response found.)"


def run_comparison() -> None:
    if not os.getenv("OPENAI_API_KEY"):
        raise SystemExit(
            "OPENAI_API_KEY is not set. Copy .env.example to .env and add your key."
        )

    initial_messages = [
        SystemMessage(
            content=(
                "You are a helpful assistant. For weather questions, you must call "
                "the get_weather tool before answering."
            )
        ),
        HumanMessage(content=QUESTION),
    ]
    langchain_flow = build_langchain_flow()
    manual_flow = build_manual_flow()

    print(f"Model: {MODEL}")
    print(f"Prompt: {QUESTION}")
    print("Timing is wall-clock end-to-end for each flow (including model API calls).\n")

    started = time.perf_counter()
    langchain_state = langchain_flow.invoke(
        {"messages": initial_messages}, config={"recursion_limit": 10}
    )
    langchain_seconds = time.perf_counter() - started
    langchain_messages = langchain_state["messages"]
    langchain_tool_count = sum(
        isinstance(message, ToolMessage) for message in langchain_messages
    )
    print("**With LangChain")
    print(f"Answer: {langchain_answer(langchain_messages)}")
    print(f"Tool calls: {langchain_tool_count}")
    print(f"Elapsed: {langchain_seconds:.2f} seconds\n")

    manual_input: list[dict[str, Any]] = [
        {
            "role": "user",
            "content": QUESTION,
        }
    ]
    started = time.perf_counter()
    manual_state = manual_flow.invoke(
        {"messages": manual_input, "pending_calls": []},
        config={"recursion_limit": 10},
    )
    manual_seconds = time.perf_counter() - started
    manual_messages = manual_state["messages"]
    manual_tool_count = sum(
        message.get("type") == "function_call_output" for message in manual_messages
    )
    print("**Without LangChain (OpenAI SDK + LangGraph)**")
    print(f"Answer: {manual_answer(manual_messages)}")
    print(f"Tool calls: {manual_tool_count}")
    print(f"Elapsed: {manual_seconds:.2f} seconds\n")

if __name__ == "__main__":
    run_comparison()