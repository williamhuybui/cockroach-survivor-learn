# LangGraph: With and Without LangChain

This small demo runs two equivalent weather-tool flows:

- **With LangChain:** `ChatOpenAI`, `@tool`, `ToolNode`, and `tools_condition` handle most tool plumbing.
- **Without LangChain:** the OpenAI Python SDK is called directly; the code defines the tool JSON schema, parses function calls, dispatches the tool, formats outputs, manages history, and routes the graph. LangGraph itself is still used in both examples.

The output shows each answer, tool-call count, and end-to-end elapsed time. Both flows use the same model, tool data, and question, and execute sequentially. The weather function returns fixed demo data; it does not call a weather service. API/network/model latency can vary between runs, so timing is illustrative rather than a benchmark.

## Run

1. Use Python 3.10 or newer and install dependencies:

   ```sh
   pip install -r requirements.txt
   ```

2. Copy `.env.example` to `.env` and replace the placeholder with your OpenAI API key.
3. Run `python main.py`.

Optionally set `OPENAI_MODEL` or `QUESTION` in `.env`. The default model is `gpt-4o-mini`.

## What the comparison demonstrates

LangChain shortens integration code by standardizing tool definitions, tool execution, and routing. The direct-SDK flow exposes those steps explicitly, which can be useful when custom control is needed, but means more code to maintain. The actual response time is dominated by API/network latency; fewer lines do not necessarily make the model respond faster.