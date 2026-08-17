import os
import logging

from dotenv import load_dotenv

from fastapi import FastAPI
from langserve import add_routes
import uvicorn

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.agents import AgentExecutor, create_react_agent, Tool
from langchain_community.tools import DuckDuckGoSearchRun, WikipediaQueryRun
from langchain_community.utilities import WikipediaAPIWrapper
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnableLambda

# ----------------------------
# Load Environment
# ----------------------------
load_dotenv()

GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")

if not GOOGLE_API_KEY:
    raise ValueError("GOOGLE_API_KEY is missing. Add it to your .env file.")

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("job_ready_agent")

# ----------------------------
# Initialize LLM
# ----------------------------
llm = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash",
    google_api_key=GOOGLE_API_KEY,
    temperature=0,
)

# ----------------------------
# Tools
# ----------------------------
search_tool = DuckDuckGoSearchRun(
    name="web_search"
)

wiki_wrapper = WikipediaAPIWrapper(
    top_k_results=2,
    doc_content_chars_max=2000,
)

wiki_tool = WikipediaQueryRun(
    api_wrapper=wiki_wrapper
)


def calculator(expression: str) -> str:
    """Calculate a basic mathematical expression safely."""
    allowed = set("0123456789+-*/().% ")
    if not expression or any(char not in allowed for char in expression):
        return "Invalid mathematical expression."

    try:
        result = eval(expression, {"__builtins__": {}}, {})
        return str(result)
    except Exception as exc:
        return f"Calculation error: {exc}"


calculator_tool = Tool(
    name="calculator",
    func=calculator,
    description=(
        "Use this tool for arithmetic calculations. "
        "Input should be a mathematical expression such as "
        "'23 * 47 + 12'."
    ),
)

tools = [
    Tool(
        name="web_search",
        func=search_tool.run,
        description=(
            "Search the web for current events, recent facts, "
            "or information you are unsure about."
        ),
    ),
    Tool(
        name="wikipedia",
        func=wiki_tool.run,
        description=(
            "Look up general knowledge, historical facts, "
            "people, places, and background information."
        ),
    ),
    calculator_tool,
]

# ----------------------------
# ReAct Prompt
# ----------------------------
prompt_template = """You are a Job Ready AI Agent.

You can use tools to solve user requests.

Available tools:
{tools}

Use the following format:

Question: the user's question
Thought: decide what to do
Action: one of [{tool_names}]
Action Input: the input for the tool
Observation: the tool result
... repeat if needed ...
Thought: I now know the final answer
Final Answer: the answer to the user

If a tool is not required, answer directly.

Question: {input}
Thought:{agent_scratchpad}
"""

prompt = PromptTemplate.from_template(prompt_template)

# ----------------------------
# Build Agent
# ----------------------------
agent = create_react_agent(
    llm=llm,
    tools=tools,
    prompt=prompt,
)

agent_executor = AgentExecutor(
    agent=agent,
    tools=tools,
    verbose=True,
    handle_parsing_errors=True,
    max_iterations=8,
    max_execution_time=60,
    return_intermediate_steps=True,
)

# ----------------------------
# LangServe Runnable
# ----------------------------
def run_agent(request):
    query = request.get("input", "")
    if not query:
        return "Please provide an input question."

    logger.info("Running Job Ready Agent: %s", query)

    try:
        result = agent_executor.invoke({"input": query})
        return result.get("output", "")
    except Exception as exc:
        logger.exception("Agent execution failed")
        return f"Agent error: {exc}"


agent_runnable = RunnableLambda(run_agent)

# ----------------------------
# FastAPI
# ----------------------------
app = FastAPI(
    title="Job Ready LangChain Agent",
    version="1.0",
    description=(
        "A LangChain ReAct agent using Gemini, web search, "
        "Wikipedia, and a calculator tool."
    ),
)

add_routes(
    app,
    agent_runnable,
    path="/agent",
)

# ----------------------------
# Health Check
# ----------------------------
@app.get("/")
def root():
    return {
        "agent": "LangChain Job Ready Agent",
        "status": "running",
        "endpoint": "/agent",
    }


@app.get("/health")
def health():
    return {"status": "healthy"}


# ----------------------------
# Run
# ----------------------------
if __name__ == "__main__":
    uvicorn.run(
        "langchain_job_ready_agent:app",
        host="0.0.0.0",
        port=int(os.environ.get("PORT", 8000)),
    )
