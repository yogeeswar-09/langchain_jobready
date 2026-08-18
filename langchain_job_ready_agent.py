import os
import logging

from dotenv import load_dotenv

from fastapi import FastAPI
from langserve import add_routes
import uvicorn

from pydantic import BaseModel

from langchain_google_genai import ChatGoogleGenerativeAI

from langchain_classic.agents import (
    AgentExecutor,
    create_react_agent,
)

from langchain_core.tools import Tool
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnableLambda

from langchain_community.tools import (
    DuckDuckGoSearchRun,
    WikipediaQueryRun,
)

from langchain_community.utilities import (
    WikipediaAPIWrapper,
)


# ============================================================
# Load Environment Variables
# ============================================================

load_dotenv()

GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")

if not GOOGLE_API_KEY:
    raise ValueError(
        "GOOGLE_API_KEY is missing. "
        "Add it to the Render Environment Variables."
    )


# ============================================================
# Logging
# ============================================================

logging.basicConfig(level=logging.INFO)

logger = logging.getLogger("job_ready_agent")


# ============================================================
# Input Schema
# ============================================================

class AgentInput(BaseModel):
    input: str


# ============================================================
# Initialize Gemini LLM
# ============================================================

llm = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash",
    google_api_key=GOOGLE_API_KEY,
    temperature=0,
)


# ============================================================
# Tool 1 — Web Search
# ============================================================

search_tool = DuckDuckGoSearchRun(
    name="web_search"
)


# ============================================================
# Tool 2 — Wikipedia
# ============================================================

wiki_wrapper = WikipediaAPIWrapper(
    top_k_results=2,
    doc_content_chars_max=2000,
)

wiki_tool = WikipediaQueryRun(
    api_wrapper=wiki_wrapper
)


# ============================================================
# Tool 3 — Calculator
# ============================================================

def calculator(expression: str) -> str:
    """
    Calculate a basic mathematical expression.
    """

    allowed_characters = set(
        "0123456789+-*/().% "
    )

    if not expression:
        return "No mathematical expression was provided."

    if any(
        character not in allowed_characters
        for character in expression
    ):
        return "Invalid mathematical expression."

    try:

        result = eval(
            expression,
            {"__builtins__": {}},
            {}
        )

        return str(result)

    except Exception as exc:

        return f"Calculation error: {exc}"


calculator_tool = Tool(
    name="calculator",
    func=calculator,
    description=(
        "Use this tool for arithmetic calculations. "
        "Input should be a mathematical expression. "
        "Example: 23 * 47 + 12."
    ),
)


# ============================================================
# Register Tools
# ============================================================

tools = [

    Tool(
        name="web_search",
        func=search_tool.run,
        description=(
            "Search the internet for current events, "
            "recent information, facts, or information "
            "you are unsure about."
        ),
    ),

    Tool(
        name="wikipedia",
        func=wiki_tool.run,
        description=(
            "Search Wikipedia for general knowledge, "
            "historical information, people, places, "
            "technology, and background information."
        ),
    ),

    calculator_tool,

]


# ============================================================
# ReAct Prompt
# ============================================================

prompt_template = """

You are a Job Ready AI Agent.

Your job is to answer the user's questions accurately.

You have access to the following tools:

{tools}


Use the following format:

Question: the user's question

Thought: think about what information is required

Action: the action to take. Choose one of:
[{tool_names}]

Action Input: the input for the selected tool

Observation: the result returned by the tool

... repeat the Thought / Action / Action Input /
Observation process when necessary ...

Thought: I now know the final answer

Final Answer: provide the final answer to the user


Tool usage rules:

- Use web_search for current events and recent information.
- Use wikipedia for general knowledge and background information.
- Use calculator for mathematical calculations.
- Do not use a tool when it is unnecessary.
- Never invent tool results.
- Give a clear and concise final answer.


Question: {input}

Thought:{agent_scratchpad}
"""


prompt = PromptTemplate.from_template(
    prompt_template
)


# ============================================================
# Create ReAct Agent
# ============================================================

agent = create_react_agent(
    llm=llm,
    tools=tools,
    prompt=prompt,
)


# ============================================================
# Agent Executor
# ============================================================

agent_executor = AgentExecutor(
    agent=agent,
    tools=tools,

    verbose=True,

    handle_parsing_errors=True,

    max_iterations=8,

    max_execution_time=60,

    return_intermediate_steps=True,
)


# ============================================================
# LangServe Runnable
# ============================================================

def run_agent(request: AgentInput) -> str:
    """
    Execute the LangChain ReAct agent.

    LangServe provides an AgentInput object containing
    the user's question in the 'input' field.
    """

    query = request.input

    if not query.strip():
        return "Please provide an input question."

    logger.info(
        "Running Job Ready Agent: %s",
        query
    )

    try:

        result = agent_executor.invoke(
            {
                "input": query
            }
        )

        answer = result.get(
            "output",
            ""
        )

        return answer

    except Exception as exc:

        logger.exception(
            "Agent execution failed"
        )

        return f"Agent error: {exc}"


# ============================================================
# Create Typed Runnable
# ============================================================

agent_runnable = RunnableLambda(
    run_agent
).with_types(
    input_type=AgentInput,
    output_type=str,
)


# ============================================================
# FastAPI Application
# ============================================================

app = FastAPI(

    title="LangChain Job Ready Agent",

    version="1.0",

    description=(
        "A production-style LangChain ReAct AI Agent "
        "powered by Google Gemini with Web Search, "
        "Wikipedia, and Calculator tools."
    ),

)


# ============================================================
# LangServe Routes
# ============================================================

add_routes(

    app,

    agent_runnable,

    path="/agent",

    input_type=AgentInput,

    output_type=str,

)


# ============================================================
# Root Endpoint
# ============================================================

@app.get("/")
def root():

    return {

        "agent": "LangChain Job Ready Agent",

        "status": "running",

        "framework": "LangChain",

        "agent_type": "ReAct",

        "model": "Gemini 2.5 Flash",

        "tools": [
            "Web Search",
            "Wikipedia",
            "Calculator",
        ],

        "playground": "/agent/playground/",

    }


# ============================================================
# Health Check
# ============================================================

@app.get("/health")
def health():

    return {

        "status": "healthy"

    }


# ============================================================
# Run Application
# ============================================================

if __name__ == "__main__":

    uvicorn.run(

        "langchain_job_ready_agent:app",

        host="0.0.0.0",

        port=int(
            os.environ.get(
                "PORT",
                8000
            )
        ),

    )
