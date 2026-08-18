import os
import logging

from dotenv import load_dotenv
from fastapi import FastAPI
from langserve import add_routes
import uvicorn

from pydantic import BaseModel, Field

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

from langchain_community.utilities import WikipediaAPIWrapper


# ============================================================
# Load Environment Variables
# ============================================================

load_dotenv()

GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")

if not GOOGLE_API_KEY:
    raise ValueError(
        "GOOGLE_API_KEY is missing. "
        "Add it in Render Environment Variables."
    )


# ============================================================
# Logging
# ============================================================

logging.basicConfig(level=logging.INFO)

logger = logging.getLogger("job_ready_agent")


# ============================================================
# Playground Input Schema
# ============================================================
#
# IMPORTANT:
# input has a default value of "".
#
# This prevents LangServe Playground from immediately showing:
# "must have required property 'input'"
# before the user has typed anything.
#
# ============================================================

class AgentInput(BaseModel):
    input: str = Field(
        default="",
        description="Enter your question for the Job Ready AI Agent",
    )


# ============================================================
# Initialize Gemini
# ============================================================

llm = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash",
    google_api_key=GOOGLE_API_KEY,
    temperature=0,
)


# ============================================================
# Web Search Tool
# ============================================================

duckduckgo_search = DuckDuckGoSearchRun()


def web_search(query: str) -> str:
    """
    Search the web for current or recent information.
    """

    try:
        return duckduckgo_search.run(query)

    except Exception as exc:
        logger.exception("Web search failed")
        return f"Web search error: {exc}"


web_search_tool = Tool(
    name="web_search",
    func=web_search,
    description=(
        "Search the internet for current events, recent facts, "
        "news, or information that may have changed recently."
    ),
)


# ============================================================
# Wikipedia Tool
# ============================================================

wikipedia_api = WikipediaAPIWrapper(
    top_k_results=2,
    doc_content_chars_max=2000,
)

wikipedia_search = WikipediaQueryRun(
    api_wrapper=wikipedia_api
)


def search_wikipedia(query: str) -> str:
    """
    Search Wikipedia for general knowledge.
    """

    try:
        return wikipedia_search.run(query)

    except Exception as exc:
        logger.exception("Wikipedia search failed")
        return f"Wikipedia error: {exc}"


wikipedia_tool = Tool(
    name="wikipedia",
    func=search_wikipedia,
    description=(
        "Search Wikipedia for general knowledge, people, "
        "places, history, science, technology and background "
        "information."
    ),
)


# ============================================================
# Calculator Tool
# ============================================================

def calculator(expression: str) -> str:
    """
    Perform basic arithmetic calculations.
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
            {},
        )

        return str(result)

    except Exception as exc:

        logger.exception("Calculator failed")

        return f"Calculation error: {exc}"


calculator_tool = Tool(
    name="calculator",
    func=calculator,
    description=(
        "Perform arithmetic calculations. "
        "Input must be a mathematical expression. "
        "Example: 25 * 48"
    ),
)


# ============================================================
# Register Tools
# ============================================================

tools = [
    web_search_tool,
    wikipedia_tool,
    calculator_tool,
]


# ============================================================
# ReAct Prompt
# ============================================================

prompt_template = """
You are a Job Ready AI Agent powered by Google Gemini.

Your goal is to answer the user's question accurately.

You have access to these tools:

{tools}

TOOLS:

web_search
Use this when the user asks about current events,
recent information, news, or information that may have changed.

wikipedia
Use this for general knowledge, historical information,
people, places, science, technology and background information.

calculator
Use this for mathematical calculations.


You MUST follow this ReAct format when using a tool:

Question: the user's question

Thought: decide whether a tool is required

Action: one of [{tool_names}]

Action Input: input for the selected tool

Observation: result from the tool

Thought: I now know the final answer

Final Answer: give the final answer


If another tool call is required, repeat:

Thought
Action
Action Input
Observation


IMPORTANT RULES:

1. Use a tool when it is useful.
2. Do not invent tool results.
3. Do not expose unnecessary internal reasoning.
4. Give the user a clear final answer.
5. If no tool is required, answer directly.
6. After obtaining enough information, always provide a Final Answer.


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
    max_iterations=6,
    max_execution_time=60,
    return_intermediate_steps=False,
)


# ============================================================
# LangServe Adapter
# ============================================================

def run_agent(request) -> str:
    """
    Receive input from LangServe Playground
    and execute the LangChain ReAct agent.
    """

    # LangServe normally provides a dictionary.
    if isinstance(request, dict):
        query = request.get("input", "")

    # Support Pydantic input as a fallback.
    elif isinstance(request, AgentInput):
        query = request.input

    else:
        query = str(request)


    query = query.strip()


    # Handle empty Playground input ourselves.
    if not query:
        return "Please enter a question."


    logger.info(
        "User question: %s",
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


        if not answer:

            return (
                "The agent completed the request "
                "but did not return an answer."
            )


        return answer


    except Exception as exc:

        logger.exception(
            "Agent execution failed"
        )

        return (
            "The agent encountered an error: "
            f"{str(exc)}"
        )


# ============================================================
# Typed Runnable
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
        "LangChain ReAct Agent powered by Google Gemini "
        "with Web Search, Wikipedia and Calculator tools."
    ),
)


# ============================================================
# LangServe
# ============================================================

add_routes(
    app,
    agent_runnable,
    path="/agent",
    playground_type="default",
)


# ============================================================
# Root
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
# Run
# ============================================================

if __name__ == "__main__":

    uvicorn.run(
        "langchain_job_ready_agent:app",
        host="0.0.0.0",
        port=int(
            os.environ.get(
                "PORT",
                8000,
            )
        ),
    )
