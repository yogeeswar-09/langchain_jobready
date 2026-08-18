import os
import json
import logging
from typing import Optional

import requests
from dotenv import load_dotenv

from fastapi import FastAPI, UploadFile, File, HTTPException
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

from langchain_community.tools import DuckDuckGoSearchRun

from pypdf import PdfReader


# ============================================================
# ENVIRONMENT
# ============================================================

load_dotenv()

GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")

if not GOOGLE_API_KEY:
    raise ValueError(
        "GOOGLE_API_KEY is missing. "
        "Add GOOGLE_API_KEY in Render Environment Variables."
    )


# ============================================================
# LOGGING
# ============================================================

logging.basicConfig(level=logging.INFO)

logger = logging.getLogger("placement_agent")


# ============================================================
# GLOBAL STUDENT SESSION
# ============================================================

student_resume = ""

student_resume_filename = ""

student_role = ""

student_github_id = ""


# ============================================================
# INPUT SCHEMA
# ============================================================

class PlacementInput(BaseModel):
    role: str = Field(
        default="AI/ML Engineer",
        description="Target placement/job role",
    )

    github_id: str = Field(
        default="",
        description="GitHub username, for example yogeeswar-09",
    )


# ============================================================
# LLM
# ============================================================

llm = ChatGoogleGenerativeAI(
    model="gemma-4-31b-it",
    google_api_key=GOOGLE_API_KEY,
    temperature=0,
)


# ============================================================
# WEB SEARCH
# ============================================================

duckduckgo = DuckDuckGoSearchRun()


def search_jobs(query: str) -> str:
    """
    Search the web for placement/job opportunities.
    """

    try:
        result = duckduckgo.run(
            f"""
            Find current job and internship opportunities related to:

            {query}

            Focus on:
            - internships
            - entry-level jobs
            - campus placement opportunities
            - required technologies
            - required skills
            """
        )

        return result

    except Exception as exc:
        logger.exception("Job search failed")

        return f"Job search failed: {exc}"


job_search_tool = Tool(
    name="job_search",
    func=search_jobs,
    description=(
        "Search the web for current internship and entry-level "
        "job opportunities related to the student's target role. "
        "Use this to identify common job requirements and "
        "placement opportunities."
    ),
)


# ============================================================
# SKILL GAP ANALYSIS
# ============================================================

def analyze_skill_gap(role: str) -> str:
    """
    Compare the student's resume against the target role.
    """

    if not student_resume:
        return (
            "No resume has been uploaded yet. "
            "Please upload a resume PDF using /upload-resume "
            "before running the placement analysis."
        )

    prompt = f"""
You are a professional campus-placement skill-gap analyst.

Analyze the student's resume against the target role.

TARGET ROLE:
{role}

STUDENT RESUME:
{student_resume}

Provide:

1. Skills already demonstrated by the student
2. Important skills expected for the target role
3. Missing or weak skills
4. Priority level for each missing skill
5. Short explanation of why each missing skill matters

Return a concise structured report.
"""

    try:

        response = llm.invoke(prompt)

        return response.content

    except Exception as exc:

        logger.exception("Skill gap analysis failed")

        return f"Skill gap analysis failed: {exc}"


skill_gap_tool = Tool(
    name="skill_gap_analysis",
    func=analyze_skill_gap,
    description=(
        "Analyze the uploaded student resume against the target "
        "placement role and identify existing skills, missing "
        "skills, weak areas, and skill priorities."
    ),
)


# ============================================================
# PROJECT RECOMMENDATION
# ============================================================

def recommend_projects(role: str) -> str:
    """
    Recommend projects based on the student's resume and target role.
    """

    if not student_resume:
        return (
            "No resume is available. Upload the student's resume "
            "before generating project recommendations."
        )

    prompt = f"""
You are a campus-placement project mentor.

Recommend practical projects that will improve this student's
placement readiness.

TARGET ROLE:
{role}

STUDENT RESUME:
{student_resume}

Create 3 project recommendations.

For each project provide:

- Project title
- Problem it solves
- Main features
- Technologies
- Skills it develops
- Why it helps for the target role
- Difficulty level

Prioritize projects that fill likely skill gaps and can be
demonstrated on GitHub.
"""

    try:

        response = llm.invoke(prompt)

        return response.content

    except Exception as exc:

        logger.exception("Project recommendation failed")

        return f"Project recommendation failed: {exc}"


project_tool = Tool(
    name="project_recommendation",
    func=recommend_projects,
    description=(
        "Recommend practical portfolio projects based on the "
        "student's resume, target placement role, and likely "
        "skill gaps. Projects should improve GitHub portfolio "
        "strength and placement readiness."
    ),
)


# ============================================================
# GITHUB EVALUATION
# ============================================================

def evaluate_github(username: str) -> str:
    """
    Evaluate a public GitHub profile using GitHub's public API.
    """

    if not username:
        return (
            "No GitHub username was provided. "
            "Provide a GitHub username to evaluate the profile."
        )

    username = username.strip()

    # Remove URL if user accidentally provides one
    username = username.rstrip("/")

    if "github.com/" in username:
        username = username.split("github.com/")[-1]

    username = username.split("/")[0]

    try:

        headers = {
            "Accept": "application/vnd.github+json",
            "User-Agent": "Placement-Ready-AI-Agent",
        }

        profile_url = (
            f"https://api.github.com/users/{username}"
        )

        repos_url = (
            f"https://api.github.com/users/{username}/repos"
            "?per_page=100&sort=updated"
        )

        profile_response = requests.get(
            profile_url,
            headers=headers,
            timeout=15,
        )

        if profile_response.status_code == 404:
            return f"GitHub user '{username}' was not found."

        profile_response.raise_for_status()

        profile = profile_response.json()

        repos_response = requests.get(
            repos_url,
            headers=headers,
            timeout=15,
        )

        repos_response.raise_for_status()

        repos = repos_response.json()

        if not isinstance(repos, list):
            repos = []

        public_repos = profile.get(
            "public_repos",
            0,
        )

        followers = profile.get(
            "followers",
            0,
        )

        following = profile.get(
            "following",
            0,
        )

        repo_summary = []

        for repo in repos[:15]:

            repo_summary.append(
                {
                    "name": repo.get("name"),
                    "description": repo.get("description"),
                    "language": repo.get("language"),
                    "stars": repo.get("stargazers_count"),
                    "forks": repo.get("forks_count"),
                    "updated": repo.get("updated_at"),
                    "url": repo.get("html_url"),
                }
            )

        github_data = {
            "username": username,
            "name": profile.get("name"),
            "bio": profile.get("bio"),
            "profile_url": profile.get("html_url"),
            "public_repositories": public_repos,
            "followers": followers,
            "following": following,
            "repositories": repo_summary,
        }

        prompt = f"""
You are a technical recruiter evaluating a student's GitHub profile
for campus placements.

Target role:
{student_role}

GitHub profile data:

{json.dumps(github_data, indent=2)}

Evaluate:

1. Overall GitHub profile strength
2. Project relevance to the target role
3. Technology diversity
4. Repository activity
5. Project quality indicators
6. README/documentation opportunities
7. Areas that could improve
8. Recruiter impression
9. A GitHub readiness score out of 100

Do not claim to inspect source code that was not provided.

Return a concise recruiter-style report.
"""

        response = llm.invoke(prompt)

        return response.content

    except requests.RequestException as exc:

        logger.exception("GitHub API failed")

        return f"GitHub API error: {exc}"

    except Exception as exc:

        logger.exception("GitHub evaluation failed")

        return f"GitHub evaluation failed: {exc}"


github_tool = Tool(
    name="github_evaluation",
    func=evaluate_github,
    description=(
        "Evaluate a student's public GitHub profile using the "
        "GitHub API. Analyze repositories, activity, technologies, "
        "project relevance, documentation opportunities, and "
        "overall GitHub readiness."
    ),
)


# ============================================================
# TOOL LIST
# ============================================================

tools = [
    job_search_tool,
    skill_gap_tool,
    project_tool,
    github_tool,
]


# ============================================================
# REACT PROMPT
# ============================================================

prompt_template = """
You are a Placement-Ready AI Agent designed to help students
prepare for campus placements.

Your job is to analyze the student's profile and provide
actionable placement guidance.

You have access to these tools:

{tools}

TOOLS:

job_search
Search for current job and internship opportunities and identify
skills commonly requested for the target role.

skill_gap_analysis
Compare the student's uploaded resume with the target role and
identify missing or weak skills.

project_recommendation
Recommend practical projects that address skill gaps and improve
the student's portfolio.

github_evaluation
Evaluate the student's public GitHub profile for placement readiness.


IMPORTANT:

You should decide which tools are useful for the student's request.

A complete placement analysis normally uses:

1. job_search
2. skill_gap_analysis
3. project_recommendation
4. github_evaluation

You may call tools multiple times when useful.

After gathering enough information, provide a FINAL SYNTHESIS.

The final synthesis should contain:

PLACEMENT READINESS REPORT

Target Role:
...

Job Opportunities / Market Requirements:
...

Current Skills:
...

Skill Gaps:
...

Recommended Projects:
...

GitHub Evaluation:
...

Priority Actions:
1.
2.
3.

Overall Placement Readiness:
__/100


Use the following ReAct format:

Question: the user's question

Thought: decide which tool is useful

Action: one of [{tool_names}]

Action Input: the tool input

Observation: tool result

Thought: decide whether another tool is needed

Action: another tool if needed

Action Input: tool input

Observation: tool result

Thought: I now know the final answer

Final Answer: final placement-readiness report


Do not expose unnecessary internal reasoning in the final answer.


Question: {input}

Thought:{agent_scratchpad}
"""


prompt = PromptTemplate.from_template(
    prompt_template
)


# ============================================================
# CREATE LANGCHAIN AGENT
# ============================================================

agent = create_react_agent(
    llm=llm,
    tools=tools,
    prompt=prompt,
)


# ============================================================
# AGENT EXECUTOR
# ============================================================

agent_executor = AgentExecutor(
    agent=agent,
    tools=tools,
    verbose=True,
    handle_parsing_errors=True,
    max_iterations=10,
    max_execution_time=120,
    return_intermediate_steps=False,
)


# ============================================================
# RUN AGENT
# ============================================================

def run_agent(request) -> str:

    global student_role
    global student_github_id

    if isinstance(request, dict):

        role = request.get(
            "role",
            "AI/ML Engineer",
        )

        github_id = request.get(
            "github_id",
            "",
        )

    elif isinstance(request, PlacementInput):

        role = request.role

        github_id = request.github_id

    else:

        role = "AI/ML Engineer"

        github_id = ""


    role = role.strip()

    github_id = github_id.strip()


    if not role:

        role = "AI/ML Engineer"


    student_role = role

    student_github_id = github_id


    if not student_resume:

        return (
            "Please upload your Resume PDF first using "
            "the /upload-resume endpoint.\n\n"
            "Then return to the Playground and enter your "
            "target role and GitHub username."
        )


    question = f"""
Perform a complete placement-readiness analysis for me.

Target Role:
{role}

GitHub Username:
{github_id}

The student's resume has already been uploaded.

Please:

1. Analyze relevant job opportunities and market requirements.
2. Identify the student's skill gaps.
3. Recommend projects to close those gaps.
4. Evaluate the GitHub profile.
5. Produce a final placement-readiness report.

Use the appropriate tools.
"""


    logger.info(
        "Starting placement analysis | role=%s | github=%s",
        role,
        github_id,
    )


    try:

        result = agent_executor.invoke(
            {
                "input": question
            }
        )

        answer = result.get(
            "output",
            "",
        )

        if not answer:

            return (
                "The placement agent completed the analysis "
                "but did not return a final report."
            )

        return answer


    except Exception as exc:

        logger.exception(
            "Placement agent execution failed"
        )

        return (
            "Placement Agent Error:\n"
            f"{str(exc)}"
        )


# ============================================================
# LANGSERVE INPUT
# ============================================================

placement_runnable = RunnableLambda(
    run_agent
).with_types(
    input_type=PlacementInput,
    output_type=str,
)


# ============================================================
# FASTAPI
# ============================================================

app = FastAPI(
    title="Placement-Ready AI Agent",
    version="1.0",
    description=(
        "LangChain Placement-Ready AI Agent that analyzes "
        "job opportunities, identifies skill gaps, recommends "
        "projects, and evaluates GitHub profiles."
    ),
)


# ============================================================
# LANGSERVE
# ============================================================

add_routes(
    app,
    placement_runnable,
    path="/agent",
)


# ============================================================
# RESUME UPLOAD
# ============================================================

@app.post("/upload-resume")
async def upload_resume(
    file: UploadFile = File(...)
):

    global student_resume
    global student_resume_filename

    if not file.filename:
        raise HTTPException(
            status_code=400,
            detail="Please select a PDF file.",
        )


    if not file.filename.lower().endswith(".pdf"):

        raise HTTPException(
            status_code=400,
            detail="Only PDF files are supported.",
        )


    try:

        contents = await file.read()

        from io import BytesIO

        pdf = PdfReader(
            BytesIO(contents)
        )

        extracted_text = []

        for page in pdf.pages:

            text = page.extract_text()

            if text:
                extracted_text.append(text)


        student_resume = "\n\n".join(
            extracted_text
        ).strip()


        if not student_resume:

            raise HTTPException(
                status_code=400,
                detail=(
                    "Could not extract text from the PDF. "
                    "Please upload a text-based PDF."
                ),
            )


        student_resume_filename = file.filename


        logger.info(
            "Resume uploaded: %s",
            file.filename,
        )


        return {
            "status": "success",
            "filename": file.filename,
            "pages": len(pdf.pages),
            "characters_extracted": len(student_resume),
            "message": (
                "Resume uploaded and parsed successfully. "
                "You can now use /agent/playground/."
            ),
        }


    except HTTPException:
        raise

    except Exception as exc:

        logger.exception(
            "Resume processing failed"
        )

        raise HTTPException(
            status_code=500,
            detail=f"Resume processing failed: {exc}",
        )


# ============================================================
# ROOT
# ============================================================

@app.get("/")
def root():

    return {
        "agent": "Placement-Ready AI Agent",
        "status": "running",
        "framework": "LangChain",
        "agent_type": "ReAct",
        "model": "gemma-4-31b-it",

        "capabilities": [
            "Job Opportunity Analysis",
            "Skill Gap Analysis",
            "Project Recommendation",
            "GitHub Evaluation",
            "Final Placement Synthesis",
        ],

        "resume_upload": "/upload-resume",

        "playground": "/agent/playground/",
    }


# ============================================================
# HEALTH
# ============================================================

@app.get("/health")
def health():

    return {
        "status": "healthy",
        "resume_uploaded": bool(student_resume),
        "resume_filename": student_resume_filename,
    }


# ============================================================
# RUN
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
