import os
import logging
from io import BytesIO
from typing import Optional

import requests
from dotenv import load_dotenv

from fastapi import (
    FastAPI,
    UploadFile,
    File,
    Form,
    HTTPException,
)
from fastapi.responses import HTMLResponse

from langserve import add_routes
import uvicorn

from pydantic import BaseModel, Field

from pypdf import PdfReader

from langchain_google_genai import ChatGoogleGenerativeAI

from langchain_classic.agents import (
    AgentExecutor,
    create_react_agent,
)

from langchain_core.tools import Tool
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnableLambda

from langchain_community.tools import DuckDuckGoSearchRun


# ============================================================
# ENVIRONMENT
# ============================================================

load_dotenv()

GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")

if not GOOGLE_API_KEY:
    raise ValueError(
        "GOOGLE_API_KEY is missing. "
        "Add it to Render Environment Variables."
    )


# ============================================================
# LOGGING
# ============================================================

logging.basicConfig(level=logging.INFO)

logger = logging.getLogger("placement_agent")


# ============================================================
# LLM
# ============================================================

llm = ChatGoogleGenerativeAI(
    model="gemma-4-31b-it",
    google_api_key=GOOGLE_API_KEY,
    temperature=0,
)


# ============================================================
# HTML FRONTEND
# ============================================================

HTML_PAGE = """
<!DOCTYPE html>

<html lang="en">

<head>

<meta charset="UTF-8">

<meta
    name="viewport"
    content="width=device-width, initial-scale=1.0"
>

<title>Placement-Ready AI Agent</title>

<style>

* {
    box-sizing: border-box;
}

body {

    margin: 0;

    min-height: 100vh;

    font-family:
        Inter,
        system-ui,
        -apple-system,
        BlinkMacSystemFont,
        "Segoe UI",
        sans-serif;

    background:
        radial-gradient(
            circle at top left,
            #172554 0,
            #090d18 35%,
            #05070d 100%
        );

    color: #f8fafc;

}


.container {

    width: min(1100px, 92%);

    margin: 0 auto;

    padding: 50px 0 70px;

}


.header {

    text-align: center;

    margin-bottom: 38px;

}


.badge {

    display: inline-block;

    padding: 8px 14px;

    border: 1px solid #334155;

    border-radius: 999px;

    background: rgba(15, 23, 42, 0.8);

    color: #93c5fd;

    font-size: 13px;

    font-weight: 600;

    margin-bottom: 16px;

}


h1 {

    margin: 0;

    font-size: clamp(32px, 5vw, 56px);

    line-height: 1.05;

    letter-spacing: -1.5px;

}


.subtitle {

    max-width: 720px;

    margin: 18px auto 0;

    color: #94a3b8;

    font-size: 17px;

    line-height: 1.7;

}


.card {

    background: rgba(15, 23, 42, 0.82);

    border: 1px solid #1e293b;

    border-radius: 20px;

    padding: 28px;

    box-shadow:
        0 25px 70px rgba(0, 0, 0, 0.35);

    backdrop-filter: blur(15px);

}


.grid {

    display: grid;

    grid-template-columns:
        repeat(2, minmax(0, 1fr));

    gap: 20px;

}


.field {

    margin-bottom: 20px;

}


.full {

    grid-column: 1 / -1;

}


label {

    display: block;

    margin-bottom: 9px;

    color: #e2e8f0;

    font-size: 14px;

    font-weight: 650;

}


input[type="text"] {

    width: 100%;

    padding: 14px 15px;

    border: 1px solid #334155;

    border-radius: 12px;

    outline: none;

    background: #0b1120;

    color: #f8fafc;

    font-size: 15px;

}


input[type="text"]:focus {

    border-color: #60a5fa;

    box-shadow:
        0 0 0 3px rgba(96, 165, 250, 0.12);

}


.file-box {

    border: 1.5px dashed #475569;

    border-radius: 14px;

    padding: 22px;

    background: #0b1120;

    transition: 0.2s;

}


.file-box:hover {

    border-color: #60a5fa;

}


input[type="file"] {

    width: 100%;

    color: #94a3b8;

}


.help {

    margin-top: 8px;

    color: #64748b;

    font-size: 12px;

}


.button {

    width: 100%;

    margin-top: 8px;

    padding: 15px 20px;

    border: 0;

    border-radius: 12px;

    background:
        linear-gradient(
            135deg,
            #2563eb,
            #4f46e5
        );

    color: white;

    font-size: 16px;

    font-weight: 700;

    cursor: pointer;

    transition: transform 0.15s, opacity 0.15s;

}


.button:hover {

    transform: translateY(-1px);

}


.button:disabled {

    cursor: not-allowed;

    opacity: 0.55;

    transform: none;

}


.status {

    display: none;

    margin-top: 22px;

    padding: 14px 16px;

    border-radius: 12px;

    background: #0f172a;

    border: 1px solid #1e293b;

    color: #94a3b8;

}


.status.show {

    display: block;

}


.status.error {

    color: #fca5a5;

    border-color: #7f1d1d;

    background: #1c0b0b;

}


.report {

    display: none;

    margin-top: 28px;

}


.report.show {

    display: block;

}


.report-header {

    display: flex;

    align-items: center;

    justify-content: space-between;

    gap: 20px;

    margin-bottom: 18px;

}


.report-title {

    font-size: 24px;

    font-weight: 750;

}


.score {

    min-width: 110px;

    padding: 12px 16px;

    text-align: center;

    border: 1px solid #334155;

    border-radius: 14px;

    background: #0b1120;

}


.score-number {

    font-size: 27px;

    font-weight: 800;

    color: #60a5fa;

}


.score-label {

    color: #64748b;

    font-size: 11px;

    text-transform: uppercase;

    letter-spacing: 1px;

}


.report-body {

    white-space: pre-wrap;

    line-height: 1.75;

    color: #cbd5e1;

    font-size: 15px;

}


.footer {

    text-align: center;

    margin-top: 25px;

    color: #475569;

    font-size: 12px;

}


.links {

    margin-top: 18px;

    text-align: center;

}


.links a {

    color: #60a5fa;

    text-decoration: none;

    margin: 0 8px;

    font-size: 13px;

}


@media (max-width: 700px) {

    .grid {

        grid-template-columns: 1fr;

    }

    .full {

        grid-column: auto;
    }

    .card {

        padding: 20px;
    }

    .report-header {

        align-items: flex-start;

        flex-direction: column;

    }

}

</style>

</head>


<body>

<div class="container">


    <div class="header">

        <div class="badge">
            LANGCHAIN • GEMMA • AI AGENT
        </div>

        <h1>
            Placement-Ready AI Agent
        </h1>

        <p class="subtitle">
            Upload your resume, select your target role,
            provide your GitHub profile, and get an AI-powered
            placement readiness analysis.
        </p>

    </div>


    <div class="card">

        <form id="analysisForm">


            <div class="grid">


                <div class="field full">

                    <label for="resume">
                        Resume PDF
                    </label>

                    <div class="file-box">

                        <input
                            id="resume"
                            name="resume"
                            type="file"
                            accept=".pdf,application/pdf"
                            required
                        >

                        <div class="help">
                            Upload a text-based PDF resume.
                        </div>

                    </div>

                </div>


                <div class="field">

                    <label for="role">
                        Target Placement Role
                    </label>

                    <input
                        id="role"
                        name="role"
                        type="text"
                        placeholder="e.g. AI/ML Engineer"
                        value="AI/ML Engineer"
                        required
                    >

                </div>


                <div class="field">

                    <label for="github_id">
                        GitHub Username
                    </label>

                    <input
                        id="github_id"
                        name="github_id"
                        type="text"
                        placeholder="e.g. yogeeswar-09"
                        required
                    >

                </div>


            </div>


            <button
                id="analyzeButton"
                class="button"
                type="submit"
            >
                Analyze Placement Readiness
            </button>


        </form>


        <div
            id="status"
            class="status"
        ></div>


        <div
            id="report"
            class="report"
        >

            <div class="report-header">

                <div class="report-title">
                    Placement Readiness Report
                </div>

                <div class="score">

                    <div
                        id="scoreNumber"
                        class="score-number"
                    >
                        —
                    </div>

                    <div class="score-label">
                        Readiness
                    </div>

                </div>

            </div>


            <div
                id="reportBody"
                class="report-body"
            ></div>

        </div>


    </div>


    <div class="links">

        <a href="/agent/playground/" target="_blank">
            LangServe Playground
        </a>

        <a href="/docs" target="_blank">
            API Docs
        </a>

        <a href="/health" target="_blank">
            Health
        </a>

    </div>


    <div class="footer">

        Placement-Ready AI Agent • LangChain • FastAPI • Render

    </div>


</div>


<script>

const form =
    document.getElementById("analysisForm");

const button =
    document.getElementById("analyzeButton");

const status =
    document.getElementById("status");

const report =
    document.getElementById("report");

const reportBody =
    document.getElementById("reportBody");

const scoreNumber =
    document.getElementById("scoreNumber");


form.addEventListener(
    "submit",
    async function(event) {

        event.preventDefault();


        const resume =
            document.getElementById("resume").files[0];

        const role =
            document.getElementById("role").value.trim();

        const githubId =
            document.getElementById("github_id").value.trim();


        if (!resume) {

            showError(
                "Please select your Resume PDF."
            );

            return;
        }


        if (!resume.name.toLowerCase().endsWith(".pdf")) {

            showError(
                "Only PDF resumes are supported."
            );

            return;
        }


        if (!role) {

            showError(
                "Please enter your target role."
            );

            return;
        }


        if (!githubId) {

            showError(
                "Please enter your GitHub username."
            );

            return;
        }


        button.disabled = true;

        button.textContent =
            "Analyzing... Please wait";


        report.classList.remove("show");

        showStatus(
            "Reading resume and running placement analysis..."
        );


        const formData =
            new FormData();

        formData.append(
            "resume",
            resume
        );

        formData.append(
            "role",
            role
        );

        formData.append(
            "github_id",
            githubId
        );


        try {

            const response =
                await fetch(
                    "/api/analyze",
                    {
                        method: "POST",
                        body: formData
                    }
                );


            const data =
                await response.json();


            if (!response.ok) {

                throw new Error(
                    data.detail ||
                    "Analysis failed."
                );

            }


            reportBody.textContent =
                data.report || "No report returned.";


            scoreNumber.textContent =
                extractScore(
                    data.report
                );


            report.classList.add("show");


            showStatus(
                "Analysis completed successfully."
            );


            window.scrollTo({
                top:
                    report.offsetTop - 30,
                behavior: "smooth"
            });


        }

        catch (error) {

            showError(
                error.message ||
                "Something went wrong."
            );

        }

        finally {

            button.disabled = false;

            button.textContent =
                "Analyze Placement Readiness";

        }

    }
);


function showStatus(message) {

    status.className =
        "status show";

    status.textContent =
        message;

}


function showError(message) {

    status.className =
        "status show error";

    status.textContent =
        message;

}


function extractScore(text) {

    if (!text) {
        return "—";
    }


    const patterns = [

        /(?:Overall Placement Readiness|Placement Readiness|Readiness)[^\\d]{0,30}(\\d{1,3})\\s*\\/\\s*100/i,

        /(?:Score)[^\\d]{0,20}(\\d{1,3})\\s*\\/\\s*100/i

    ];


    for (
        const pattern of patterns
        ) {

        const match =
            text.match(pattern);

        if (match) {

            return (
                match[1] +
                "/100"
            );

        }

    }


    return "—";

}

</script>

</body>

</html>
"""


# ============================================================
# PDF EXTRACTION
# ============================================================

def extract_pdf_text(
    file_bytes: bytes
) -> str:

    try:

        reader = PdfReader(
            BytesIO(file_bytes)
        )

        pages = []

        for page in reader.pages:

            text = page.extract_text()

            if text:

                pages.append(text)


        result = "\n\n".join(
            pages
        ).strip()


        if not result:

            raise ValueError(
                "No readable text could be extracted "
                "from the PDF."
            )


        return result


    except Exception as exc:

        raise ValueError(
            f"Could not read the PDF: {exc}"
        )


# ============================================================
# JOB SEARCH TOOL
# ============================================================

def create_job_search_tool():

    search = DuckDuckGoSearchRun()


    def job_search(
        query: str
    ) -> str:

        try:

            result = search.run(
                f"""
                Search for current internship, entry-level,
                and campus placement opportunities related to:

                {query}

                Identify:

                - relevant job roles
                - common required skills
                - programming languages
                - frameworks
                - databases
                - cloud/devops requirements
                - AI/ML requirements where relevant

                Prefer recent information.
                """
            )

            return result

        except Exception as exc:

            logger.exception(
                "Job search failed"
            )

            return (
                f"Job search error: {exc}"
            )


    return Tool(
        name="job_search",
        func=job_search,
        description=(
            "Search the web for current job and internship "
            "requirements for the student's target placement role."
        )
    )


# ============================================================
# SKILL GAP TOOL
# ============================================================

def create_skill_gap_tool(
    resume_text: str
):

    def skill_gap(
        role: str
    ) -> str:

        prompt = f"""
You are a campus placement skill-gap analyst.

TARGET ROLE:
{role}

STUDENT RESUME:
{resume_text}

Analyze the resume against the target role.

Return:

CURRENT SKILLS:
- skills clearly demonstrated

SKILL GAPS:
- missing skills
- weak skills

PRIORITY:
- High
- Medium
- Low

Explain why each important missing skill matters
for the target placement role.

Do not invent skills that are not present in the resume.
"""


        try:

            response =
                llm.invoke(prompt)

            return response.content

        except Exception as exc:

            logger.exception(
                "Skill gap analysis failed"
            )

            return (
                f"Skill gap analysis error: {exc}"
            )


    return Tool(
        name="skill_gap_analysis",
        func=skill_gap,
        description=(
            "Compare the student's resume with the target role "
            "and identify current skills, missing skills, weak "
            "areas, and skill priorities."
        )
    )


# ============================================================
# PROJECT RECOMMENDATION TOOL
# ============================================================

def create_project_tool(
    resume_text: str
):

    def recommend_projects(
        role: str
    ) -> str:

        prompt = f"""
You are a campus placement project mentor.

TARGET ROLE:
{role}

STUDENT RESUME:
{resume_text}

Recommend 3 practical portfolio projects.

Projects must:

- address likely skill gaps
- be realistic for a student
- improve placement readiness
- be suitable for GitHub
- demonstrate relevant technologies

For every project provide:

PROJECT:
Problem:
Features:
Technology Stack:
Skills Developed:
Why It Helps:
Difficulty:

Do not recommend projects unrelated to the target role.
"""


        try:

            response =
                llm.invoke(prompt)

            return response.content

        except Exception as exc:

            logger.exception(
                "Project recommendation failed"
            )

            return (
                f"Project recommendation error: {exc}"
            )


    return Tool(
        name="project_recommendation",
        func=recommend_projects,
        description=(
            "Recommend practical portfolio projects that "
            "address the student's skill gaps and improve "
            "placement readiness."
        )
    )


# ============================================================
# GITHUB TOOL
# ============================================================

def create_github_tool(
    target_role: str
):

    def github_evaluation(
        username: str
    ) -> str:

        username = username.strip()

        if "github.com/" in username:

            username = username.split(
                "github.com/"
            )[-1]

        username = username.rstrip(
            "/"
        ).split("/")[0]


        if not username:

            return (
                "No GitHub username was provided."
            )


        headers = {
            "Accept":
                "application/vnd.github+json",

            "User-Agent":
                "Placement-Ready-AI-Agent",
        }


        try:

            profile_url = (
                f"https://api.github.com/users/"
                f"{username}"
            )


            repos_url = (
                f"https://api.github.com/users/"
                f"{username}/repos"
                f"?per_page=100&sort=updated"
            )


            profile_response =
                requests.get(
                    profile_url,
                    headers=headers,
                    timeout=15
                )


            if profile_response.status_code == 404:

                return (
                    f"GitHub user '{username}' "
                    "was not found."
                )


            profile_response.raise_for_status()


            profile =
                profile_response.json()


            repos_response =
                requests.get(
                    repos_url,
                    headers=headers,
                    timeout=15
                )


            repos_response.raise_for_status()


            repos =
                repos_response.json()


            if not isinstance(
                repos,
                list
            ):

                repos = []


            repo_data = []


            for repo in repos[:20]:

                repo_data.append({

                    "name":
                        repo.get("name"),

                    "description":
                        repo.get("description"),

                    "language":
                        repo.get("language"),

                    "stars":
                        repo.get(
                            "stargazers_count"
                        ),

                    "forks":
                        repo.get(
                            "forks_count"
                        ),

                    "updated":
                        repo.get(
                            "updated_at"
                        ),

                    "url":
                        repo.get(
                            "html_url"
                        ),

                })


            github_data = {

                "username":
                    username,

                "name":
                    profile.get("name"),

                "bio":
                    profile.get("bio"),

                "profile_url":
                    profile.get("html_url"),

                "public_repositories":
                    profile.get(
                        "public_repos",
                        0
                    ),

                "followers":
                    profile.get(
                        "followers",
                        0
                    ),

                "following":
                    profile.get(
                        "following",
                        0
                    ),

                "repositories":
                    repo_data,

            }


            prompt = f"""
You are a technical recruiter evaluating a student's
GitHub profile for campus placements.

TARGET ROLE:
{target_role}

GITHUB DATA:
{github_data}

Evaluate:

1. Profile strength
2. Project relevance
3. Technology relevance
4. Repository activity
5. Project presentation
6. Documentation/README quality opportunities
7. Areas for improvement
8. Recruiter impression
9. GitHub readiness score out of 100

IMPORTANT:

Only evaluate information actually provided by the
GitHub API data.

Do not claim that you inspected source code unless
source code information is actually available.

Return a concise recruiter-style report.
"""


            response =
                llm.invoke(prompt)


            return response.content


        except requests.RequestException as exc:

            logger.exception(
                "GitHub API failed"
            )

            return (
                f"GitHub API error: {exc}"
            )


        except Exception as exc:

            logger.exception(
                "GitHub evaluation failed"
            )

            return (
                f"GitHub evaluation error: {exc}"
            )


    return Tool(
        name="github_evaluation",
        func=github_evaluation,
        description=(
            "Evaluate a student's public GitHub profile "
            "using the GitHub API. Analyze repository count, "
            "activity, technologies, project relevance, "
            "presentation, and recruiter readiness."
        )
    )


# ============================================================
# BUILD PLACEMENT AGENT
# ============================================================

def build_agent(
    resume_text: str,
    target_role: str,
    github_username: str
):

    tools = [

        create_job_search_tool(),

        create_skill_gap_tool(
            resume_text
        ),

        create_project_tool(
            resume_text
        ),

        create_github_tool(
            target_role
        ),

    ]


    prompt_template = """
You are a Placement-Ready AI Agent.

You help students prepare for campus placements.

The student provides:

- Resume PDF
- Target job role
- GitHub username

Your available tools are:

{tools}


TOOLS:

job_search
Search current job/internship requirements.

skill_gap_analysis
Compare the resume against the target role.

project_recommendation
Recommend projects that address skill gaps.

github_evaluation
Evaluate the student's public GitHub profile.


FOR A COMPLETE ANALYSIS:

You should normally use ALL FOUR tools:

1. job_search
2. skill_gap_analysis
3. project_recommendation
4. github_evaluation


The final answer must synthesize the tool results.

FINAL REPORT FORMAT:

PLACEMENT READINESS REPORT

Target Role:
...

1. JOB OPPORTUNITY ANALYSIS
...

2. CURRENT SKILLS
...

3. SKILL GAP ANALYSIS
...

4. RECOMMENDED PROJECTS
...

5. GITHUB EVALUATION
...

6. PRIORITY ACTION PLAN
1.
2.
3.
4.
5.

7. OVERALL PLACEMENT READINESS
__/100


Important:

- Do not invent resume information.
- Do not invent GitHub information.
- Clearly distinguish available data from recommendations.
- Give actionable advice.
- Keep the final answer organized and readable.


Use this ReAct format:

Question: the user's request

Thought: decide which tool to use

Action: one of [{tool_names}]

Action Input: input for the tool

Observation: result from the tool

Thought: decide whether another tool is required

Action: tool

Action Input: input

Observation: result

Thought: I now know the final answer

Final Answer: final report


Question:
{input}

Thought:
{agent_scratchpad}
"""


    prompt = PromptTemplate.from_template(
        prompt_template
    )


    agent =
        create_react_agent(
            llm=llm,
            tools=tools,
            prompt=prompt,
        )


    executor =
        AgentExecutor(
            agent=agent,
            tools=tools,

            verbose=True,

            handle_parsing_errors=True,

            max_iterations=12,

            max_execution_time=180,

            return_intermediate_steps=False,
        )


    return executor


# ============================================================
# COMPLETE ANALYSIS
# ============================================================

def run_complete_analysis(
    resume_text: str,
    target_role: str,
    github_username: str
) -> str:

    logger.info(
        "Starting placement analysis | role=%s | github=%s",
        target_role,
        github_username
    )


    executor =
        build_agent(
            resume_text=resume_text,
            target_role=target_role,
            github_username=github_username
        )


    question = f"""
Perform a complete placement-readiness analysis.

TARGET ROLE:
{target_role}

GITHUB USERNAME:
{github_username}

The student's resume has been parsed and is available
to your skill-gap and project recommendation tools.

Use all four tools:

1. Job Search
2. Skill Gap Analysis
3. Project Recommendation
4. GitHub Evaluation

Then synthesize the results into the requested
PLACEMENT READINESS REPORT.
"""


    try:

        result =
            executor.invoke(
                {
                    "input": question
                }
            )


        answer =
            result.get(
                "output",
                ""
            )


        if not answer:

            return (
                "The agent completed the workflow "
                "but did not return a final report."
            )


        return answer


    except Exception as exc:

        logger.exception(
            "Placement analysis failed"
        )

        return (
            "Placement Agent Error:\n\n"
            f"{str(exc)}"
        )


# ============================================================
# FASTAPI APP
# ============================================================

app = FastAPI(

    title="Placement-Ready AI Agent",

    version="1.0",

    description=(
        "LangChain AI Agent for campus placement "
        "preparation."
    ),

)


# ============================================================
# MAIN STUDENT UI
# ============================================================

@app.get(
    "/",
    response_class=HTMLResponse
)
def home():

    return HTML_PAGE


# ============================================================
# COMPLETE WORKFLOW API
# ============================================================

@app.post("/api/analyze")
async def analyze(

    resume: UploadFile =
        File(...),

    role: str =
        Form(...),

    github_id: str =
        Form(...),

):

    if not resume.filename:

        raise HTTPException(
            status_code=400,
            detail="Resume PDF is required."
        )


    if not resume.filename.lower().endswith(
        ".pdf"
    ):

        raise HTTPException(
            status_code=400,
            detail="Only PDF resumes are supported."
        )


    role = role.strip()

    github_id = github_id.strip()


    if not role:

        raise HTTPException(
            status_code=400,
            detail="Target role is required."
        )


    if not github_id:

        raise HTTPException(
            status_code=400,
            detail="GitHub username is required."
        )


    try:

        pdf_bytes =
            await resume.read()


        # 10 MB protection
        if len(pdf_bytes) > 10 * 1024 * 1024:

            raise HTTPException(
                status_code=400,
                detail="Resume must be smaller than 10 MB."
            )


        resume_text =
            extract_pdf_text(
                pdf_bytes
            )


        report =
            run_complete_analysis(
                resume_text=resume_text,
                target_role=role,
                github_username=github_id
            )


        return {

            "status":
                "success",

            "filename":
                resume.filename,

            "role":
                role,

            "github_id":
                github_id,

            "report":
                report,

        }


    except HTTPException:

        raise


    except Exception as exc:

        logger.exception(
            "Complete workflow failed"
        )

        raise HTTPException(
            status_code=500,
            detail=str(exc)
        )


# ============================================================
# LANGSERVE PLAYGROUND
# ============================================================

class PlaygroundInput(BaseModel):

    role: str = Field(
        default="AI/ML Engineer"
    )

    github_id: str = Field(
        default=""
    )


def playground_agent(
    request
):

    role =
        request.get(
            "role",
            "AI/ML Engineer"
        )

    github_id =
        request.get(
            "github_id",
            ""
        )


    return (
        "The complete PDF workflow is available "
        "at the main page (/). "
        "The LangServe Playground accepts role "
        "and GitHub ID, while the main UI handles "
        "the Resume PDF upload."
    )


playground_runnable =
    RunnableLambda(
        playground_agent
    ).with_types(
        input_type=PlaygroundInput,
        output_type=str,
    )


add_routes(

    app,

    playground_runnable,

    path="/agent",

)


# ============================================================
# HEALTH
# ============================================================

@app.get("/health")
def health():

    return {

        "status":
            "healthy",

        "agent":
            "Placement-Ready AI Agent",

        "model":
            "gemma-4-31b-it",

        "capabilities": [

            "Job Search",

            "Skill Gap Analysis",

            "Project Recommendation",

            "GitHub Evaluation",

            "Final Placement Synthesis",

        ],

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
                8000
            )
        ),

    )
