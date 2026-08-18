import os
import json
import logging
from typing import Any, Dict

import requests
from dotenv import load_dotenv

from fastapi import FastAPI, File, Form, UploadFile
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware

from langserve import add_routes

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableLambda

from pypdf import PdfReader


# ============================================================
# CONFIG
# ============================================================

load_dotenv()

GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")

if not GOOGLE_API_KEY:
    raise ValueError("GOOGLE_API_KEY is missing.")

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("placement-agent")


# ============================================================
# LLM
# ============================================================

llm = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash",
    google_api_key=GOOGLE_API_KEY,
    temperature=0.2,
)


# ============================================================
# FASTAPI
# ============================================================

app = FastAPI(
    title="Placement-Ready AI Agent",
    version="2.0",
    description=(
        "LangChain-powered placement assistant for job analysis, "
        "skill gaps, project recommendations and GitHub evaluation."
    ),
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ============================================================
# GLOBAL STORAGE
# ============================================================

LAST_RESUME_TEXT = ""


# ============================================================
# PDF EXTRACTION
# ============================================================

def extract_resume_text(pdf_bytes: bytes) -> str:
    """
    Extract text from a PDF resume.
    """

    try:
        temp_path = "/tmp/resume.pdf"

        with open(temp_path, "wb") as f:
            f.write(pdf_bytes)

        reader = PdfReader(temp_path)

        pages = []

        for page in reader.pages:
            text = page.extract_text() or ""
            pages.append(text)

        resume_text = "\n".join(pages).strip()

        if not resume_text:
            return "No readable text could be extracted from the PDF."

        return resume_text

    except Exception as exc:
        logger.exception("PDF extraction failed")
        return f"Resume extraction failed: {exc}"


# ============================================================
# GITHUB ANALYSIS
# ============================================================

def get_github_profile(username: str) -> Dict[str, Any]:

    username = username.strip()

    if not username:
        return {
            "error": "GitHub username was not provided."
        }

    headers = {
        "Accept": "application/vnd.github+json",
        "User-Agent": "Placement-Ready-AI-Agent",
    }

    try:

        profile_url = f"https://api.github.com/users/{username}"

        profile_response = requests.get(
            profile_url,
            headers=headers,
            timeout=10,
        )

        if profile_response.status_code != 200:
            return {
                "error": f"GitHub profile not found: {username}"
            }

        profile = profile_response.json()

        repos_url = f"https://api.github.com/users/{username}/repos"

        repos_response = requests.get(
            repos_url,
            headers=headers,
            params={
                "per_page": 30,
                "sort": "updated",
            },
            timeout=10,
        )

        repositories = []

        if repos_response.status_code == 200:

            for repo in repos_response.json():

                repositories.append(
                    {
                        "name": repo.get("name"),
                        "description": repo.get("description"),
                        "language": repo.get("language"),
                        "stars": repo.get("stargazers_count", 0),
                        "forks": repo.get("forks_count", 0),
                        "updated_at": repo.get("updated_at"),
                        "html_url": repo.get("html_url"),
                        "fork": repo.get("fork", False),
                    }
                )

        return {
            "username": profile.get("login"),
            "name": profile.get("name"),
            "bio": profile.get("bio"),
            "public_repos": profile.get("public_repos", 0),
            "followers": profile.get("followers", 0),
            "following": profile.get("following", 0),
            "profile_url": profile.get("html_url"),
            "repositories": repositories,
        }

    except Exception as exc:

        logger.exception("GitHub analysis failed")

        return {
            "error": f"GitHub API error: {exc}"
        }


# ============================================================
# JOB SEARCH
# ============================================================

def search_jobs(role: str) -> str:
    """
    Lightweight public job search.
    Uses DuckDuckGo HTML search so the deployment does not depend
    on the old DuckDuckGo LangChain tool.
    """

    try:

        query = f"{role} fresher jobs India placement internship"

        url = "https://html.duckduckgo.com/html/"

        response = requests.post(
            url,
            data={"q": query},
            headers={
                "User-Agent": "Mozilla/5.0"
            },
            timeout=10,
        )

        if response.status_code != 200:
            return "Job search temporarily unavailable."

        from bs4 import BeautifulSoup

        soup = BeautifulSoup(
            response.text,
            "html.parser",
        )

        results = []

        for result in soup.select(".result")[:6]:

            title_element = result.select_one(".result__a")
            snippet_element = result.select_one(".result__snippet")

            if title_element:

                title = title_element.get_text(
                    " ",
                    strip=True,
                )

                link = title_element.get(
                    "href",
                    "",
                )

                snippet = ""

                if snippet_element:
                    snippet = snippet_element.get_text(
                        " ",
                        strip=True,
                    )

                results.append(
                    {
                        "title": title,
                        "link": link,
                        "snippet": snippet,
                    }
                )

        if not results:
            return "No current public job-search results were found."

        return json.dumps(
            results,
            indent=2,
        )

    except Exception as exc:

        logger.warning(
            "Job search failed: %s",
            exc,
        )

        return (
            "Job search was unavailable. "
            "Analyze the target role using general placement requirements."
        )


# ============================================================
# MAIN LANGCHAIN ANALYSIS
# ============================================================

analysis_prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """
You are a professional campus-placement AI advisor.

Your job is to analyze a student's:

1. Resume
2. Target placement role
3. Current job opportunities
4. GitHub profile

Then produce a practical placement-readiness report.

IMPORTANT:

- Do not invent facts about the student's resume.
- Do not invent GitHub repositories.
- Use the supplied GitHub information.
- Clearly distinguish observed information from recommendations.
- Give realistic scores.
- Be helpful to a college student.
- Avoid generic motivational filler.
- Make the report actionable.

Return ONLY valid JSON.

The JSON must have EXACTLY this structure:

{
  "overall_score": 0,
  "github_score": 0,

  "job_analysis": {
    "target_role": "",
    "market_expectations": [],
    "relevant_opportunities": []
  },

  "current_skills": [],

  "skill_gaps": [
    {
      "skill": "",
      "importance": "High",
      "reason": ""
    }
  ],

  "recommended_projects": [
    {
      "title": "",
      "description": "",
      "skills": [],
      "why_it_helps": ""
    }
  ],

  "github_evaluation": {
    "strengths": [],
    "weaknesses": [],
    "recommendations": []
  },

  "action_plan": [
    {
      "priority": 1,
      "action": "",
      "reason": ""
    }
  ],

  "human_report": ""
}

For overall_score and github_score use integers from 0 to 100.

The human_report must sound like a real placement mentor speaking directly
to the student.

It should:

- address the student naturally
- explain where they currently stand
- explain their strongest areas
- explain their biggest weaknesses
- explain what they should learn next
- recommend a practical project strategy
- discuss GitHub honestly
- give a clear next-step roadmap

Keep human_report between roughly 250 and 450 words.

Do NOT use markdown inside the JSON string.
""",
        ),
        (
            "human",
            """
TARGET ROLE:
{role}

RESUME:
{resume}

JOB SEARCH INFORMATION:
{jobs}

GITHUB PROFILE:
{github}
""",
        ),
    ]
)


analysis_chain = analysis_prompt | llm


# ============================================================
# SAFE JSON PARSER
# ============================================================

def parse_json_response(response: Any) -> Dict[str, Any]:

    try:

        if hasattr(response, "content"):
            content = response.content
        else:
            content = str(response)

        if isinstance(content, list):

            parts = []

            for item in content:

                if isinstance(item, dict):
                    parts.append(
                        item.get("text", "")
                    )

                else:
                    parts.append(str(item))

            content = "".join(parts)

        content = str(content).strip()

        # Remove markdown code fences if Gemini adds them.

        if content.startswith("```"):

            content = content.replace(
                "```json",
                "",
                1,
            )

            content = content.replace(
                "```",
                "",
            )

            content = content.strip()

        # Locate JSON object.

        start = content.find("{")
        end = content.rfind("}")

        if start >= 0 and end >= 0:

            content = content[start:end + 1]

        data = json.loads(content)

        return data

    except Exception as exc:

        logger.exception(
            "JSON parsing failed: %s",
            exc,
        )

        return {
            "overall_score": 0,
            "github_score": 0,
            "job_analysis": {
                "target_role": "",
                "market_expectations": [],
                "relevant_opportunities": [],
            },
            "current_skills": [],
            "skill_gaps": [],
            "recommended_projects": [],
            "github_evaluation": {
                "strengths": [],
                "weaknesses": [],
                "recommendations": [],
            },
            "action_plan": [],
            "human_report": (
                "The AI analysis could not be parsed correctly. "
                "Please try the analysis again."
            ),
        }


# ============================================================
# PLACEMENT WORKFLOW
# ============================================================

def run_placement_workflow(
    resume_text: str,
    role: str,
    github_username: str,
) -> Dict[str, Any]:

    logger.info(
        "Starting placement workflow for role=%s github=%s",
        role,
        github_username,
    )

    # --------------------------------------------------------
    # 1. JOB SEARCH
    # --------------------------------------------------------

    jobs = search_jobs(role)

    # --------------------------------------------------------
    # 2. GITHUB
    # --------------------------------------------------------

    github_data = get_github_profile(
        github_username
    )

    # --------------------------------------------------------
    # 3. LANGCHAIN + GEMINI ANALYSIS
    # --------------------------------------------------------

    response = analysis_chain.invoke(
        {
            "role": role,
            "resume": resume_text[:20000],
            "jobs": jobs[:12000],
            "github": json.dumps(
                github_data,
                indent=2,
            )[:15000],
        }
    )

    report = parse_json_response(
        response
    )

    # --------------------------------------------------------
    # 4. NORMALIZE OUTPUT
    # --------------------------------------------------------

    report["overall_score"] = int(
        report.get(
            "overall_score",
            0,
        )
    )

    report["github_score"] = int(
        report.get(
            "github_score",
            0,
        )
    )

    report["target_role"] = role

    report["github_username"] = github_username

    return report


# ============================================================
# LANGSERVE INPUT
# ============================================================

def playground_workflow(data: Dict[str, Any]) -> Dict[str, Any]:

    resume = data.get(
        "resume",
        "",
    )

    role = data.get(
        "role",
        "",
    )

    github = data.get(
        "github_username",
        "",
    )

    if not resume:
        return {
            "error": "Please provide resume text."
        }

    if not role:
        return {
            "error": "Please provide a target placement role."
        }

    return run_placement_workflow(
        resume,
        role,
        github,
    )


placement_runnable = RunnableLambda(
    playground_workflow
)


# ============================================================
# LANGSERVE ROUTE
# ============================================================

add_routes(
    app,
    placement_runnable,
    path="/agent",
)


# ============================================================
# CUSTOM WEB UI
# ============================================================

HTML_PAGE = r"""
<!DOCTYPE html>

<html lang="en">

<head>

<meta charset="UTF-8">

<meta
name="viewport"
content="width=device-width, initial-scale=1.0"
>

<title>
Placement-Ready AI Agent
</title>

<style>

* {
    box-sizing: border-box;
}

body {

    margin: 0;

    font-family:
        Inter,
        Arial,
        sans-serif;

    color: #f8fafc;

    background:
        radial-gradient(
            circle at top left,
            #172554 0%,
            #090d1b 42%,
            #050713 100%
        );

    min-height: 100vh;
}

.container {

    width: min(
        1200px,
        94%
    );

    margin: auto;

    padding: 45px 0 80px;
}

.badge {

    display: inline-block;

    padding: 9px 16px;

    border: 1px solid
        rgba(
            96,
            165,
            250,
            0.4
        );

    border-radius: 999px;

    color: #93c5fd;

    background:
        rgba(
            15,
            23,
            42,
            0.6
        );

    font-size: 14px;

    font-weight: 700;
}

.hero {

    text-align: center;

    margin-bottom: 38px;
}

.hero h1 {

    font-size:
        clamp(
            42px,
            7vw,
            72px
        );

    margin: 18px 0 10px;

    letter-spacing: -3px;
}

.gradient {

    background:
        linear-gradient(
            90deg,
            #60a5fa,
            #818cf8,
            #c084fc
        );

    -webkit-background-clip: text;

    background-clip: text;

    color: transparent;
}

.hero p {

    max-width: 850px;

    margin: auto;

    color: #94a3b8;

    font-size: 18px;

    line-height: 1.6;
}

.card {

    background:
        rgba(
            15,
            23,
            42,
            0.85
        );

    border: 1px solid
        rgba(
            148,
            163,
            184,
            0.18
        );

    border-radius: 22px;

    padding: 32px;

    box-shadow:
        0 30px 80px
        rgba(
            0,
            0,
            0,
            0.35
        );

    backdrop-filter: blur(18px);
}

.input-grid {

    display: grid;

    grid-template-columns:
        1fr 1fr;

    gap: 24px;

    align-items: start;
}

.field label {

    display: block;

    margin-bottom: 10px;

    font-size: 13px;

    font-weight: 800;

    letter-spacing: 0.5px;

    text-transform: uppercase;

    color: #cbd5e1;
}

input[type="text"] {

    width: 100%;

    padding: 17px;

    border-radius: 13px;

    border: 1px solid
        #334155;

    background: #eef4ff;

    color: #0f172a;

    font-size: 16px;

    outline: none;
}

.upload {

    border:
        1px dashed
        #475569;

    border-radius: 18px;

    padding: 28px;

    text-align: center;

    min-height: 190px;

    display: flex;

    align-items: center;

    justify-content: center;

    flex-direction: column;

    cursor: pointer;

    background:
        rgba(
            2,
            6,
            23,
            0.55
        );
}

.upload:hover {

    border-color: #60a5fa;
}

.upload-icon {

    font-size: 38px;

    margin-bottom: 8px;
}

.upload strong {

    font-size: 17px;
}

.upload small {

    color: #64748b;

    margin-top: 8px;
}

.file-name {

    color: #60a5fa;

    margin-top: 12px;

    font-weight: 700;

    word-break: break-word;
}

input[type="file"] {

    display: none;
}

.analyze {

    width: 100%;

    margin-top: 28px;

    padding: 18px;

    border: none;

    border-radius: 14px;

    color: white;

    background:
        linear-gradient(
            90deg,
            #2563eb,
            #4f46e5,
            #6366f1
        );

    font-size: 18px;

    font-weight: 800;

    cursor: pointer;

    transition: 0.2s;
}

.analyze:hover {

    transform: translateY(-2px);

    box-shadow:
        0 15px 40px
        rgba(
            79,
            70,
            229,
            0.35
        );
}

.analyze:disabled {

    opacity: 0.6;

    cursor: wait;

    transform: none;
}

.workflow {

    margin-top: 30px;
}

.workflow-header {

    display: flex;

    justify-content: space-between;

    margin-bottom: 10px;

    font-weight: 800;
}

.progress {

    height: 9px;

    background: #1e293b;

    border-radius: 999px;

    overflow: hidden;
}

.progress-bar {

    height: 100%;

    width: 0%;

    background:
        linear-gradient(
            90deg,
            #3b82f6,
            #8b5cf6
        );

    transition:
        width 0.5s ease;
}

.steps {

    display: grid;

    grid-template-columns:
        repeat(
            5,
            1fr
        );

    gap: 9px;

    margin-top: 22px;
}

.step {

    text-align: center;

    padding: 13px 8px;

    border: 1px solid
        #334155;

    border-radius: 12px;

    color: #64748b;

    font-size: 13px;

    font-weight: 800;
}

.step.active {

    color: #86efac;

    border-color: #22c55e;

    background:
        rgba(
            34,
            197,
            94,
            0.08
        );
}

.status {

    margin-top: 20px;

    padding: 17px;

    border-radius: 13px;

    border: 1px solid
        #1e293b;

    color: #93c5fd;

    background:
        rgba(
            15,
            23,
            42,
            0.6
        );
}


/* =====================================================
   REPORT
===================================================== */

.report {

    margin-top: 35px;

    display: none;
}

.report-title {

    font-size: 32px;

    margin-bottom: 20px;
}

.report-grid {

    display: grid;

    grid-template-columns:
        1fr 1fr;

    gap: 22px;

    align-items: stretch;
}

.report-panel {

    border-radius: 20px;

    padding: 27px;

    border: 1px solid
        rgba(
            148,
            163,
            184,
            0.18
        );

    background:
        rgba(
            15,
            23,
            42,
            0.9
        );
}

.report-panel h3 {

    margin-top: 0;

    font-size: 21px;
}

.panel-label {

    color: #60a5fa;

    font-size: 12px;

    font-weight: 900;

    text-transform: uppercase;

    letter-spacing: 1px;
}

.score {

    display: flex;

    align-items: center;

    gap: 18px;

    margin: 20px 0 25px;
}

.score-number {

    font-size: 54px;

    font-weight: 900;

    background:
        linear-gradient(
            90deg,
            #60a5fa,
            #a78bfa
        );

    -webkit-background-clip: text;

    color: transparent;
}

.score-label {

    color: #94a3b8;

    line-height: 1.5;
}

.section {

    margin-top: 25px;
}

.section h4 {

    margin-bottom: 10px;

    color: #e2e8f0;
}

ul {

    padding-left: 20px;

    color: #94a3b8;

    line-height: 1.7;
}

li {

    margin-bottom: 5px;
}

.gap {

    padding: 13px;

    margin-bottom: 9px;

    border-radius: 12px;

    background:
        rgba(
            30,
            41,
            59,
            0.7
        );

    border: 1px solid
        #334155;
}

.gap strong {

    color: #f8fafc;
}

.priority {

    display: flex;

    gap: 12px;

    margin-bottom: 12px;

    padding: 13px;

    border-radius: 12px;

    background:
        rgba(
            30,
            41,
            59,
            0.7
        );
}

.priority-number {

    min-width: 28px;

    height: 28px;

    border-radius: 50%;

    display: flex;

    align-items: center;

    justify-content: center;

    background: #4f46e5;

    font-weight: 800;
}

.human-report {

    color: #cbd5e1;

    font-size: 16px;

    line-height: 1.85;

    white-space: pre-wrap;
}

.stat-row {

    display: grid;

    grid-template-columns:
        1fr 1fr;

    gap: 12px;

    margin-top: 15px;
}

.stat {

    padding: 16px;

    border-radius: 13px;

    background:
        #0b1222;

    border: 1px solid
        #1e293b;
}

.stat-value {

    font-size: 25px;

    font-weight: 900;

    color: #93c5fd;
}

.stat-name {

    color: #64748b;

    font-size: 12px;

    margin-top: 3px;
}

footer {

    text-align: center;

    color: #475569;

    margin-top: 35px;

    font-size: 13px;
}


@media (max-width: 850px) {

    .input-grid,
    .report-grid {

        grid-template-columns:
            1fr;
    }

    .steps {

        grid-template-columns:
            1fr 1fr;
    }

    .hero h1 {

        letter-spacing: -1px;
    }
}

</style>

</head>


<body>

<div class="container">

    <div class="hero">

        <div class="badge">
            ✦ LangChain · Gemini · GitHub · Placement AI
        </div>

        <h1>
            Placement-Ready
            <span class="gradient">
                AI Agent
            </span>
        </h1>

        <p>
            Upload your resume, choose your target role,
            and let the agent analyze job requirements,
            identify skill gaps, recommend projects,
            and evaluate your GitHub profile.
        </p>

    </div>


    <div class="card">

        <div class="input-grid">

            <div class="field">

                <label>
                    Resume PDF
                </label>

                <label
                    class="upload"
                    for="resume"
                >

                    <div class="upload-icon">
                        📄
                    </div>

                    <strong>
                        Drop your resume here
                    </strong>

                    <small>
                        or click to browse · PDF up to 10 MB
                    </small>

                    <div
                        id="fileName"
                        class="file-name"
                    >
                    </div>

                </label>

                <input
                    id="resume"
                    type="file"
                    accept=".pdf"
                >

            </div>


            <div>

                <div class="field">

                    <label>
                        Target Placement Role
                    </label>

                    <input
                        id="role"
                        type="text"
                        value="Web developer"
                    >

                </div>


                <div
                    class="field"
                    style="margin-top:22px;"
                >

                    <label>
                        GitHub Username
                    </label>

                    <input
                        id="github"
                        type="text"
                        value="yogeeswar-09"
                    >

                </div>

            </div>

        </div>


        <button
            id="analyze"
            class="analyze"
            onclick="analyzePlacement()"
        >
            Analyze Placement Readiness →
        </button>


        <div
            id="workflow"
            class="workflow"
        >

            <div class="workflow-header">

                <span>
                    Agent workflow
                </span>

                <span id="percent">
                    0%
                </span>

            </div>

            <div class="progress">

                <div
                    id="progressBar"
                    class="progress-bar"
                ></div>

            </div>


            <div class="steps">

                <div
                    id="step1"
                    class="step"
                >
                    Resume
                </div>

                <div
                    id="step2"
                    class="step"
                >
                    Jobs
                </div>

                <div
                    id="step3"
                    class="step"
                >
                    Skill Gaps
                </div>

                <div
                    id="step4"
                    class="step"
                >
                    Projects
                </div>

                <div
                    id="step5"
                    class="step"
                >
                    GitHub
                </div>

            </div>


            <div
                id="status"
                class="status"
            >
                Ready to analyze your placement readiness.
            </div>

        </div>

    </div>


    <!-- ==================================================
         REPORT
    =================================================== -->

    <div
        id="report"
        class="report"
    >

        <h2 class="report-title">
            Placement Analysis
        </h2>


        <div class="report-grid">


            <!-- ==========================================
                 STRUCTURED REPORT
            =========================================== -->

            <div class="report-panel">

                <div class="panel-label">
                    Structured assessment
                </div>

                <h3>
                    📊 Placement Readiness Report
                </h3>


                <div class="score">

                    <div
                        id="overallScore"
                        class="score-number"
                    >
                        0/100
                    </div>

                    <div class="score-label">
                        Overall placement<br>
                        readiness
                    </div>

                </div>


                <div class="stat-row">

                    <div class="stat">

                        <div
                            id="githubScore"
                            class="stat-value"
                        >
                            0/100
                        </div>

                        <div class="stat-name">
                            GitHub readiness
                        </div>

                    </div>

                    <div class="stat">

                        <div
                            id="targetRole"
                            class="stat-value"
                        >
                            -
                        </div>

                        <div class="stat-name">
                            Target role
                        </div>

                    </div>

                </div>


                <div class="section">

                    <h4>
                        💼 Job Opportunity Analysis
                    </h4>

                    <ul id="jobAnalysis"></ul>

                </div>


                <div class="section">

                    <h4>
                        🧠 Current Skills
                    </h4>

                    <ul id="skills"></ul>

                </div>


                <div class="section">

                    <h4>
                        ⚠️ Skill Gaps
                    </h4>

                    <div id="skillGaps"></div>

                </div>


                <div class="section">

                    <h4>
                        🚀 Recommended Projects
                    </h4>

                    <div id="projects"></div>

                </div>


                <div class="section">

                    <h4>
                        🐙 GitHub Evaluation
                    </h4>

                    <ul id="githubStrengths"></ul>

                    <strong>
                        Areas to improve
                    </strong>

                    <ul id="githubWeaknesses"></ul>

                </div>


                <div class="section">

                    <h4>
                        🎯 Priority Action Plan
                    </h4>

                    <div id="actionPlan"></div>

                </div>

            </div>


            <!-- ==========================================
                 HUMAN REPORT
            =========================================== -->

            <div class="report-panel">

                <div class="panel-label">
                    Personalized advisor
                </div>

                <h3>
                    🧑‍💼 Your Placement Advisor
                </h3>

                <div
                    id="humanReport"
                    class="human-report"
                >
                    Your personalized placement
                    advisor report will appear here.
                </div>

            </div>


        </div>

    </div>


    <footer>

        Placement-Ready AI Agent ·
        LangChain · Gemini · FastAPI · GitHub

    </footer>

</div>


<script>

const resumeInput =
    document.getElementById("resume");

const fileName =
    document.getElementById("fileName");

resumeInput.addEventListener(
    "change",
    function () {

        if (this.files.length) {

            fileName.textContent =
                this.files[0].name;

        } else {

            fileName.textContent = "";

        }

    }
);


function setProgress(
    percent,
    activeSteps
) {

    document
        .getElementById("progressBar")
        .style.width =
        percent + "%";

    document
        .getElementById("percent")
        .textContent =
        percent + "%";


    for (
        let i = 1;
        i <= 5;
        i++
    ) {

        const element =
            document.getElementById(
                "step" + i
            );

        if (
            activeSteps.includes(i)
        ) {

            element.classList.add(
                "active"
            );

        } else {

            element.classList.remove(
                "active"
            );

        }

    }

}


function listItems(
    elementId,
    items
) {

    const element =
        document.getElementById(
            elementId
        );

    element.innerHTML = "";

    if (
        !items ||
        items.length === 0
    ) {

        const li =
            document.createElement(
                "li"
            );

        li.textContent =
            "No specific information available.";

        element.appendChild(li);

        return;

    }


    items.forEach(
        item => {

            const li =
                document.createElement(
                    "li"
                );

            li.textContent =
                typeof item === "string"
                    ? item
                    : JSON.stringify(item);

            element.appendChild(li);

        }
    );

}


function renderReport(data) {

    document
        .getElementById("report")
        .style.display =
        "block";


    document
        .getElementById("overallScore")
        .textContent =
        `${data.overall_score}/100`;


    document
        .getElementById("githubScore")
        .textContent =
        `${data.github_score}/100`;


    document
        .getElementById("targetRole")
        .textContent =
        data.target_role || "-";


    const job =
        data.job_analysis || {};


    listItems(
        "jobAnalysis",
        [
            `Target role: ${job.target_role || data.target_role || "-"}`,
            ...(job.market_expectations || []),
            ...(job.relevant_opportunities || [])
                .map(
                    item =>
                        typeof item === "string"
                            ? item
                            : item.title || JSON.stringify(item)
                )
        ]
    );


    listItems(
        "skills",
        data.current_skills || []
    );


    const gaps =
        document.getElementById(
            "skillGaps"
        );

    gaps.innerHTML = "";


    (
        data.skill_gaps || []
    ).forEach(
        gap => {

            const div =
                document.createElement(
                    "div"
                );

            div.className =
                "gap";

            div.innerHTML = `
                <strong>
                    ${escapeHtml(
                        gap.skill || ""
                    )}
                </strong>
                <br>
                <span style="color:#fbbf24;">
                    ${escapeHtml(
                        gap.importance || ""
                    )}
                </span>
                <br>
                <span style="color:#94a3b8;">
                    ${escapeHtml(
                        gap.reason || ""
                    )}
                </span>
            `;

            gaps.appendChild(div);

        }
    );


    const projects =
        document.getElementById(
            "projects"
        );

    projects.innerHTML = "";


    (
        data.recommended_projects || []
    ).forEach(
        project => {

            const div =
                document.createElement(
                    "div"
                );

            div.className =
                "gap";

            div.innerHTML = `
                <strong>
                    ${escapeHtml(
                        project.title || ""
                    )}
                </strong>

                <br>

                <span style="color:#94a3b8;">
                    ${escapeHtml(
                        project.description || ""
                    )}
                </span>

                <br><br>

                <strong>
                    Skills:
                </strong>

                <span style="color:#94a3b8;">
                    ${escapeHtml(
                        (
                            project.skills || []
                        ).join(", ")
                    )}
                </span>

                <br><br>

                <span style="color:#93c5fd;">
                    ${escapeHtml(
                        project.why_it_helps || ""
                    )}
                </span>
            `;

            projects.appendChild(div);

        }
    );


    const github =
        data.github_evaluation || {};


    listItems(
        "githubStrengths",
        github.strengths || []
    );


    listItems(
        "githubWeaknesses",
        github.weaknesses || []
    );


    const actionPlan =
        document.getElementById(
            "actionPlan"
        );

    actionPlan.innerHTML = "";


    (
        data.action_plan || []
    ).forEach(
        item => {

            const div =
                document.createElement(
                    "div"
                );

            div.className =
                "priority";

            div.innerHTML = `
                <div class="priority-number">
                    ${escapeHtml(
                        String(
                            item.priority || ""
                        )
                    )}
                </div>

                <div>
                    <strong>
                        ${escapeHtml(
                            item.action || ""
                        )}
                    </strong>

                    <br>

                    <span style="color:#94a3b8;">
                        ${escapeHtml(
                            item.reason || ""
                        )}
                    </span>
                </div>
            `;

            actionPlan.appendChild(div);

        }
    );


    document
        .getElementById(
            "humanReport"
        )
        .textContent =
        data.human_report ||
        "No human advisor report was generated.";

}


function escapeHtml(
    value
) {

    return String(value)
        .replaceAll("&", "&amp;")
        .replaceAll("<", "&lt;")
        .replaceAll(">", "&gt;")
        .replaceAll('"', "&quot;")
        .replaceAll("'", "&#039;");

}


async function analyzePlacement() {

    const file =
        resumeInput.files[0];

    const role =
        document
            .getElementById("role")
            .value
            .trim();

    const github =
        document
            .getElementById("github")
            .value
            .trim();


    if (!file) {

        alert(
            "Please upload your Resume PDF."
        );

        return;

    }


    if (!role) {

        alert(
            "Please enter your target placement role."
        );

        return;

    }


    const button =
        document.getElementById(
            "analyze"
        );

    const status =
        document.getElementById(
            "status"
        );


    button.disabled = true;

    button.textContent =
        "Analyzing... Please wait";

    document
        .getElementById("report")
        .style.display =
        "none";


    try {

        setProgress(
            10,
            [1]
        );

        status.textContent =
            "Parsing your resume...";


        const formData =
            new FormData();

        formData.append(
            "file",
            file
        );


        const uploadResponse =
            await fetch(
                "/upload-resume",
                {
                    method: "POST",
                    body: formData
                }
            );


        if (
            !uploadResponse.ok
        ) {

            throw new Error(
                "Resume upload failed."
            );

        }


        const uploadData =
            await uploadResponse.json();


        setProgress(
            30,
            [1, 2]
        );

        status.textContent =
            "Searching placement opportunities...";


        await new Promise(
            resolve =>
                setTimeout(
                    resolve,
                    400
                )
        );


        setProgress(
            50,
            [1, 2, 3]
        );

        status.textContent =
            "Identifying your skill gaps...";


        await new Promise(
            resolve =>
                setTimeout(
                    resolve,
                    400
                )
        );


        setProgress(
            65,
            [1, 2, 3, 4]
        );

        status.textContent =
            "Generating project recommendations...";


        await new Promise(
            resolve =>
                setTimeout(
                    resolve,
                    400
                )
        );


        setProgress(
            80,
            [1, 2, 3, 4, 5]
        );

        status.textContent =
            "Evaluating your GitHub profile...";


        const response =
            await fetch(
                "/analyze",
                {
                    method: "POST",
                    headers: {
                        "Content-Type":
                            "application/json"
                    },

                    body: JSON.stringify(
                        {
                            role: role,
                            github_username:
                                github
                        }
                    )
                }
            );


        if (
            !response.ok
        ) {

            const errorText =
                await response.text();

            throw new Error(
                errorText ||
                "Analysis failed."
            );

        }


        const result =
            await response.json();


        setProgress(
            100,
            [1, 2, 3, 4, 5]
        );

        status.textContent =
            "Placement analysis completed successfully.";


        renderReport(
            result
        );


        window.scrollTo(
            {
                top:
                    document
                        .getElementById(
                            "report"
                        )
                        .offsetTop - 25,

                behavior: "smooth"
            }
        );


    } catch (error) {

        console.error(error);

        status.textContent =
            "Analysis failed. Please try again.";

        alert(
            "Analysis failed:\n\n" +
            error.message
        );


    } finally {

        button.disabled = false;

        button.textContent =
            "Analyze Again →";

    }

}

</script>

</body>

</html>
"""


# ============================================================
# CUSTOM UI ROUTE
# ============================================================

@app.get(
    "/",
    response_class=HTMLResponse,
)
async def home():

    return HTML_PAGE


# ============================================================
# RESUME UPLOAD
# ============================================================

@app.post("/upload-resume")
async def upload_resume(
    file: UploadFile = File(...)
):

    global LAST_RESUME_TEXT

    if not file.filename.lower().endswith(".pdf"):

        return JSONResponse(
            status_code=400,
            content={
                "error":
                    "Please upload a PDF resume."
            },
        )


    pdf_bytes =
        await file.read()


    if len(pdf_bytes) > 10 * 1024 * 1024:

        return JSONResponse(
            status_code=400,
            content={
                "error":
                    "Resume must be smaller than 10 MB."
            },
        )


    LAST_RESUME_TEXT =
        extract_resume_text(
            pdf_bytes
        )


    if LAST_RESUME_TEXT.startswith(
        "Resume extraction failed"
    ):

        return JSONResponse(
            status_code=400,
            content={
                "error":
                    LAST_RESUME_TEXT
            },
        )


    return {
        "success": True,
        "filename": file.filename,
        "message":
            "Resume uploaded successfully.",
    }


# ============================================================
# ANALYZE ENDPOINT
# ============================================================

@app.post("/analyze")
async def analyze(
    data: Dict[str, Any]
):

    global LAST_RESUME_TEXT

    if not LAST_RESUME_TEXT:

        return JSONResponse(
            status_code=400,
            content={
                "error":
                    "Please upload your resume first."
            },
        )


    role =
        str(
            data.get(
                "role",
                ""
            )
        ).strip()


    github =
        str(
            data.get(
                "github_username",
                ""
            )
        ).strip()


    if not role:

        return JSONResponse(
            status_code=400,
            content={
                "error":
                    "Target placement role is required."
            },
        )


    try:

        result =
            run_placement_workflow(
                resume_text=
                    LAST_RESUME_TEXT,
                role=role,
                github_username=github,
            )

        return result

    except Exception as exc:

        logger.exception(
            "Placement workflow failed"
        )

        return JSONResponse(
            status_code=500,
            content={
                "error":
                    f"Placement analysis failed: {exc}"
            },
        )


# ============================================================
# HEALTH
# ============================================================

@app.get("/health")
async def health():

    return {
        "status": "healthy",
        "agent":
            "Placement-Ready AI Agent",
        "version": "2.0",
    }


# ============================================================
# START
# ============================================================

if __name__ == "__main__":

    import uvicorn

    uvicorn.run(
        app,
        host="0.0.0.0",
        port=int(
            os.environ.get(
                "PORT",
                8000
            )
        ),
    )
