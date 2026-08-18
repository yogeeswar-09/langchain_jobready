import os
import logging
from io import BytesIO

import requests
from dotenv import load_dotenv
from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.responses import HTMLResponse
from pydantic import BaseModel, Field
from pypdf import PdfReader

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.runnables import RunnableLambda
from langchain_community.tools import DuckDuckGoSearchRun
from langserve import add_routes

# ============================================================
# CONFIG
# ============================================================

load_dotenv()

GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
if not GOOGLE_API_KEY:
    raise ValueError("GOOGLE_API_KEY is missing.")

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("placement_agent")

llm = ChatGoogleGenerativeAI(
    model="gemini-3.6-flash",
    google_api_key=GOOGLE_API_KEY,
    temperature=0,
)

# ============================================================
# PDF
# ============================================================

def extract_pdf_text(file_bytes: bytes) -> str:
    reader = PdfReader(BytesIO(file_bytes))
    pages = []

    for page in reader.pages:
        text = page.extract_text() or ""
        if text.strip():
            pages.append(text.strip())

    result = "\n\n".join(pages).strip()

    if not result:
        raise ValueError("No readable text was found in the PDF.")

    return result

# ============================================================
# JOB SEARCH
# ============================================================

def search_job_requirements(role: str) -> str:
    try:
        search = DuckDuckGoSearchRun()
        query = (
            f"current skills requirements for {role} internship "
            f"entry level campus placement jobs programming frameworks "
            f"databases cloud AI ML tools"
        )
        result = search.invoke(query)
        return str(result)[:10000]
    except Exception as exc:
        logger.exception("Job search failed")
        return f"Job search unavailable: {exc}"

# ============================================================
# SKILL GAP
# ============================================================

def analyze_skill_gap(resume: str, role: str) -> str:
    prompt = f"""
You are a campus placement skill-gap analyst.

Target role:
{role}

Student resume:
{resume[:14000]}

Identify:
1. Current relevant skills
2. Missing skills
3. Weak/basic skills
4. High-priority skills to learn
5. Medium-priority skills to learn

Only use evidence from the resume. Do not invent experience.
"""
    try:
        return str(llm.invoke(prompt).content)
    except Exception as exc:
        logger.exception("Skill gap analysis failed")
        return f"Skill gap analysis unavailable: {exc}"

# ============================================================
# PROJECT RECOMMENDATION
# ============================================================

def recommend_projects(resume: str, role: str, gaps: str) -> str:
    prompt = f"""
You are a placement project mentor.

Target role:
{role}

Student resume:
{resume[:10000]}

Skill gaps:
{gaps[:7000]}

Recommend exactly 3 realistic portfolio projects.

For each project include:
- Project name
- Problem
- Key features
- Technology stack
- Skills demonstrated
- Why it helps placement
- Difficulty

Projects should directly address the skill gaps and target role.
"""
    try:
        return str(llm.invoke(prompt).content)
    except Exception as exc:
        logger.exception("Project recommendation failed")
        return f"Project recommendations unavailable: {exc}"

# ============================================================
# GITHUB
# ============================================================

def evaluate_github(username: str, role: str) -> str:
    username = username.strip()

    if "github.com/" in username:
        username = username.split("github.com/", 1)[1]

    username = username.strip("/").split("/")[0]

    if not username:
        return "No GitHub username provided."

    headers = {
        "Accept": "application/vnd.github+json",
        "User-Agent": "Placement-Ready-AI-Agent",
    }

    try:
        profile_url = f"https://api.github.com/users/{username}"
        repos_url = (
            f"https://api.github.com/users/{username}/repos"
            "?per_page=30&sort=updated"
        )

        profile_response = requests.get(
            profile_url, headers=headers, timeout=15
        )

        if profile_response.status_code == 404:
            return f"GitHub user '{username}' was not found."

        profile_response.raise_for_status()

        repos_response = requests.get(
            repos_url, headers=headers, timeout=15
        )
        repos_response.raise_for_status()

        profile = profile_response.json()
        repos = repos_response.json()

        repo_lines = []
        for repo in repos[:30]:
            repo_lines.append(
                f"- {repo.get('name')} | "
                f"language={repo.get('language')} | "
                f"stars={repo.get('stargazers_count', 0)} | "
                f"description={repo.get('description') or 'No description'}"
            )

        github_data = f"""
Username: {username}
Name: {profile.get('name')}
Bio: {profile.get('bio')}
Public repositories: {profile.get('public_repos', 0)}
Followers: {profile.get('followers', 0)}
Profile: {profile.get('html_url')}

Repositories:
{chr(10).join(repo_lines)}
"""

        prompt = f"""
You are a technical recruiter evaluating GitHub for a campus placement.

Target role:
{role}

GitHub information:
{github_data}

Evaluate:
1. Profile strength
2. Project relevance
3. Technology relevance
4. Repository presentation
5. Activity signals available in the supplied data
6. Improvements
7. GitHub readiness score out of 100

Do not claim to have inspected source code, commits, or README files unless
that information is present in the supplied data.
"""
        return str(llm.invoke(prompt).content)

    except Exception as exc:
        logger.exception("GitHub evaluation failed")
        return f"GitHub evaluation unavailable: {exc}"

# ============================================================
# FINAL REPORT
# ============================================================

def final_report(
    role: str,
    resume: str,
    jobs: str,
    gaps: str,
    projects: str,
    github: str,
) -> str:
    prompt = f"""
You are a Placement-Ready AI Agent.

Create a professional campus-placement readiness report.

TARGET ROLE:
{role}

RESUME:
{resume[:10000]}

JOB REQUIREMENTS:
{jobs[:8000]}

SKILL GAP:
{gaps[:7000]}

PROJECT RECOMMENDATIONS:
{projects[:7000]}

GITHUB EVALUATION:
{github[:7000]}

Use this structure:

PLACEMENT READINESS REPORT

Target Role: {role}

1. JOB OPPORTUNITY ANALYSIS
Summarize the most important requirements.

2. CURRENT SKILLS
List the student's strongest relevant skills.

3. SKILL GAP ANALYSIS
Separate high, medium and low priority gaps.

4. RECOMMENDED PROJECTS
Give the best recommended projects and explain why.

5. GITHUB EVALUATION
Give strengths, weaknesses and improvements.

6. PRIORITY ACTION PLAN
Give exactly 5 practical next steps.

7. OVERALL PLACEMENT READINESS
Give a score in exactly this format:
Overall Placement Readiness: XX/100

Then give a short explanation.

Be honest. Do not invent achievements or experience.
"""
    try:
        return str(llm.invoke(prompt).content)
    except Exception as exc:
        logger.exception("Final report failed")
        return f"Final report unavailable: {exc}"

# ============================================================
# COMPLETE WORKFLOW
# ============================================================

def run_workflow(resume: str, role: str, github: str) -> str:
    logger.info("Starting placement workflow")

    jobs = search_job_requirements(role)
    logger.info("Job analysis complete")

    gaps = analyze_skill_gap(resume, role)
    logger.info("Skill gap analysis complete")

    projects = recommend_projects(resume, role, gaps)
    logger.info("Project recommendations complete")

    github_report = evaluate_github(github, role)
    logger.info("GitHub evaluation complete")

    report = final_report(
        role=role,
        resume=resume,
        jobs=jobs,
        gaps=gaps,
        projects=projects,
        github=github_report,
    )

    logger.info("Placement workflow complete")
    return report

# ============================================================
# FASTAPI
# ============================================================

app = FastAPI(
    title="Placement-Ready AI Agent",
    version="1.0",
    description=(
        "LangChain placement agent for job analysis, skill gaps, "
        "project recommendations and GitHub evaluation."
    ),
)

# ============================================================
# MAIN WEB APP
# ============================================================

HTML = """
<!DOCTYPE html>
<html>
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>Placement-Ready AI Agent</title>
<style>
body{
    margin:0;
    background:#070b14;
    color:#e5e7eb;
    font-family:Arial,sans-serif;
}
.wrap{
    width:min(950px,92%);
    margin:50px auto;
}
h1{text-align:center;margin-bottom:10px}
.sub{text-align:center;color:#94a3b8;margin-bottom:30px}
.card{
    background:#0f172a;
    border:1px solid #263244;
    border-radius:18px;
    padding:28px;
}
label{
    display:block;
    margin:18px 0 8px;
    font-weight:bold;
}
input{
    width:100%;
    box-sizing:border-box;
    padding:13px;
    border-radius:10px;
    border:1px solid #334155;
    background:#080d18;
    color:white;
}
button{
    width:100%;
    margin-top:25px;
    padding:15px;
    border:0;
    border-radius:10px;
    background:#2563eb;
    color:white;
    font-size:16px;
    font-weight:bold;
    cursor:pointer;
}
button:disabled{opacity:.5}
#status{
    margin-top:20px;
    color:#93c5fd;
    display:none;
}
#report{
    margin-top:25px;
    padding:25px;
    background:#080d18;
    border:1px solid #263244;
    border-radius:14px;
    white-space:pre-wrap;
    line-height:1.65;
    display:none;
}
.links{
    text-align:center;
    margin-top:20px;
}
a{color:#60a5fa;margin:0 8px}
</style>
</head>
<body>
<div class="wrap">
<h1>Placement-Ready AI Agent</h1>
<div class="sub">
Analyze job opportunities, identify skill gaps, recommend projects,
and evaluate your GitHub profile.
</div>

<div class="card">
<form id="form">

<label>Resume PDF</label>
<input id="resume" type="file" accept=".pdf" required>

<label>Target Placement Role</label>
<input id="role" type="text" placeholder="AI/ML Engineer" required>

<label>GitHub Username</label>
<input id="github" type="text" placeholder="yogeeswar-09" required>

<button id="button" type="submit">
Analyze Placement Readiness
</button>

</form>

<div id="status"></div>
<div id="report"></div>
</div>

<div class="links">
<a href="/agent/playground/" target="_blank">LangServe Playground</a>
<a href="/docs" target="_blank">API Docs</a>
<a href="/health" target="_blank">Health</a>
</div>
</div>

<script>
const form=document.getElementById("form");
const button=document.getElementById("button");
const status=document.getElementById("status");
const report=document.getElementById("report");

form.addEventListener("submit",async(e)=>{
    e.preventDefault();

    const file=document.getElementById("resume").files[0];
    const role=document.getElementById("role").value.trim();
    const github=document.getElementById("github").value.trim();

    if(!file){
        alert("Please select a PDF resume.");
        return;
    }

    button.disabled=true;
    button.textContent="Analyzing... Please wait";
    status.style.display="block";
    status.textContent="Running placement workflow...";
    report.style.display="none";

    const data=new FormData();
    data.append("resume",file);
    data.append("role",role);
    data.append("github_id",github);

    try{
        const response=await fetch("/api/analyze",{
            method:"POST",
            body:data
        });

        const result=await response.json();

        if(!response.ok){
            throw new Error(result.detail || "Analysis failed.");
        }

        report.textContent=result.report;
        report.style.display="block";
        status.textContent="Analysis completed successfully.";

        report.scrollIntoView({behavior:"smooth"});
    }catch(error){
        status.textContent="Analysis failed: "+error.message;
    }finally{
        button.disabled=false;
        button.textContent="Analyze Placement Readiness";
    }
});
</script>
</body>
</html>
"""

@app.get("/", response_class=HTMLResponse)
def home():
    return HTML

# ============================================================
# ANALYSIS API
# ============================================================

@app.post("/api/analyze")
async def analyze(
    resume: UploadFile = File(...),
    role: str = Form(...),
    github_id: str = Form(...),
):
    if not resume.filename:
        raise HTTPException(
            status_code=400,
            detail="Resume PDF is required.",
        )

    if not resume.filename.lower().endswith(".pdf"):
        raise HTTPException(
            status_code=400,
            detail="Only PDF files are supported.",
        )

    role = role.strip()
    github_id = github_id.strip()

    if not role:
        raise HTTPException(
            status_code=400,
            detail="Target role is required.",
        )

    if not github_id:
        raise HTTPException(
            status_code=400,
            detail="GitHub username is required.",
        )

    try:
        pdf_bytes = await resume.read()

        if len(pdf_bytes) > 10 * 1024 * 1024:
            raise HTTPException(
                status_code=400,
                detail="Resume must be smaller than 10 MB.",
            )

        resume_text = extract_pdf_text(pdf_bytes)

        report = run_workflow(
            resume=resume_text,
            role=role,
            github=github_id,
        )

        return {
            "status": "success",
            "filename": resume.filename,
            "role": role,
            "github_id": github_id,
            "report": report,
        }

    except HTTPException:
        raise

    except Exception as exc:
        logger.exception("Analysis request failed")
        raise HTTPException(
            status_code=500,
            detail=f"Analysis failed: {exc}",
        )

# ============================================================
# LANGSERVE PLAYGROUND
# ============================================================

class PlaygroundInput(BaseModel):
    role: str = Field(default="AI/ML Engineer")
    github_id: str = Field(default="")

def playground_function(data):
    role = data.get("role", "AI/ML Engineer")
    github_id = data.get("github_id", "")

    return run_playground_workflow(role, github_id)

def run_playground_workflow(role, github_id):
    if not github_id:
        return (
            "Enter a GitHub username to run the placement workflow. "
            "For PDF resume analysis, use the main application at /."
        )

    github_report = evaluate_github(github_id, role)

    prompt = f"""
You are a placement mentor.

Target role: {role}

GitHub evaluation:
{github_report}

Give a concise placement-readiness assessment with:
1. GitHub strengths
2. Weaknesses
3. Recommended improvements
4. Five action items
5. Overall readiness score out of 100
"""
    try:
        return str(llm.invoke(prompt).content)
    except Exception as exc:
        return f"Playground analysis failed: {exc}"

playground_runnable = RunnableLambda(
    playground_function
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
        "status": "healthy",
        "agent": "Placement-Ready AI Agent",
        "framework": "LangChain",
        "model": "gemini-2.5-flash",
        "workflow": [
            "PDF Resume Parsing",
            "Job Opportunity Analysis",
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
    import uvicorn

    uvicorn.run(
        "langchain_job_ready_agent:app",
        host="0.0.0.0",
        port=int(os.environ.get("PORT", "8000")),
    )
