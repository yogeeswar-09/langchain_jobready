import os
import logging
import uuid
from io import BytesIO

import requests
from dotenv import load_dotenv
from fastapi import FastAPI, UploadFile, File, Form, HTTPException, BackgroundTasks
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

# ============================================================
# WORKFLOW STATUS
# ============================================================

jobs_store = {}


def set_job(job_id: str, status: str, message: str, progress: int, report=None):
    jobs_store[job_id] = {
        "status": status,
        "message": message,
        "progress": progress,
        "report": report,
    }


def run_workflow(
    resume: str,
    role: str,
    github: str,
    job_id: str | None = None,
) -> str:
    if job_id:
        set_job(job_id, "running", "Reading your resume...", 10)

    logger.info("Starting placement workflow")

    if job_id:
        set_job(job_id, "running", "Analyzing current job requirements...", 25)
    jobs = search_job_requirements(role)
    logger.info("Job analysis complete")

    if job_id:
        set_job(job_id, "running", "Identifying your skill gaps...", 42)
    gaps = analyze_skill_gap(resume, role)
    logger.info("Skill gap analysis complete")

    if job_id:
        set_job(job_id, "running", "Recommending portfolio projects...", 60)
    projects = recommend_projects(resume, role, gaps)
    logger.info("Project recommendations complete")

    if job_id:
        set_job(job_id, "running", "Evaluating your GitHub profile...", 77)
    github_report = evaluate_github(github, role)
    logger.info("GitHub evaluation complete")

    if job_id:
        set_job(job_id, "running", "Preparing your final placement report...", 90)

    report = final_report(
        role=role,
        resume=resume,
        jobs=jobs,
        gaps=gaps,
        projects=projects,
        github=github_report,
    )

    if job_id:
        set_job(job_id, "completed", "Placement analysis completed.", 100, report)

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
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>Placement-Ready AI Agent</title>
<style>
:root{
    --bg:#050816;
    --panel:rgba(15,23,42,.78);
    --panel2:#0b1222;
    --border:#263552;
    --text:#f8fafc;
    --muted:#94a3b8;
    --blue:#3b82f6;
    --purple:#7c3aed;
    --green:#22c55e;
}
*{box-sizing:border-box}
body{
    margin:0;
    min-height:100vh;
    color:var(--text);
    font-family:Inter,ui-sans-serif,system-ui,-apple-system,BlinkMacSystemFont,"Segoe UI",sans-serif;
    background:
      radial-gradient(circle at 15% 10%,rgba(59,130,246,.18),transparent 30%),
      radial-gradient(circle at 85% 20%,rgba(124,58,237,.16),transparent 28%),
      var(--bg);
}
.wrap{width:min(1080px,92%);margin:42px auto 70px}
.hero{text-align:center;margin-bottom:34px}
.badge{
    display:inline-flex;align-items:center;gap:8px;
    padding:7px 13px;border:1px solid #29406a;
    border-radius:999px;background:rgba(15,23,42,.65);
    color:#93c5fd;font-size:13px;font-weight:700;
}
h1{
    font-size:clamp(34px,5vw,58px);
    line-height:1.05;margin:18px 0 12px;
    letter-spacing:-2px;
}
.gradient{
    background:linear-gradient(90deg,#60a5fa,#a78bfa);
    -webkit-background-clip:text;background-clip:text;color:transparent;
}
.hero p{max-width:760px;margin:auto;color:var(--muted);font-size:17px;line-height:1.6}
.card{
    background:var(--panel);
    backdrop-filter:blur(18px);
    border:1px solid var(--border);
    border-radius:24px;
    padding:30px;
    box-shadow:0 25px 80px rgba(0,0,0,.28);
}
.grid{
    display:grid;grid-template-columns:1.2fr 1fr;gap:20px;
}
.field{margin-bottom:20px}
label{display:block;font-size:14px;font-weight:800;margin-bottom:9px}
input[type=text]{
    width:100%;padding:14px 15px;border-radius:12px;
    border:1px solid #334155;background:#080e1d;color:white;
    outline:none;font-size:15px;transition:.2s;
}
input[type=text]:focus{border-color:#5b8def;box-shadow:0 0 0 4px rgba(59,130,246,.12)}
.drop{
    border:1.5px dashed #3a4b6c;border-radius:16px;
    padding:25px;text-align:center;background:#080e1d;
    transition:.2s;cursor:pointer;
}
.drop:hover,.drop.drag{border-color:#60a5fa;background:#0b1428}
.drop input{display:none}
.file-icon{font-size:34px}
.file-title{font-weight:800;margin-top:8px}
.file-help{font-size:13px;color:var(--muted);margin-top:5px}
.file-name{color:#93c5fd;margin-top:10px;word-break:break-all}
.button{
    width:100%;padding:16px;border:0;border-radius:13px;
    background:linear-gradient(90deg,#2563eb,#4f46e5);
    color:white;font-size:16px;font-weight:900;cursor:pointer;
    box-shadow:0 12px 28px rgba(37,99,235,.22);
}
.button:hover{filter:brightness(1.08)}
.button:disabled{opacity:.55;cursor:not-allowed}
.workflow{margin-top:26px}
.workflow-head{display:flex;justify-content:space-between;gap:15px;align-items:center}
.workflow-title{font-weight:900}
.percent{color:#93c5fd;font-weight:800}
.progress{
    height:8px;background:#111b2e;border-radius:999px;
    overflow:hidden;margin:12px 0 20px;
}
.bar{
    height:100%;width:0;
    background:linear-gradient(90deg,#3b82f6,#8b5cf6);
    transition:width .5s ease;
}
.steps{display:grid;grid-template-columns:repeat(5,1fr);gap:8px}
.step{
    padding:12px 8px;border:1px solid #263552;border-radius:12px;
    text-align:center;color:#64748b;font-size:12px;font-weight:700;
}
.step.active{border-color:#4f8cff;color:#bfdbfe;background:rgba(59,130,246,.08)}
.step.done{border-color:#267347;color:#86efac;background:rgba(34,197,94,.07)}
.status{
    margin-top:16px;padding:13px 15px;border-radius:12px;
    background:#080e1d;border:1px solid #263552;color:#93c5fd;
}
.report{
    margin-top:26px;padding:28px;border-radius:18px;
    background:#070d1a;border:1px solid #263552;
    line-height:1.75;white-space:pre-wrap;
    display:none;
}
.report-head{display:flex;justify-content:space-between;align-items:center;gap:12px;margin-bottom:15px}
.report h2{margin:0}
.copy{
    border:1px solid #334155;background:#111a2c;color:#cbd5e1;
    border-radius:9px;padding:8px 12px;cursor:pointer;
}
.info{
    display:grid;grid-template-columns:repeat(3,1fr);gap:12px;
    margin:24px 0;
}
.info-card{
    padding:17px;border:1px solid #263552;border-radius:15px;
    background:rgba(8,14,29,.7);
}
.info-card strong{display:block;margin-bottom:5px}
.info-card span{color:var(--muted);font-size:13px}
.links{text-align:center;margin-top:22px}
.links a{color:#60a5fa;margin:0 10px;text-decoration:none}
.links a:hover{text-decoration:underline}
.footer{text-align:center;color:#475569;font-size:12px;margin-top:25px}
@media(max-width:760px){
    .grid,.info{grid-template-columns:1fr}
    .steps{grid-template-columns:1fr 1fr}
    .card{padding:20px}
}
</style>
</head>

<body>
<div class="wrap">

<section class="hero">
    <div class="badge">✦ LangChain · Gemini · GitHub · Placement AI</div>
    <h1>Placement-Ready <span class="gradient">AI Agent</span></h1>
    <p>
        Upload your resume, choose your target role, and let the agent analyze
        job requirements, identify skill gaps, recommend projects, and evaluate
        your GitHub profile.
    </p>
</section>

<section class="card">
<form id="form">

<div class="grid">
<div class="field">
    <label>RESUME PDF</label>
    <label class="drop" id="drop">
        <input id="resume" type="file" accept=".pdf" required>
        <div class="file-icon">📄</div>
        <div class="file-title">Drop your resume here</div>
        <div class="file-help">or click to browse · PDF up to 10 MB</div>
        <div class="file-name" id="fileName">No file selected</div>
    </label>
</div>

<div>
<div class="field">
    <label for="role">TARGET PLACEMENT ROLE</label>
    <input id="role" type="text" placeholder="AI/ML Engineer" required>
</div>

<div class="field">
    <label for="github">GITHUB USERNAME</label>
    <input id="github" type="text" placeholder="yogeeswar-09" required>
</div>
</div>
</div>

<button class="button" id="button" type="submit">
    Analyze My Placement Readiness →
</button>
</form>

<div class="workflow" id="workflow" style="display:none">
    <div class="workflow-head">
        <div class="workflow-title">Agent workflow</div>
        <div class="percent" id="percent">0%</div>
    </div>
    <div class="progress"><div class="bar" id="bar"></div></div>

    <div class="steps">
        <div class="step" id="s1">Resume</div>
        <div class="step" id="s2">Jobs</div>
        <div class="step" id="s3">Skill Gaps</div>
        <div class="step" id="s4">Projects</div>
        <div class="step" id="s5">GitHub</div>
    </div>

    <div class="status" id="status">Preparing analysis...</div>
</div>

<div class="report" id="report">
    <div class="report-head">
        <h2>Placement Readiness Report</h2>
        <button class="copy" id="copy">Copy</button>
    </div>
    <div id="reportText"></div>
</div>
</section>

<div class="info">
    <div class="info-card">
        <strong>01 · Job Analysis</strong>
        <span>Finds skills commonly requested for your target role.</span>
    </div>
    <div class="info-card">
        <strong>02 · Skill Gap</strong>
        <span>Compares your resume against role requirements.</span>
    </div>
    <div class="info-card">
        <strong>03 · GitHub Review</strong>
        <span>Evaluates your public GitHub profile and repositories.</span>
    </div>
</div>

<div class="links">
    <a href="/agent/playground/" target="_blank">LangServe Playground</a>
    <a href="/docs" target="_blank">API Docs</a>
    <a href="/health" target="_blank">Health</a>
</div>

<div class="footer">Placement-Ready AI Agent · LangChain · FastAPI · Render</div>
</div>

<script>
const form=document.getElementById("form");
const fileInput=document.getElementById("resume");
const drop=document.getElementById("drop");
const fileName=document.getElementById("fileName");
const button=document.getElementById("button");
const workflow=document.getElementById("workflow");
const status=document.getElementById("status");
const bar=document.getElementById("bar");
const percent=document.getElementById("percent");
const report=document.getElementById("report");
const reportText=document.getElementById("reportText");
const copy=document.getElementById("copy");

const steps=[
    [10,"s1"],
    [25,"s2"],
    [42,"s3"],
    [60,"s4"],
    [77,"s5"]
];

function updateProgress(value,message){
    bar.style.width=value+"%";
    percent.textContent=value+"%";
    status.textContent=message;

    document.querySelectorAll(".step").forEach(x=>{
        x.classList.remove("active","done");
    });

    steps.forEach(([point,id])=>{
        const el=document.getElementById(id);
        if(value>=point) el.classList.add(value>=point+12?"done":"active");
    });
}

fileInput.addEventListener("change",()=>{
    fileName.textContent=fileInput.files[0]
        ? fileInput.files[0].name
        : "No file selected";
});

["dragenter","dragover"].forEach(event=>{
    drop.addEventListener(event,e=>{
        e.preventDefault();
        drop.classList.add("drag");
    });
});

["dragleave","drop"].forEach(event=>{
    drop.addEventListener(event,e=>{
        e.preventDefault();
        drop.classList.remove("drag");
    });
});

drop.addEventListener("drop",e=>{
    const file=e.dataTransfer.files[0];
    if(file && file.type==="application/pdf"){
        fileInput.files=e.dataTransfer.files;
        fileName.textContent=file.name;
    }
});

form.addEventListener("submit",async e=>{
    e.preventDefault();

    const file=fileInput.files[0];
    const role=document.getElementById("role").value.trim();
    const github=document.getElementById("github").value.trim();

    if(!file){
        alert("Please select your resume PDF.");
        return;
    }

    if(!file.name.toLowerCase().endsWith(".pdf")){
        alert("Please upload a PDF file.");
        return;
    }

    button.disabled=true;
    button.textContent="Analyzing...";
    workflow.style.display="block";
    report.style.display="none";
    updateProgress(5,"Uploading resume and starting the agent...");

    const data=new FormData();
    data.append("resume",file);
    data.append("role",role);
    data.append("github_id",github);

    try{
        const start=await fetch("/api/analyze",{
            method:"POST",
            body:data
        });

        const started=await start.json();

        if(!start.ok){
            throw new Error(started.detail || "Could not start analysis.");
        }

        const jobId=started.job_id;

        while(true){
            await new Promise(resolve=>setTimeout(resolve,1200));

            const response=await fetch("/api/status/"+jobId);
            const result=await response.json();

            if(!response.ok){
                throw new Error(result.detail || "Could not read analysis status.");
            }

            updateProgress(result.progress,result.message);

            if(result.status==="completed"){
                reportText.textContent=result.report || "No report returned.";
                report.style.display="block";
                button.textContent="Analyze Again →";
                status.textContent="Analysis completed successfully.";
                updateProgress(100,"Placement analysis completed successfully.");
                report.scrollIntoView({behavior:"smooth",block:"start"});
                break;
            }

            if(result.status==="failed"){
                throw new Error(result.message || "Analysis failed.");
            }
        }
    }catch(error){
        status.textContent="Analysis failed: "+error.message;
        button.textContent="Try Again";
    }finally{
        button.disabled=false;
    }
});

copy.addEventListener("click",async()=>{
    await navigator.clipboard.writeText(reportText.textContent);
    copy.textContent="Copied ✓";
    setTimeout(()=>copy.textContent="Copy",1500);
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
    background_tasks: BackgroundTasks,
    resume: UploadFile = File(...),
    role: str = Form(...),
    github_id: str = Form(...),
):
    if not resume.filename:
        raise HTTPException(status_code=400, detail="Resume PDF is required.")

    if not resume.filename.lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF files are supported.")

    role = role.strip()
    github_id = github_id.strip()

    if not role:
        raise HTTPException(status_code=400, detail="Target role is required.")

    if not github_id:
        raise HTTPException(status_code=400, detail="GitHub username is required.")

    try:
        pdf_bytes = await resume.read()

        if len(pdf_bytes) > 10 * 1024 * 1024:
            raise HTTPException(
                status_code=400,
                detail="Resume must be smaller than 10 MB.",
            )

        resume_text = extract_pdf_text(pdf_bytes)

        job_id = uuid.uuid4().hex
        set_job(job_id, "queued", "Analysis queued...", 5)

        background_tasks.add_task(
            run_workflow,
            resume_text,
            role,
            github_id,
            job_id,
        )

        return {
            "status": "started",
            "job_id": job_id,
        }

    except HTTPException:
        raise
    except Exception as exc:
        logger.exception("Could not start analysis")
        raise HTTPException(
            status_code=500,
            detail=f"Could not start analysis: {exc}",
        )


@app.get("/api/status/{job_id}")
def analysis_status(job_id: str):
    job = jobs_store.get(job_id)

    if not job:
        raise HTTPException(status_code=404, detail="Analysis job not found.")

    return job


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
        "model": "gemini-3.6-flash",
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
