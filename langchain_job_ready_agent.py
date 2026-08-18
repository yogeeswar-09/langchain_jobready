import os
import json
import logging
import re
from typing import Any, Dict

import requests
from dotenv import load_dotenv
from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, JSONResponse
from pydantic import BaseModel, Field
from pypdf import PdfReader

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableLambda
from langchain_google_genai import ChatGoogleGenerativeAI
from langserve import add_routes


# ============================================================
# CONFIG
# ============================================================

load_dotenv()

API_KEY = os.getenv("GOOGLE_API_KEY") or os.getenv("GEMINI_API_KEY")

if not API_KEY:
    raise RuntimeError(
        "Missing GOOGLE_API_KEY or GEMINI_API_KEY environment variable."
    )

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("placement-agent")


# ============================================================
# APP
# ============================================================

app = FastAPI(
    title="Placement-Ready AI Agent",
    version="3.0",
    description=(
        "Placement agent that analyzes resumes, jobs, skill gaps, "
        "projects and GitHub profiles."
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
# LLM
# ============================================================

llm = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash",
    google_api_key=API_KEY,
    temperature=0.2,
)


# ============================================================
# MEMORY FOR THE CURRENT DEPLOYMENT
# ============================================================

LAST_RESUME_TEXT = ""


# ============================================================
# LANGSERVE INPUT MODEL
# ============================================================

class PlaygroundInput(BaseModel):
    resume: str = Field(
        default="",
        description="Resume text"
    )
    role: str = Field(
        default="Web Developer",
        description="Target placement role"
    )
    github_username: str = Field(
        default="",
        description="GitHub username"
    )


# ============================================================
# PDF READER
# ============================================================

def extract_pdf_text(pdf_bytes: bytes) -> str:
    try:
        path = "/tmp/placement_resume.pdf"

        with open(path, "wb") as file:
            file.write(pdf_bytes)

        reader = PdfReader(path)

        pages = []

        for page in reader.pages:
            pages.append(page.extract_text() or "")

        text = "\n".join(pages).strip()

        if not text:
            return "No readable text was found in the PDF."

        return text

    except Exception as exc:
        logger.exception("PDF extraction failed")
        raise RuntimeError(f"Could not read resume PDF: {exc}")


# ============================================================
# JOB SEARCH
# ============================================================

def search_jobs(role: str) -> list:
    try:
        query = f"{role} fresher jobs India internship"

        response = requests.get(
            "https://html.duckduckgo.com/html/",
            params={"q": query},
            headers={"User-Agent": "Mozilla/5.0"},
            timeout=8,
        )

        if response.status_code != 200:
            return []

        # Parse DuckDuckGo result blocks without BeautifulSoup.
        # This avoids an unnecessary third-party dependency on Render.
        results = []

        blocks = re.findall(
            r'<div[^>]+class="[^"]*result[^"]*"[^>]*>(.*?)</div>\s*</div>',
            response.text,
            flags=re.IGNORECASE | re.DOTALL,
        )

        if not blocks:
            blocks = re.findall(
                r'<div[^>]+class="[^"]*result[^"]*"[^>]*>(.*?)(?=<div[^>]+class="[^"]*result[^"]*"|$)',
                response.text,
                flags=re.IGNORECASE | re.DOTALL,
            )

        def clean_html(value: str) -> str:
            value = re.sub(r"<[^>]+>", " ", value)
            value = re.sub(r"\s+", " ", value)
            return (
                value.replace("&amp;", "&")
                .replace("&quot;", '"')
                .replace("&#x27;", "'")
                .replace("&#39;", "'")
                .replace("&lt;", "<")
                .replace("&gt;", ">")
                .strip()
            )

        for block in blocks[:6]:
            title_match = re.search(
                r'class="[^"]*result__a[^"]*"[^>]*href="([^"]+)"[^>]*>(.*?)</a>',
                block,
                flags=re.IGNORECASE | re.DOTALL,
            )

            if not title_match:
                title_match = re.search(
                    r'href="([^"]+)"[^>]*class="[^"]*result__a[^"]*"[^>]*>(.*?)</a>',
                    block,
                    flags=re.IGNORECASE | re.DOTALL,
                )

            if not title_match:
                continue

            snippet_match = re.search(
                r'class="[^"]*result__snippet[^"]*"[^>]*>(.*?)</(?:a|div|span)>',
                block,
                flags=re.IGNORECASE | re.DOTALL,
            )

            results.append(
                {
                    "title": clean_html(title_match.group(2)),
                    "url": title_match.group(1),
                    "snippet": (
                        clean_html(snippet_match.group(1))
                        if snippet_match
                        else ""
                    ),
                }
            )

        return results

    except Exception as exc:
        logger.warning("Job search unavailable: %s", exc)
        return []


# ============================================================
# GITHUB
# ============================================================

def get_github(username: str) -> Dict[str, Any]:
    if not username:
        return {
            "available": False,
            "message": "GitHub username was not provided."
        }

    headers = {
        "Accept": "application/vnd.github+json",
        "User-Agent": "Placement-Ready-AI-Agent",
    }

    try:
        profile_response = requests.get(
            f"https://api.github.com/users/{username}",
            headers=headers,
            timeout=8,
        )

        if profile_response.status_code != 200:
            return {
                "available": False,
                "message": f"GitHub user '{username}' was not found."
            }

        profile = profile_response.json()

        repos_response = requests.get(
            f"https://api.github.com/users/{username}/repos",
            params={
                "per_page": 30,
                "sort": "updated",
            },
            headers=headers,
            timeout=8,
        )

        repos = []

        if repos_response.status_code == 200:
            for repo in repos_response.json():
                repos.append(
                    {
                        "name": repo.get("name"),
                        "description": repo.get("description"),
                        "language": repo.get("language"),
                        "stars": repo.get("stargazers_count", 0),
                        "forks": repo.get("forks_count", 0),
                        "updated_at": repo.get("updated_at"),
                        "url": repo.get("html_url"),
                        "fork": repo.get("fork", False),
                    }
                )

        return {
            "available": True,
            "username": profile.get("login"),
            "name": profile.get("name"),
            "bio": profile.get("bio"),
            "public_repos": profile.get("public_repos", 0),
            "followers": profile.get("followers", 0),
            "profile_url": profile.get("html_url"),
            "repositories": repos,
        }

    except Exception as exc:
        logger.warning("GitHub lookup failed: %s", exc)
        return {
            "available": False,
            "message": f"GitHub lookup failed: {exc}"
        }


# ============================================================
# AI PROMPT
# ============================================================

PROMPT = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """
You are a professional campus-placement AI advisor.

Analyze the student's resume, target role, public job information,
and GitHub profile.

The objective is to help the student become placement-ready.

Be factual. Never invent resume facts, GitHub repositories, or job
facts. If information is unavailable, say so.

Return ONLY valid JSON. Do not use markdown fences.

Use exactly this structure:

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

Scores must be integers from 0 to 100.

The human_report should be 250-450 words and sound like a
real placement mentor speaking directly to the student.

It should explain:
- where the student currently stands
- strongest areas
- biggest weaknesses
- what to learn next
- project strategy
- GitHub improvements
- a practical roadmap
""",
        ),
        (
            "human",
            """
TARGET ROLE:
{role}

RESUME:
{resume}

JOB INFORMATION:
{jobs}

GITHUB INFORMATION:
{github}
""",
        ),
    ]
)

analysis_chain = PROMPT | llm


# ============================================================
# JSON NORMALIZER
# ============================================================

def parse_ai_response(response: Any) -> Dict[str, Any]:
    content = getattr(response, "content", response)

    if isinstance(content, list):
        pieces = []

        for item in content:
            if isinstance(item, dict):
                pieces.append(str(item.get("text", "")))
            else:
                pieces.append(str(item))

        content = "".join(pieces)

    text = str(content).strip()

    if text.startswith("```"):
        text = text.replace("```json", "", 1)
        text = text.replace("```", "")
        text = text.strip()

    start = text.find("{")
    end = text.rfind("}")

    if start == -1 or end == -1:
        raise ValueError("Gemini did not return valid JSON.")

    return json.loads(text[start:end + 1])


# ============================================================
# MAIN WORKFLOW
# ============================================================

def run_workflow(
    resume: str,
    role: str,
    github_username: str
) -> Dict[str, Any]:

    jobs = search_jobs(role)
    github = get_github(github_username)

    response = analysis_chain.invoke(
        {
            "resume": resume[:20000],
            "role": role,
            "jobs": json.dumps(
                jobs,
                indent=2
            )[:12000],
            "github": json.dumps(
                github,
                indent=2
            )[:15000],
        }
    )

    report = parse_ai_response(response)

    report["overall_score"] = int(
        report.get("overall_score", 0)
    )

    report["github_score"] = int(
        report.get("github_score", 0)
    )

    report["target_role"] = role
    report["github_username"] = github_username

    return report


# ============================================================
# LANGSERVE WORKFLOW
# ============================================================

def playground_function(data: PlaygroundInput) -> Dict[str, Any]:
    if not data.resume.strip():
        return {
            "error": "Resume text is required."
        }

    if not data.role.strip():
        return {
            "error": "Target role is required."
        }

    return run_workflow(
        data.resume,
        data.role,
        data.github_username,
    )


playground_runnable = RunnableLambda(
    playground_function
).with_types(
    input_type=PlaygroundInput
)


# ============================================================
# LANGSERVE
# ============================================================

add_routes(
    app,
    playground_runnable,
    path="/agent",
)


# ============================================================
# UI
# ============================================================

HTML = r"""
<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width,initial-scale=1.0">
<title>Placement-Ready AI Agent</title>

<style>
*{box-sizing:border-box}

body{
margin:0;
font-family:Inter,Arial,sans-serif;
color:#f8fafc;
background:
radial-gradient(circle at 10% 0%,#172554 0%,transparent 35%),
radial-gradient(circle at 90% 20%,#28134d 0%,transparent 35%),
#050713;
min-height:100vh
}

.container{
width:min(1200px,94%);
margin:auto;
padding:40px 0 80px
}

.hero{text-align:center;margin-bottom:35px}

.badge{
display:inline-block;
padding:9px 16px;
border:1px solid #334d7c;
border-radius:999px;
color:#93c5fd;
background:#0f172acc;
font-size:13px;
font-weight:800
}

h1{
font-size:clamp(42px,7vw,76px);
margin:18px 0 10px;
letter-spacing:-3px
}

.gradient{
background:linear-gradient(90deg,#60a5fa,#818cf8,#c084fc);
-webkit-background-clip:text;
background-clip:text;
color:transparent
}

.hero p{
max-width:900px;
margin:auto;
color:#94a3b8;
font-size:18px;
line-height:1.6
}

.card{
background:#0f172ae8;
border:1px solid #33415555;
border-radius:24px;
padding:32px;
box-shadow:0 30px 90px #0008;
backdrop-filter:blur(18px)
}

.grid{
display:grid;
grid-template-columns:1fr 1fr;
gap:24px
}

label.title{
display:block;
font-size:13px;
font-weight:900;
text-transform:uppercase;
letter-spacing:.6px;
margin-bottom:10px;
color:#cbd5e1
}

.upload{
min-height:190px;
border:1px dashed #475569;
border-radius:18px;
display:flex;
align-items:center;
justify-content:center;
flex-direction:column;
cursor:pointer;
background:#02061788;
text-align:center;
padding:20px
}

.upload:hover{border-color:#60a5fa}

.upload-icon{font-size:38px}

.upload small{
color:#64748b;
margin-top:8px
}

.file-name{
margin-top:12px;
color:#60a5fa;
font-weight:800;
word-break:break-word
}

input[type=file]{display:none}

input[type=text]{
width:100%;
padding:17px;
border-radius:13px;
border:1px solid #334155;
background:#eef4ff;
color:#0f172a;
font-size:16px;
outline:none
}

.field+.field{margin-top:22px}

.analyze{
width:100%;
margin-top:28px;
padding:18px;
border:0;
border-radius:14px;
color:white;
background:linear-gradient(90deg,#2563eb,#4f46e5,#6366f1);
font-size:18px;
font-weight:900;
cursor:pointer
}

.analyze:disabled{opacity:.6;cursor:wait}

.workflow{margin-top:28px}

.workflow-head{
display:flex;
justify-content:space-between;
font-weight:900;
margin-bottom:9px
}

.progress{
height:9px;
background:#1e293b;
border-radius:999px;
overflow:hidden
}

.bar{
height:100%;
width:0;
background:linear-gradient(90deg,#3b82f6,#8b5cf6);
transition:width .4s
}

.steps{
display:grid;
grid-template-columns:repeat(5,1fr);
gap:9px;
margin-top:18px
}

.step{
padding:12px 6px;
text-align:center;
border:1px solid #334155;
border-radius:12px;
color:#64748b;
font-size:13px;
font-weight:800
}

.step.active{
color:#86efac;
border-color:#22c55e;
background:#22c55e12
}

.status{
margin-top:18px;
padding:16px;
border:1px solid #1e293b;
border-radius:13px;
color:#93c5fd;
background:#0f172acc
}

.report{display:none;margin-top:35px}

.report-title{
font-size:32px;
margin:0 0 20px
}

.report-grid{
display:grid;
grid-template-columns:1fr 1fr;
gap:22px
}

.panel{
background:#0f172af2;
border:1px solid #33415555;
border-radius:20px;
padding:28px
}

.panel-label{
color:#60a5fa;
font-size:12px;
font-weight:900;
text-transform:uppercase;
letter-spacing:1px
}

.panel h3{font-size:22px;margin:8px 0 20px}

.score{
font-size:58px;
font-weight:900;
background:linear-gradient(90deg,#60a5fa,#a78bfa);
-webkit-background-clip:text;
color:transparent
}

.stat-row{
display:grid;
grid-template-columns:1fr 1fr;
gap:12px
}

.stat{
padding:16px;
border-radius:13px;
background:#0b1222;
border:1px solid #1e293b
}

.stat-value{
font-size:24px;
font-weight:900;
color:#93c5fd
}

.stat-name{
color:#64748b;
font-size:12px;
margin-top:4px
}

.section{margin-top:25px}

.section h4{margin-bottom:10px}

ul{
padding-left:20px;
color:#94a3b8;
line-height:1.7
}

.gap,.project,.priority{
padding:14px;
margin-bottom:10px;
border-radius:12px;
background:#1e293bb8;
border:1px solid #334155
}

.gap strong,.project strong{color:#f8fafc}

.human{
color:#cbd5e1;
font-size:16px;
line-height:1.85;
white-space:pre-wrap
}

.priority{
display:flex;
gap:12px
}

.priority-number{
min-width:28px;
height:28px;
border-radius:50%;
background:#4f46e5;
display:flex;
align-items:center;
justify-content:center;
font-weight:900
}

footer{
text-align:center;
color:#475569;
margin-top:30px;
font-size:13px
}

@media(max-width:850px){
.grid,.report-grid{grid-template-columns:1fr}
.steps{grid-template-columns:1fr 1fr}
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
<span class="gradient">AI Agent</span>
</h1>

<p>
Upload your resume, choose your target role,
and let the agent analyze jobs, skill gaps,
projects and GitHub readiness.
</p>

</div>

<div class="card">

<div class="grid">

<div>

<label class="title">Resume PDF</label>

<label class="upload" for="resume">

<div class="upload-icon">📄</div>

<strong>Drop your resume here</strong>

<small>
or click to browse · PDF up to 10 MB
</small>

<div id="fileName" class="file-name"></div>

</label>

<input id="resume" type="file" accept=".pdf">

</div>

<div>

<div class="field">

<label class="title">
Target Placement Role
</label>

<input
id="role"
type="text"
value="Web developer"
>

</div>

<div class="field">

<label class="title">
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
onclick="analyze()"
>
Analyze Placement Readiness →
</button>

<div class="workflow">

<div class="workflow-head">
<span>Agent workflow</span>
<span id="percent">0%</span>
</div>

<div class="progress">
<div id="bar" class="bar"></div>
</div>

<div class="steps">

<div id="s1" class="step">Resume</div>
<div id="s2" class="step">Jobs</div>
<div id="s3" class="step">Skill Gaps</div>
<div id="s4" class="step">Projects</div>
<div id="s5" class="step">GitHub</div>

</div>

<div id="status" class="status">
Ready to analyze your placement readiness.
</div>

</div>

</div>


<div id="report" class="report">

<h2 class="report-title">
Placement Analysis
</h2>

<div class="report-grid">


<div class="panel">

<div class="panel-label">
Structured assessment
</div>

<h3>
📊 Placement Readiness Report
</h3>

<div id="overall" class="score">
0/100
</div>

<div class="stat-row">

<div class="stat">
<div id="githubScore" class="stat-value">
0/100
</div>
<div class="stat-name">
GitHub readiness
</div>
</div>

<div class="stat">
<div id="roleOutput" class="stat-value">
-
</div>
<div class="stat-name">
Target role
</div>
</div>

</div>

<div class="section">
<h4>💼 Job Analysis</h4>
<ul id="jobs"></ul>
</div>

<div class="section">
<h4>🧠 Current Skills</h4>
<ul id="skills"></ul>
</div>

<div class="section">
<h4>⚠️ Skill Gaps</h4>
<div id="gaps"></div>
</div>

<div class="section">
<h4>🚀 Recommended Projects</h4>
<div id="projects"></div>
</div>

<div class="section">
<h4>🐙 GitHub Evaluation</h4>
<strong>Strengths</strong>
<ul id="ghStrengths"></ul>
<strong>Improve</strong>
<ul id="ghWeaknesses"></ul>
</div>

<div class="section">
<h4>🎯 Action Plan</h4>
<div id="actions"></div>
</div>

</div>


<div class="panel">

<div class="panel-label">
Personalized advisor
</div>

<h3>
🧑‍💼 Human Placement Advisor
</h3>

<div id="human" class="human">
Your personalized mentor report will appear here.
</div>

</div>

</div>

</div>

<footer>
Placement-Ready AI Agent · LangChain · Gemini · FastAPI · GitHub
</footer>

</div>


<script>

const resume =
document.getElementById("resume");

resume.addEventListener(
"change",
function(){
document.getElementById("fileName").textContent =
this.files.length ? this.files[0].name : "";
}
);

function progress(percent, active){

document.getElementById("bar").style.width =
percent + "%";

document.getElementById("percent").textContent =
percent + "%";

for(let i=1;i<=5;i++){

const el=document.getElementById("s"+i);

if(active.includes(i)){
el.classList.add("active");
}else{
el.classList.remove("active");
}

}

}

function safe(value){

return String(value ?? "")
.replaceAll("&","&amp;")
.replaceAll("<","&lt;")
.replaceAll(">","&gt;")
.replaceAll('"',"&quot;")
.replaceAll("'","&#039;");
}

function makeList(id,items){

const el=document.getElementById(id);

el.innerHTML="";

if(!Array.isArray(items) || items.length===0){

el.innerHTML="<li>No specific information available.</li>";

return;
}

items.forEach(item=>{

const li=document.createElement("li");

if(typeof item==="string"){
li.textContent=item;
}else{
li.textContent=JSON.stringify(item);
}

el.appendChild(li);

});

}

function render(data){

document.getElementById("report").style.display="block";

document.getElementById("overall").textContent =
(data.overall_score ?? 0) + "/100";

document.getElementById("githubScore").textContent =
(data.github_score ?? 0) + "/100";

document.getElementById("roleOutput").textContent =
data.target_role || "-";

const job=data.job_analysis || {};

makeList(
"jobs",
[
"Target role: " + (job.target_role || data.target_role || "-"),
...(job.market_expectations || []),
...(job.relevant_opportunities || [])
.map(x => typeof x === "string" ? x : (x.title || JSON.stringify(x)))
]
);

makeList("skills",data.current_skills || []);

const gaps=document.getElementById("gaps");
gaps.innerHTML="";

(data.skill_gaps || []).forEach(g=>{

const div=document.createElement("div");
div.className="gap";

div.innerHTML=
"<strong>"+safe(g.skill)+"</strong><br>"+
"<span style='color:#fbbf24'>"+safe(g.importance)+"</span><br>"+
"<span style='color:#94a3b8'>"+safe(g.reason)+"</span>";

gaps.appendChild(div);

});

const projects=document.getElementById("projects");
projects.innerHTML="";

(data.recommended_projects || []).forEach(p=>{

const div=document.createElement("div");
div.className="project";

div.innerHTML=
"<strong>"+safe(p.title)+"</strong><br>"+
"<span style='color:#94a3b8'>"+safe(p.description)+"</span><br><br>"+
"<strong>Skills:</strong> "+
"<span style='color:#94a3b8'>"+
safe((p.skills || []).join(", "))+
"</span><br><br>"+
"<span style='color:#93c5fd'>"+
safe(p.why_it_helps)+
"</span>";

projects.appendChild(div);

});

const gh=data.github_evaluation || {};

makeList(
"ghStrengths",
gh.strengths || []
);

makeList(
"ghWeaknesses",
gh.weaknesses || []
);

const actions=document.getElementById("actions");
actions.innerHTML="";

(data.action_plan || []).forEach(a=>{

const div=document.createElement("div");
div.className="priority";

div.innerHTML=
"<div class='priority-number'>"+
safe(a.priority)+
"</div>"+
"<div>"+
"<strong>"+safe(a.action)+"</strong><br>"+
"<span style='color:#94a3b8'>"+
safe(a.reason)+
"</span>"+
"</div>";

actions.appendChild(div);

});

document.getElementById("human").textContent =
data.human_report ||
"No human report was generated.";

}

async function analyze(){

const file=resume.files[0];

const role=document
.getElementById("role")
.value.trim();

const github=document
.getElementById("github")
.value.trim();

if(!file){
alert("Please upload your Resume PDF.");
return;
}

if(!role){
alert("Please enter your target placement role.");
return;
}

const button=document.getElementById("analyze");
const status=document.getElementById("status");

button.disabled=true;
button.textContent="Analyzing... Please wait";

document.getElementById("report").style.display="none";

try{

progress(10,[1]);
status.textContent="Uploading and parsing your resume...";

const form=new FormData();
form.append("file",file);

const upload=await fetch(
"/upload-resume",
{
method:"POST",
body:form
}
);

if(!upload.ok){

const error=await upload.text();
throw new Error(error || "Resume upload failed.");

}

progress(25,[1,2]);
status.textContent="Searching relevant placement opportunities...";

await new Promise(r=>setTimeout(r,350));

progress(45,[1,2,3]);
status.textContent="Identifying your skill gaps...";

await new Promise(r=>setTimeout(r,350));

progress(60,[1,2,3,4]);
status.textContent="Creating project recommendations...";

await new Promise(r=>setTimeout(r,350));

progress(75,[1,2,3,4,5]);
status.textContent="Evaluating your GitHub profile...";

const response=await fetch(
"/analyze",
{
method:"POST",
headers:{
"Content-Type":"application/json"
},
body:JSON.stringify({
role:role,
github_username:github
})
}
);

if(!response.ok){

const error=await response.text();

throw new Error(
error || "Placement analysis failed."
);

}

const data=await response.json();

progress(100,[1,2,3,4,5]);

status.textContent =
"Placement analysis completed successfully.";

render(data);

setTimeout(
()=>{
document.getElementById("report")
.scrollIntoView({
behavior:"smooth",
block:"start"
});
},
100
);

}catch(error){

console.error(error);

status.textContent =
"Analysis failed. Please try again.";

alert(
"Analysis failed:\n\n" +
error.message
);

}finally{

button.disabled=false;
button.textContent="Analyze Again →";

}

}

</script>

</body>
</html>
"""


# ============================================================
# HOME
# ============================================================

@app.get("/", response_class=HTMLResponse)
async def home():
    return HTML


# ============================================================
# RESUME UPLOAD
# ============================================================

@app.post("/upload-resume")
async def upload_resume(file: UploadFile = File(...)):
    global LAST_RESUME_TEXT

    filename = file.filename or ""

    if not filename.lower().endswith(".pdf"):
        return JSONResponse(
            status_code=400,
            content={
                "error": "Please upload a PDF resume."
            },
        )

    pdf_bytes = await file.read()

    if len(pdf_bytes) > 10 * 1024 * 1024:
        return JSONResponse(
            status_code=400,
            content={
                "error": "Resume must be smaller than 10 MB."
            },
        )

    try:
        LAST_RESUME_TEXT = extract_pdf_text(pdf_bytes)

        return {
            "success": True,
            "filename": filename,
            "message": "Resume uploaded successfully.",
        }

    except Exception as exc:
        return JSONResponse(
            status_code=400,
            content={
                "error": str(exc)
            },
        )


# ============================================================
# ANALYZE
# ============================================================

@app.post("/analyze")
async def analyze(data: Dict[str, Any]):
    global LAST_RESUME_TEXT

    if not LAST_RESUME_TEXT:
        return JSONResponse(
            status_code=400,
            content={
                "error": "Please upload your resume first."
            },
        )

    role = str(data.get("role", "")).strip()
    github = str(
        data.get("github_username", "")
    ).strip()

    if not role:
        return JSONResponse(
            status_code=400,
            content={
                "error": "Target placement role is required."
            },
        )

    try:
        result = run_workflow(
            resume=LAST_RESUME_TEXT,
            role=role,
            github_username=github,
        )

        return result

    except Exception as exc:
        logger.exception("Workflow failed")

        return JSONResponse(
            status_code=500,
            content={
                "error": f"Placement analysis failed: {exc}"
            },
        )


# ============================================================
# HEALTH
# ============================================================

@app.get("/health")
async def health():
    return {
        "status": "healthy",
        "agent": "Placement-Ready AI Agent",
        "version": "3.0",
    }


# ============================================================
# LOCAL START
# ============================================================

if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        app,
        host="0.0.0.0",
        port=int(os.environ.get("PORT", "8000")),
    )
