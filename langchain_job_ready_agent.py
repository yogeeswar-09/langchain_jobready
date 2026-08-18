import os
import logging
from io import BytesIO
from concurrent.futures import ThreadPoolExecutor

import requests
from dotenv import load_dotenv

from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.responses import HTMLResponse

from langserve import add_routes

from pydantic import BaseModel, Field
from pypdf import PdfReader

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.runnables import RunnableLambda
from langchain_core.prompts import ChatPromptTemplate

from langchain_community.tools import DuckDuckGoSearchRun


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
# LLM
# ============================================================

llm = ChatGoogleGenerativeAI(
    model="gemma-4-31b-it",
    google_api_key=GOOGLE_API_KEY,
    temperature=0,
)


# ============================================================
# PDF TEXT EXTRACTION
# ============================================================

def extract_pdf_text(file_bytes: bytes) -> str:

    reader = PdfReader(
        BytesIO(file_bytes)
    )

    pages = []

    for page in reader.pages:

        text = page.extract_text()

        if text:
            pages.append(text)

    result = "\n\n".join(pages).strip()

    if not result:
        raise ValueError(
            "Could not extract readable text from the PDF."
        )

    return result


# ============================================================
# JOB SEARCH TOOL
# ============================================================

def search_job_requirements(role: str) -> str:

    logger.info(
        "Searching job requirements for: %s",
        role
    )

    try:

        search = DuckDuckGoSearchRun()

        query = f"""
        Search for recent internship and entry-level
        campus placement job requirements for:

        {role}

        Find commonly requested:
        - programming languages
        - frameworks
        - databases
        - cloud technologies
        - AI/ML skills
        - tools
        - soft skills

        Focus on current industry requirements.
        """

        result = search.invoke(query)

        return str(result)[:12000]

    except Exception as exc:

        logger.exception(
            "Job search failed"
        )

        return (
            "Job search was unavailable. "
            f"Reason: {exc}"
        )


# ============================================================
# SKILL GAP ANALYSIS
# ============================================================

def analyze_skill_gap(
    resume_text: str,
    role: str
) -> str:

    logger.info(
        "Running skill gap analysis"
    )

    prompt = f"""
You are a campus placement skill-gap analyst.

TARGET ROLE:
{role}

STUDENT RESUME:
{resume_text[:16000]}

Analyze the student's current skills against
the target placement role.

Return:

CURRENT SKILLS
- Skills clearly demonstrated in the resume.

MISSING SKILLS
- Important skills that are not demonstrated.

WEAK / NEEDS IMPROVEMENT
- Skills that appear limited or basic.

PRIORITY
- High priority
- Medium priority
- Low priority

Be evidence-based.

Do not invent skills that are not present
in the resume.
"""

    try:

        response = llm.invoke(
            prompt
        )

        return response.content

    except Exception as exc:

        logger.exception(
            "Skill gap analysis failed"
        )

        return (
            f"Skill gap analysis failed: {exc}"
        )


# ============================================================
# PROJECT RECOMMENDATION
# ============================================================

def recommend_projects(
    resume_text: str,
    role: str,
    skill_gap: str
) -> str:

    logger.info(
        "Generating project recommendations"
    )

    prompt = f"""
You are a campus placement project mentor.

TARGET ROLE:
{role}

STUDENT RESUME:
{resume_text[:12000]}

SKILL GAP ANALYSIS:
{skill_gap[:8000]}

Recommend 3 practical portfolio projects.

The projects must:

- address important skill gaps
- match the target role
- be realistic for a college student
- be suitable for GitHub
- demonstrate meaningful technical skills
- help during campus placements

For every project provide:

PROJECT NAME
PROBLEM
KEY FEATURES
TECHNOLOGY STACK
SKILLS DEVELOPED
WHY IT HELPS PLACEMENT
DIFFICULTY

Avoid generic or unrelated projects.
"""

    try:

        response = llm.invoke(
            prompt
        )

        return response.content

    except Exception as exc:

        logger.exception(
            "Project recommendation failed"
        )

        return (
            f"Project recommendation failed: {exc}"
        )


# ============================================================
# GITHUB EVALUATION
# ============================================================

def evaluate_github(
    username: str,
    role: str
) -> str:

    logger.info(
        "Evaluating GitHub: %s",
        username
    )

    username = username.strip()

    if "github.com/" in username:

        username = username.split(
            "github.com/"
        )[-1]

    username = username.rstrip(
        "/"
    ).split("/")[0]

    if not username:

        return "No GitHub username provided."

    headers = {
        "Accept":
            "application/vnd.github+json",

        "User-Agent":
            "Placement-Ready-AI-Agent"
    }

    try:

        profile_url = (
            f"https://api.github.com/users/{username}"
        )

        repos_url = (
            f"https://api.github.com/users/{username}/repos"
            f"?per_page=100&sort=updated"
        )

        profile_response = requests.get(
            profile_url,
            headers=headers,
            timeout=15
        )

        if profile_response.status_code == 404:

            return (
                f"GitHub user '{username}' was not found."
            )

        profile_response.raise_for_status()

        profile = profile_response.json()


        repos_response = requests.get(
            repos_url,
            headers=headers,
            timeout=15
        )

        repos_response.raise_for_status()

        repos = repos_response.json()

        if not isinstance(repos, list):

            repos = []


        repo_data = []

        for repo in repos[:25]:

            repo_data.append({

                "name":
                    repo.get("name"),

                "description":
                    repo.get("description"),

                "language":
                    repo.get("language"),

                "stars":
                    repo.get("stargazers_count"),

                "forks":
                    repo.get("forks_count"),

                "updated":
                    repo.get("updated_at"),

                "url":
                    repo.get("html_url")

            })


        github_data = {

            "username":
                username,

            "name":
                profile.get("name"),

            "bio":
                profile.get("bio"),

            "profile":
                profile.get("html_url"),

            "public_repositories":
                profile.get("public_repos", 0),

            "followers":
                profile.get("followers", 0),

            "following":
                profile.get("following", 0),

            "repositories":
                repo_data
        }


        prompt = f"""
You are a technical recruiter.

Evaluate this public GitHub profile
for the target campus placement role.

TARGET ROLE:
{role}

GITHUB DATA:
{github_data}

Evaluate:

1. Profile strength
2. Number and quality of projects
3. Technology relevance
4. Repository activity
5. Project descriptions
6. Documentation
7. Portfolio presentation
8. Areas to improve
9. Recruiter impression

Give a GitHub readiness score out of 100.

IMPORTANT:

Only use the GitHub data provided.

Do not claim that you inspected source code,
commits, README files, or code quality unless
that information is explicitly available.
"""

        response = llm.invoke(
            prompt
        )

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
            f"GitHub evaluation failed: {exc}"
        )


# ============================================================
# FINAL SYNTHESIS
# ============================================================

def generate_final_report(
    role: str,
    resume_text: str,
    job_requirements: str,
    skill_gap: str,
    projects: str,
    github_report: str
) -> str:

    logger.info(
        "Generating final placement report"
    )

    prompt = f"""
You are the final Placement-Ready AI Agent.

Your job is to synthesize the complete analysis
for a college student preparing for campus placements.

TARGET ROLE:
{role}


STUDENT RESUME:
{resume_text[:12000]}


JOB REQUIREMENTS:
{job_requirements[:10000]}


SKILL GAP ANALYSIS:
{skill_gap[:8000]}


PROJECT RECOMMENDATIONS:
{projects[:8000]}


GITHUB EVALUATION:
{github_report[:8000]}


Create a professional placement-readiness report.

Use exactly this structure:


PLACEMENT READINESS REPORT

Target Role:
{role}


1. JOB OPPORTUNITY ANALYSIS

Explain the important skills and requirements
currently relevant to this role.


2. CURRENT SKILLS

List the student's strongest relevant skills
based only on the resume.


3. SKILL GAP ANALYSIS

Explain:

- High priority gaps
- Medium priority gaps
- Low priority gaps


4. RECOMMENDED PROJECTS

Recommend the best projects from the analysis.

Explain why each project helps.


5. GITHUB EVALUATION

Summarize:

- strengths
- weaknesses
- improvements
- recruiter impression


6. PRIORITY ACTION PLAN

Give exactly 5 actionable steps.

Example:

1. Learn ...
2. Build ...
3. Improve GitHub ...
4. Practice ...
5. Apply ...


7. OVERALL PLACEMENT READINESS

Give a score from 0 to 100.

Format:

Overall Placement Readiness: XX/100

Then give a short explanation.


IMPORTANT:

- Do not invent information.
- Do not claim unsupported achievements.
- Keep recommendations realistic.
- Be honest about weaknesses.
- Make the report useful for campus placement preparation.
"""

    try:

        response = llm.invoke(
            prompt
        )

        return response.content

    except Exception as exc:

        logger.exception(
            "Final synthesis failed"
        )

        return (
            "Final synthesis failed: "
            f"{exc}"
        )


# ============================================================
# COMPLETE PLACEMENT WORKFLOW
# ============================================================

def run_complete_analysis(
    resume_text: str,
    role: str,
    github_id: str
) -> str:

    logger.info(
        "Starting complete placement workflow"
    )


    # --------------------------------------------------------
    # STEP 1
    # Job search
    # --------------------------------------------------------

    job_requirements = search_job_requirements(
        role
    )


    # --------------------------------------------------------
    # STEP 2
    # Skill gap
    # --------------------------------------------------------

    skill_gap = analyze_skill_gap(
        resume_text,
        role
    )


    # --------------------------------------------------------
    # STEP 3
    # Project recommendations
    # --------------------------------------------------------

    projects = recommend_projects(
        resume_text,
        role,
        skill_gap
    )


    # --------------------------------------------------------
    # STEP 4
    # GitHub
    # --------------------------------------------------------

    github_report = evaluate_github(
        github_id,
        role
    )


    # --------------------------------------------------------
    # STEP 5
    # Final synthesis
    # --------------------------------------------------------

    final_report = generate_final_report(
        role=role,
        resume_text=resume_text,
        job_requirements=job_requirements,
        skill_gap=skill_gap,
        projects=projects,
        github_report=github_report
    )


    logger.info(
        "Placement workflow completed"
    )


    return final_report


# ============================================================
# FASTAPI APP
# ============================================================

app = FastAPI(
    title="Placement-Ready AI Agent",
    version="2.0",
    description=(
        "LangChain-powered AI Agent for campus placement "
        "preparation."
    )
)


# ============================================================
# MAIN APPLICATION UI
# ============================================================

HTML_PAGE = r"""
<!DOCTYPE html>

<html lang="en">

<head>

<meta charset="UTF-8">

<meta name="viewport"
      content="width=device-width, initial-scale=1.0">

<title>
Placement-Ready AI Agent
</title>


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

    width: min(
        1100px,
        92%
    );

    margin: auto;

    padding:
        50px 0 70px;
}


.header {

    text-align: center;

    margin-bottom: 35px;
}


.badge {

    display: inline-block;

    padding:
        8px 14px;

    border:
        1px solid #334155;

    border-radius:
        999px;

    background:
        rgba(
            15,
            23,
            42,
            0.8
        );

    color:
        #93c5fd;

    font-size:
        13px;

    font-weight:
        600;

    margin-bottom:
        16px;
}


h1 {

    margin: 0;

    font-size:
        clamp(
            32px,
            5vw,
            56px
        );
}


.subtitle {

    max-width:
        720px;

    margin:
        18px auto 0;

    color:
        #94a3b8;

    font-size:
        17px;

    line-height:
        1.7;
}


.card {

    background:
        rgba(
            15,
            23,
            42,
            0.88
        );

    border:
        1px solid #1e293b;

    border-radius:
        20px;

    padding:
        28px;

    box-shadow:
        0 25px 70px
        rgba(
            0,
            0,
            0,
            0.35
        );
}


.grid {

    display:
        grid;

    grid-template-columns:
        repeat(
            2,
            minmax(
                0,
                1fr
            )
        );

    gap:
        20px;
}


.full {

    grid-column:
        1 / -1;
}


.field {

    margin-bottom:
        20px;
}


label {

    display:
        block;

    margin-bottom:
        9px;

    color:
        #e2e8f0;

    font-size:
        14px;

    font-weight:
        650;
}


input[type="text"] {

    width:
        100%;

    padding:
        14px 15px;

    border:
        1px solid #334155;

    border-radius:
        12px;

    outline:
        none;

    background:
        #0b1120;

    color:
        #f8fafc;

    font-size:
        15px;
}


input[type="text"]:focus {

    border-color:
        #60a5fa;
}


.file-box {

    border:
        1.5px dashed #475569;

    border-radius:
        14px;

    padding:
        22px;

    background:
        #0b1120;
}


input[type="file"] {

    width:
        100%;

    color:
        #94a3b8;
}


.help {

    margin-top:
        8px;

    color:
        #64748b;

    font-size:
        12px;
}


.button {

    width:
        100%;

    margin-top:
        8px;

    padding:
        15px 20px;

    border:
        0;

    border-radius:
        12px;

    background:
        linear-gradient(
            135deg,
            #2563eb,
            #4f46e5
        );

    color:
        white;

    font-size:
        16px;

    font-weight:
        700;

    cursor:
        pointer;
}


.button:disabled {

    opacity:
        0.55;

    cursor:
        not-allowed;
}


.status {

    display:
        none;

    margin-top:
        22px;

    padding:
        14px 16px;

    border-radius:
        12px;

    background:
        #0f172a;

    border:
        1px solid #1e293b;

    color:
        #94a3b8;
}


.status.show {

    display:
        block;
}


.status.error {

    color:
        #fca5a5;

    border-color:
        #7f1d1d;

    background:
        #1c0b0b;
}


.report {

    display:
        none;

    margin-top:
        28px;
}


.report.show {

    display:
        block;
}


.report-header {

    display:
        flex;

    align-items:
        center;

    justify-content:
        space-between;

    gap:
        20px;

    margin-bottom:
        18px;
}


.report-title {

    font-size:
        24px;

    font-weight:
        750;
}


.score {

    min-width:
        120px;

    padding:
        12px 16px;

    text-align:
        center;

    border:
        1px solid #334155;

    border-radius:
        14px;

    background:
        #0b1120;
}


.score-number {

    font-size:
        27px;

    font-weight:
        800;

    color:
        #60a5fa;
}


.score-label {

    color:
        #64748b;

    font-size:
        11px;

    text-transform:
        uppercase;
}


.report-body {

    white-space:
        pre-wrap;

    line-height:
        1.75;

    color:
        #cbd5e1;

    font-size:
        15px;
}


.links {

    margin-top:
        20px;

    text-align:
        center;
}


.links a {

    color:
        #60a5fa;

    text-decoration:
        none;

    margin:
        0 8px;

    font-size:
        13px;
}


.footer {

    text-align:
        center;

    margin-top:
        25px;

    color:
        #475569;

    font-size:
        12px;
}


@media (
    max-width: 700px
) {

    .grid {

        grid-template-columns:
            1fr;
    }

    .full {

        grid-column:
            auto;
    }

    .report-header {

        flex-direction:
            column;

        align-items:
            flex-start;
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

Analyze job opportunities,
identify skill gaps,
recommend projects,
and evaluate your GitHub profile.

</p>

</div>



<div class="card">


<form id="analysisForm">


<div class="grid">


<div class="field full">


<label>
Resume PDF
</label>


<div class="file-box">


<input
    id="resume"
    type="file"
    accept=".pdf,application/pdf"
    required
>


<div class="help">
Upload your text-based PDF resume.
</div>


</div>

</div>



<div class="field">


<label>
Target Placement Role
</label>


<input
    id="role"
    type="text"
    placeholder="AI/ML Engineer"
    required
>


</div>



<div class="field">


<label>
GitHub Username
</label>


<input
    id="github_id"
    type="text"
    placeholder="yogeeswar-09"
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

<a
    href="/agent/playground/"
    target="_blank"
>
LangServe Playground
</a>


<a
    href="/docs"
    target="_blank"
>
API Docs
</a>


<a
    href="/health"
    target="_blank"
>
Health
</a>

</div>


<div class="footer">

Placement-Ready AI Agent
•
LangChain
•
FastAPI
•
Render

</div>


</div>



<script>


const form =
    document.getElementById(
        "analysisForm"
    );


const button =
    document.getElementById(
        "analyzeButton"
    );


const status =
    document.getElementById(
        "status"
    );


const report =
    document.getElementById(
        "report"
    );


const reportBody =
    document.getElementById(
        "reportBody"
    );


const scoreNumber =
    document.getElementById(
        "scoreNumber"
    );


form.addEventListener(
    "submit",
    async function(event) {

        event.preventDefault();


        const resume =
            document.getElementById(
                "resume"
            ).files[0];


        const role =
            document.getElementById(
                "role"
            ).value.trim();


        const githubId =
            document.getElementById(
                "github_id"
            ).value.trim();


        if (!resume) {

            showError(
                "Please select your Resume PDF."
            );

            return;
        }


        if (
            !resume.name
                .toLowerCase()
                .endsWith(".pdf")
        ) {

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


        button.disabled =
            true;


        button.textContent =
            "Analyzing... Please wait";


        report.classList.remove(
            "show"
        );


        showStatus(
            "Running placement analysis..."
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
                        method:
                            "POST",

                        body:
                            formData
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
                data.report ||
                "No report returned.";


            scoreNumber.textContent =
                extractScore(
                    data.report
                );


            report.classList.add(
                "show"
            );


            showStatus(
                "Placement analysis completed successfully."
            );


            setTimeout(
                function() {

                    report.scrollIntoView({
                        behavior:
                            "smooth"
                    });

                },
                100
            );


        }
        catch (error) {

            showError(
                error.message ||
                "Something went wrong."
            );

        }
        finally {

            button.disabled =
                false;


            button.textContent =
                "Analyze Placement Readiness";

        }

    }
);


function showStatus(
    message
) {

    status.className =
        "status show";

    status.textContent =
        message;
}


function showError(
    message
) {

    status.className =
        "status show error";

    status.textContent =
        message;
}


function extractScore(
    text
) {

    if (!text) {
        return "—";
    }


    const patterns = [

        /Overall Placement Readiness\s*:\s*(\d{1,3})\s*\/\s*100/i,

        /Placement Readiness\s*:\s*(\d{1,3})\s*\/\s*100/i,

        /Readiness\s*:\s*(\d{1,3})\s*\/\s*100/i

    ];


    for (
        const pattern of patterns
    ) {

        const match =
            text.match(
                pattern
            );


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
# HOME
# ============================================================

@app.get(
    "/",
    response_class=HTMLResponse
)
def home():

    return HTML_PAGE


# ============================================================
# COMPLETE ANALYSIS API
# ============================================================

@app.post(
    "/api/analyze"
)
async def analyze(

    resume: UploadFile = File(...),

    role: str = Form(...),

    github_id: str = Form(...)

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
            detail="Only PDF files are supported."
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

        pdf_bytes = await resume.read()


        if len(pdf_bytes) > (
            10 * 1024 * 1024
        ):

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
                resume_text,
                role,
                github_id
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
                report

        }


    except HTTPException:

        raise


    except Exception as exc:

        logger.exception(
            "Analysis failed"
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


def playground_function(
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
        "Placement-Ready AI Agent is running. "
        "For the complete workflow including Resume PDF "
        "upload, use the main application at /. "
        f"Target role: {role}. "
        f"GitHub username: {github_id}."
    )


playground_runnable =
    RunnableLambda(
        playground_function
    ).with_types(
        input_type=
            PlaygroundInput,

        output_type=
            str
    )


add_routes(
    app,
    playground_runnable,
    path="/agent"
)


# ============================================================
# HEALTH
# ============================================================

@app.get(
    "/health"
)
def health():

    return {

        "status":
            "healthy",

        "agent":
            "Placement-Ready AI Agent",

        "framework":
            "LangChain",

        "model":
            "gemma-4-31b-it",

        "workflow": [

            "Resume PDF Parsing",

            "Job Opportunity Analysis",

            "Skill Gap Analysis",

            "Project Recommendation",

            "GitHub Evaluation",

            "Final Placement Synthesis"

        ]

    }


# ============================================================
# RUN
# ============================================================

if __name__ == "__main__":

    import uvicorn

    uvicorn.run(
        "langchain_job_ready_agent:app",
        host="0.0.0.0",
        port=int(
            os.environ.get(
                "PORT",
                8000
            )
        )
    )
