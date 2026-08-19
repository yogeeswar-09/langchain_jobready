# 🎯 Placement-Ready AI Agent

> An AI-powered career assistant that helps students prepare for campus placements through job analysis, skill-gap identification, project recommendations, resume evaluation, and GitHub profile analysis.

[![LangChain](https://img.shields.io/badge/LangChain-Framework-green)](https://www.langchain.com/)
[![Python](https://img.shields.io/badge/Python-3.x-blue)](https://www.python.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-API-009688)](https://fastapi.tiangolo.com/)
[![Gemini](https://img.shields.io/badge/Google-Gemini-orange)](https://ai.google.dev/)
[![Render](https://img.shields.io/badge/Deployed-Render-purple)](https://render.com/)

---

## 🚀 Overview

The **Placement-Ready AI Agent** is a LangChain-based AI application designed to assist students throughout their campus-placement preparation journey.

Instead of providing generic career advice, the agent analyzes a student's profile and generates actionable recommendations based on their skills, projects, target roles, resume, and GitHub presence.

### The agent focuses on:

- 📊 Job opportunity analysis
- 🧩 Skill-gap identification
- 🛠️ Project recommendations
- 📄 Resume/profile evaluation
- 🐙 GitHub profile analysis
- 🎯 Placement-readiness assessment
- 💡 Personalized improvement recommendations

---

## ✨ Key Features

### 📊 Job Analysis

Analyzes a target job opportunity and identifies:

- Required skills
- Preferred technologies
- Role expectations
- Relevant qualifications
- Important preparation areas

### 🧩 Skill Gap Analysis

Compares the student's existing skill set against the requirements of the target role.

The agent highlights:

- Existing strengths
- Missing skills
- Priority areas
- Recommended learning path

### 🛠️ Project Recommendations

Suggests projects that can strengthen the candidate's portfolio based on:

- Target job role
- Current skills
- Missing technologies
- Placement relevance

### 📄 Resume Evaluation

Evaluates the student's resume/profile and provides actionable feedback on:

- Technical skills
- Project presentation
- Role alignment
- Resume strengths
- Areas for improvement

### 🐙 GitHub Analysis

Evaluates GitHub-related information to identify:

- Project quality
- Technical relevance
- Portfolio strength
- Areas that could improve recruiter perception

---

## 🧠 How It Works

```text
             Student Profile
                    │
                    ▼
             Target Job
                    │
                    ▼
          ┌───────────────────┐
          │   LangChain Agent │
          └─────────┬─────────┘
                    │
       ┌────────────┼────────────┐
       ▼            ▼            ▼
 Job Analysis   Skill Gap    Profile Review
       │            │            │
       └────────────┼────────────┘
                    ▼
          Project Recommendations
                    │
                    ▼
        Placement Readiness Report
