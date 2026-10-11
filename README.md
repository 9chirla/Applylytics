# Applylytics

<p align="center">
  <strong>ATS‑optimise your CV with AI</strong><br/>
  Upload a PDF, paste a job description → match score, missing keywords, hiring-manager feedback, downloadable Word CV.
</p>

<p align="center">
  <img src="https://img.shields.io/badge/Python-Streamlit-FF4B4B?style=flat-square&logo=streamlit&logoColor=white" alt="Streamlit" />
  <img src="https://img.shields.io/badge/AI-Groq%20%2B%20Gemini-111?style=flat-square" alt="AI" />
  <img src="https://img.shields.io/badge/tests-pytest-0A9EDC?style=flat-square&logo=pytest&logoColor=white" alt="pytest" />
  <img src="https://img.shields.io/badge/deploy-Render-46E3B7?style=flat-square&logo=render&logoColor=white" alt="Render" />
  <img src="https://img.shields.io/badge/license-MIT-blue?style=flat-square" alt="MIT" />
</p>

## Features

- Keyword + semantic ATS scoring against a job description  
- AI coaching / hiring-manager style feedback  
- Optimised CV export as `.docx`  
- Optional email delivery via Redis + `rq` worker (Brevo)  

## Quick start

```bash
git clone https://github.com/9chirla/Applylytics.git
cd Applylytics
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env   # GROQ_API_KEY; optional GEMINI_API_KEY, BREVO_API_KEY
streamlit run app.py
```

```bash
pytest
```

## Background email worker

```bash
# terminal 1 — Redis
redis-server

# terminal 2 — worker
rq worker --with-scheduler
```

If the worker is down, the app can fall back to sending from the web process when possible.

## Deploy

`render.yaml` defines Redis + Streamlit web + `rq` worker (free plan). Set `GROQ_API_KEY`, `GEMINI_API_KEY`, `BREVO_API_KEY` on web and worker in the Render dashboard.

## License

[MIT](LICENSE) © 2026 9chirla
