# Applylytics

Streamlit app for ATS scoring, AI coaching, and CV optimisation against a job description.

## Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env   # add GROQ_API_KEY, optional GEMINI_API_KEY, BREVO_API_KEY
streamlit run app.py
```

## Background email worker (rq)

Optional email delivery uses Redis and an [rq](https://python-rq.org/) worker.

1. Start Redis locally (e.g. `redis-server`) or set `REDIS_URL` in `.env`.
2. In a second terminal, from the project root:

```bash
rq worker --with-scheduler
```

The Streamlit app enqueues `deliver_optimised_cv_email_task` when you optimise with an email address. If the worker is down, the app falls back to sending from the web process when possible.

## Tests

```bash
pytest
```
