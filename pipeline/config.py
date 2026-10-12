"""Shared settings. Override with environment variables (PowerShell: $env:NAME = "value")."""

import os
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"
DB_PATH = Path(os.environ.get("TRIAGE_DB", DATA_DIR / "triage.sqlite"))

# Any OpenAI-compatible server works; LM Studio's default local address is used here.
LLM_BASE_URL = os.environ.get("LLM_BASE_URL", "http://localhost:1234/v1")
LLM_MODEL = os.environ.get("LLM_MODEL", "gemma-4-12b-it")

# Fictional company the synthetic tickets are written for.
COMPANY = "Nimbus Notes"
PRODUCT_BLURB = (
    "Nimbus Notes is a fictional note-taking and team collaboration app with web, "
    "desktop and mobile clients. Plans: Free, Pro (monthly or annual) and Team. "
    "Features include shared workspaces, offline sync, PDF/Markdown export, "
    "calendar integration and an AI summary button."
)
