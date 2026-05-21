"""Application settings loaded from environment, .env, or Streamlit secrets."""

from __future__ import annotations

import os
from typing import Any

from pydantic_settings import BaseSettings, SettingsConfigDict

# Streamlit Cloud secrets.toml / dashboard keys → Settings fields
_SECRETS_FIELD_MAP: dict[str, str] = {
    "GROQ_API_KEY": "groq_api_key",
    "GEMINI_API_KEY": "gemini_api_key",
    "GEMINI_MODEL": "gemini_model",
    "GROQ_MODEL": "groq_model",
    "APP_PASSWORD": "app_password",
    "PORTFOLIO_URL": "portfolio_url",
    "DEBUG_MODE": "debug_mode",
}

_PLACEHOLDER_MARKERS = (
    "your_actual_key_here",
    "your-key",
    "your_key",
    "changeme",
    "replace_me",
    "gsk_xxx",
)

_API_KEY_HELP = (
    "Set GROQ_API_KEY in a local `.env` file, or in Streamlit Cloud under "
    "App settings → Secrets (TOML: GROQ_API_KEY = \"your-key\"). "
    "Do not reuse OPENAI_API_KEY — they go to different servers."
)


def sync_streamlit_secrets_to_environ() -> None:
    """Expose Streamlit secrets as env vars so pydantic-settings can read them."""
    try:
        import streamlit as st

        for secret_key in _SECRETS_FIELD_MAP:
            try:
                value = st.secrets[secret_key]
            except (KeyError, TypeError):
                continue
            if value is None:
                continue
            text = str(value).strip()
            if text:
                os.environ[secret_key] = text
    except FileNotFoundError:
        pass
    except Exception:
        pass


def _streamlit_secrets_overrides() -> dict[str, Any]:
    """Direct kwargs for Settings from st.secrets (call-time, not import-time)."""
    overrides: dict[str, Any] = {}
    try:
        import streamlit as st

        for secret_key, field_name in _SECRETS_FIELD_MAP.items():
            try:
                value = st.secrets[secret_key]
            except (KeyError, TypeError):
                continue
            if value is None:
                continue
            text = str(value).strip()
            if text:
                overrides[field_name] = value
    except FileNotFoundError:
        pass
    except Exception:
        pass
    return overrides


def _is_placeholder_key(key: str) -> bool:
    lowered = key.strip().lower()
    if not lowered:
        return True
    return any(marker in lowered for marker in _PLACEHOLDER_MARKERS)


class Settings(BaseSettings):
    """Runtime configuration for Applylytics."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    groq_api_key: str = ""
    gemini_api_key: str = ""
    gemini_model: str = "gemini-2.5-flash"
    groq_model: str = "llama-3.3-70b-versatile"
    app_password: str | None = None
    portfolio_url: str = "https://github.com"
    debug_mode: bool = False


def load_settings() -> Settings:
    """Build settings after syncing Streamlit secrets into the environment."""
    sync_streamlit_secrets_to_environ()
    return Settings(**_streamlit_secrets_overrides())


def bootstrap_settings() -> Settings:
    """Load settings once at app startup (safe for Streamlit Cloud secrets)."""
    global settings
    settings = load_settings()
    return settings


def resolve_groq_api_key() -> str:
    """Return Groq API key from secrets, env, or .env (refreshed each call)."""
    sync_streamlit_secrets_to_environ()
    current = load_settings()
    key = (current.groq_api_key or "").strip()
    if not key:
        key = (os.getenv("GROQ_API_KEY") or os.getenv("OPENAI_API_KEY") or "").strip()
    if key and not _is_placeholder_key(key):
        return key
    return ""


def api_key_help_message() -> str:
    return _API_KEY_HELP


# Populated by bootstrap_settings() in app.py before other applylytics imports.
settings: Settings = Settings()
