"""Application settings loaded from environment, .env, or Streamlit secrets."""

from __future__ import annotations

from typing import Any

from pydantic_settings import BaseSettings, SettingsConfigDict

# Streamlit Cloud secrets.toml keys → Settings fields
_SECRETS_FIELD_MAP: dict[str, str] = {
    "GROQ_API_KEY": "groq_api_key",
    "GEMINI_API_KEY": "gemini_api_key",
    "GEMINI_MODEL": "gemini_model",
    "GROQ_MODEL": "groq_model",
    "APP_PASSWORD": "app_password",
    "PORTFOLIO_URL": "portfolio_url",
    "DEBUG_MODE": "debug_mode",
}


def _streamlit_secrets_overrides() -> dict[str, Any]:
    """Read Streamlit Cloud secrets when env vars / .env are absent."""
    overrides: dict[str, Any] = {}
    try:
        import streamlit as st

        secrets = st.secrets
        for secret_key, field_name in _SECRETS_FIELD_MAP.items():
            if secret_key in secrets:
                overrides[field_name] = secrets[secret_key]
    except Exception:
        pass
    return overrides


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


settings = Settings(**_streamlit_secrets_overrides())
