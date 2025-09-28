import os
import importlib
import pytest

def test_config_variables(monkeypatch):
    # Set dummy env vars
    monkeypatch.setenv("ASTRA_DB_API_ENDPOINT", "https://dummy.endpoint")
    monkeypatch.setenv("ASTRA_DB_APPLICATION_TOKEN", "dummy-token")
    monkeypatch.setenv("ASTRA_DB_KEYSPACE", "dummy-keyspace")
    monkeypatch.setenv("GROQ_API_KEY", "dummy-groq-key")

    # Reload config module so Config class picks up env vars
    from flipkart import config
    importlib.reload(config)
    Config = config.Config

    assert Config.EMBEDDING_MODEL == "BAAI/bge-base-en-v1.5"
    assert Config.LLM_MODEL == "llama-3.1-8b-instant"
    assert Config.ASTRA_DB_API_ENDPOINT == "https://dummy.endpoint"
    assert Config.ASTRA_DB_APPLICATION_TOKEN == "dummy-token"
    assert Config.ASTRA_DB_KEYSPACE == "dummy-keyspace"
    assert Config.GROQ_API_KEY == "dummy-groq-key"
