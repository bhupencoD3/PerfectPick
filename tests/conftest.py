import pytest
from unittest.mock import patch
import os

@pytest.fixture(autouse=True)
def mock_env_vars(monkeypatch):
    monkeypatch.setenv("ASTRA_DB_API_ENDPOINT", "https://dummy.endpoint")
    monkeypatch.setenv("ASTRA_DB_APPLICATION_TOKEN", "dummy-token")
    monkeypatch.setenv("ASTRA_DB_KEYSPACE", "dummy-keyspace")
    monkeypatch.setenv("GROQ_API_KEY", "dummy-groq-key")
