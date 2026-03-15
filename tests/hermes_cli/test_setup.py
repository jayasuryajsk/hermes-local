import json

from hermes_cli.auth import get_active_provider
from hermes_cli.config import load_config, save_config
from hermes_cli.setup import setup_model_provider


def _clear_provider_env(monkeypatch):
    for key in (
        "NOUS_API_KEY",
        "OPENROUTER_API_KEY",
        "OPENAI_BASE_URL",
        "OPENAI_API_KEY",
        "LLM_MODEL",
    ):
        monkeypatch.delenv(key, raising=False)


def test_local_setup_detects_and_selects_ollama_model(tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    _clear_provider_env(monkeypatch)

    config = load_config()

    # provider choice, then model choice from detected list
    prompt_choices = iter([0, 1])
    monkeypatch.setattr(
        "hermes_cli.setup.prompt_choice",
        lambda *args, **kwargs: next(prompt_choices),
    )
    monkeypatch.setattr("hermes_cli.setup._detect_ollama_models", lambda: ["llama3.2", "qwen2.5-coder:latest"])

    prompt_values = iter([
        "http://localhost:11434/v1",
    ])
    monkeypatch.setattr(
        "hermes_cli.setup.prompt",
        lambda *args, **kwargs: next(prompt_values),
    )

    setup_model_provider(config)
    save_config(config)

    reloaded = load_config()

    assert isinstance(reloaded["model"], dict)
    assert reloaded["model"]["provider"] == "custom"
    assert reloaded["model"]["base_url"] == "http://localhost:11434/v1"
    assert reloaded["model"]["default"] == "qwen2.5-coder:latest"


def test_local_setup_clears_active_oauth_provider(tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    _clear_provider_env(monkeypatch)

    auth_path = tmp_path / "auth.json"
    auth_path.write_text(json.dumps({"active_provider": "nous", "providers": {}}))

    config = load_config()

    prompt_choices = iter([0, 0])
    monkeypatch.setattr(
        "hermes_cli.setup.prompt_choice",
        lambda *args, **kwargs: next(prompt_choices),
    )
    monkeypatch.setattr("hermes_cli.setup._detect_ollama_models", lambda: ["qwen2.5-coder:latest"])

    prompt_values = iter([
        "http://localhost:11434/v1",
    ])
    monkeypatch.setattr(
        "hermes_cli.setup.prompt",
        lambda *args, **kwargs: next(prompt_values),
    )

    setup_model_provider(config)
    save_config(config)

    reloaded = load_config()

    assert get_active_provider() is None
    assert isinstance(reloaded["model"], dict)
    assert reloaded["model"]["provider"] == "custom"
    assert reloaded["model"]["base_url"] == "http://localhost:11434/v1"
    assert reloaded["model"]["default"] == "qwen2.5-coder:latest"


def test_detect_ollama_models_parses_cli_output(monkeypatch):
    class _Proc:
        returncode = 0
        stdout = """NAME ID SIZE MODIFIED
qwen2.5-coder:latest abc 4GB now
llama3.2 def 2GB now
"""

    monkeypatch.setattr("subprocess.run", lambda *args, **kwargs: _Proc())

    from hermes_cli.setup import _detect_ollama_models

    assert _detect_ollama_models() == ["qwen2.5-coder:latest", "llama3.2"]


def test_detect_ollama_models_returns_empty_when_unavailable(monkeypatch):
    def _boom(*args, **kwargs):
        raise FileNotFoundError("ollama not found")

    monkeypatch.setattr("subprocess.run", _boom)

    from hermes_cli.setup import _detect_ollama_models

    assert _detect_ollama_models() == []
