"""
Unit tests for the VLM provider layer (``modules.models.vlm_providers``) and its use in
``identify_object_from_views``. No network: ``anthropic.Anthropic`` and ``ollama.Client`` are mocked.
"""

from __future__ import annotations

import base64
import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import anthropic
import httpx2
import pytest
import yaml

from modules.models import vlm_tools
from modules.models.vlm_providers import VLMFatalError, get_vlm_provider, reset_vlm_provider
from modules.models.vlm_providers.claude_provider import RESPONSE_SCHEMA, ClaudeProvider
from modules.models.vlm_providers.ollama_provider import OllamaProvider
from modules.utils import constants as C
from modules.utils.config import SegmentorConfig

MINIMAL_PNG_BYTES = base64.b64decode(
    "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mP8z8BQDwAEhQGAhKmMIQAAAABJRU5ErkJggg=="
)
_VIEWS = 3


@pytest.fixture(autouse=True)
def _reset_singletons():
    SegmentorConfig._instance = None
    SegmentorConfig._config_path = None
    reset_vlm_provider()
    yield
    SegmentorConfig._instance = None
    SegmentorConfig._config_path = None
    reset_vlm_provider()


@pytest.fixture(autouse=True)
def _no_retry_sleep():
    with patch("modules.models.vlm_tools.time.sleep"):
        yield


def _init_config(tmp_path: Path, vlm: dict) -> None:
    cfg = tmp_path / "config.yml"
    cfg.write_text(yaml.safe_dump({"input_file": "x.las", "vlm": vlm}), encoding="utf-8")
    SegmentorConfig.initialize(str(cfg))


def _views(tmp_path: Path) -> list[str]:
    paths = []
    for i in range(_VIEWS):
        p = tmp_path / f"{i + 1:02d}.png"
        p.write_bytes(MINIMAL_PNG_BYTES)
        paths.append(str(p))
    return paths


def _claude_message(text: str, stop_reason: str = "end_turn", category: str | None = None):
    return SimpleNamespace(
        model="claude-opus-5-5",
        stop_reason=stop_reason,
        stop_details=SimpleNamespace(category=category) if category else None,
        content=[SimpleNamespace(type="text", text=text)] if text else [],
        usage=SimpleNamespace(input_tokens=10, output_tokens=5),
        _request_id="req_test",
    )


def _api_error(cls, status: int):
    request = httpx2.Request("POST", "https://api.anthropic.com/v1/messages")
    return cls("boom", response=httpx2.Response(status, request=request), body=None)


_ANSWER = json.dumps({"class": "VEHICLE", "label": "Van", "description": "a white van"})


# --- provider selection -------------------------------------------------------------------------


def test_default_provider_is_ollama_with_legacy_flat_keys(tmp_path: Path):
    _init_config(tmp_path, {"views_count": 4, "model": "legacy:1b", "host": "http://h:1"})
    provider = get_vlm_provider()
    assert isinstance(provider, OllamaProvider)
    assert (provider.model, provider.host) == ("legacy:1b", "http://h:1")


def test_nested_ollama_keys_win_over_flat_ones(tmp_path: Path):
    _init_config(tmp_path, {"provider": "ollama", "model": "flat", "ollama": {"model": "nested", "host": "http://n"}})
    provider = get_vlm_provider()
    assert (provider.model, provider.host) == ("nested", "http://n")


def test_claude_provider_selected_and_cached(tmp_path: Path):
    _init_config(tmp_path, {"provider": "claude", "claude": {"model": "claude-sonnet-5-5", "effort": "medium"}})
    with patch("anthropic.Anthropic") as client_cls:
        provider = get_vlm_provider()
        assert get_vlm_provider() is provider
    assert isinstance(provider, ClaudeProvider)
    assert (provider.model, provider.effort) == ("claude-sonnet-5-5", "medium")
    client_cls.assert_called_once_with(max_retries=2, timeout=120.0)


def test_unknown_provider_rejected_by_config(tmp_path: Path):
    with pytest.raises(ValueError, match="vlm.provider"):
        _init_config(tmp_path, {"provider": "openai"})


def test_bad_claude_effort_rejected_by_config(tmp_path: Path):
    with pytest.raises(ValueError, match="vlm.claude.effort"):
        _init_config(tmp_path, {"provider": "claude", "claude": {"effort": "extreme"}})


# --- Claude request / response ------------------------------------------------------------------


def _claude_provider(refusal_fallback: bool = True) -> tuple[ClaudeProvider, MagicMock]:
    with patch("anthropic.Anthropic") as client_cls:
        provider = ClaudeProvider(
            model="claude-opus-5-5",
            effort="low",
            max_tokens=4096,
            max_retries=2,
            timeout_seconds=120,
            refusal_fallback=refusal_fallback,
        )
    return provider, client_cls.return_value


def test_claude_request_shape_with_fallback():
    provider, client = _claude_provider(refusal_fallback=True)
    client.beta.messages.create.return_value = _claude_message(_ANSWER)

    assert provider.generate("PROMPT", ["aaa", "bbb", "ccc"]) == _ANSWER

    client.messages.create.assert_not_called()
    kwargs = client.beta.messages.create.call_args.kwargs
    assert kwargs["betas"] == ["server-side-fallback-2026-07-01"]
    assert kwargs["fallbacks"] == "default"
    assert kwargs["model"] == "claude-opus-5-5"
    assert kwargs["max_tokens"] == 4096
    assert kwargs["output_config"]["effort"] == "low"
    assert kwargs["output_config"]["format"] == {"type": "json_schema", "schema": RESPONSE_SCHEMA}
    assert RESPONSE_SCHEMA["properties"]["class"]["enum"] == list(C.VLM_CLASS_TO_CLASSIFICATION)

    content = kwargs["messages"][0]["content"]
    assert [b["type"] for b in content] == ["image", "image", "image", "text"]
    assert content[0]["source"] == {"type": "base64", "media_type": "image/png", "data": "aaa"}
    assert content[-1]["text"] == "PROMPT"


def test_claude_without_fallback_uses_plain_endpoint():
    provider, client = _claude_provider(refusal_fallback=False)
    client.messages.create.return_value = _claude_message(_ANSWER)
    provider.generate("PROMPT", ["aaa"])
    client.beta.messages.create.assert_not_called()
    assert "fallbacks" not in client.messages.create.call_args.kwargs


@pytest.mark.parametrize(
    "error",
    [
        _api_error(anthropic.AuthenticationError, 401),
        _api_error(anthropic.NotFoundError, 404),
        _api_error(anthropic.BadRequestError, 400),
    ],
)
def test_claude_non_retryable_errors_are_fatal(error):
    provider, client = _claude_provider()
    client.beta.messages.create.side_effect = error
    with pytest.raises(VLMFatalError):
        provider.generate("PROMPT", ["aaa"])


def test_claude_rate_limit_is_not_fatal():
    provider, client = _claude_provider()
    client.beta.messages.create.side_effect = _api_error(anthropic.RateLimitError, 429)
    with pytest.raises(anthropic.RateLimitError):
        provider.generate("PROMPT", ["aaa"])


@pytest.mark.parametrize("stop_reason", ["refusal", "max_tokens"])
def test_claude_refusal_and_truncation_raise_retryable(stop_reason):
    provider, client = _claude_provider()
    client.beta.messages.create.return_value = _claude_message("", stop_reason=stop_reason, category="cyber")
    with pytest.raises(RuntimeError) as exc_info:
        provider.generate("PROMPT", ["aaa"])
    assert not isinstance(exc_info.value, VLMFatalError)


# --- identify_object_from_views through the provider --------------------------------------------


def test_identify_with_claude_builds_record(tmp_path: Path):
    _init_config(tmp_path, {"provider": "claude"})
    with patch("anthropic.Anthropic") as client_cls:
        client_cls.return_value.beta.messages.create.return_value = _claude_message(_ANSWER)
        record = vlm_tools.identify_object_from_views(_views(tmp_path), cluster_id=7, chunk_index=0)

    content = client_cls.return_value.beta.messages.create.call_args.kwargs["messages"][0]["content"]
    assert sum(b["type"] == "image" for b in content) == _VIEWS
    assert record["cluster_id"] == 7
    assert record["vlm_class"] == "VEHICLE"
    assert record["name"] == "van"
    assert record["classification"] == C.PCB_CLASSIFICATION_VEHICLE


def test_identify_retries_refusals_then_unknown(tmp_path: Path):
    _init_config(tmp_path, {"provider": "claude"})
    with patch("anthropic.Anthropic") as client_cls:
        create = client_cls.return_value.beta.messages.create
        create.return_value = _claude_message("", stop_reason="refusal")
        record = vlm_tools.identify_object_from_views(_views(tmp_path), cluster_id=1, chunk_index=0)
    assert create.call_count == 3
    assert record["vlm_class"] == C.VLM_CLASS_UNKNOWN
    assert record["classification"] == C.PCB_CLASSIFICATION_UNKNOWN


def test_identify_does_not_retry_fatal_errors(tmp_path: Path):
    _init_config(tmp_path, {"provider": "claude"})
    with patch("anthropic.Anthropic") as client_cls:
        create = client_cls.return_value.beta.messages.create
        create.side_effect = _api_error(anthropic.AuthenticationError, 401)
        with pytest.raises(VLMFatalError):
            vlm_tools.identify_object_from_views(_views(tmp_path), cluster_id=1, chunk_index=0)
    assert create.call_count == 1


def test_identify_with_ollama_unchanged(tmp_path: Path):
    _init_config(tmp_path, {"provider": "ollama", "ollama": {"model": "gemma3:4b", "host": "http://localhost:11434"}})
    with patch("modules.models.vlm_providers.ollama_provider.Client") as client_cls:
        client_cls.return_value.chat.return_value = {"message": {"content": "Sure: " + _ANSWER}}
        record = vlm_tools.identify_object_from_views(_views(tmp_path), cluster_id=2, chunk_index=0)

    client_cls.assert_called_with(host="http://localhost:11434")
    chat_kwargs = client_cls.return_value.chat.call_args.kwargs
    assert chat_kwargs["model"] == "gemma3:4b"
    assert len(chat_kwargs["messages"][0]["images"]) == _VIEWS
    assert record["vlm_class"] == "VEHICLE"
    assert record["name"] == "van"


def test_claude_missing_credentials_is_fatal(tmp_path: Path, monkeypatch):
    for var in ("ANTHROPIC_API_KEY", "ANTHROPIC_AUTH_TOKEN", "ANTHROPIC_PROFILE"):
        monkeypatch.delenv(var, raising=False)
    monkeypatch.setenv("ANTHROPIC_CONFIG_DIR", str(tmp_path / "no-profile"))
    _init_config(tmp_path, {"provider": "claude"})
    with pytest.raises(VLMFatalError, match="ANTHROPIC_API_KEY"):
        get_vlm_provider()
