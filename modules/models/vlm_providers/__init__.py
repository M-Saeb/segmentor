"""
VLM backends for object identification. ``vlm.provider`` in config.yml selects one;
prompt, retries and response parsing stay in ``modules.models.vlm_tools``.
"""

from modules.models.vlm_providers.base import VLMFatalError, VLMProvider
from modules.utils import constants
from modules.utils.config import SegmentorConfig

__all__ = ["VLMFatalError", "VLMProvider", "get_vlm_provider", "reset_vlm_provider"]

_provider: VLMProvider | None = None
_provider_config = None  # the SegmentorConfig instance _provider was built from


def _build_provider(vlm_cfg: dict) -> VLMProvider:
    provider_name = str(vlm_cfg.get("provider", "ollama")).strip().lower()

    if provider_name == "ollama":
        from modules.models.vlm_providers.ollama_provider import OllamaProvider

        # Older configs keep model/host directly under `vlm`.
        ollama_cfg = vlm_cfg.get("ollama") or {}
        return OllamaProvider(
            model=ollama_cfg.get("model", vlm_cfg.get("model", "gemma3:4b")),
            host=ollama_cfg.get("host", vlm_cfg.get("host", "http://localhost:11434")),
        )

    if provider_name == "claude":
        from modules.models.vlm_providers.claude_provider import ClaudeProvider

        claude_cfg = vlm_cfg.get("claude") or {}
        return ClaudeProvider(
            model=claude_cfg.get("model", "claude-opus-5-5"),
            effort=claude_cfg.get("effort", "low"),
            max_tokens=claude_cfg.get("max_tokens", 4096),
            max_retries=claude_cfg.get("max_retries", 2),
            timeout_seconds=claude_cfg.get("timeout_seconds", 120),
            refusal_fallback=claude_cfg.get("refusal_fallback", True),
        )

    raise ValueError(f"vlm.provider must be one of {constants.VLM_PROVIDERS}, got {provider_name!r}")


def get_vlm_provider() -> VLMProvider:
    """Build the configured provider once and reuse it (keeps the HTTP client's connections alive)."""
    global _provider, _provider_config
    cfg = SegmentorConfig.get_instance()
    if _provider is None or _provider_config is not cfg:
        _provider = _build_provider(getattr(cfg, "vlm", {}) or {})
        _provider_config = cfg
    return _provider


def reset_vlm_provider() -> None:
    global _provider, _provider_config
    _provider = None
    _provider_config = None
