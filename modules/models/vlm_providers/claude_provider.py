from modules.models.vlm_providers.base import VLMFatalError
from modules.utils import constants
from modules.utils.log_utils import get_dual_logger

log = get_dual_logger(__name__)

_FALLBACK_BETA = "server-side-fallback-2026-07-01"

# Keys match the JSON the prompt asks for, so vlm_tools parses Claude and Ollama answers the same way.
RESPONSE_SCHEMA = {
    "type": "object",
    "properties": {
        "class": {"type": "string", "enum": list(constants.VLM_CLASS_TO_CLASSIFICATION)},
        "label": {"type": "string"},
        "description": {"type": "string"},
    },
    "required": ["class", "label", "description"],
    "additionalProperties": False,
}


class ClaudeProvider:
    name = "claude"

    def __init__(
        self,
        model: str,
        effort: str,
        max_tokens: int,
        max_retries: int,
        timeout_seconds: float,
        refusal_fallback: bool,
    ):
        try:
            import anthropic
        except ImportError as ex:
            raise VLMFatalError(
                "vlm.provider is 'claude' but the 'anthropic' package is not installed (pip install anthropic)."
            ) from ex

        self._anthropic = anthropic
        self.model = model
        self.effort = effort
        self.max_tokens = int(max_tokens)
        self.refusal_fallback = bool(refusal_fallback)
        # Credentials come from ANTHROPIC_API_KEY or an `ant auth login` profile, never from config.yml.
        try:
            self._client = anthropic.Anthropic(max_retries=int(max_retries), timeout=float(timeout_seconds))
        except anthropic.CredentialsError as ex:
            raise VLMFatalError(
                "vlm.provider is 'claude' but no Anthropic credentials were found. "
                "Set the ANTHROPIC_API_KEY environment variable (or run `ant auth login`)."
            ) from ex

    def _build_content(self, prompt: str, images_b64: list[str]) -> list[dict]:
        content: list[dict] = [
            {
                "type": "image",
                "source": {"type": "base64", "media_type": "image/png", "data": data},
            }
            for data in images_b64
        ]
        content.append({"type": "text", "text": prompt})
        return content

    def generate(self, prompt: str, images_b64: list[str]) -> str:
        anthropic = self._anthropic
        params = {
            "model": self.model,
            "max_tokens": self.max_tokens,
            "output_config": {
                "effort": self.effort,
                "format": {"type": "json_schema", "schema": RESPONSE_SCHEMA},
            },
            "messages": [{"role": "user", "content": self._build_content(prompt, images_b64)}],
        }

        try:
            if self.refusal_fallback:
                response = self._client.beta.messages.create(
                    betas=[_FALLBACK_BETA],
                    fallbacks="default",
                    **params,
                )
            else:
                response = self._client.messages.create(**params)
        except (
            anthropic.CredentialsError,
            anthropic.AuthenticationError,
            anthropic.PermissionDeniedError,
            anthropic.NotFoundError,
            anthropic.BadRequestError,
        ) as ex:
            raise VLMFatalError(f"Claude request rejected ({type(ex).__name__}): {ex}") from ex
        # RateLimitError / 5xx / connection errors propagate so the caller's retry loop handles them.

        log.debug(
            "Claude %s: input_tokens=%s output_tokens=%s request_id=%s",
            response.model,
            response.usage.input_tokens,
            response.usage.output_tokens,
            response._request_id,
        )

        if response.stop_reason == "refusal":
            category = response.stop_details.category if response.stop_details else None
            raise RuntimeError(f"Claude declined the request (category={category})")
        if response.stop_reason == "max_tokens":
            raise RuntimeError(
                f"Claude hit max_tokens={self.max_tokens} before answering; raise vlm.claude.max_tokens."
            )

        text = next((block.text for block in response.content if block.type == "text"), "")
        if not text:
            raise RuntimeError(f"Claude returned no text (stop_reason={response.stop_reason})")
        return text
