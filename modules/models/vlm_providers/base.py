from typing import Protocol


class VLMFatalError(RuntimeError):
    """A VLM error that retrying won't fix (bad credentials, unknown model, missing package)."""


class VLMProvider(Protocol):
    name: str

    def generate(self, prompt: str, images_b64: list[str]) -> str:
        """Send the prompt and base64 PNG views to the model; return its raw text answer."""
        ...
