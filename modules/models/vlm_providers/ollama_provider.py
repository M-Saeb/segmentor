from ollama import Client


class OllamaProvider:
    name = "ollama"

    def __init__(self, model: str, host: str):
        self.model = model
        self.host = host

    def generate(self, prompt: str, images_b64: list[str]) -> str:
        response = Client(host=self.host).chat(
            model=self.model,
            messages=[
                {
                    "role": "user",
                    "content": prompt,
                    "images": images_b64,
                }
            ],
        )

        if isinstance(response, dict):
            return (response.get("message") or {}).get("content") or ""
        msg = getattr(response, "message", None)
        return getattr(msg, "content", "") if msg else str(response)
