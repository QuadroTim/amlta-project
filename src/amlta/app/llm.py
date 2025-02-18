from langchain_ollama import ChatOllama

_DEFAULT_MODEL = "llama3.2"
_DEFAULT_BASE_URL = "127.0.0.1:11434"


def get_model(model: str | None = None, base_url: str | None = None) -> ChatOllama:
    model = model or _DEFAULT_MODEL
    base_url = base_url or _DEFAULT_BASE_URL

    return ChatOllama(model=model, base_url=base_url)
