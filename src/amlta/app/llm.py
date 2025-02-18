from langchain_ollama import ChatOllama

_DEFAULT_MODEL = "llama3.2"


def get_model(model: str | None = None, base_url: str | None = None) -> ChatOllama:
    model = model or _DEFAULT_MODEL

    return ChatOllama(model=model, base_url=base_url)
