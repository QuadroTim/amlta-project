from langchain_ollama import ChatOllama


def get_ollama(model: str = "llama3.2", base_url: str | None = None) -> ChatOllama:
    return ChatOllama(model=model, base_url=base_url, num_ctx=2048)
