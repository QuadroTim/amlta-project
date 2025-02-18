from typing import cast

from langchain_core.runnables import ConfigurableField
from langchain_ollama import ChatOllama

ollama = cast(
    ChatOllama,
    ChatOllama(model="llama3.2", num_ctx=512).configurable_fields(
        model=ConfigurableField(
            id="ollama_model",
            description="Ollama model to use",
        ),
        base_url=ConfigurableField(
            id="ollama_base_url",
            description="Ollama base URL",
        ),
    ),
)
