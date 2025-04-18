from typing import Optional
from langchain.chat_models.base import BaseChatModel
from langchain.chat_models import ChatOpenAI
from langchain_google_genai import ChatGoogleGenerativeAI

PROVIDER_ALIASES = {
    "openai": "openai",
    "google": "google",
    "gemini": "google"
}

MODEL_ALIASES = {
    "openai": {
        "gpt-4o": "gpt-4o",
        "gpt-4o-mini": "gpt-4o-mini",
        "default": "gpt-4o"
    },
    "google": {
        "gemini-1.5-flash": "gemini-1.5-flash",
        "gemini-2.0-flash": "gemini-2.0-flash",
        "default": "gemini-2.0-flash"
    }
}

def get_llm(provider: str, model_name: Optional[str] = None, **kwargs) -> BaseChatModel:
    """
    Returns a LangChain-compatible LLM instance for the given provider and model.
    Supported providers: 'openai', 'google' (Gemini)
    """
    provider_key = PROVIDER_ALIASES.get(provider.lower(), provider.lower())
    if provider_key not in MODEL_ALIASES:
        raise ValueError(f"Unsupported provider: {provider}")
    model_map = MODEL_ALIASES[provider_key]
    model = model_map.get(model_name, model_map["default"]) if model_name else model_map["default"]

    if provider_key == "openai":
        return ChatOpenAI(model=model, **kwargs)
    elif provider_key == "google":
        return ChatGoogleGenerativeAI(model=model, **kwargs)
    else:
        raise ValueError(f"Provider '{provider}' is not supported yet.")