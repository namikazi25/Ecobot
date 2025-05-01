from typing import Optional
from langchain_community.chat_models import ChatOpenAI
from langchain_google_genai import ChatGoogleGenerativeAI

PROVIDERS = {
    "openai": {
        "gpt-4o": "gpt-4o",
        "gpt-4o-mini": "gpt-4o-mini",
        "default": "gpt-4o"
    },
    "google": {
        "gemini-2.0-flash": "gemini-2.0-flash",
        "default": "gemini-2.0-flash"
    }
}

def get_llm(provider: str = "google", model_name: Optional[str] = None, temperature: float = 0.2, **kwargs):
    provider_key = provider.lower()
    model = model_name or PROVIDERS[provider_key]["default"]
    print(f"LLM → {provider}:{model}")
    if provider_key == "openai":
        return ChatOpenAI(model=model, temperature=temperature, **kwargs)
    elif provider_key == "google":
        return ChatGoogleGenerativeAI(model=model, temperature=temperature, **kwargs)
    else:
        raise ValueError(f"Unknown LLM provider: {provider}")