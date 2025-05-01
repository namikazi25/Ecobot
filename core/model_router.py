class ModelRouter:
    """Stub for ModelRouter. Implement routing logic as needed."""
    def __init__(self):
        pass

    def route(self, input_data=None):
        # Routing logic: always return a default LLM instance for now
        return get_llm()
from core.llm_factory import get_llm