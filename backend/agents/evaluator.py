import sys
import os

# Add the project root to sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from core.llm_factory import get_llm
from core.prompts import EVALUATOR_PROMPT
from langchain.chains import LLMChain

class EvaluatingAgent:
    """Evaluates the plan using an LLM to determine its validity."""

    def __init__(self, llm=None):
        self.llm = llm or get_llm("google", "gemini-2.0-flash")
        self.chain = LLMChain(llm=self.llm, prompt=EVALUATOR_PROMPT)

    def evaluate(self, plan, history=None):
        """Evaluates the plan using prior messages for better decision-making."""
        history = history or []
        tool = plan.get("tool")
        data = plan.get("data")
        # If it's a GPT-based response, check for repeated queries in history
        if tool == "gpt":
            last_messages = " ".join([msg["content"] for msg in history[-5:]])
            if data in last_messages:
                return {"error": "This query was already answered recently."}
        # Use LangChain LLMChain for evaluation
        plan_str = str(plan)
        history_str = "\n".join([f"{msg['role']}: {msg['content']}" for msg in history[-3:]])
        result = self.chain.run(plan=plan_str, history=history_str)
        result_obj = self.chain.invoke({"plan": plan_str, "history": history_str})
        result = result_obj["text"] if isinstance(result_obj, dict) and "text" in result_obj else str(result_obj)
        return result
