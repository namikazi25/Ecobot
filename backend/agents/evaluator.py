import sys
import os

# Add the project root to sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from backend.gpt_handler import process_with_llm

class EvaluatingAgent:
    """Evaluates the plan using GPT-4o-mini to determine its validity."""

    def evaluate(self, plan, history=None):
        history = history or []

        tool = plan.get("tool")
        data = plan.get("data")

        # If it's a GPT-based response, check for repeated queries in history
        if tool == "gpt":
            last_messages = " ".join([msg["content"] for msg in history[-5:]])
            if data in last_messages:
                return {"error": "This query was already answered recently."}

        # Validate tool type
        valid_tools = ["gpt", "image", "pdf", "wiki", "wiki_full"]
        if tool not in valid_tools:
            return {"error": f"❌ Unknown tool selected: {tool}"}
            
        if tool == "gpt":
            return process_with_llm(data)
        return plan  # Otherwise, proceed with the validated plan
