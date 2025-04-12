import sys
import os

# Add the project root to sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from backend.gpt_handler import evaluate_plan_with_gpt4o

class EvaluatingAgent:
    """Agent responsible for evaluating plans using GPT-4o-mini to determine their scientific validity and avoid duplicate queries."""

    def evaluate(self, plan, history=None):
        """
        Evaluates the plan using prior chat messages for better decision-making.
        If a GPT-based response repeats a recent query from the chat history, returns an error response for the duplication.

        Args:
            plan (dict): The plan dictionary specifying the tool and data.
            history (list, optional): List of past chat messages (as dicts). Defaults to None.

        Returns:
            dict: The original plan if valid, otherwise an error response.
        """
        history = history or []

        tool = plan.get("tool")
        data = plan.get("data")

        # Prevent repeating the same GPT-based response if just answered
        if tool == "gpt":
            last_messages = " ".join([msg["content"] for msg in history[-5:]])
            if data and isinstance(data, str) and data.strip() in last_messages:
                return {"error": "This query was already answered recently."}

        return plan  # Otherwise, proceed normally
