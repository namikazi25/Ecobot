from typing import Optional, List, Dict, Any
from langchain.chains import LLMChain
from langchain.prompts import ChatPromptTemplate, SystemMessagePromptTemplate, HumanMessagePromptTemplate
from langchain.schema import BaseMessage
from core.llm_factory import get_llm

class EvaluatingChain:
    """LangChain-based evaluator that determines plan validity."""
    
    def __init__(self):
        self.llm = get_llm()
        self.prompt = ChatPromptTemplate(
            messages=[
                SystemMessagePromptTemplate.from_template(
                    """You are an evaluation agent responsible for validating action plans.
                    Your role is to ensure that the proposed tool and data are appropriate for the task.
                    Available tools: gpt, image, pdf, wiki, wiki_full
                    """
                ),
                HumanMessagePromptTemplate.from_template(
                    """Please evaluate this plan:
                    Tool: {tool}
                    Data: {data}
                    Recent History: {history}
                    
                    Respond with either:
                    1. The original plan if valid
                    2. An error message if invalid
                    """
                )
            ]
        )
        self.chain = LLMChain(llm=self.llm, prompt=self.prompt)
    
    def _format_history(self, history: List[Dict[str, str]]) -> str:
        if not history:
            return "No recent interactions"
        return "\n".join([f"{msg['role']}: {msg['content']}" for msg in history[-5:]])
    
    def evaluate(self, plan: Dict[str, Any], history: Optional[List[Dict[str, str]]] = None) -> Dict[str, Any]:
        """Evaluates the plan using LangChain components."""
        history = history or []
        
        tool = plan.get("tool")
        data = plan.get("data")
        
        # Basic validation
        if not tool or not data:
            return {"error": "❌ Invalid plan structure: missing tool or data"}
            
        # Check for repeated queries in history
        if tool == "gpt":
            recent_content = self._format_history(history)
            if data in recent_content:
                return {"error": "This query was already answered recently."}
        
        # Validate tool type
        valid_tools = ["gpt", "image", "pdf", "wiki", "wiki_full"]
        if tool not in valid_tools:
            return {"error": f"❌ Unknown tool selected: {tool}"}
        
        # Use LangChain for deeper evaluation
        try:
            result = self.chain.invoke({
                "tool": tool,
                "data": data,
                "history": self._format_history(history)
            })
            
            # If the chain detected any issues
            if "error" in result:
                return {"error": result["error"]}
                
            return plan
            
        except Exception as e:
            return {"error": f"❌ Evaluation error: {str(e)}"}