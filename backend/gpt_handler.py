import os
from core.llm_factory import get_llm
from backend.agents.planner_langchain import PlanningChain
from backend.agents.evaluator_langchain import EvaluatingChain
from backend.agents.executor_langchain import ExecutorChain

# Initialize LangChain-based agents
planner = PlanningChain()
evaluator = EvaluatingChain()
executor = ExecutorChain()

def process_with_llm(query, context=None, history=None):
    """Process queries using the LangChain-based agent pipeline."""
    try:
        # Generate plan using PlanningChain
        plan = planner.plan(query, context=context)
        
        # Evaluate plan using EvaluatingChain
        evaluation = evaluator.evaluate(plan)
        if "error" in evaluation:
            return {"response": f"❌ Evaluation error: {evaluation['error']}", "sources": []}
            
        # Execute plan using ExecutorChain
        result = executor.execute(evaluation, history=history)
        return result
    except Exception as e:
        return {"response": f"❌ Error processing query: {str(e)}", "sources": []}

