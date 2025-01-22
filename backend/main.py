from fastapi import FastAPI, UploadFile, File, Form
from agents.planner import PlanningAgent
from agents.evaluator import EvaluatingAgent
from agents.executor import ExecutingAgent
import json

app = FastAPI()

# Initialize agents
planner = PlanningAgent()
evaluator = EvaluatingAgent()
executor = ExecutingAgent()

MAX_RETRIES = 3
chat_history = []  # Persistent Chat History

@app.post("/query/")
async def process_query(
    query: str = Form(...),
    file: UploadFile = File(None)
):
    """Handles user input and maintains chat history."""
    
    global chat_history  # Ensure history is shared across requests
    
    file_content = None
    file_type = None

    if file:
        file_content = await file.read()
        file_type = file.content_type

    attempt = 0
    while attempt < MAX_RETRIES:
        # **Step 1: Plan**
        plan = planner.plan(query, file_content, file_type, chat_history)

        # **Step 2: Evaluate**
        evaluation = evaluator.evaluate(plan, chat_history)

        if "error" not in evaluation:
            # **Step 3: Execute**
            result = executor.execute(evaluation, chat_history)
            chat_history.append({"role": "assistant", "content": result["response"]})  # Store response
            return result  # Successfully executed

        # **If evaluation fails, retry planning**
        attempt += 1

    return {"error": "Failed to generate a valid plan after multiple attempts."}
