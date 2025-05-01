from fastapi import FastAPI, UploadFile, File, Form
from typing import List
from agents.planner_langchain import PlanningChain
from agents.evaluator_langchain import EvaluatingChain
from agents.executor_langchain import ExecutorChain
import json

app = FastAPI()

# Initialize LangChain-based agents
planner = PlanningChain()
evaluator = EvaluatingChain()
executor = ExecutorChain()

MAX_RETRIES = 3
chat_history = []  # Persistent Chat History

@app.post("/query/")
async def process_query(
    query: str = Form(...),
    pdf_context: str = Form(None),
    files: List[UploadFile] = File(None)
    
):
    global chat_history
    file_contents = []  # We'll gather (bytes, content_type) for each file
    # print("[DEBUG] Received request. files param = ", files)
    # Convert each UploadFile into bytes for the pipeline
    if files:
        for f in files:
            content = await f.read()
            file_contents.append((content, f.content_type))
    
    # Use PDF context if available
    if pdf_context and not files:
        query = f"{query}\nPDF Context: {pdf_context}"

    attempt = 0
    while attempt < MAX_RETRIES:
        plan = planner.plan(query, file_contents, chat_history)
        evaluation = evaluator.evaluate(plan, chat_history)

        if "error" not in evaluation:
            result = executor.execute(evaluation, chat_history)
            if files and any("pdf" in f.content_type for f in files):
                # If there's at least one PDF, set pdf_context
                result["pdf_context"] = evaluation.get("extracted_text", "")
            if not result.get("response"):
                result["response"] = "⚠️ No response generated"
            if "sources" not in result:
                result["sources"] = []
            chat_history.append({"role": "assistant", "content": result["response"]})
            return result
        attempt += 1

    return {"response": "⚠️ Failed to generate a valid plan after multiple attempts.", "sources": []}