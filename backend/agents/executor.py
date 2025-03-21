import sys
import os
from PIL import Image
import io

# Add the project root to sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from backend.tools.image_tools import process_image_with_gpt4o
from backend.gpt_handler import process_with_gpt4o
from backend.tools.pdf_tools import process_pdf_with_gpt4o
from backend.tools.valyu_tool import ValyuClient

class ExecutingAgent:
    """Executes the validated plan and retrieves results."""
    
    def __init__(self):
        self.valyu_client = ValyuClient()

    def execute(self, plan, history=None):
        """Executes the validated plan based on the tool selection."""
        history = history or []
        response = {"response": "", "sources": []}

        try:
            tool = plan.get("tool")
            data = plan.get("data")
            file_type = plan.get("file_type")

            if tool == "gpt":
                response["response"] = process_with_gpt4o(data)
            
            elif tool == "image":
                # 'data' contains {"file_bytes": ..., "user_query": ...}
                file_bytes = data.get("file_bytes")
                user_query = data.get("user_query", "")

                try:
                    img = Image.open(io.BytesIO(file_bytes))
                    img.verify()
                except Exception as e:
                    return {
                        "response": f"❌ Invalid/Corrupted Image: {str(e)}. Please upload a valid JPEG/PNG.",
                        "sources": []
                    }

                if file_type and file_bytes:
                    response["response"] = process_image_with_gpt4o(file_bytes, file_type, user_query)
                else:
                    response["response"] = "❌ Missing file or file type for image processing"
            
            elif tool == "pdf":
                extracted_text = data.get("extracted_text", "")
                user_query = data.get("user_query", "Summarize this document.")
                response["response"] = process_pdf_with_gpt4o(extracted_text, user_query)
            
            elif tool == "valyu":
                # Use Valyu to retrieve relevant context
                result = self.valyu_client.context_search(data)
                if "error" in result or not result.get("results"):
                    response["response"] = (
                        f"❌ Valyu Error: {result.get('error', 'No results found')}. "
                        f"GPT fallback:\n{process_with_gpt4o(data)}"
                    )
                    response["sources"] = []
                else:
                    formatted_results = []
                    sources = []
                    for res in result.get("results", []):
                        formatted_results.append(
                            f"Title: {res.get('title')}\n"
                            f"Content: {res.get('content')}\n"
                            f"URL: {res.get('url')}\n"
                            f"Relevance: {res.get('relevance_score')}"
                        )
                        sources.append(res.get("url"))
                    response["response"] = "Using Valyu for context enrichment:\n\n" + "\n\n".join(formatted_results)
                    response["sources"] = sources
                    response["valyu_used"] = True
            
            else:
                response["response"] = "❌ Unknown tool selected"

        except Exception as e:
            response["response"] = f"⚠️ Execution Error: {str(e)}"

        history.append({"role": "assistant", "content": response["response"]})
        
        return {
            "response": response["response"],
            "sources": response.get("sources", []),
            "history": history
        }
