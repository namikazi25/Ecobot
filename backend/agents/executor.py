import sys
import os
from PIL import Image
import io

# Add the project root to sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from backend.tools.image_tools import process_image_with_llm
from backend.gpt_handler import process_with_gpt4o
from backend.tools.pdf_tools import process_pdf_with_llm
from backend.tools.wiki_tool import search_wikipedia, fetch_full_page
from core.llm_factory import get_llm
from core.prompts import EXECUTOR_PROMPT
from langchain.chains import LLMChain

class ExecutingAgent:
    """Executes the validated plan and retrieves results."""

    def __init__(self, llm=None):
        self.llm = llm or get_llm("google", "gemini-2.0-flash")
        self.chain = LLMChain(llm=self.llm, prompt=EXECUTOR_PROMPT)

    def execute(self, plan, history=None):
        """Executes the validated plan based on the tool selection"""
        history = history or []
        response = {"response": "", "sources": []}

        try:
            tool = plan.get("tool")
            data = plan.get("data")
            file_type = plan.get("file_type")

            if tool == "gpt":
               # Use LangChain LLMChain for execution
                context = "\n".join([f"{msg['role']}: {msg['content']}" for msg in history[-3:]])
                response["response"] = self.chain.run(tool=tool, data=data, context=context)
            elif tool == "image":
                # Now 'data' is a dict containing {"file_bytes": ..., "user_query": ...}
                file_bytes = data.get("file_bytes")
                user_query = data.get("user_query", "")  # default to empty if missing
                try:
                    img = Image.open(io.BytesIO(file_bytes))
                    img.verify()
                except Exception as e:
                    return {
                        "response": f"❌ Invalid/Corrupted Image: {str(e)}. Please upload a valid JPEG/PNG.",
                        "sources": []
                    }
                if file_type and file_bytes:
                    response["response"] = process_image_with_llm(file_bytes, file_type, user_query)
                else:
                    response["response"] = "❌ Missing file or file type for image processing"
            elif tool == "pdf":
                extracted_text = data.get("extracted_text", "")
                user_query = data.get("user_query", "Summarize this document.")
                response["response"] = process_pdf_with_llm(extracted_text, user_query)
            elif tool == "wiki":
                result = search_wikipedia(data)
                if "error" in result:
                    return self.fallback_response(data, result)
                response["response"] = self.format_wiki_summary(result)
                response["sources"] = [result["url"]]
            elif tool == "wiki_full":
                result = fetch_full_page(data)
                if "error" in result:
                    return self.fallback_response(data, result)
                response["response"] = self.format_wiki_summary(result)
                response["sources"] = [result["url"]]
            else:
                response["response"] = f"❌ Unknown tool: {tool}"
        except Exception as e:
            response["response"] = f"❌ Execution error: {str(e)}"
        return response

    @staticmethod
    def format_wiki_summary(result: dict) -> str:
        return f"""🌿 **{result['title']}**  
{result['summary']}  
📅 Last Updated: {result['last_updated'][:10]}  
🔗 [Read More]({result['url']})"""

    @staticmethod
    def format_full_wiki(result: dict) -> str:
        return f"""📖 **Full Article**: {result['url']}  
{result['content'][:2500]}...  
**Sections**: {', '.join(result['sections'][:5])}"""

    @staticmethod
    def fallback_response(query: str, error: dict) -> dict:
        return {
            "response": f"❌ Wikipedia Error: {error.get('error', 'Unknown error')}. GPT Response:\n{process_with_gpt4o(query)}",
            "sources": [],
            "history": []
        }