import sys
import os

# Add the project root to sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from backend.tools.image_tools import process_image_with_gpt4o
from backend.gpt_handler import process_with_gpt4o
from backend.tools.pdf_tools import process_pdf_with_gpt4o
from backend.tools.wiki_tool import search_wikipedia, fetch_full_page

class ExecutingAgent:
    """Executes the validated plan and retrieves results."""

    def execute(self, plan, history=None):
        """Executes the validated plan based on the tool selection"""
        history = history or []
        response = {"response": "", "sources": []}

        try:
            tool = plan.get("tool")
            data = plan.get("data")
            file_type = plan.get("file_type")

            if tool == "gpt":
                response["response"] = process_with_gpt4o(data)
            
            elif tool == "image":
                # Now 'data' is a dict containing {"file_bytes": ..., "user_query": ...}
                file_bytes = data.get("file_bytes")
                user_query = data.get("user_query", "")  # default to empty if missing

                if file_type and file_bytes:
                    response["response"] = process_image_with_gpt4o(file_bytes, file_type, user_query)
                else:
                    response["response"] = "❌ Missing file or file type for image processing"
            
            elif tool == "pdf":
                extracted_text = data.get("extracted_text", "")
                user_query = data.get("user_query", "Summarize this document.")
                response["response"] = process_pdf_with_gpt4o(extracted_text, user_query)
            
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
                response["response"] = self.format_full_wiki(result)
                response["sources"] = [result["url"]]
            
            else:
                response["response"] = "❌ Unknown tool selected"

        except Exception as e:
            response["response"] = f"⚠️ Execution Error: {str(e)}"

        # Update chat history
        history.append({"role": "assistant", "content": response["response"]})
        
        return {
            "response": response["response"],
            "sources": response.get("sources", []),
            "history": history
        }

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