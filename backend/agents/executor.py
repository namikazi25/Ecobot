import sys
import os
from PIL import Image
import io

# Add the project root to sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from backend.tools.image_tools import process_image_with_gpt4o
from backend.gpt_handler import process_with_gpt4o
from backend.tools.pdf_tools import process_pdf_with_gpt4o
from backend.tools.wiki_tool import search_wikipedia, fetch_full_page

class ExecutingAgent:
    """Executes validated user plans and orchestrates calls to the appropriate ecological tools (image, wiki, pdf, GPT), formatting the output for the user."""

    def execute(self, plan, history=None):
        """
        Executes the validated plan by delegating work to the selected data processing tool (GPT, image, PDF, or Wiki API).
        Handles validation and error messaging, returning both the response and relevant sources.

        Args:
            plan (dict): A dictionary specifying the selected tool and its data payload.
            history (list, optional): Chat history as a list of dicts. Defaults to None.

        Returns:
            dict: Output with keys 'response' (str) and optional 'sources' (list) or error keys.
        """
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

                try:
                    # Pre-check before GPT call
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
            
            elif tool == "wiki":
                result = search_wikipedia(data)
                if "error" in result:
                    return self.fallback_response(data, result)
                response["response"] = self.format_wiki_summary(result)
                response["sources"] = [result["url"]]

            elif tool == "wiki_full":
                page_result = fetch_full_page(data)
                if "error" in page_result:
                    return self.fallback_response(data, page_result)
                response["response"] = page_result.get("extract", "No page extract found.")
                response["sources"] = [page_result.get("fullurl", "")]

            else:
                response["response"] = "❌ Unknown tool in plan."

        except Exception as e:
            response["response"] = f"❌ Execution error: {str(e)}"
        return response

    def fallback_response(self, query, result):
        """Return an error message with optional fallback guidance."""
        return {
            "response": f"Failed retrieving Wikipedia info: {result.get('error', '')}\nTry rephrasing your query or check your internet connection.",
            "sources": []
        }

    def format_wiki_summary(self, wiki_result):
        """Format the brief Wikipedia extract and relevant metadata for presentation."""
        summary = wiki_result.get("extract", "[No summary available]")
        url = wiki_result.get("url", "")
        return f"{summary}\n\n[🔗 Source]({url})" if url else summary

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