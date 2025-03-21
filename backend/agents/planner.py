import sys
import os
import re

# Add the project root to sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from backend.gpt_handler import generate_plan_with_gpt4o
from backend.tools.image_tools import process_image_with_gpt4o
from backend.tools.pdf_tools import extract_text_from_pdf

class PlanningAgent:
    """Generates execution plans using GPT-4o and domain-specific heuristics"""

    # Merged triggers from Valyu and previous Wikipedia triggers
    VALYU_TRIGGERS = [
        "biodiversity index", "conservation status", "habitat loss",
        "ecological impact", "species migration patterns", "climate change effects",
        "ecosystem services", "carbon sequestration", "invasive species",
        "protected areas", "sustainability metrics", "environmental policy",
        "wildlife corridors", "restoration potential", "land use change",
        "ecological data", "scientific consensus", "peer-reviewed study",
        "environmental impact assessment", "species recovery plan", "ecosystem health indicators",
        "wikipedia", "verified source", "scientific name", "taxonomy of",
        "habitat of", "academic sources", "species classification",
        "kingdom", "phylum", "genus", "family"
    ]

    def plan(self, query, file_contents=None, history=None):
        """
        Generate execution plan considering multiple data sources.

        :param query: The text query from the user.
        :param file_contents: A list of tuples [(bytes, content_type), ...].
        :param history: Chat history for context.
        """
        history = history or []
        file_contents = file_contents or []
        plan = {"tool": "gpt", "data": query}

        try:
            if file_contents:
                file_bytes, file_type = file_contents[0]
                plan = self._handle_single_file(query, file_bytes, file_type)
            else:
                if self._requires_valyu(query):
                    plan = self._create_valyu_plan(query)
                else:
                    plan = self._create_gpt_plan(query, history)
        except Exception as e:
            plan = self._create_error_plan(f"Planning error: {str(e)}")

        return plan

    def _handle_single_file(self, query, file_bytes, file_type):
        """Process a single file with validation and error handling."""
        if "image" in file_type:
            return {
                "tool": "image",
                "data": {"file_bytes": file_bytes, "user_query": query},
                "file_type": file_type,
                "rationale": "Image analysis required"
            }
        if "pdf" in file_type:
            extracted_text = extract_text_from_pdf(file_bytes)
            if "❌" in extracted_text:
                raise ValueError(extracted_text)
            return {
                "tool": "pdf",
                "data": {"extracted_text": extracted_text, "user_query": query},
                "rationale": "PDF document processing"
            }
        raise ValueError(f"Unsupported file type: {file_type}")

    def _create_valyu_plan(self, query: str) -> dict:
        """Create Valyu-specific execution plan for ecological data."""
        return {
            "tool": "valyu",
            "data": self._clean_valyu_query(query),
            "rationale": "Valyu context required for ecological verification",
            "sources": [
                "valyu/valyu-conservation",
                "valyu/valyu-climate",
                "valyu/valyu-biodiversity"
            ]
        }

    def _create_gpt_plan(self, query, history):
        """Create GPT plan with conversation context."""
        return {
            "tool": "gpt",
            "data": f"{self._build_conversation_context(history)}\n\n{query}",
            "rationale": "General ecological query with context"
        }

    def _create_error_plan(self, error_msg):
        return {
            "tool": "gpt",
            "data": error_msg,
            "rationale": "Error handling fallback"
        }

    def _requires_valyu(self, query: str) -> bool:
        """Check if query requires Valyu context search."""
        lower_query = query.lower()
        return any(trigger in lower_query for trigger in self.VALYU_TRIGGERS)

    def _clean_valyu_query(self, query: str) -> str:
        """Prepare query for Valyu context search by removing common environmental terms."""
        return re.sub(r'\b(environmental|ecological|conservation)\b', '', query, flags=re.IGNORECASE).strip()

    def _build_conversation_context(self, history):
        """Build context from last 3 messages."""
        return "\n".join(
            f"{msg['role']}: {msg['content']}" for msg in history[-3:]
        )
