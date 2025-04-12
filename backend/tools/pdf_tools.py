import io
import pdfplumber
from backend.tools.openai_client import client  # Use shared OpenAI client

def extract_text_from_pdf(file_content: bytes) -> str:
    """Extract text from a PDF file using pdfplumber.

    Args:
        file_content (bytes): Raw binary content of the uploaded PDF.
    Returns:
        str: Extracted document text or user-facing error if failure.
    """
    try:
        if not isinstance(file_content, bytes):  # ✅ Ensure input is bytes
            raise ValueError("Expected bytes, but received a different format.")

        with pdfplumber.open(io.BytesIO(file_content)) as pdf:
            extracted_text = "\n".join([page.extract_text() for page in pdf.pages if page.extract_text()])
        return extracted_text.strip() if extracted_text else "No readable text found in the PDF."
    except Exception as e:
        return f"❌ Error extracting text from PDF: {str(e)}"

def process_pdf_with_gpt4o(extracted_text: str, query: str) -> str:
    """Send extracted PDF text and a user query to GPT-4o for summarization/analysis.

    Args:
        extracted_text (str): PDF content as a string.
        query (str): User question/request about the document.
    Returns:
        str: Model's response or error message.
    """
    if not extracted_text:
        return "No text extracted from the PDF."
    try:
        response = client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {  # Add system message for context
                    "role": "system",
                    "content": """You are EcoBot, an AI-powered ecological assistant. \n                    Provide scientific and informative responses about biodiversity, \n                    species identification, and ecosystems using the provided document text."""
                },
                {
                    "role": "user", 
                    "content": f"{query}\n\nExtracted text:\n{extracted_text}"
                }
            ],
            max_tokens=500,
        )
        return response.choices[0].message.content
    except Exception as e:
        return f"❌ Error processing PDF with GPT-4o: {str(e)}"
