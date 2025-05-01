import io
import pdfplumber
from core.model_router import ModelRouter

router = ModelRouter()

def extract_text_from_pdf(file_content: bytes) -> str:
    """Extracts text from a PDF file."""
    try:
        if not isinstance(file_content, bytes):  # ✅ Ensure input is bytes
            raise ValueError("Expected bytes, but received a different format.")

        with pdfplumber.open(io.BytesIO(file_content)) as pdf:
            extracted_text = "\n".join([page.extract_text() for page in pdf.pages if page.extract_text()])
        return extracted_text.strip() if extracted_text else "No readable text found in the PDF."
    except Exception as e:
        return f"❌ Error extracting text from PDF: {str(e)}"

def process_pdf_with_llm(file_bytes: bytes, user_query: str) -> str:
    """Sends PDF bytes to Gemini (via router) for processing using the user's query."""
    if not file_bytes:
        return "No PDF bytes provided."
    try:
        b64 = base64.b64encode(file_bytes).decode("utf-8")
        file_block = {"type": "file", "source_type": "base64", "mime_type": "application/pdf", "data": b64}
        llm = router.route()
        result = llm.invoke([
            {"role": "user", "content": [
                {"type": "text", "text": user_query},
                file_block
            ]}
        ])
        return result.content if hasattr(result, "content") else result
    except Exception as e:
        return f"❌ Error processing PDF: {str(e)}"
