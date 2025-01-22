import io
import pdfplumber
from backend.tools.openai_client import client  # Use shared OpenAI client

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

def process_pdf_with_gpt4o(file_content: bytes, query: str = "Summarize this document.") -> str:
    """Sends extracted PDF text to GPT-4o for processing."""
    extracted_text = extract_text_from_pdf(file_content)

    if "❌" in extracted_text:
        return extracted_text  # Return the extraction error directly

    try:
        response = client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {"role": "user", "content": f"{query}\n\nExtracted text:\n{extracted_text}"}
            ],
            max_tokens=500,
        )
        return response.choices[0].message.content
    except Exception as e:
        return f"❌ Error processing PDF with GPT-4o: {str(e)}"
