import io
import pdfplumber
from core.llm_factory import get_llm
from langchain.chains import LLMChain
from core.prompts import EXECUTOR_PROMPT

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

def process_pdf_with_llm(extracted_text: str, query: str, llm=None) -> str:
    """Sends extracted PDF text to the selected LLM for processing using the user's query."""
    if not extracted_text:
        return "No text extracted from the PDF."
    llm = llm or get_llm("google", "gemini-2.0-flash")
    chain = LLMChain(llm=llm, prompt=EXECUTOR_PROMPT)
    try:
        context = f"Extracted text:\n{extracted_text}"
        return chain.run(tool="pdf", data=query, context=context)
    except Exception as e:
        return f"❌ Error processing PDF with LLM: {str(e)}"
