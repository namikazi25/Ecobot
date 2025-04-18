import base64
from PIL import Image
import io
from core.llm_factory import get_llm
from langchain.chains import LLMChain
from core.prompts import EXECUTOR_PROMPT

def encode_image(file_content: bytes, file_type: str) -> str:
    """Validate and re-encode images properly"""
    try:
        # First validation pass
        with Image.open(io.BytesIO(file_content)) as img:
            img.verify()

        # Second pass - recreate image for processing
        with Image.open(io.BytesIO(file_content)) as img:
            if img.mode != 'RGB':
                img = img.convert('RGB')
            output_buffer = io.BytesIO()
            img.save(
                output_buffer, 
                format="JPEG" if "jpeg" in file_type else "PNG",
                quality=95
            )
            fresh_bytes = output_buffer.getvalue()
        base64_encoded = base64.b64encode(fresh_bytes).decode("utf-8")
        return f"data:image/{file_type.split('/')[-1]};base64,{base64_encoded}"
    except Exception as e:
        print(f"❌ Image Encoding Failed: {str(e)}")
        return None

def process_image_with_llm(file_content: bytes, file_type: str, user_query: str, llm=None):
    """
    Sends an image + the user's actual question to the selected LLM (via LangChain).
    """
    image_data_url = encode_image(file_content, file_type)
    if not image_data_url:
        return "❌ Error: Image encoding failed."
    llm = llm or get_llm("google", "gemini-2.0-flash")
    chain = LLMChain(llm=llm, prompt=EXECUTOR_PROMPT)
    try:
        # For Gemini, pass image and text as input variables if supported
        # For now, we use the prompt template for text and pass image as context if needed
        return chain.run(tool="image", data=user_query, context=image_data_url)
    except Exception as e:
        return f"❌ Error processing image with LLM: {str(e)}"
