import base64
from PIL import Image
import io
from core.model_router import ModelRouter

router = ModelRouter()

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
        return base64_encoded
    except Exception as e:
        print(f"❌ Image Encoding Failed: {str(e)}")
        return None

def process_image_with_llm(file_content: bytes, file_type: str, user_query: str):
    """
    Sends an image + the user's actual question to Gemini (via router).
    """
    b64 = encode_image(file_content, file_type)
    if not b64:
        return "❌ Error: Image encoding failed."
    mime = file_type
    image_block = {"type": "image", "source_type": "base64", "mime_type": mime, "data": b64}
    try:
        llm = router.route()
        result = llm.invoke([
            {"role": "user", "content": [
                {"type": "text", "text": user_query},
                image_block
            ]}
        ])
        return result.content if hasattr(result, "content") else result
    except Exception as e:
        return f"❌ Error processing image: {str(e)}"
