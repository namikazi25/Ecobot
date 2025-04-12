import base64
from PIL import Image
import io
from backend.tools.openai_client import client  # Use shared OpenAI client

def encode_image(file_content: bytes, file_type: str) -> str:
    """Validate and re-encode an image to base64 data URL for use with LLMs and vision APIs.

    Args:
        file_content (bytes): Raw binary content of the image file.
        file_type (str): MIME type (e.g., 'image/jpeg', 'image/png').
    Returns:
        str: Base64-encoded data URL for the image ("data:image/...;base64,...") or None if failure.
    """
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
                quality=95  # Maintain quality
            )
            fresh_bytes = output_buffer.getvalue()
        base64_encoded = base64.b64encode(fresh_bytes).decode("utf-8")
        return f"data:image/{file_type.split('/')[-1]};base64,{base64_encoded}"
    except Exception as e:
        print(f"❌ Image Encoding Failed: {str(e)}")
        return None

def process_image_with_gpt4o(file_content: bytes, file_type: str, user_query: str) -> str:
    """Send an image and an associated user question to GPT-4o for ecological analysis or species identification.

    Args:
        file_content (bytes): Image file content.
        file_type (str): The MIME type of the image.
        user_query (str): User's ecological or identification question.
    Returns:
        str: LLM model response or user-facing error.
    """
    image_data_url = encode_image(file_content, file_type)
    if not image_data_url:
        return "❌ Error: Image encoding failed."
    try:
        print("[DEBUG] user_query:", user_query)
        print("[DEBUG] image_data_url:", image_data_url[:100], "...")  # Print partial if large
        response = client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": user_query},
                        {"type": "image_url", "image_url": {"url": image_data_url, "detail": "high"}},
                    ],
                }
            ],
            max_tokens=500,
        )
        return response.choices[0].message.content
    except Exception as e:
        return f"❌ Error processing image with GPT-4o: {str(e)}"
