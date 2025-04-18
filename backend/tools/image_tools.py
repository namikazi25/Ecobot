import base64
from PIL import Image
import io
from backend.tools.openai_client import client  # Use shared OpenAI client

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
                quality=95  # Maintain quality
            )
            fresh_bytes = output_buffer.getvalue()

        base64_encoded = base64.b64encode(fresh_bytes).decode("utf-8")
        return f"data:image/{file_type.split('/')[-1]};base64,{base64_encoded}"
    
    except Exception as e:
        print(f"❌ Image Encoding Failed: {str(e)}")
        return None
def process_image_with_gpt4o(file_content: bytes, file_type: str, user_query: str):
    """
    Sends an image + the user's actual question to GPT-4o.
    """

    image_data_url = encode_image(file_content, file_type)
    if not image_data_url:
        return "❌ Error: Image encoding failed."

    try:
        # Now we pass user_query in the 'text' portion:
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
        return f"❌ Error processing image: {str(e)}"
