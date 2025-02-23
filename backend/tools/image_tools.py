import base64
import io
from backend.tools.openai_client import client  # Use shared OpenAI client

def encode_image(file_content: bytes, file_type: str) -> str:
    """Encodes an image to Base64 format for GPT-4o processing."""
    try:
        base64_encoded = base64.b64encode(file_content).decode("utf-8")
        return f"data:image/{file_type.split('/')[-1]};base64,{base64_encoded}"
    except Exception as e:
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
