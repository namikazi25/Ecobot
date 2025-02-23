import gradio as gr
import requests
import mimetypes
import os

API_URL = "http://localhost:8000/query/"  # Your running FastAPI endpoint

def call_backend(text: str, file_paths: list[str]):
    """
    Send 'text' + optional files to the FastAPI backend.
    file_paths is a list of local paths on the server side.
    """
    data = {"query": text or "No text"}
    files = {}
    for i, path in enumerate(file_paths):
        with open(path, "rb") as f:
            file_bytes = f.read()
        mime_type = mimetypes.guess_type(path)[0] or "application/octet-stream"
        # Bracket notation so that the server sees them as files[0], files[1], etc.
        files[f"files[{i}]"] = (os.path.basename(path), file_bytes, mime_type)

    try:
        response = requests.post(API_URL, data=data, files=files)
        return response.json()
    except Exception as e:
        return {"response": f"Backend Error: {str(e)}", "sources": []}

def handle_message(message, history):
    """
    - message: {"text": "...", "files": [...list of file paths...]}
    - history: chat history (list of dicts in OpenAI style)
    Returns a string (the assistant's new reply).
    """
    user_text = message.get("text", "")
    file_paths = message.get("files", [])

    # Send everything to your FastAPI backend
    backend_data = call_backend(user_text, file_paths)
    bot_text = backend_data.get("response", "No response received.")
    sources = backend_data.get("sources", [])

    # Optionally append sources at the end
    if sources:
        formatted_sources = "\n".join(f"- {src}" for src in sources)
        bot_text += f"\n\n**Sources**:\n{formatted_sources}"

    # Return the assistant's message as a string
    return bot_text

demo = gr.ChatInterface(
    fn=handle_message,
    # We'll store the entire conversation as a list of openAI-style messages:
    type="messages",
    # This enables text + file upload in a single chat bubble
    multimodal=True,
    title="🌿 EcoBot: Your Ecological Assistant",
    description="Upload images/PDFs directly in your chat turns!",
    # If you want to limit file types, pass a custom MultimodalTextbox:
    textbox=gr.MultimodalTextbox(
        file_types=[".pdf", ".jpg", ".jpeg", ".png"],  
        file_count="multiple",  # user can attach multiple files in one turn
        label="Type your question or request, then optionally add files."
    )
)

if __name__ == "__main__":
    # By default, Gradio will run on 127.0.0.1:7860
    demo.launch()
