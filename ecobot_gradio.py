import gradio as gr
import requests
import mimetypes
import os
import json

API_URL = "http://localhost:8000/query/"  # FastAPI endpoint

def call_backend(text: str, file_paths: list[str]):
    data = {"query": text or "No text"}
    files = []
    for path in file_paths:
        with open(path, "rb") as f:
            file_bytes = f.read()
        mime_type = mimetypes.guess_type(path)[0] or "application/octet-stream"
        files.append(("files", (os.path.basename(path), file_bytes, mime_type)))

    try:
        resp = requests.post(API_URL, data=data, files=files, timeout=15)
        # Debug: print status and raw body to console
        print(f"\U0001F50D Backend returned {resp.status_code}: {resp.text}")

        resp.raise_for_status()
        payload = resp.json()
    except requests.HTTPError as http_err:
        return {"response": f"HTTP error {resp.status_code}: {resp.text}", "sources": []}
    except ValueError:
        # JSON decode failed
        return {"response": f"Invalid JSON from backend: {resp.text}", "sources": []}
    except Exception as e:
        return {"response": f"Backend request failed: {e}", "sources": []}

    return payload

def handle_message(message, history):
    user_text = message.get("text", "")
    file_paths = message.get("files", [])

    backend_data = call_backend(user_text, file_paths)

    # If backend_data has a “detail” key (FastAPI error), expose it
    if "detail" in backend_data:
        bot_text = f"Backend error: {backend_data['detail']}"
    else:
        bot_text = backend_data.get("response")
        if bot_text is None:
            # Show entire payload so you can diagnose missing fields
            bot_text = f"No 'response' field in backend payload:\n{json.dumps(backend_data, indent=2)}"

    # Append sources if present
    sources = backend_data.get("sources", [])
    if sources:
        src_lines = "\n".join(f"- {s}" for s in sources)
        bot_text += f"\n\n**Sources**:\n{src_lines}"

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
