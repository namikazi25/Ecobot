import streamlit as st
import requests
import json
import os

# FastAPI backend URL - Read from environment variable or default
API_URL = os.getenv("BACKEND_API_URL", "http://127.0.0.1:8000")

st.set_page_config(page_title="🌿 EcoBot", layout="wide")
st.title("🌿 EcoBot: Your Ecological Assistant")

# Initialize session state for chat history
if "messages" not in st.session_state:
    st.session_state.messages = []
if "last_uploaded_file" not in st.session_state:
    st.session_state.last_uploaded_file = None
if "pdf_context" not in st.session_state:
    st.session_state.pdf_context = None

# Display chat history
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        if "file" in message:
            st.write(f"📄 Uploaded File: {message['file']}")
        st.markdown(message["content"])
        if "sources" in message:
            with st.expander("📚 Sources"):
                for source in message["sources"]:
                    st.markdown(f"[🔗 {source}]({source})")

# File uploader
uploaded_file = st.file_uploader("Upload an image or PDF (optional)", type=["jpg", "jpeg", "png", "pdf"])

# Chat input
user_input = st.chat_input("Enter your query...")

if user_input:
    user_message = {"role": "user", "content": user_input}
    
    if uploaded_file:
        user_message["file"] = uploaded_file.name
        st.session_state.last_uploaded_file = uploaded_file.name

    st.session_state.messages.append(user_message)

    # Display user message
    with st.chat_message("user"):
        st.markdown(user_input)
        if uploaded_file:
            st.write(f"📄 Uploaded File: {uploaded_file.name}")

    # Assistant response area
    with st.chat_message("assistant"):
        message_placeholder = st.empty()
        message_placeholder.write("Processing...")

        # Prepare request payload
        history = json.dumps([{"role": msg["role"], "content": msg["content"]} for msg in st.session_state.messages])
        data = {
            "query": user_input if user_input else "Analyze the uploaded file.",
            "history": history,
            "pdf_context": st.session_state.pdf_context
        }
        
        files = None
        if uploaded_file:
            files = {"file": (uploaded_file.name, uploaded_file.getvalue(), uploaded_file.type)}

        # Get response from backend
        try:
            response = requests.post(f"{API_URL}/query/", data=data, files=files)
            response_data = response.json()
            bot_response = response_data.get("response", "No response received.")
            sources = response_data.get("sources", [])
            
            if "pdf_context" in response_data:
                st.session_state.pdf_context = response_data["pdf_context"]
                
        except requests.exceptions.RequestException as e:
            bot_response = f"⚠️ Backend Error: {str(e)}"
            sources = []

        # Display response and sources
        message_placeholder.markdown(bot_response)
        if sources:
            with st.expander("📚 Sources"):
                for source in sources:
                    st.markdown(f"[🔗 {source}]({source})")

    # Store in session state
    assistant_message = {"role": "assistant", "content": bot_response}
    if sources:
        assistant_message["sources"] = sources
    st.session_state.messages.append(assistant_message)

    # Clear uploaded file
    st.session_state.last_uploaded_file = None
    st.rerun()