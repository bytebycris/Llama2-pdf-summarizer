import streamlit as st
import replicate
import PyPDF2
import os

st.set_page_config(page_title="🖊️PDF Summarizer Chatbot")

# ---------------------------------------------------------------------------
# Theme toggle (light / dark)
# ---------------------------------------------------------------------------
ms = st.session_state
if "themes" not in ms:
    ms.themes = {
        "current_theme": "light",
        "refreshed": True,
        "light": {
            "theme.base": "dark",
            "theme.backgroundColor": "#FFFFFF",
            "theme.primaryColor": "#6200EE",
            "theme.secondaryBackgroundColor": "#F5F5F5",
            "theme.textColor": "#000000",
            "button_face": "🌜",
        },
        "dark": {
            "theme.base": "light",
            "theme.backgroundColor": "#121212",
            "theme.primaryColor": "#BB86FC",
            "theme.secondaryBackgroundColor": "#1F1B24",
            "theme.textColor": "#E0E0E0",
            "button_face": "🌞",
        },
    }


def change_theme():
    previous_theme = ms.themes["current_theme"]
    tdict = ms.themes[previous_theme]
    for vkey, vval in tdict.items():
        if vkey.startswith("theme"):
            st._config.set_option(vkey, vval)
    ms.themes["refreshed"] = False
    ms.themes["current_theme"] = "dark" if previous_theme == "light" else "light"


btn_face = ms.themes[ms.themes["current_theme"]]["button_face"]
st.button(btn_face, on_click=change_theme)

if not ms.themes["refreshed"]:
    ms.themes["refreshed"] = True
    st.rerun()


# ---------------------------------------------------------------------------
# PDF text extraction (cached in session state so it isn't repeated on rerun)
# ---------------------------------------------------------------------------
def extract_text_from_pdf(pdf_file):
    """Extract text from an uploaded PDF file."""
    reader = PyPDF2.PdfReader(pdf_file)
    pages = [page.extract_text() or "" for page in reader.pages]
    return "".join(pages)


# ---------------------------------------------------------------------------
# Sidebar – credentials, file upload, clear button
# ---------------------------------------------------------------------------
with st.sidebar:
    st.title("🖊️PDF Summarizer Chatbot")

    # --- API token -----------------------------------------------------------
    replicate_api = ""
    if "REPLICATE_API_TOKEN" in st.secrets:
        st.success("API key already provided!", icon="✅")
        replicate_api = st.secrets["REPLICATE_API_TOKEN"]
    else:
        replicate_api = st.text_input("Enter Replicate API token:", type="password")
        if not (replicate_api.startswith("r8_") and len(replicate_api) == 40):
            st.warning("Please enter your credentials!", icon="⚠️")
        else:
            st.success("Proceed to entering your prompt message!", icon="👉")

    # --- PDF upload (cached in session state) --------------------------------
    uploaded_file = st.file_uploader("Upload a PDF file", type="pdf")

    if uploaded_file:
        file_id = f"{uploaded_file.name}_{uploaded_file.size}"
        if st.session_state.get("pdf_file_id") != file_id:
            with st.spinner("Extracting text from PDF..."):
                st.session_state["pdf_text"] = extract_text_from_pdf(uploaded_file)
                st.session_state["pdf_file_id"] = file_id
            st.success("PDF uploaded and text extracted!")
        else:
            st.success("PDF already loaded.")

    # --- Clear chat ----------------------------------------------------------
    def clear_chat_history():
        st.session_state.messages = [
            {"role": "assistant", "content": "Upload a PDF file from the sidebar to get started."}
        ]

    st.button("Clear Chat History", on_click=clear_chat_history)

    st.markdown(
        """
        Developed by Ichikawa Hiroshi - 2024  
        Visit my GitHub profile <a href="https://github.com/0xichikawa"
        style="color:white; background-color:#3187A2; padding:3px 5px;
        text-decoration:none; border-radius:5px;">here</a>
        """,
        unsafe_allow_html=True,
    )


# ---------------------------------------------------------------------------
# Set Replicate API token (only when a valid key is available)
# ---------------------------------------------------------------------------
if replicate_api:
    os.environ["REPLICATE_API_TOKEN"] = replicate_api


# ---------------------------------------------------------------------------
# Chat history
# ---------------------------------------------------------------------------
if "messages" not in st.session_state:
    st.session_state.messages = [
        {"role": "assistant", "content": "Upload a PDF file from the sidebar to get started."}
    ]

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.write(message["content"])


# ---------------------------------------------------------------------------
# LLM response generation (uses chat history for multi-turn context)
# ---------------------------------------------------------------------------
def generate_llama2_response(text, question):
    """Build a prompt with conversation history + PDF context and query the model."""
    system_msg = (
        "You are a helpful assistant. You do not respond as 'User' "
        "or pretend to be 'User'. You only respond once as 'Assistant'."
    )

    # Build conversation history
    history_parts = []
    for msg in st.session_state.messages:
        role = "User" if msg["role"] == "user" else "Assistant"
        history_parts.append(f"{role}: {msg['content']}")
    conversation_history = "\n\n".join(history_parts)

    prompt = (
        f"{system_msg}\n\n"
        f"Here is the PDF context:\n\n{text[:5000]}\n\n"
        f"Conversation so far:\n{conversation_history}\n\n"
        f"User: {question}\n\nAssistant:"
    )

    try:
        output = replicate.run(
            "a16z-infra/llama13b-v2-chat:df7690f1994d94e96ad9d568eac121aecf50684a0b0963b25a41cc40061269e5",
            input={
                "prompt": prompt,
                "temperature": 0.1,
                "top_p": 0.9,
                "max_length": 2000,
                "repetition_penalty": 1,
            },
        )
        return output
    except Exception as e:
        st.error(f"Error generating response: {e}")
        return None


# ---------------------------------------------------------------------------
# Main chat interaction
# ---------------------------------------------------------------------------
pdf_text = st.session_state.get("pdf_text", "")

if pdf_text:
    if question := st.chat_input("Ask a question about the PDF..."):
        # Save the user message
        st.session_state.messages.append({"role": "user", "content": question})
        with st.chat_message("user"):
            st.write(question)

        # Generate and stream the assistant response
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                response = generate_llama2_response(pdf_text, question)
                if response is not None:
                    placeholder = st.empty()
                    full_response = ""
                    for item in response:
                        full_response += item
                        placeholder.markdown(full_response)
                    placeholder.markdown(full_response)
                    st.session_state.messages.append(
                        {"role": "assistant", "content": full_response}
                    )
else:
    st.chat_input("Upload a PDF to start chatting...", disabled=True)
