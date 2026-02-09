import streamlit as st
import replicate
import PyPDF2
import os

st.set_page_config(page_title="🖊️PDF Summarizer Chatbot")

# Toggle light and dark mode themes
ms = st.session_state
if "themes" not in ms: 
    ms.themes = {"current_theme": "light",
                 "refreshed": True,
                    
                "light": {"theme.base": "dark",
                          "theme.backgroundColor": "#FFFFFF",
                          "theme.primaryColor": "#6200EE",
                          "theme.secondaryBackgroundColor": "#F5F5F5",
                          "theme.textColor": "000000",
                          "button_face": "🌜"},

                "dark":  {"theme.base": "light",
                          "theme.backgroundColor": "#121212",
                          "theme.primaryColor": "#BB86FC",
                          "theme.secondaryBackgroundColor": "#1F1B24",
                          "theme.textColor": "#E0E0E0",
                          "button_face": "🌞"},
                          }


def ChangeTheme():
    previous_theme = ms.themes["current_theme"]
    tdict = ms.themes["light"] if ms.themes["current_theme"] == "light" else ms.themes["dark"]
    for vkey, vval in tdict.items(): 
        if vkey.startswith("theme"): st._config.set_option(vkey, vval)

    ms.themes["refreshed"] = False
    if previous_theme == "dark": ms.themes["current_theme"] = "light"
    elif previous_theme == "light": ms.themes["current_theme"] = "dark"

btn_face = ms.themes["light"]["button_face"] if ms.themes["current_theme"] == "light" else ms.themes["dark"]["button_face"]
st.button(btn_face, on_click=ChangeTheme)

if ms.themes["refreshed"] == False:
    ms.themes["refreshed"] = True
    st.rerun()


# Function to extract pdf to text format
def extract_text_from_pdf(pdf_file):
    """Extract text from an uploaded PDF file."""
    reader = PyPDF2.PdfReader(pdf_file)
    text = ""
    for page in reader.pages:
        text += page.extract_text()
    return text


# Replicate Credentials
with st.sidebar:
    st.title('🖊️PDF Summarizer Chatbot')
    if 'REPLICATE_API_TOKEN' in st.secrets:
        st.success('API key already provided!', icon='✅')
        replicate_api = st.secrets['REPLICATE_API_TOKEN']
    else:
        replicate_api = st.text_input('Enter Replicate API token:', type='password')
        if not (replicate_api.startswith('r8_') and len(replicate_api)==40):
            st.warning('Please enter your credentials!', icon='⚠️')
        else:
            st.success('Proceed to entering your prompt message!', icon='👉')

    uploaded_file = st.file_uploader("Upload a PDF file", type="pdf")
    pdf_text = ""
    
    if uploaded_file:
        with st.spinner("Extracting text from PDF..."):
            pdf_text = extract_text_from_pdf(uploaded_file)
        st.success("PDF uploaded and text extracted!")

    st.markdown('''
        Developed by Ichikawa Hiroshi - 2024  
        Visit my GitHub profile <a href="https://github.com/0xichikawa" style="color:white; background-color:#3187A2; padding:3px 5px; text-decoration:none; border-radius:5px;">here</a>
        ''', unsafe_allow_html=True)


os.environ['REPLICATE_API_TOKEN'] = replicate_api

# Store LLM generated responses
if "messages" not in st.session_state.keys():
    st.session_state.messages = [{"role": "assistant", "content": "Upload a PDF file from the sidebar to get started."}]

# Display or clear chat messages
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.write(message["content"])

def clear_chat_history():
    st.session_state.messages = [{"role": "assistant", "content": "Upload a PDF file from the sidebar to get started."}]
    st.sidebar.button('Clear Chat History', on_click=clear_chat_history)


def generate_llama2_response(text, question):
    string_dialogue = "You are a helpful assistant. You do not respond as 'User' or pretend to be 'User'. You only respond once as 'Assistant'."
    prompt = f"Here is the context:\n\n{text[:5000]}\n\nNow answer this question:\n{question}"
    for dict_message in st.session_state.messages:
        if dict_message["role"] == "user":
            string_dialogue += "User: " + dict_message["content"] + "\n\n"
        else:
            string_dialogue += "Assistant: " + dict_message["content"] + "\n\n"
    output = replicate.run('a16z-infra/llama13b-v2-chat:df7690f1994d94e96ad9d568eac121aecf50684a0b0963b25a41cc40061269e5', 
                           input={"prompt" : prompt,
                                  "temperature":0.1, "top_p":0.9, "max_length":2000, "repetition_penalty":1})
    return output


# Generate a new response if last message is not from assistant
if pdf_text:
    question = st.text_input("Enter your question:")
    if question:
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                response = generate_llama2_response(pdf_text, question)
                placeholder = st.empty()
                full_response = ''
                for item in response:
                    full_response += item
                    placeholder.markdown(full_response)
                placeholder.markdown(full_response)
        message = {"role": "assistant", "content": full_response}
        st.session_state.messages.append(message)

