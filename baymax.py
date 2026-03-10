import streamlit as st
import datetime
from google import genai
from google.genai import types

# 1. FIX: Initialize Gemini Client with named argument 'api_key'
# Replace the string below with your actual API key
client = genai.Client(api_key="AIzaSyDGVB16gbZ0K6TwxmhbwARQafU_p40dSzs")

# Page config
st.set_page_config(page_title="🤖 Baymax Assistant", layout="centered")

# Sidebar information
with st.sidebar:
    st.markdown("### About Us")
    st.markdown("""
    **Suraj**
    
    We are dedicated to building helpful AI-powered chat assistants.
    📧 sn740698@gmail.com
    """)
    st.markdown("---")
    st.markdown("""
    ### User Guidance
    - **Searching Mode:** Factual summaries.
    - **Health:** General health info (Not a doctor!).
    - **Mathematics:** Step-by-step math.
    - **Code:** Programming help.
    """)

# Initialize session state
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "edit_index" not in st.session_state:
    st.session_state.edit_index = None
if "todo_list" not in st.session_state:
    st.session_state.todo_list = []
if "health_logs" not in st.session_state:
    st.session_state.health_logs = []

# Logic Functions
def add_task(task):
    st.session_state.todo_list.append(task)

def list_tasks():
    return "\n".join(f"- {t}" for t in st.session_state.todo_list) if st.session_state.todo_list else "Empty."

def log_symptom(symptom):
    st.session_state.health_logs.append({'date': datetime.date.today().isoformat(), 'symptom': symptom})

def get_bot_response(user_text, mode):
    instructions = {
        "Searching Mode": "You are Baymax, an encyclopedia. Provide factual summaries.",
        "Health": "You are Baymax, a health companion. Be empathetic. Remind user to see a doctor.",
        "Mathematics": "You are Baymax, a math tutor. Show step-by-step work.",
        "Code": "You are Baymax, a senior coder. Provide clean code and explanations."
    }
    
    try:
        response = client.models.generate_content(
            model='gemini-2.0-flash',
            contents=user_text,
            config=types.GenerateContentConfig(
                system_instruction=instructions.get(mode, ""),
                temperature=0.7,
            )
        )
        return response.text
    except Exception as e:
        return f"Error: {str(e)}"

# UI: Mode selection
mode = st.radio("Select Mode", options=["Searching Mode", "Health", "Mathematics", "Code"], horizontal=True)

st.title(f"🤖 Baymax: {mode}")

# Chat History Display
for idx, message in enumerate(st.session_state.chat_history):
    if message["role"] == "user":
        st.chat_message("user").write(message["msg"])
    else:
        st.chat_message("assistant").write(message["msg"])

# Input Area
with st.container():
    user_input = st.chat_input("How can I help you?")
    if user_input:
        # Add user message
        st.session_state.chat_history.append({"role": "user", "msg": user_input})
        
        # Get and add bot response
        response = get_bot_response(user_input, mode)
        st.session_state.chat_history.append({"role": "bot", "msg": response})
        st.rerun()

# Tools Section
st.markdown("---")
col1, col2 = st.columns(2)

with col1:
    with st.expander("📝 To-Do List"):
        task = st.text_input("New task", key="t_in")
        if st.button("Add"):
            add_task(task)
            st.rerun()
        st.text(list_tasks())

with col2:
    if mode == "Health":
        with st.expander("🏥 Health Tracker"):
            symp = st.text_input("Log symptom", key="h_in")
            if st.button("Log"):
                log_symptom(symp)
                st.rerun()
            for log in st.session_state.health_logs:
                st.write(f"**{log['date']}**: {log['symptom']}")

if st.button("Clear Chat"):
    st.session_state.chat_history = []
    st.rerun()


