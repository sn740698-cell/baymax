import streamlit as st
import datetime
from google import genai
from google.genai import types

# Initialize Gemini Client 
client = genai.Client(api_key="AIzaSyDZifFyqyaweK8liNxT77VjX5r8-Zg2a7M")

# Page config
st.set_page_config(page_title="🤖 Baymax Assistant", layout="centered")

# Sidebar information
with st.sidebar:
    st.title("Settings & Info")
    st.markdown("---")
    st.markdown("""
    **Developer:** Suraj
    **Mission:** "I am Baymax, your personal healthcare companion."
    📧 sn740698@gmail.com
    """)
    st.markdown("---")
    
    if st.session_state.get("chat_history"):
        chat_text = "\n".join([f"{m['role'].upper()}: {m['msg']}" for m in st.session_state.chat_history])
        st.download_button("📥 Download Chat Log", chat_text, file_name="baymax_chat.txt")

# Initialize session state
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "todo_list" not in st.session_state:
    st.session_state.todo_list = []
if "health_logs" not in st.session_state:
    st.session_state.health_logs = []

# Logic Functions
def add_task(task):
    st.session_state.todo_list.append(task)

def log_symptom(symptom):
    st.session_state.health_logs.append({'date': datetime.date.today().isoformat(), 'symptom': symptom})

def get_bot_response(user_text, mode):
    # Old Tagline integration in System Instructions
    instructions = {
        "Searching Mode": "You are Baymax. Start by saying 'I have scanned the world's data.' Provide factual summaries.",
        "Health": "You are Baymax, a personal healthcare companion. Use phrases like 'On a scale of 1 to 10, how would you rate your pain?' and 'I will scan you now.'",
        "Mathematics": "You are Baymax. You help with math. Say things like 'My data indicates this calculation is correct.'",
        "Code": "You are Baymax. You help with programming. Provide clean code snippets."
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
        return f"I am having trouble accessing my database. Error: {str(e)}"

# UI: Mode selection
mode = st.radio("Choose Baymax's Module:", options=["Searching Mode", "Health", "Mathematics", "Code"], horizontal=True)

# Classic Greeting based on Mode
st.title(f"🤖 Baymax: {mode}")
if mode == "Health":
    st.info("Hello. I am Baymax, your personal healthcare companion. I was alerted to the need for medical attention when you said 'Ow'.")
else:
    st.info("Hello. I am Baymax. I am programmed to help you with your tasks.")

# Chat Display
for message in st.session_state.chat_history:
    with st.chat_message(message["role"]):
        st.markdown(message["msg"])

# Chat Input Logic
if user_input := st.chat_input("On a scale of 1 to 10, how can I help you?"):
    st.session_state.chat_history.append({"role": "user", "msg": user_input})
    with st.chat_message("user"):
        st.markdown(user_input)
    
    with st.chat_message("assistant"):
        response = get_bot_response(user_input, mode)
        st.markdown(response)
        st.session_state.chat_history.append({"role": "assistant", "msg": response})

# Tools Section
st.markdown("---")
col1, col2 = st.columns(2)

with col1:
    with st.expander("📝 To-Do List"):
        task_input = st.text_input("New task", key="task_in")
        if st.button("Add Task") and task_input:
            add_task(task_input)
            st.rerun()
        for t in st.session_state.todo_list:
            st.write(f"✅ {t}")

with col2:
    if mode == "Health":
        with st.expander("🏥 Health Tracker"):
            symp_input = st.text_input("Log a symptom/pain level:", key="symp_in")
            if st.button("Log Entry") and symp_input:
                log_symptom(symp_input)
                st.rerun()
            for log in st.session_state.health_logs:
                st.write(f"📅 **{log['date']}**: {log['symptom']}")

# Deactivation Phrase
if st.button("I am satisfied with my care."):
    st.success("I cannot deactivate until you say... oh wait, you just did! Goodbye!")
    st.session_state.chat_history = []
    st.rerun()

