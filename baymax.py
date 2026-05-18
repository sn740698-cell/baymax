import streamlit as st
import datetime
import time
from openai import OpenAI

# Initialize OpenAI Client (configured for OpenRouter API base URL since the key is an OpenRouter key)
client = OpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key="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",
    default_headers={
        "HTTP-Referer": "http://localhost:8501", # Required by OpenRouter
        "X-Title": "Baymax Assistant" # Required by OpenRouter
    }
)

# Page config
st.set_page_config(page_title="🤖 Baymax Assistant", layout="centered")

# --- CUSTOM CSS UI/UX OVERHAUL ---
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;600;800&display=swap');

    html, body, [class*="css"]  {
        font-family: 'Outfit', sans-serif;
    }

    /* Animated Dynamic Background with Floating Orbs */
    .stApp {
        background-color: #0b1120;
        background-image: 
            radial-gradient(at 0% 0%, hsla(253,16%,7%,1) 0, transparent 50%), 
            radial-gradient(at 50% 0%, hsla(225,39%,30%,0.2) 0, transparent 50%), 
            radial-gradient(at 100% 0%, hsla(339,49%,30%,0.2) 0, transparent 50%);
        color: #f8fafc;
        overflow-x: hidden;
    }
    
    /* Animated Orbs using pseudo-elements */
    .stApp::before {
        content: '';
        position: fixed;
        top: -100px;
        left: -100px;
        width: 300px;
        height: 300px;
        background: #38bdf8;
        filter: blur(100px);
        opacity: 0.3;
        border-radius: 50%;
        animation: float 8s ease-in-out infinite alternate;
        z-index: -1;
    }
    .stApp::after {
        content: '';
        position: fixed;
        bottom: -100px;
        right: -100px;
        width: 400px;
        height: 400px;
        background: #818cf8;
        filter: blur(120px);
        opacity: 0.3;
        border-radius: 50%;
        animation: float 12s ease-in-out infinite alternate-reverse;
        z-index: -1;
    }

    /* Sidebar Ultimate Glassmorphism */
    [data-testid="stSidebar"] {
        background: rgba(11, 17, 32, 0.4) !important;
        backdrop-filter: blur(16px);
        -webkit-backdrop-filter: blur(16px);
        border-right: 1px solid rgba(255, 255, 255, 0.05);
        box-shadow: 5px 0 20px rgba(0,0,0,0.3);
    }
    
    /* Input Box styling with Hover Glow */
    .stChatInputContainer {
        border-radius: 30px !important;
        background: rgba(255,255,255,0.03) !important;
        border: 1px solid rgba(255,255,255,0.1) !important;
        backdrop-filter: blur(20px);
        transition: all 0.3s ease-in-out;
    }
    .stChatInputContainer:focus-within, .stChatInputContainer:hover {
        border-color: rgba(56, 189, 248, 0.6) !important;
        box-shadow: 0 0 20px rgba(56, 189, 248, 0.2) !important;
        transform: translateY(-2px);
    }

    /* Chat Messages hover scaling and borders */
    [data-testid="stChatMessage"] {
        background: rgba(255,255,255,0.02);
        border: 1px solid rgba(255,255,255,0.05);
        border-radius: 20px;
        padding: 15px;
        margin-bottom: 12px;
        backdrop-filter: blur(10px);
        animation: slideUp 0.5s cubic-bezier(0.2, 0.8, 0.2, 1);
        transition: all 0.3s ease;
    }
    [data-testid="stChatMessage"]:hover {
        background: rgba(255,255,255,0.05);
        border-color: rgba(129, 140, 248, 0.4);
        transform: scale(1.02) translateY(-2px);
        box-shadow: 0 10px 25px rgba(0,0,0,0.2);
    }
    
    /* Crazy Glowing Title */
    h1 {
        background: linear-gradient(120deg, #38bdf8, #818cf8, #e879f9);
        background-size: 200% 200%;
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-weight: 800;
        text-shadow: 0px 4px 30px rgba(129, 140, 248, 0.4);
        animation: gradientShift 5s ease infinite;
        letter-spacing: -1px;
    }

    /* Expanders & Tools */
    [data-testid="stExpander"] {
        background: rgba(255,255,255,0.02) !important;
        border: 1px solid rgba(255,255,255,0.05) !important;
        border-radius: 20px !important;
        transition: all 0.3s ease;
    }
    [data-testid="stExpander"]:hover {
        border-color: rgba(56, 189, 248, 0.5) !important;
        box-shadow: 0 5px 20px rgba(56, 189, 248, 0.15);
    }

    /* Magic Button hover effects */
    .stButton>button {
        border-radius: 25px;
        background: linear-gradient(45deg, #38bdf8, #818cf8);
        color: white;
        border: none;
        font-weight: 600;
        letter-spacing: 0.5px;
        transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
        position: relative;
        overflow: hidden;
        z-index: 1;
    }
    .stButton>button::before {
        content: '';
        position: absolute;
        top: 0; left: -100%;
        width: 100%; height: 100%;
        background: linear-gradient(90deg, transparent, rgba(255,255,255,0.4), transparent);
        transition: all 0.5s ease;
        z-index: -1;
    }
    .stButton>button:hover::before {
        left: 100%;
    }
    .stButton>button:hover {
        transform: translateY(-4px) scale(1.05);
        box-shadow: 0 10px 25px rgba(56, 189, 248, 0.5);
        color: white !important;
    }

    /* Animations */
    @keyframes slideUp {
        from { opacity: 0; transform: translateY(30px) scale(0.95); }
        to { opacity: 1; transform: translateY(0) scale(1); }
    }
    @keyframes float {
        0% { transform: translateY(0px) translateX(0px); }
        50% { transform: translateY(20px) translateX(20px); }
        100% { transform: translateY(0px) translateX(0px); }
    }
    @keyframes gradientShift {
        0% { background-position: 0% 50%; }
        50% { background-position: 100% 50%; }
        100% { background-position: 0% 50%; }
    }
</style>
""", unsafe_allow_html=True)

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
    
    st.markdown("### System Status")
    cols = st.columns(2)
    cols[0].metric(label="Battery", value="100%", delta="-0.01%")
    cols[1].metric(label="Empathy Chip", value="Active", delta="Optimal", delta_color="normal")
    st.progress(100, text="Database Connection")
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
        response = client.chat.completions.create(
            model="openai/gpt-4o",
            messages=[
                {"role": "system", "content": instructions.get(mode, "")},
                {"role": "user", "content": user_text}
            ],
            temperature=0.7,
            stream=True
        )
        for chunk in response:
            if chunk.choices[0].delta.content is not None:
                yield chunk.choices[0].delta.content
    except Exception as e:
        yield f"I am having trouble accessing my database. Error: {str(e)}"

# UI: Mode selection
mode = st.radio("Choose Baymax's Module:", options=["Searching Mode", "Health", "Mathematics", "Code"], horizontal=True)

# Creative Mode Icons and Effects
mode_icons = {
    "Searching Mode": "🔍",
    "Health": "❤️",
    "Mathematics": "📐",
    "Code": "💻"
}
icon = mode_icons.get(mode, "🤖")
st.title(f"{icon} Baymax: {mode}")

if mode == "Health":
    st.info("Hello. I am Baymax, your personal healthcare companion. I was alerted to the need for medical attention when you said 'Ow'.")
elif mode == "Code":
    st.info("Initializing programming protocols. I am ready to compile.")
elif mode == "Mathematics":
    st.info("Calculating optimal parameters. How may I assist your equations?")
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
        stream = get_bot_response(user_input, mode)
        response_text = st.write_stream(stream)
        st.session_state.chat_history.append({"role": "assistant", "msg": response_text})

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
    st.balloons()
    st.success("I cannot deactivate until you say... oh wait, you just did! Goodbye!")
    time.sleep(2)
    st.session_state.chat_history = []
    st.rerun()
