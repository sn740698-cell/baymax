import streamlit as st
import pandas as pd
import wikipedia
import re
import datetime
from collections import defaultdict
import google.generativeai as genai
from google.generativeai.types import HarmCategory, HarmBlockThreshold

# Gemini API configuration
GEMINI_API_KEY = "AIzaSyBhBl-GjVq9ulWZ_aHLOevWNmxYjxwFrHs"
genai.configure(api_key=GEMINI_API_KEY)
model = genai.GenerativeModel('gemini-2.5-pro-exp-03-25')

# Page config
st.set_page_config(page_title="🤖 Baymax Assistant", layout="centered")

# Sidebar
with st.sidebar:
    st.markdown("### About Us")
    st.markdown("""
    **Suraj**

    AI assistant project.

    📧 sn740698@gmail.com
    """)

# Load CSV
@st.cache_data
def load_responses():
    try:
        return pd.read_csv("conversational_responses.csv")
    except:
        data = {
            "Query": ["hello", "hi"],
            "Response": [
                "Hello! I'm Baymax. How can I help?",
                "Hi! What would you like to know?"
            ]
        }
        return pd.DataFrame(data)

responses = load_responses()

# Medical dictionary
medical_symptoms = {
    "fever": "Stay hydrated and take rest. Consult doctor if high fever.",
    "cough": "Drink warm fluids. Persistent cough requires doctor visit.",
    "headache": "Rest and hydrate. Severe headache requires medical attention.",
}

# Session states
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

if "edit_index" not in st.session_state:
    st.session_state.edit_index = None

if "read_more_states" not in st.session_state:
    st.session_state.read_more_states = {}

if "wiki_search_history" not in st.session_state:
    st.session_state.wiki_search_history = []

if "todo_list" not in st.session_state:
    st.session_state.todo_list = []

if "health_logs" not in st.session_state:
    st.session_state.health_logs = []

if "related_topics" not in st.session_state:
    st.session_state.related_topics = defaultdict(list)

# Math check
def is_math_expression(text):
    return re.match(r'^[\d\s\.\+\-\*\/\^\(\)]+$', text.strip()) is not None

# Simple math
def evaluate_math_expression(expr):
    try:
        result = eval(expr)
        return f"The result is: {result}"
    except:
        return "Invalid math expression."

# Gemini response
def get_gemini_response(prompt, mode):

    safety_settings = {
        HarmCategory.HARM_CATEGORY_HARASSMENT: HarmBlockThreshold.BLOCK_MEDIUM_AND_ABOVE,
        HarmCategory.HARM_CATEGORY_HATE_SPEECH: HarmBlockThreshold.BLOCK_MEDIUM_AND_ABOVE,
    }

    system_prompt = f"""
    You are Baymax AI assistant.

    Mode: {mode}

    User: {prompt}
    """

    try:
        response = model.generate_content(system_prompt)
        return response.text
    except:
        return "AI connection error."

# Wikipedia
def get_wikipedia_info(topic):

    try:
        page = wikipedia.page(topic)
        summary = wikipedia.summary(topic, sentences=5)
        return summary, None, page.url

    except wikipedia.exceptions.DisambiguationError as e:
        return f"Be more specific. Options: {e.options[:5]}", None, None

    except:
        return "Topic not found.", None, None


# Bot response
def get_bot_response(user_text, mode):

    text = user_text.lower()

    if mode == "Wikipedia":
        return get_wikipedia_info(user_text)

    if mode == "Health":
        if text in medical_symptoms:
            return medical_symptoms[text], None, None

    if mode == "Mathematics":
        return evaluate_math_expression(user_text), None, None

    if is_math_expression(user_text):
        return evaluate_math_expression(user_text), None, None

    match = responses[responses["Query"].str.lower() == text]

    if not match.empty:
        return match.iloc[0]["Response"], None, None

    return get_gemini_response(user_text, mode), None, None


# Add message
def add_message(user_msg, mode):

    st.session_state.chat_history.append({
        "role": "user",
        "msg": user_msg
    })

    resp, img, url = get_bot_response(user_msg, mode)

    st.session_state.chat_history.append({
        "role": "bot",
        "msg": resp,
        "image": img,
        "wiki_url": url
    })


# Mode select
mode = st.radio(
    "Mode",
    ["Wikipedia", "Health", "Mathematics"],
    horizontal=True
)

st.title("🤖 Baymax Assistant")

# Input
user_text = st.text_input("Ask something")

if st.button("Send"):

    if user_text:
        add_message(user_text, mode)


# Clear
if st.button("Clear Chat"):
    st.session_state.chat_history = []
    st.rerun()


# Display chat
for idx, message in enumerate(st.session_state.chat_history):

    raw_msg = message["msg"]
    safe_msg = str(raw_msg).replace('"', '&quot;')

    if message["role"] == "user":

        cols = st.columns([8,1])

        with cols[0]:
            st.markdown(f"🧑 {raw_msg}")

        with cols[1]:

            st.markdown(
                f"""
                <button onclick="navigator.clipboard.writeText('{safe_msg}')">
                📋
                </button>
                """,
                unsafe_allow_html=True
            )

    else:

        cols = st.columns([8,1])

        with cols[0]:
            st.markdown(f"**Baymax:** {raw_msg}")

        with cols[1]:

            st.markdown(
                f"""
                <button onclick="navigator.clipboard.writeText('{safe_msg}')">
                📋
                </button>
                """,
                unsafe_allow_html=True
            )


# To-Do
with st.expander("📝 To-Do List"):

    task = st.text_input("Add task")

    if st.button("Add"):

        if task:
            st.session_state.todo_list.append(task)

    st.write(st.session_state.todo_list)


# Health log
if mode == "Health":

    with st.expander("🏥 Health Log"):

        sym = st.text_input("Log symptom")

        if st.button("Save symptom"):

            if sym:

                st.session_state.health_logs.append({
                    "date": datetime.date.today(),
                    "symptom": sym
                })

        st.write(st.session_state.health_logs)


# Footer
st.markdown(
"""
<style>
footer {visibility:hidden;}
</style>

<div style="text-align:center;color:gray;margin-top:40px">
Baymax Assistant • Streamlit
</div>
""",
unsafe_allow_html=True
)

