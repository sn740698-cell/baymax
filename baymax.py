import streamlit as st
import datetime
from google import genai
from google.genai import types

# Initialize Gemini Client
# Note: Ensure your API key is kept secure!
client = genai.Client(api_key="AIzaSyDEHCrbpbH6CLYrmK3B9x3xDn4SDmPaHkE")

# Page config
st.set_page_config(page_title="🤖 Baymax Assistant", layout="centered")

# Sidebar information
with st.sidebar:
    st.markdown("### About Us")
    st.markdown("""
    **Suraj**
    
    We are dedicated to building helpful AI-powered chat assistants to make your life better.
    📧 sn740698@gmail.com suraj ( 9206881748 )
    """)
    st.markdown("---")
    st.markdown("""
    ### User Guidance
    
    - **Searching Mode:** Explore the world's knowledge with factual summaries.
    - **Health:** Ask about medical symptoms like "fever" or "trouble sleeping".
    - **Mathematics:** Solve complex math expressions or word problems.
    - **Code:** Generate, debug, or explain programming code.
    - **Tools:** Use the To-Do list and Health Tracker to log your day.
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

# To-Do List functions
def add_task(task):
    st.session_state.todo_list.append(task)

def list_tasks():
    if st.session_state.todo_list:
        return "\n".join(f"- {t}" for t in st.session_state.todo_list)
    return "Your to-do list is empty."

# Health tracker functions
def log_symptom(symptom):
    entry = {'date': datetime.date.today().isoformat(), 'symptom': symptom}
    st.session_state.health_logs.append(entry)

def display_health_logs():
    if not st.session_state.health_logs:
        return "No symptoms logged yet."
    return "Symptom Logs:\n" + "".join(f"{log['date']}: {log['symptom']}\n" for log in st.session_state.health_logs)

# Generate response using Gemini API
def get_bot_response(user_text, mode):
    # Set the AI's personality based on the selected mode
    system_instruction = ""
    if mode == "Searching Mode":
        system_instruction = "You are Baymax, a highly knowledgeable encyclopedia assistant. Provide accurate, detailed, and factual summaries similar to Wikipedia articles."
    elif mode == "Health":
        system_instruction = "You are Baymax, a personal healthcare companion. Provide helpful, empathetic health information. Always remind the user to consult a real doctor for serious issues."
    elif mode == "Mathematics":
        system_instruction = "You are Baymax, a math assistant. Evaluate mathematical expressions and solve problems step-by-step."
    elif mode == "Code":
        system_instruction = "You are Baymax, an expert programmer. Provide clean, efficient code snippets, explain how they work, and help debug if asked."

    try:
        response = client.models.generate_content(
            model='gemini-2.0-flash', # Updated to 2.0 Flash as per standard capabilities
            contents=user_text,
            config=types.GenerateContentConfig(
                system_instruction=system_instruction,
                temperature=0.7,
            )
        )
        return response.text
    except Exception as e:
        return f"Oops! I had trouble connecting to my AI brain. Error: {str(e)}"

# Chat message handling
def add_message(user_msg, mode):
    st.session_state.chat_history.append({"role": "user", "msg": user_msg})
    bot_resp = get_bot_response(user_msg, mode)
    st.session_state.chat_history.append({"role": "bot", "msg": bot_resp})

def update_message(pos, new_msg, mode):
    st.session_state.chat_history[pos]["msg"] = new_msg
    bot_pos = pos + 1
    bot_resp = get_bot_response(new_msg, mode)
    
    if bot_pos < len(st.session_state.chat_history) and st.session_state.chat_history[bot_pos]["role"] == "bot":
        st.session_state.chat_history[bot_pos].update({"msg": bot_resp})
    else:
        st.session_state.chat_history.insert(bot_pos, {"role": "bot", "msg": bot_resp})

# Mode selection UI
mode = st.radio("Select Mode", options=["Searching Mode", "Health", "Mathematics", "Code"], index=0, horizontal=True)

# Header and intro for each mode
if mode == "Searching Mode":
    st.title("🌐 Baymax: Searching Mode")
    st.subheader("Your Gateway to Global Knowledge")
    
    # Sliding-style image gallery
    cols = st.columns(4)
    images = [
        "https://images.unsplash.com/photo-1507413245164-6160d8298b31?w=400", # Science
        "https://images.unsplash.com/photo-1451187580459-43490279c0fa?w=400", # Tech
        "https://images.unsplash.com/photo-1447069387593-a5de0862481e?w=400", # History
        "https://images.unsplash.com/photo-1516321318423-f06f85e504b3?w=400"  # Info
    ]
    captions = ["Science", "Space", "History", "Digital"]
    for i, col in enumerate(cols):
        col.image(images[i], caption=captions[i], use_container_width=True)
    
    st.markdown("### Hello! I'm Baymax. What would you like to discover today?")

elif mode == "Health":
    st.title("🏥 Baymax: Healthcare Companion")
    st.subheader("Personalized care for a healthier you")
    st.markdown("### Hello! I'm Baymax. How are you feeling today?")

elif mode == "Mathematics":
    st.title("🔢 Baymax: Mathematics Assistant")
    st.subheader("Solving the world, one equation at a time")
    st.markdown("### Hello! I'm Baymax. Enter a mathematical problem:")

elif mode == "Code":
    st.title("💻 Baymax: Code Assistant")
    st.subheader("Ready to write the code")
    st.markdown("### Hello! I'm Baymax. What are we building today?")

# Input form for user queries
with st.form("input_form", clear_on_submit=True):
    if st.session_state.edit_index is None:
        user_text_input = st.text_input("Ask me anything:", key="input_text")
    else:
        user_text_input = st.text_input(
            "Edit your message:",
            value=st.session_state.chat_history[st.session_state.edit_index]["msg"],
            key="input_text"
        )
    submitted = st.form_submit_button("Send")

# On form submission
if submitted and user_text_input.strip():
    if st.session_state.edit_index is None:
        add_message(user_text_input.strip(), mode)
    else:
        update_message(st.session_state.edit_index, user_text_input.strip(), mode)
        st.session_state.edit_index = None
    st.rerun()

# Button to clear chat
if st.button("Clear Chat"):
    st.session_state.chat_history.clear()
    st.session_state.edit_index = None
    st.rerun()

# Display chat history
for idx, message in reversed(list(enumerate(st.session_state.chat_history))):
    raw_msg = message["msg"]
    msg_text = raw_msg.replace('"', '&quot;').replace('\n', '\\n') if isinstance(raw_msg, str) else str(raw_msg)
    
    if message["role"] == "user":
        cols = st.columns([0.7, 0.15, 0.15])
        with cols[0]:
            st.markdown(f"🧑‍💻 **You:** {raw_msg}")
        with cols[1]:
            if st.button("✏️ Edit", key=f"edit_{idx}"):
                st.session_state.edit_index = idx
                st.rerun()
        with cols[2]:
            st.markdown(f'<button onclick="navigator.clipboard.writeText(\'{msg_text}\')">📋 Copy</button>', unsafe_allow_html=True)
    else:
        cols = st.columns([0.85, 0.15])
        with cols[0]:
            st.markdown(f"**Baymax:**\n\n{raw_msg}")
        with cols[1]:
            st.markdown(f'<button onclick="navigator.clipboard.writeText(\'{msg_text}\')">📋 Copy</button>', unsafe_allow_html=True)

st.markdown("---")

# To-Do List expander
with st.expander("📝 To-Do List"):
    new_task = st.text_input("Add a task to your list:", key="todo_task")
    if st.button("Add Task"):
        if new_task.strip():
            add_task(new_task.strip())
            st.success(f"Added task: {new_task}")
    st.text("Current tasks:")
    st.text(list_tasks())

# Health Tracker expander (Only shows in Health mode)
if mode == "Health":
    with st.expander("🏥 Health Tracker"):
        symptom_input = st.text_input("Log a symptom:", key="health_symptom")
        if st.button("Log Symptom"):
            if symptom_input.strip():
                log_symptom(symptom_input.strip())
                st.success(f"Logged symptom: {symptom_input}")
        st.text(display_health_logs())

# Footer
st.markdown("""
    <style>
    footer {visibility: hidden;}
    </style>
    <div style="margin-top: 50px; text-align: center; color: gray;">
        Powered by Google Gemini AI & Streamlit • Baymax Assistant v3.0
    </div>
""", unsafe_allow_html=True)
