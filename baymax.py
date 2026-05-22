import streamlit as st
import datetime
import time
import random
import json
import requests
from openai import OpenAI as DeepSeekClient
import threading
from streamlit.runtime.scriptrunner import add_script_run_ctx

BAYMAX_DIALOGUES = {
    "Searching Mode": [
        "Accessing global knowledge base...",
        "Scanning the web for relevant information...",
        "Cross-referencing historical archives...",
        "Querying search engines...",
        "Fetching the latest data...",
        "Synthesizing information from multiple sources...",
        "Analyzing search results...",
        "Extracting key facts...",
        "Compiling a comprehensive answer...",
        "Reviewing digital encyclopedias...",
        "Processing global data streams...",
        "Gathering context for your query...",
        "Searching academic and public databases...",
        "Evaluating the credibility of sources...",
        "Indexing relevant articles...",
        "Translating complex information...",
        "Summarizing extensive documents...",
        "Connecting data points...",
        "Navigating the information superhighway...",
        "Retrieving requested data..."
    ],
    "Health": [
        "Scanning medical database...",
        "Hello. I am Baymax, your personal healthcare companion.",
        "Analyzing symptoms and protocols...",
        "Accessing global health networks...",
        "On a scale of 1 to 10, how would you rate your pain?",
        "Cross-referencing medical journals...",
        "Checking vital signs parameters...",
        "Evaluating potential treatments...",
        "Reviewing healthcare guidelines...",
        "Consulting digital medical records...",
        "Processing health inquiries...",
        "Scanning neural pathways...",
        "Assessing physical condition metrics...",
        "Retrieving nutritional information...",
        "Analyzing physiological data...",
        "Formulating a wellness plan...",
        "Checking medication interactions...",
        "Reviewing first aid procedures...",
        "Calibrating empathy protocols...",
        "Preparing medical advice..."
    ],
    "Mathematics": [
        "Calculating optimal parameters...",
        "Formulating mathematical equations...",
        "Solving complex algorithms...",
        "Processing numerical data...",
        "Evaluating algebraic expressions...",
        "Computing probabilities...",
        "Analyzing geometric structures...",
        "Deriving calculus theorems...",
        "Cross-referencing mathematical constants...",
        "Running statistical models...",
        "Solving differential equations...",
        "Processing trigonometric functions...",
        "Calculating permutations and combinations...",
        "Evaluating matrix operations...",
        "Computing mathematical limits...",
        "Reviewing number theory...",
        "Factoring polynomials...",
        "Executing arithmetic operations...",
        "Analyzing quantitative data...",
        "Graphing mathematical functions..."
    ],
    "Code": [
        "Initializing programming protocols...",
        "I am ready to compile...",
        "Scanning code repositories...",
        "Debugging syntax errors...",
        "Analyzing algorithm efficiency...",
        "Reviewing software architecture...",
        "Parsing data structures...",
        "Evaluating code logic...",
        "Compiling source code...",
        "Checking for memory leaks...",
        "Accessing developer documentation...",
        "Running unit tests...",
        "Formatting code blocks...",
        "Reviewing object-oriented principles...",
        "Optimizing database queries...",
        "Resolving dependency conflicts...",
        "Deploying virtual environments...",
        "Analyzing backend logic...",
        "Parsing JSON data...",
        "Building application interfaces..."
    ]
}

def cycle_dialogues(status_container, dialogues, stop_event):
    while not stop_event.is_set():
        status_container.update(label=random.choice(dialogues))
        time.sleep(1.2)

# Retrieve API key and Base URL from st.secrets if available, else use defaults
try:
    api_key = st.secrets.get("POLLINATIONS_API_KEY", "sk_PqexpJokRElrFvPFxt5uy5vx8DWWS0ni")
    base_url = st.secrets.get("POLLINATIONS_BASE_URL", "https://gen.pollinations.ai/v1")
except Exception:
    api_key = "sk_PqexpJokRElrFvPFxt5uy5vx8DWWS0ni"
    base_url = "https://gen.pollinations.ai/v1"

# Initialize OpenAI Client (via Pollinations API)
client = DeepSeekClient(
    base_url=base_url,
    api_key=api_key,
    timeout=15.0
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

    /* Hide Streamlit Default UI Elements */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}

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
    
    /* Input Box styling with Futuristic Arc-Reactor Glow */
    .stChatInputContainer {
        border-radius: 30px !important;
        background: rgba(20, 25, 40, 0.7) !important;
        border: 2px solid rgba(56, 189, 248, 0.4) !important;
        box-shadow: 0 0 15px rgba(56, 189, 248, 0.2), inset 0 0 10px rgba(56, 189, 248, 0.1) !important;
        backdrop-filter: blur(20px);
        transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
        animation: pulseGlow 3s infinite alternate;
    }
    @keyframes pulseGlow {
        0% { box-shadow: 0 0 15px rgba(56, 189, 248, 0.2), inset 0 0 10px rgba(56, 189, 248, 0.1); border-color: rgba(56, 189, 248, 0.4); }
        100% { box-shadow: 0 0 35px rgba(129, 140, 248, 0.6), inset 0 0 20px rgba(129, 140, 248, 0.3); border-color: rgba(129, 140, 248, 0.8); }
    }
    .stChatInputContainer:focus-within, .stChatInputContainer:hover {
        transform: translateY(-5px) scale(1.02);
        animation: none;
        box-shadow: 0 0 45px rgba(232, 121, 249, 0.7), inset 0 0 25px rgba(232, 121, 249, 0.3) !important;
        border-color: rgba(232, 121, 249, 0.9) !important;
    }
    [data-testid="stChatInput"] {
        background: transparent !important;
    }
    [data-testid="stChatInputTextArea"] {
        background: transparent !important;
        color: #fff !important;
    }

    /* Send Button Glowing Hover Effect */
    [data-testid="stChatInput"] button {
        transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1) !important;
        border-radius: 50% !important;
    }
    [data-testid="stChatInput"] button:hover {
        background: linear-gradient(135deg, #38bdf8, #e879f9) !important;
        transform: scale(1.2) rotate(15deg) !important;
        box-shadow: 0 0 20px rgba(232, 121, 249, 0.8) !important;
    }
    [data-testid="stChatInput"] button:hover svg {
        fill: white !important;
    }

    /* Awesome Radio Options */
    [data-testid="stRadio"] > div[role="radiogroup"] > label {
        background: rgba(20, 25, 40, 0.6);
        border: 1px solid rgba(255, 255, 255, 0.1);
        border-radius: 25px;
        padding: 8px 15px 8px 10px;
        margin-right: 10px;
        transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
    }
    [data-testid="stRadio"] > div[role="radiogroup"] > label:hover {
        background: rgba(56, 189, 248, 0.15);
        border-color: rgba(56, 189, 248, 0.5);
        box-shadow: 0 0 20px rgba(56, 189, 248, 0.3);
        transform: scale(1.05) translateY(-3px);
    }

    /* Custom style for the small refresh button */
    div[data-testid="stColumn"]:has(.refresh-btn) .stButton > button {
        background: #000000 !important;
        border-radius: 50% !important;
        width: 45px !important;
        height: 45px !important;
        min-height: 0 !important;
        padding: 0 !important;
        display: flex !important;
        align-items: center !important;
        justify-content: center !important;
        box-shadow: 0 5px 15px rgba(0,0,0,0.5) !important;
        transition: all 0.4s cubic-bezier(0.4, 0, 0.2, 1) !important;
        border: none !important;
    }
    div[data-testid="stColumn"]:has(.refresh-btn) .stButton > button::before {
        display: none !important;
    }
    div[data-testid="stColumn"]:has(.refresh-btn) .stButton > button:hover {
        transform: rotate(180deg) scale(1.15) !important;
        background: #222222 !important;
        box-shadow: 0 5px 20px rgba(0,0,0,0.8) !important;
    }
    div[data-testid="stColumn"]:has(.refresh-btn) .stButton > button p {
        margin: 0 !important;
        font-size: 18px !important;
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
    📧 sn740698@gmail.com
    """)
    st.markdown("---")
    
    st.markdown("### System Status")
    cols = st.columns(2)
    
    cols[0].metric(label="System Core", value="Online", delta="Stable", delta_color="normal")
    cols[1].metric(label="Empathy Chip", value="Active", delta="Optimal", delta_color="normal")
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
    instructions = {
        "Searching Mode": "You are Baymax. You must provide highly detailed, extensive, and very large answers that exhaustively cover the topic (including definitions, how it works step-by-step, real-world examples, and interesting facts). However, keep the explanation incredibly simple, friendly, and easy-to-understand (ELI5 style). Organize the response into clear headings, brief bullet points, and short, crisp sentences so that even though the answer is large and thorough, it remains very easy to scan, read, and digest. Use plenty of helpful emojis 🚀🌟 to keep it highly engaging!",
        "Health": "You are Baymax, a personal healthcare companion. Provide very detailed, empathetic, and highly comprehensive medical explanations. Break down your answers so they are very easy to understand, using clear formatting and actionable advice.",
        "Mathematics": "You are Baymax. You help with math. Provide detailed, step-by-step solutions that are very easy to follow.",
        "Code": "You are Baymax. Provide comprehensive, deeply explained programming help. Do not just give short answers. Provide full code snippets and thoroughly explain how they work step-by-step."
    }
    
    if mode == "Searching Mode":
        try:
            from duckduckgo_search import DDGS
            results = DDGS().text(user_text, max_results=5)
            if results:
                user_text += "\n\n[Real-time web search results for context:]\n" + "\n".join([f"- {r.get('title', 'No Title')}: {r.get('body', 'No Body')}" for r in results])
        except Exception:
            pass
            
    openrouter_engines = [
        {"model": "deepseek/deepseek-chat", "key": "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"},
        {"model": "openai/gpt-4o-mini", "key": "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"},
        {"model": "openai/gpt-4o-mini", "key": "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"},
        {"model": "qwen/qwen-2.5-72b-instruct", "key": "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"},
        {"model": "qwen/qwen-2.5-72b-instruct", "key": "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"}
    ]
    
    pollinations_models = ["openai", "gemini", "mistral"]
    last_error = ""
    
    # --- PRIMARY ENGINE: OpenRouter Keys ---
    for engine in openrouter_engines:
        try:
            url = "https://openrouter.ai/api/v1/chat/completions"
            headers = {
                "Authorization": f"Bearer {engine['key']}",
                "Content-Type": "application/json"
            }
            payload = {
                "model": engine["model"],
                "messages": [
                    {"role": "system", "content": instructions.get(mode, "")},
                    {"role": "user", "content": user_text}
                ],
                "stream": True
            }
            with requests.post(url, json=payload, headers=headers, stream=True, timeout=8.0) as response:
                if response.status_code != 200:
                    last_error = f"OR HTTP {response.status_code}: {response.text[:100]}"
                    continue
                
                got_content = False
                for line in response.iter_lines():
                    if line:
                        decoded_line = line.decode('utf-8').strip()
                        if decoded_line.startswith("data: "):
                            data_str = decoded_line[6:]
                            if data_str == "[DONE]":
                                break
                            try:
                                data_json = json.loads(data_str)
                                content = data_json["choices"][0]["delta"].get("content", "")
                                if content:
                                    got_content = True
                                    yield content
                            except Exception:
                                continue
                if got_content:
                    return
                else:
                    last_error = f"OR Model {engine['model']} returned empty response."
                    continue
        except Exception as e:
            last_error = str(e)
            continue
            
    # --- BACKUP ENGINE: Pollinations ---
    active_key = st.session_state.get("pollinations_api_key", "").strip()
    if not active_key:
        try:
            active_key = st.secrets.get("POLLINATIONS_API_KEY", "sk_PqexpJokRElrFvPFxt5uy5vx8DWWS0ni").strip()
        except Exception:
            active_key = "sk_PqexpJokRElrFvPFxt5uy5vx8DWWS0ni"
        
    for fallback_model in pollinations_models:
        for attempt in range(2):
            try:
                # Wrap in with-statement to guarantee the HTTP socket closes instantly!
                if active_key:
                    # Authenticated POST request to the unified Pollinations endpoint
                    url = "https://gen.pollinations.ai/v1/chat/completions"
                    headers = {
                        "Authorization": f"Bearer {active_key}",
                        "Content-Type": "application/json"
                    }
                    payload = {
                        "messages": [
                            {"role": "system", "content": instructions.get(mode, "")},
                            {"role": "user", "content": user_text}
                        ],
                        "model": fallback_model,
                        "stream": True
                    }
                    
                    with requests.post(url, json=payload, headers=headers, stream=True, timeout=8.0) as response:
                        if response.status_code != 200:
                            last_error = f"HTTP {response.status_code}: {response.text[:100]}"
                            if response.status_code in [429, 503] or "queue full" in response.text.lower():
                                time.sleep(random.uniform(1.5, 3.0))
                                continue
                            else:
                                break
                        
                        got_content = False
                        for line in response.iter_lines():
                            if line:
                                decoded_line = line.decode('utf-8').strip()
                                if decoded_line.startswith("data: "):
                                    data_str = decoded_line[6:]
                                    if data_str == "[DONE]":
                                        break
                                    try:
                                        data_json = json.loads(data_str)
                                        content = data_json["choices"][0]["delta"].get("content", "")
                                        if content:
                                            got_content = True
                                            yield content
                                    except Exception:
                                        continue
                        if got_content:
                            return
                        else:
                            last_error = f"Model {fallback_model} returned empty response."
                            break
                else:
                    # Anonymous GET request to the public legacy endpoint
                    import urllib.parse
                    encoded_prompt = urllib.parse.quote(user_text)
                    encoded_system = urllib.parse.quote(instructions.get(mode, ""))
                    url = f"https://text.pollinations.ai/{encoded_prompt}?model={fallback_model}&system={encoded_system}&stream=true"
                    
                    with requests.get(url, stream=True, timeout=8.0) as response:
                        if response.status_code != 200:
                            last_error = f"HTTP {response.status_code}: {response.text[:100]}"
                            if response.status_code in [429, 503] or "queue full" in response.text.lower():
                                time.sleep(random.uniform(1.5, 3.0))
                                continue
                            else:
                                break
                        
                        got_content = False
                        for line in response.iter_lines():
                            if line:
                                decoded_line = line.decode('utf-8').strip()
                                if decoded_line.startswith("data: "):
                                    data_str = decoded_line[6:]
                                    if data_str == "[DONE]":
                                        break
                                    try:
                                        data_json = json.loads(data_str)
                                        content = data_json["choices"][0]["delta"].get("content", "")
                                        if content:
                                            got_content = True
                                            yield content
                                    except Exception:
                                        continue
                        if got_content:
                            return
                        else:
                            last_error = f"Model {fallback_model} returned empty response."
                            break
            except Exception as e:
                last_error = str(e)
                time.sleep(random.uniform(1.5, 3.0))
                continue
            
    # If we get here, all models failed.
    if "429" in last_error or "queue full" in last_error.lower() or "503" in last_error:
        yield "My servers are currently experiencing heavy traffic! 🚦 Please wait a few moments and ask me again.\n\n*Tip: Your API key is successfully integrated! If you see this, please try asking again in a moment.*"
    else:
        yield f"All of my free AI data-cores are currently overwhelmed or offline. Please try asking again in a few minutes! (Last internal error: {last_error[:100]}...)"

# UI: Mode selection
is_gen = st.session_state.get("is_generating", False)
mode = st.radio("Choose Baymax's Module:", options=["Searching Mode", "Health", "Mathematics", "Code"], horizontal=True, disabled=is_gen)

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

# Suggestions based on Mode
all_suggestions = {
    "Searching Mode": [
        "Summarize the latest AI news", "Find facts about quantum physics", "Explain how black holes work",
        "What is the history of the Internet?", "How do airplanes fly?", "Explain the theory of relativity",
        "Who was Leonardo da Vinci?", "How does the stock market work?", "What causes the Northern Lights?",
        "Why is the sky blue?", "Explain the string theory", "How do submarines work?", 
        "What is dark matter?", "Explain the blockchain like I'm 5", "How do solar panels generate electricity?",
        "What are the 7 wonders of the ancient world?", "How do volcanoes erupt?", "What is the speed of light?",
        "How do noise-canceling headphones work?", "What is the Fermi Paradox?", "Explain the placebo effect",
        "How do search engines work?", "Who invented the first computer?", "What are the rings of Saturn made of?",
        "How does a microwave heat food?", "Explain the water cycle", "What is the Mariana Trench?",
        "How do chameleons change color?", "Explain the butterfly effect", "What is CRISPR technology?",
        "How do self-driving cars work?", "What is the tallest mountain in the solar system?", "Explain the greenhouse effect",
        "How do bees make honey?", "What is the origin of the Olympic Games?", "How do batteries store energy?",
        "Explain the concept of parallel universes", "What is the history of tea?"
    ],
    "Health": [
        "I have a headache, what should I do?", "What are good exercises for back pain?", "Explain the symptoms of a cold",
        "How can I improve my sleep quality?", "What are the benefits of drinking water?", "How do I lower my blood pressure naturally?",
        "What is a balanced diet?", "How to reduce stress and anxiety?", "Explain the difference between a virus and a bacteria",
        "What are the best foods for gut health?", "How to prevent eye strain from screens?", "Explain the keto diet",
        "What causes migraines?", "How much protein do I need daily?", "What are the signs of dehydration?",
        "How does the immune system work?", "What is BMI and does it matter?", "How to treat a minor burn?",
        "What are the benefits of intermittent fasting?", "How to tell if a cut is infected?", "What is the best way to treat a fever?",
        "Explain the importance of Vitamin D", "How do vaccines work?", "What are the symptoms of food poisoning?",
        "How to do CPR?", "What are the benefits of meditation?", "How to naturally boost energy levels?",
        "What is the Heimlich maneuver?", "How does caffeine affect the brain?", "What are the stages of sleep?",
        "How to recognize a stroke?", "What causes allergies?", "How to heal a sprained ankle?",
        "What are the benefits of probiotics?", "Explain the role of insulin in the body", "How to establish a healthy morning routine?",
        "What is melatonin and how does it work?", "How to practice mindfulness?"
    ],
    "Mathematics": [
        "Solve for x: 2x + 5 = 15", "Explain the Pythagorean theorem", "Calculate the derivative of x^2",
        "What is the Fibonacci sequence?", "How does probability work?", "Explain the concept of infinity",
        "What are prime numbers?", "How to calculate compound interest?", "Solve the quadratic equation x^2 - 4x + 4 = 0",
        "What is the Golden Ratio?", "Explain Calculus to a beginner", "How to find the area of a circle?",
        "What is a prime factorization?", "Explain the Monty Hall problem", "What is an imaginary number?",
        "How do logarithms work?", "What is standard deviation?", "Explain trigonometry basics",
        "What is a matrix?", "Explain the concept of a limit", "How to calculate percentages?",
        "What is Euler's number (e)?", "Explain the difference between permutations and combinations", "What is a vector?",
        "How to find the volume of a sphere?", "Explain the Riemann Hypothesis simply", "What is a fractal?",
        "How to calculate the mean, median, and mode?", "What is a differential equation?", "Explain the concept of zero",
        "What is a factorial?", "How does binary math work?", "Explain Boolean algebra",
        "What is a tessellation?", "How to calculate permutations?", "Explain the concept of Pi",
        "What is a complex plane?", "How to solve a system of linear equations?"
    ],
    "Code": [
        "Write a python script to scrape a website", "How do I reverse a string in JavaScript?", "Debug this SQL query",
        "Explain Object-Oriented Programming", "What is the difference between React and Angular?", "How to fix a NullPointerException?",
        "Write a REST API in Node.js", "Explain recursion with an example", "What are Docker containers?",
        "How does Git merge work?", "Explain the MVC architecture", "Write a regex to match an email address",
        "What is a memory leak?", "Explain asynchronous programming in Python", "How to setup a virtual environment?",
        "What is CI/CD?", "Explain the time complexity of QuickSort", "How to parse JSON in C++?",
        "What is a hash table?", "Explain the difference between let, const, and var in JS", "How to deploy a React app?",
        "What is a Promise in JavaScript?", "Explain dependency injection", "How to use Git rebase?",
        "What is the difference between SQL and NoSQL?", "Explain the SOLID principles", "How to read a file in Go?",
        "What is a closure in JavaScript?", "Explain the factory design pattern", "How to create a pandas DataFrame?",
        "What is Docker Compose?", "Explain the difference between GET and POST requests", "How to handle exceptions in Java?",
        "What is a web socket?", "Explain the concept of microservices", "How to write a unit test in Python?",
        "What is GraphQL?", "Explain how a load balancer works"
    ]
}

col_title, col_btn = st.columns([0.92, 0.08])
with col_title:
    st.markdown("### 💡 Suggestions")
with col_btn:
    st.markdown("<div class='refresh-btn'></div>", unsafe_allow_html=True)
    if st.button("🔄", disabled=is_gen):
        if "random_suggestions" in st.session_state:
            del st.session_state["random_suggestions"]
        st.rerun()

sug_cols = st.columns(3)

# Pick 3 random suggestions for the current mode
current_mode_suggestions = all_suggestions.get(mode, [])
if "random_suggestions" not in st.session_state or st.session_state.get("suggestion_mode") != mode:
    st.session_state.random_suggestions = random.sample(current_mode_suggestions, 3) if len(current_mode_suggestions) >= 3 else current_mode_suggestions
    st.session_state.suggestion_mode = mode

random_suggestions = st.session_state.random_suggestions

for i, suggestion in enumerate(random_suggestions):
    if sug_cols[i].button(suggestion, key=f"sug_{mode}_{i}", disabled=is_gen):
        st.session_state.chat_history.append({"role": "user", "msg": suggestion})
        st.session_state.is_generating = True
        st.rerun()

# Chat Display
for message in st.session_state.chat_history:
    with st.chat_message(message["role"]):
        st.markdown(message["msg"])

# Chat Input Logic
if user_input := st.chat_input("On a scale of 1 to 10, how can I help you?", disabled=is_gen):
    st.session_state.chat_history.append({"role": "user", "msg": user_input})
    st.session_state.is_generating = True
    st.rerun()

if st.session_state.get("is_generating", False) and st.session_state.chat_history and st.session_state.chat_history[-1]["role"] == "user":
    last_msg = st.session_state.chat_history[-1]["msg"]
    with st.chat_message("assistant"):
        stop_event = threading.Event()
        dialogues = BAYMAX_DIALOGUES.get(mode, ["Processing..."])
        status = st.status(random.choice(dialogues))
        t = threading.Thread(target=cycle_dialogues, args=(status, dialogues, stop_event))
        add_script_run_ctx(t)
        t.start()
        
        stream = get_bot_response(last_msg, mode)
        response_text = st.write_stream(stream)
        
        stop_event.set()
        t.join()
        status.update(label="Analysis complete.", state="complete", expanded=False)
    st.session_state.chat_history.append({"role": "assistant", "msg": response_text})
    st.session_state.is_generating = False
    st.rerun()

# Tools Section
# Deactivation Phrase
st.markdown("<br><br>", unsafe_allow_html=True)
st.markdown("""
<style>
    /* Make the Deactivation Button a cool glowing red gradient */
    [data-testid="stBaseButton-primary"] {
        background: linear-gradient(45deg, #ff0f7b, #f89b29) !important;
        border: none !important;
        box-shadow: 0 5px 25px rgba(255, 15, 123, 0.4) !important;
        color: white !important;
        font-weight: 800 !important;
        letter-spacing: 1px !important;
        transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1) !important;
    }
    [data-testid="stBaseButton-primary"]:hover {
        background: linear-gradient(45deg, #f89b29, #ff0f7b) !important;
        box-shadow: 0 10px 35px rgba(255, 15, 123, 0.7) !important;
        transform: scale(1.03) translateY(-3px) !important;
    }
    [data-testid="stBaseButton-primary"]::before {
        display: none !important; /* hide the previous blue hover pseudo-element */
    }
</style>
""", unsafe_allow_html=True)
if st.button("I am satisfied with my care.", type="primary", use_container_width=True, disabled=is_gen):
    st.toast('Initiating core system shutdown...', icon='⚙️')
    with st.status("Deactivating Baymax Modules...", expanded=True) as status:
        st.write("Terminating neural links...")
        time.sleep(0.5)
        st.write("Flushing memory buffers...")
        time.sleep(0.5)
        st.write("Powering down optics...")
        time.sleep(0.5)
        status.update(label="Deactivation Complete. Have a productive day.", state="complete", expanded=False)
    time.sleep(1)
    st.session_state.chat_history = []
    st.rerun()
