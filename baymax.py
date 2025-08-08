import streamlit as st
import pandas as pd
import wikipedia
import re
import datetime
from collections import defaultdict

# Page config
st.set_page_config(page_title="🤖 Baymax Assistant", layout="centered")

# Sidebar information
with st.sidebar:
    st.markdown("### About Us")
    st.markdown("""
    **Suraj and Team**
    
    We are dedicated to building helpful AI-powered chat assistants to make your life better.
    📧 sn740698@gmail.com
    """)
    st.markdown("---")
    st.markdown("""
    ### User Guidance
    
    - Medical Symptom Queries like "fever", "feeling sad", "trouble sleeping"
    - Math expressions like "12 + 36 / 4"
    - Wikipedia topics like "Albert Einstein"
    - Emotional inputs like "I feel sad"
    - To-Do list commands
    - Health Tracker symptom logs
    """)

# Load conversational responses CSV safely
@st.cache_data
def load_responses():
    try:
        return pd.read_csv(
            "conversational_responses.csv",
            on_bad_lines='skip',
            quotechar='"',
            escapechar='\\',
            encoding='utf-8'
        )
    except Exception as e:
        st.warning(f"Couldn't load responses: {str(e)}. Using default responses.")
        default_data = {
            "Query": ["hello", "hi", "how are you", "help"],
            "Response": [
                "Hello! I'm Baymax. How can I assist you today?",
                "Hi there! What would you like to know?",
                "I'm functioning normally. How about you?",
                "I can help with: 1) Medical info 2) Math calculations 3) Wikipedia searches"
            ]
        }
        return pd.DataFrame(default_data)

responses = load_responses()

# Medical symptoms dictionary (unchanged)
medical_symptoms = {
    "fever": "Hydrate well and monitor. Paracetamol can help. High or persistent fever: see your doctor.",
    "cough": "Warm drinks and cough syrup may soothe. Cough >2 weeks or with blood: contact a doctor.",
    "headache": "Rest in a quiet, dark room. Over-the-counter pain relievers may help. Severe or sudden headache requires medical attention.",
    "sore throat": "Gargle with warm salt water. Stay hydrated. If severe or lasts >1 week, see a doctor.",
    "runny nose": "Antihistamines may help if allergies. Nasal saline sprays can relieve congestion.",
    "nausea": "Sip clear fluids. Ginger may help. If persistent or with vomiting, seek medical care.",
    "vomiting": "Small sips of clear fluids. If lasts >24hrs or contains blood, see a doctor immediately.",
    "diarrhea": "Stay hydrated with electrolyte solutions. BRAT diet (bananas, rice, applesauce, toast). If severe or bloody, seek care.",
    "constipation": "Increase fiber and water intake. Gentle exercise can help. Laxatives short-term if needed.",
    "fatigue": "Ensure adequate sleep. Check for anemia or thyroid issues if persistent.",
    "dizziness": "Sit or lie down immediately. Can indicate low blood pressure or other conditions if frequent.",
    "chest pain": "EMERGENCY - seek immediate medical attention as this could indicate heart problems.",
    "shortness of breath": "Can be serious. If sudden or severe, seek emergency care immediately.",
    "abdominal pain": "Location matters. Appendicitis pain starts near belly button. Persistent pain needs evaluation.",
    "back pain": "Rest, heat/ice therapy. If severe or with leg weakness/numbness, see a doctor.",
    "muscle pain": "Rest, gentle stretching. Over-the-counter pain relievers may help.",
    "joint pain": "RICE method (Rest, Ice, Compression, Elevation). Persistent pain may need evaluation.",
    "rash": "Keep area clean and dry. Antihistamines for itching. Spreading or painful rashes need medical review.",
    "itchiness": "Moisturize skin. Antihistamines may help. If widespread or persistent, see a doctor.",
    "swelling": "Elevate affected area. Can indicate allergy or other conditions if sudden or severe.",
    "numbness": "Can indicate nerve issues. If sudden or one-sided, seek immediate care (possible stroke).",
    "tingling": "Often circulation or nerve related. Persistent symptoms need evaluation.",
    "seizures": "EMERGENCY - protect person from injury and call for help immediately.",
    "tremors": "Can be benign or indicate neurological conditions. Persistent tremors need evaluation.",
    "memory problems": "Note if sudden (needs immediate care) or gradual (schedule doctor visit).",
    "eye pain": "Sudden severe pain needs immediate care. Mild irritation may be allergies.",
    "blurred vision": "Sudden changes require emergency care. Gradual changes need eye exam.",
    "red eyes": "May be conjunctivitis. Avoid rubbing. See doctor if painful or vision affected.",
    "ear pain": "Warm compress may help. Persistent pain may indicate infection needing antibiotics.",
    "hearing loss": "Sudden hearing loss is an emergency. Gradual loss needs hearing evaluation.",
    "sinus pressure": "Steam inhalation may help. Decongestants short-term. Persistent >10 days needs evaluation.",
    "palpitations": "Note triggers. If with chest pain/shortness of breath, seek emergency care.",
    "high blood pressure": "Needs medical management. Reduce salt, exercise, manage stress.",
    "low blood pressure": "Increase fluids. Stand up slowly. If with dizziness/weakness, see doctor.",
    "wheezing": "Can indicate asthma or allergy. First-time wheezing needs medical evaluation.",
    "congestion": "Steam, saline sprays, decongestants short-term. Persistent congestion needs evaluation.",
    "heartburn": "Avoid spicy/fatty foods. Antacids may help. Frequent heartburn needs evaluation.",
    "indigestion": "Eat smaller meals. Avoid lying down after eating. Persistent symptoms need checkup.",
    "loss of appetite": "Note duration. With weight loss, needs medical evaluation.",
    "painful urination": "May indicate UTI. Increase fluids. If persists >1 day, see doctor.",
    "frequent urination": "Can indicate diabetes or UTI. Note if with thirst/weight changes.",
    "blood in urine": "Always needs medical evaluation - can indicate infection or other conditions.",
    "anxiety": "Deep breathing exercises. Persistent anxiety benefits from professional help.",
    "depression": "Reach out for support. Professional help is available and effective.",
    "insomnia": "Good sleep hygiene. Avoid screens before bed. Chronic insomnia needs evaluation.",
    "stress": "Regular exercise, relaxation techniques. Chronic stress affects health.",
    "menstrual cramps": "Heat pad, gentle exercise. Pain relievers may help. Severe pain needs evaluation.",
    "breast pain": "Often hormonal. New/lump-related pain needs medical evaluation.",
    "unexplained weight loss": "Always needs medical evaluation to determine cause.",
    "excessive thirst": "Can indicate diabetes. Note if with frequent urination.",
    "bruising easily": "May indicate blood disorder. Needs medical evaluation.",
    "swollen lymph nodes": "Persistent swelling (>2 weeks) needs medical checkup.",
    "ear pulling (in babies)": "May indicate ear infection. Check for fever, see pediatrician.",
    "excessive crying": "Check for fever, hunger, discomfort. Persistent crying needs evaluation.",
    "fever in infants": "Rectal temp >100.4°F (38°C) in babies <3 months needs immediate care.",
    "burn": "Cool with running water. Don't use ice. Cover loosely. Severe burns need emergency care.",
    "cut": "Clean with water, apply pressure to stop bleeding. Deep/large wounds need medical attention.",
    "sprain": "RICE method (Rest, Ice, Compression, Elevation). Severe pain/swelling needs evaluation.",
    "loss of taste/smell": "Common with COVID-19. Self-isolate and get tested.",
    "covid symptoms": "Fever, cough, fatigue most common. Isolate and get tested.",
    "hives": "Antihistamines may help. If with swelling/difficulty breathing - EMERGENCY.",
    "allergic reaction": "Mild: antihistamines. Severe (difficulty breathing): use epinephrine if available and call emergency services.",
    "diabetes symptoms": "Excessive thirst, urination, hunger. Unexplained weight loss. Needs medical evaluation.",
    "asthma attack": "Use rescue inhaler. If no improvement or severe distress, seek emergency care.",
    "migraine": "Rest in dark, quiet room. Hydrate. Preventive meds available for frequent migraines."
}

# Initialize session state
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

# Math expression check
def is_math_expression(text):
    return re.match(r'^[\d\s\.\+\-\*\/\^\(\)]+$', text.strip()) is not None

# Math parser class
class Parser:
    def __init__(self, expression):
        self.tokens = self.tokenize(expression)
        self.idx = -1
        self.current_token = None
        self.advance()

    def advance(self):
        self.idx += 1
        self.current_token = self.tokens[self.idx] if self.idx < len(self.tokens) else None
        return self.current_token

    def tokenize(self, expression):
        tokens = []
        i = 0
        while i < len(expression):
            c = expression[i]
            if c == ' ':
                i += 1
                continue
            if c in '()+-*/^':
                tokens.append(c)
                i += 1
            elif c.isdigit() or c == '.':
                num_str = ''
                dot_count = 0
                while i < len(expression) and (expression[i].isdigit() or expression[i] == '.'):
                    if expression[i] == '.':
                        dot_count += 1
                        if dot_count > 1:
                            raise ValueError("Invalid number with multiple decimals")
                    num_str += expression[i]
                    i += 1
                tokens.append(num_str)
            else:
                raise ValueError(f"Invalid character in expression: '{c}'")
        return tokens

    def parse(self):
        result = self.expr()
        if self.current_token is not None:
            raise ValueError(f"Unexpected token after expression: {self.current_token}")
        return result

    def expr(self):
        result = self.term()
        while self.current_token in ('+', '-'):
            if self.current_token == '+':
                self.advance()
                result += self.term()
            elif self.current_token == '-':
                self.advance()
                result -= self.term()
        return result

    def term(self):
        result = self.factor()
        while self.current_token in ('*', '/'):
            if self.current_token == '*':
                self.advance()
                result *= self.factor()
            elif self.current_token == '/':
                self.advance()
                denominator = self.factor()
                if denominator == 0:
                    raise ZeroDivisionError("Division by zero")
                result /= denominator
        return result

    def factor(self):
        result = self.power()
        while self.current_token == '^':
            self.advance()
            result **= self.factor()
        return result

    def power(self):
        if self.current_token == '(':
            self.advance()
            result = self.expr()
            if self.current_token != ')':
                raise ValueError("Mismatched parentheses")
            self.advance()
            return result
        elif self.current_token == '-':
            self.advance()
            return -self.power()
        elif self.current_token is not None and self.is_number(self.current_token):
            num = float(self.current_token)
            self.advance()
            return num
        else:
            raise ValueError(f"Unexpected token: {self.current_token}")

    def is_number(self, token):
        try:
            float(token)
            return True
        except ValueError:
            return False

def evaluate_math_expression(expr):
    try:
        expr_str = expr.strip()
        if not expr_str:
            return "Please enter a mathematical expression."
        if not is_math_expression(expr_str):
            return "Expression contains invalid characters."
        parser = Parser(expr_str)
        result = parser.parse()
        if isinstance(result, (int, float)):
            if isinstance(result, float) and result.is_integer():
                return f"The result is: {int(result)}"
            else:
                return f"The result is: {round(result,4)}"
        return "Could not evaluate the expression."
    except Exception as e:
        return f"Error evaluating expression: {str(e)}"

# Sentiment detection
def detect_sentiment(user_text):
    positive_words = ['happy', 'good', 'great', 'fantastic', 'awesome', 'love', 'nice', 'well', 'excellent']
    negative_words = ['sad', 'bad', 'terrible', 'hate', 'angry', 'upset', 'unhappy', 'not good', 'tired']
    text = user_text.lower()
    pos_score = sum(word in text for word in positive_words)
    neg_score = sum(word in text for word in negative_words)
    if pos_score > neg_score:
        return 'positive'
    if neg_score > pos_score:
        return 'negative'
    return 'neutral'

def generate_empathy_response(user_text):
    sentiment = detect_sentiment(user_text)
    if sentiment == 'positive':
        return "I'm glad to hear you're feeling good! 😊 How can I assist you further?"
    elif sentiment == 'negative':
        return "I'm sorry you're feeling down. If you want to talk, I'm here to listen. 🤗"
    else:
        return None

# To-Do List
def add_task(task):
    st.session_state.todo_list.append(task)

def list_tasks():
    if st.session_state.todo_list:
        return "\n".join(f"- {t}" for t in st.session_state.todo_list)
    return "Your to-do list is empty."

# Health tracker
def log_symptom(symptom):
    entry = {'date': datetime.date.today().isoformat(), 'symptom': symptom}
    st.session_state.health_logs.append(entry)

def display_health_logs():
    if not st.session_state.health_logs:
        return "No symptoms logged yet."
    return "Symptom Logs:\n" + "".join(f"{log['date']}: {log['symptom']}\n" for log in st.session_state.health_logs)

# Wikipedia information fetch
def get_wikipedia_info(topic):
    try:
        if topic.lower() not in [t.lower() for t in st.session_state.wiki_search_history]:
            st.session_state.wiki_search_history.append(topic)
            if len(st.session_state.wiki_search_history) > 10:
                st.session_state.wiki_search_history.pop(0)

        page = wikipedia.page(topic)
        summary = wikipedia.summary(topic, sentences=10)
        wiki_url = page.url
        images = [img for img in page.images if img.lower().endswith(('.jpg', '.jpeg', '.png'))]
        main_image = images[0] if images else None

        links = page.links[:5]
        st.session_state.related_topics[topic] = links

        return summary, main_image, wiki_url
    except wikipedia.exceptions.DisambiguationError as e:
        options = e.options[:5]
        return f"Multiple topics found, please be more specific. Options: {options}", None, None
    except wikipedia.exceptions.PageError:
        return "Sorry, I don't have information about that topic yet. 🤔", None, None
    except Exception:
        return "Sorry, I don't know how to respond to that yet. 🤔", None, None

# Generate response
def get_bot_response(user_text, mode):
    user_lower = user_text.strip().lower()
    if mode == "Wikipedia":
        return get_wikipedia_info(user_text)
    if mode == "Health" and user_lower in medical_symptoms:
        return medical_symptoms[user_lower], None, None
    if mode == "Mathematics":
        return evaluate_math_expression(user_text), None, None

    # fallback to math detect regardless mode
    if is_math_expression(user_text):
        return evaluate_math_expression(user_text), None, None

    matched = responses[responses["Query"].str.lower() == user_lower]
    if not matched.empty:
        return matched.iloc[0]["Response"], None, None

    empathy_resp = generate_empathy_response(user_text)
    if empathy_resp:
        return empathy_resp, None, None

    return get_wikipedia_info(user_text)

# Chat message handling
def add_message(user_msg, mode):
    st.session_state.chat_history.append({"role":"user", "msg": user_msg})
    bot_resp, bot_img, bot_url = get_bot_response(user_msg, mode)
    st.session_state.chat_history.append({"role":"bot", "msg": bot_resp, "image": bot_img, "wiki_url": bot_url})

def update_message(pos, new_msg, mode):
    st.session_state.chat_history[pos]["msg"] = new_msg
    bot_pos = pos + 1
    bot_resp, bot_img, bot_url = get_bot_response(new_msg, mode)
    if bot_pos < len(st.session_state.chat_history) and st.session_state.chat_history[bot_pos]["role"] == "bot":
        st.session_state.chat_history[bot_pos].update({"msg": bot_resp, "image": bot_img, "wiki_url": bot_url})
    else:
        st.session_state.chat_history.insert(bot_pos, {"role":"bot", "msg": bot_resp, "image": bot_img, "wiki_url": bot_url})

# Mode selection UI
mode = st.radio("Select Mode", options=["Wikipedia", "Health", "Mathematics"], index=0, horizontal=True)

# For Health mode, prepare symptom selection
user_input = ""
if mode == "Health":
    symptom_list = sorted(medical_symptoms.keys())
    selected_symptom = st.selectbox("Select a symptom from the list or type your symptom below:", [""] + symptom_list)
    user_input = selected_symptom if selected_symptom else ""

# Header and intro for each mode
if mode == "Wikipedia":
    st.title("🤖 Baymax: I am your info assistant")
    st.markdown("### Hello! I'm Baymax (Wiki Mode). What do you want to know about?")
    
    # Show previous Wikipedia search history
    if st.session_state.wiki_search_history:
        st.markdown("#### Continue exploring:")
        cols = st.columns(3)
        for i, topic in enumerate(st.session_state.wiki_search_history[-6:]):
            with cols[i % 3]:
                if st.button(f"🔍 {topic}", key=f"hist_sugg_{i}"):
                    add_message(topic, mode)
                    st.rerun()

    # Related topics from last bot message
    if len(st.session_state.chat_history) >= 2 and st.session_state.chat_history[-1]["role"] == "bot":
        last_topic = st.session_state.chat_history[-2]["msg"]
        if last_topic in st.session_state.related_topics:
            st.markdown("#### Related topics:")
            cols = st.columns(3)
            for i, topic in enumerate(st.session_state.related_topics[last_topic][:6]):
                with cols[i % 3]:
                    if st.button(f"📚 {topic}", key=f"rel_sugg_{i}"):
                        add_message(topic, mode)
                        st.rerun()

    # Default popular topics
    wiki_suggestions = [
        "Artificial intelligence",
        "Quantum computing",
        "Renaissance art",
        "Ancient Egypt",
        "Machine learning",
        "Human brain"
    ]
    st.markdown("#### Popular topics to explore:")
    cols = st.columns(3)
    for i, topic in enumerate(wiki_suggestions):
        with cols[i % 3]:
            if st.button(f"🌟 {topic}", key=f"wiki_sugg_{i}"):
                add_message(topic, mode)
                st.rerun()

elif mode == "Health":
    st.title("🤖 Baymax: I am your Healthcare companion")
    st.markdown("### Hello! I'm Baymax. How can I help you?")
else:
    st.title("🤖 Baymax: I am your Mathematics Assistant")
    st.markdown("### Hello! I'm Baymax (Math Mode). Enter a mathematical expression:")

# Input form for user queries
with st.form("input_form", clear_on_submit=True):
    if st.session_state.edit_index is None:
        user_text_input = st.text_input(
            "Ask me anything:" if mode != "Health" else "Or enter a different symptom or question:",
            value=user_input if mode == "Health" else "",
            key="input_text"
        )
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

# Button to clear chat
if st.button("Clear Chat"):
    st.session_state.chat_history.clear()
    st.session_state.edit_index = None
    st.session_state.read_more_states = {}
    st.rerun()

# Display chat
for idx, message in reversed(list(enumerate(st.session_state.chat_history))):
    raw_msg = message["msg"]
    msg_text = raw_msg.replace('"', '&quot;') if isinstance(raw_msg, str) else str(raw_msg)
    if message["role"] == "user":
        cols = st.columns([0.7, 0.15, 0.15])
        with cols[0]:
            st.markdown(f"🧑‍💻 {raw_msg}")
        with cols[1]:
            if st.button("✏️ Edit", key=f"edit_{idx}"):
                st.session_state.edit_index = idx
        with cols[2]:
            st.markdown(f'<button onclick="navigator.clipboard.writeText(\'{msg_text}\')">📋 Copy</button>', unsafe_allow_html=True)
    else:
        cols = st.columns([0.85, 0.15])
        with cols[0]:
            wiki_url = message.get("wiki_url")
            full_msg = raw_msg

            # For Wikipedia mode: Read more toggle for long summaries
            if mode == "Wikipedia" and wiki_url and isinstance(full_msg, str) and len(full_msg) > 50:
                if idx not in st.session_state.read_more_states:
                    st.session_state.read_more_states[idx] = False
                read_more = st.session_state.read_more_states[idx]

                if read_more:
                    st.markdown(f"**Baymax:** {full_msg}  \n\n[Read more on Wikipedia]({wiki_url})", unsafe_allow_html=True)
                    if st.button("Show less", key=f"toggle_{idx}"):
                        st.session_state.read_more_states[idx] = False
                        st.rerun()
                else:
                    short_summary = '. '.join(full_msg.split('. ')[:3]) + '.'
                    st.markdown(f"**Baymax:** {short_summary}...  \n\n[Read more on Wikipedia]({wiki_url})", unsafe_allow_html=True)
                    if st.button("Read more...", key=f"toggle_{idx}"):
                        st.session_state.read_more_states[idx] = True
                        st.rerun()
            else:
                st.markdown(f"**Baymax:** {full_msg}")

            if message.get("image"):
                st.image(message["image"], width=300)
        with cols[1]:
            st.markdown(f'<button onclick="navigator.clipboard.writeText(\'{msg_text}\')">📋 Copy</button>', unsafe_allow_html=True)

# To-Do List expander
with st.expander("📝 To-Do List"):
    new_task = st.text_input("Add a task to your list:", key="todo_task")
    if st.button("Add Task"):
        if new_task.strip():
            add_task(new_task.strip())
            st.success(f"Added task: {new_task}")
    st.text("Current tasks:")
    st.text(list_tasks())

# Health Tracker expander only in Health mode
if mode == "Health":
    with st.expander("🏥 Health Tracker"):
        symptom_input = st.text_input("Log a symptom:", key="health_symptom")
        if st.button("Log Symptom"):
            if symptom_input.strip():
                log_symptom(symptom_input.strip())
                st.success(f"Logged symptom: {symptom_input}")
        st.text(display_health_logs())

# Hide default Streamlit footer and add custom footer
st.markdown("""
    <style>
    footer {visibility: hidden;}
    </style>
    <div style="margin-top: 50px; text-align: center; color: gray;">
        Powered by Streamlit • Baymax Assistant v2.0
    </div>
""", unsafe_allow_html=True)
