# 🤖 Baymax - Personal AI Healthcare & Knowledge Companion

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://baymax-1101.streamlit.app/)

> *"Hello. I am Baymax, your personal healthcare companion."*

**Baymax** is an interactive, futuristic AI assistant application built with Streamlit. Designed with a custom Arc-Reactor glassmorphism UI, Baymax provides specialized intelligence modules across Healthcare, Web Searching, Mathematics, and Software Engineering.

---

## 🚀 Live Project Preview

Access the live deployed application here:  
👉 **[https://baymax-1101.streamlit.app/](https://baymax-1101.streamlit.app/)**

---

## ⚡ Multi-Engine AI & Failover Architecture

Baymax is engineered with a high-availability dual search engine architecture designed for seamless reliability and zero-downtime responses:

### 1. Primary AI Engine (OpenRouter Multi-Key System)
- **Multi-Key Redundancy**: Configured with **three different AI API keys** from [OpenRouter](https://openrouter.ai/).
- **Seamless Failover**: Requests cycle dynamically across multiple models (including DeepSeek Chat, GPT-4o Mini, and Qwen 2.5 72B). If one API key or model fails, times out, or encounters rate limits, the system seamlessly transitions to another key automatically—ensuring uninterrupted service for the user.

### 2. Secondary AI Engine (Pollinations AI Backup)
- **Deployed API Key System**: Integrated with an API key deployed from [Pollinations AI](https://pollinations.ai/).
- **Multi-Model Backup**: Accesses unified endpoints providing fallback access to OpenAI, Gemini, and Mistral models. If all primary OpenRouter keys encounter issues, Pollinations AI steps in to deliver the response without missing a beat.

### 3. Real-Time Web Context Engine
- Integrated with `duckduckgo_search` in **Searching Mode** to query real-time web results and provide up-to-date factual context before generating responses.

---

## ✨ Modules & Features

- 🔍 **Searching Mode**: Performs live web searches and delivers comprehensive, ELI5-style structured answers with engaging explanations.
- ❤️ **Health Module**: Empathetic healthcare companion providing structured medical guidance and wellness advice.
- 📐 **Mathematics Module**: Delivers clear, step-by-step math solutions and derivations.
- 💻 **Code Module**: Generates complete programming solutions, syntax explanations, and algorithm optimizations.
- 🎨 **Futuristic UI/UX**: Custom CSS featuring floating background orbs, glowing arc-reactor chat input, and glassmorphism panels.
- 📥 **Export Chat History**: Download full conversation logs as text files directly from the sidebar.
- ⚙️ **Satisfied Deactivation**: Deactivate Baymax with the iconic *"I am satisfied with my care"* shutdown protocol.

---

## 🛠️ Local Setup & Run

### 1. Clone the repository
```bash
git clone https://github.com/your-username/baymax.git
cd baymax
```

### 2. Install dependencies
```bash
pip install -r requirements.txt
```

### 3. Run the application
```bash
streamlit run baymax.py
```

---

## 👨‍💻 Developer Info

- **Developer**: Suraj
- **Contact**: `sn740698@gmail.com`
- **Live App**: [https://baymax-1101.streamlit.app/](https://baymax-1101.streamlit.app/)
