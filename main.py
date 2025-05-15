import os
import streamlit as st

from config import runtime_config
from utils import MemoryManager
from ui import sidebar_components, display_chat_interface

# Initialize the page config
st.set_page_config(
    page_title="Hybrid RAG+CAG Chatbot",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Initialize session state variables
if "messages" not in st.session_state:
    st.session_state.messages = []

if "user_context" not in st.session_state:
    st.session_state.user_context = {}

if "rag_instances" not in st.session_state:
    st.session_state.rag_instances = {}

if "current_rag_instance" not in st.session_state:
    st.session_state.current_rag_instance = None

# Initialize conversation context for dynamic prompting
if "conversation_context" not in st.session_state:
    st.session_state.conversation_context = {
        "topics": set(),        # Topics mentioned in conversation
        "entities": set(),      # Named entities mentioned
        "user_preferences": {}, # Inferred user preferences
        "last_queries": [],     # Recent user queries
        "unresolved_queries": []  # Queries that weren't fully answered
    }

# Dynamic Ollama Base URL
if "ollama_base_url" not in st.session_state:
    st.session_state.ollama_base_url = os.environ.get("OLLAMA_BASE_URL", "http://localhost:11434")
    runtime_config.ollama_base_url = st.session_state.ollama_base_url

if "ollama_models" not in st.session_state:
    st.session_state.ollama_models = []  # Initialize
    import requests
    try:
        response = requests.get(f"{runtime_config.ollama_base_url}/api/tags", timeout=30)
        if response.status_code == 200:
            models_data = response.json().get("models", [])
            st.session_state.ollama_models = [model["name"] for model in models_data]
        else:
            st.session_state.ollama_models = []
    except Exception:
        st.session_state.ollama_models = []

def main():
    if "update_ui" not in st.session_state: st.session_state.update_ui = False
    
    MemoryManager.log_memory_usage("app start")
    
    # This global should be updated from runtime_config
    runtime_config.ollama_base_url = st.session_state.ollama_base_url

    selected_model = sidebar_components() 
    # selected_model can be None if model_options_display is empty, handle this
    if not selected_model and len(st.session_state.ollama_models) > 0:
        selected_model = st.session_state.ollama_models[0]
    elif not selected_model:
        from config import AVAILABLE_OLLAMA_MODELS
        selected_model = AVAILABLE_OLLAMA_MODELS[0] if AVAILABLE_OLLAMA_MODELS else "gemma:2b"  # Fallback
        st.error("No Ollama models available for selection. Chat functionality may be limited.")

    display_chat_interface(selected_model)
    
    MemoryManager.log_memory_usage("app render complete")

    if st.session_state.update_ui:
        st.session_state.update_ui = False
        st.rerun()

if __name__ == "__main__":
    main()