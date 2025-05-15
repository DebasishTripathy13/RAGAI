import uuid
import logging
import streamlit as st

from config import MAX_CONTENT_SIZE, MAX_SOURCES_TO_RETRIEVE
from rag import get_rag_instance
from cag import CAGSystem
from models import OllamaClient
from prompting import DynamicPrompting
from utils import MemoryManager

def generate_response(query: str, model_name: str, include_context=True):
    if not st.session_state.current_rag_instance:
        return "No RAG instance selected. Please create or select one, and add data sources.", [], []
    
    instance = st.session_state.rag_instances[st.session_state.current_rag_instance]
    if instance.get_vector_count() == 0:
        return "The current RAG instance has no data. Please add documents or URLs first.", [], []

    try:
        # Update conversation context
        CAGSystem.update_conversation_context(query)
        
        # Get relevant documents
        relevant_docs = instance.search(query, k=MAX_SOURCES_TO_RETRIEVE)
        
        # Get user-specific context
        user_specific_context = CAGSystem.get_relevant_context(query) if include_context else {}
        
        # Generate adaptive prompts based on context
        system_prompt = DynamicPrompting.get_adaptive_system_prompt(query)
        full_prompt = DynamicPrompting.get_enhanced_user_prompt(query, relevant_docs, user_specific_context)
        
        if len(full_prompt) > MAX_CONTENT_SIZE * 0.8:  # Leave some room for model's own processing
            logging.warning(f"Prompt size {len(full_prompt)} is large, truncating.")
            full_prompt = full_prompt[:int(MAX_CONTENT_SIZE * 0.8)]
            full_prompt += "\n[PROMPT TRUNCATED DUE TO LENGTH]"
        
        ollama_cli = OllamaClient()  # Uses current OLLAMA_BASE_URL from session state
        response_text, error = ollama_cli.generate(model_name, full_prompt, system_prompt)
        
        if error: logging.warning(f"Error in Ollama response generation: {error}")
        
        # Generate follow-up questions
        follow_up_questions = DynamicPrompting.generate_follow_up_questions(query, response_text, relevant_docs)
        
        # Update conversation context with the response
        CAGSystem.update_conversation_context(query, response_text)
        
        return response_text, relevant_docs, follow_up_questions
    except Exception as e:
        logging.error(f"Error generating response: {str(e)}", exc_info=True)
        return f"An error occurred while generating response: {str(e)}", [], []

def display_chat_interface(selected_model_name: str):
    st.title("Hybrid FAQ Chatbot")
    st.markdown("""
    *Powered by ChromaDB & Ollama*
    1.  Use the sidebar to manage RAG instances, add data (URLs, files), and set user context.
    2.  Ensure your Ollama server is running and the correct model is selected.
    3.  Ask questions below!
    """)

    if not st.session_state.current_rag_instance:
        st.info("👉 Please create or select a RAG instance from the sidebar to start chatting.")
        
        # Show a welcome message with getting started tips
        st.markdown("""
        ### Getting Started
        1. Create a new RAG instance using the sidebar
        2. Add content by uploading files or providing URLs
        3. Ask questions about your content
        
        **Tip**: Use the Quick Start button in the sidebar for a faster setup!
        """)
        
    elif get_rag_instance(st.session_state.current_rag_instance).get_vector_count() == 0:
        st.warning("⚠️ The current RAG instance is empty. Add some data sources (URLs, files) via the sidebar for relevant answers.")
        
        # Show hints about adding data
        st.markdown("""
        ### Adding Data to Your RAG Instance
        1. **URLs**: Enter a website URL in the sidebar. Enable the sitemap option to automatically extract all pages.
        2. **Files**: Upload PDF, DOCX, or text files to extract their contents.
        3. **Custom Text**: You can also enter text directly through the User Context section for personalized responses.
        """)

    # Display chat messages
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])
            if msg["role"] == "assistant" and "sources" in msg and msg["sources"]:
                with st.expander("View Sources", expanded=False):
                    for i, src in enumerate(msg["sources"]):
                        meta = src.get("metadata", {})
                        title = meta.get('title', meta.get('filename', meta.get('url', f'Source {i+1}')))
                        url = meta.get('url')
                        content_preview = src.get('content', '')[:150] + "..."
                        if url:
                            st.markdown(f"**{i+1}. [{title}]({url})**")
                        else:
                            st.markdown(f"**{i+1}. {title}**")
                        st.caption(f"Match score: {src.get('score', 0.0):.2f} | *{content_preview}*")
            
            # Display follow-up questions if they exist
            if msg["role"] == "assistant" and "follow_ups" in msg and msg["follow_ups"]:
                with st.container():
                    st.markdown("#### Follow-up questions:")
                    for i, question in enumerate(msg["follow_ups"]):
                        if st.button(f"{question}", key=f"follow_up_{i}_{msg.get('id', uuid.uuid4())}"):
                            # When clicked, add this as a new user message
                            st.session_state.messages.append({"role": "user", "content": question})
                            st.session_state.update_ui = True

    # Chat input
    if prompt := st.chat_input("Ask anything about your documents..."):
        if MemoryManager.check_memory_pressure():
            st.toast("⚠️ System memory is high. Performance might be affected.", icon="⚠️")
        
        # Generate a unique ID for this message pair
        message_id = str(uuid.uuid4())
        
        st.session_state.messages.append({"role": "user", "content": prompt, "id": message_id})
        with st.chat_message("user"): st.markdown(prompt)
        
        with st.chat_message("assistant"):
            message_placeholder = st.empty()
            with st.spinner("Thinking..."):
                MemoryManager.log_memory_usage("before response gen")
                response_text, relevant_docs, follow_up_questions = generate_response(prompt, selected_model_name)
                MemoryManager.log_memory_usage("after response gen")
                
                message_placeholder.markdown(response_text)
                if relevant_docs:
                    with st.expander("View Sources", expanded=False):
                        for i, src in enumerate(relevant_docs):
                            meta = src.get("metadata", {})
                            title = meta.get('title', meta.get('filename', meta.get('url', f'Source {i+1}')))
                            url = meta.get('url')
                            content_preview = src.get('content', '')[:150] + "..."
                            if url: st.markdown(f"**{i+1}. [{title}]({url})**")
                            else: st.markdown(f"**{i+1}. {title}**")
                            st.caption(f"Match score: {src.get('score',0.0):.2f} | *{content_preview}*")
                
                # Display follow-up questions
                if follow_up_questions:
                    st.markdown("#### Follow-up questions:")
                    for i, question in enumerate(follow_up_questions):
                        if st.button(f"{question}", key=f"follow_up_{i}_{message_id}"):
                            # When clicked, add this as a new user message
                            st.session_state.messages.append({"role": "user", "content": question})
                            st.session_state.update_ui = True
            
            st.session_state.messages.append({
                "role": "assistant", 
                "content": response_text, 
                "sources": relevant_docs,
                "follow_ups": follow_up_questions,
                "id": message_id
            })
            
            # Limit chat history
            if len(st.session_state.messages) > 50:
                st.session_state.messages = st.session_state.messages[-50:]