import os
import requests
import streamlit as st

from config import REQUEST_TIMEOUT, AVAILABLE_OLLAMA_MODELS, runtime_config
from utils import MemoryManager
from rag import create_rag_instance, get_rag_instance, switch_rag_instance, delete_rag_instance
from data_store import process_url, process_uploaded_file
from cag import CAGSystem

def sidebar_components():
    st.sidebar.title("Hybrid RAG+CAG Chatbot")
    
    # Automatic RAG Instance Creation if none exist
    if not st.session_state.rag_instances:
        with st.sidebar.container():
            st.info("Welcome! You need a RAG instance to start. Create one below or use the Quick Start option.")
            if st.button("🚀 Quick Start", key="quick_start_button"):
                instance_id = create_rag_instance("My First RAG Instance", "Automatically created RAG instance")
                switch_rag_instance(instance_id)
                st.success("Created and selected your first RAG instance!")
                st.session_state.update_ui = True
    
    # Ollama Model Selection
    st.sidebar.subheader("Model Selection")
    if not st.session_state.ollama_models:
        st.sidebar.error(f"⚠️ Cannot connect to Ollama server at {runtime_config.ollama_base_url} or no models found.")
        if st.sidebar.button("Retry Ollama Connection / Refresh Model List"):
            try:
                response = requests.get(f"{runtime_config.ollama_base_url}/api/tags", timeout=REQUEST_TIMEOUT)
                if response.status_code == 200:
                    st.session_state.ollama_models = [model["name"] for model in response.json().get("models", [])]
                else: st.session_state.ollama_models = []
            except: st.session_state.ollama_models = []
            st.rerun()

        model_options_display = AVAILABLE_OLLAMA_MODELS
        if not model_options_display: model_options_display = ["gemma:2b", "llama3:8b"]  # Absolute fallback
        st.sidebar.warning("Using a predefined list of models. These might not be available on your Ollama server.")
    else:
        model_options_display = st.session_state.ollama_models
        st.sidebar.success(f"✅ Connected to Ollama. Found {len(model_options_display)} models.")
    
    default_model_idx = 0
    if model_options_display:
        preferred_models = ["llama3:8b", "gemma:7b", "mistral:7b", "phi3:mini"]
        for pref_model in preferred_models:
            if pref_model in model_options_display:
                default_model_idx = model_options_display.index(pref_model)
                break
    
    model_name = st.sidebar.selectbox(
        "Choose Ollama Model:", options=model_options_display, index=default_model_idx,
        help="Select the Ollama model for responses."
    )
    
    # RAG Instance Management
    st.sidebar.subheader("RAG Instances")
    with st.sidebar.expander("Create New RAG Instance", expanded=not st.session_state.rag_instances):
        new_name = st.text_input("Instance Name", key="new_instance_name")
        new_desc = st.text_area("Description (optional)", key="new_instance_desc")
        if st.button("Create Instance"):
            if new_name:
                instance_id = create_rag_instance(new_name, new_desc)
                switch_rag_instance(instance_id)  # Auto-switch
                st.success(f"Created and selected RAG instance: {new_name}")
                st.session_state.update_ui = True  # Trigger rerun via main loop check
            else: st.error("Instance name is required.")

    if st.session_state.rag_instances:
        instance_choices = {instance.id: instance.name for instance in st.session_state.rag_instances.values()}
        
        # Determine current selection for selectbox
        current_id = st.session_state.current_rag_instance
        ids_list = list(instance_choices.keys())
        names_list = list(instance_choices.values())
        
        current_idx = 0
        if current_id and current_id in ids_list:
            current_idx = ids_list.index(current_id)

        selected_instance_id_from_box = st.sidebar.selectbox(
            "Select RAG Instance", ids_list, format_func=lambda x: instance_choices[x], index=current_idx,
            key="select_rag_instance_box"
        )

        if selected_instance_id_from_box != st.session_state.current_rag_instance:
            switch_rag_instance(selected_instance_id_from_box)
            st.session_state.update_ui = True
        
        if st.sidebar.button("🗑️ Delete Selected RAG Instance", type="secondary"):
            if st.session_state.current_rag_instance:
                instance_to_delete = get_rag_instance(st.session_state.current_rag_instance)
                if delete_rag_instance(st.session_state.current_rag_instance):
                    st.success(f"Deleted RAG instance: {instance_to_delete.name}")
                    st.session_state.update_ui = True
                else: st.error("Failed to delete RAG instance.")
            else: st.warning("No RAG instance selected to delete.")
    else:
        st.sidebar.info("No RAG instances. Create one to begin.")

    # Data Sources
    if st.session_state.current_rag_instance:
        st.sidebar.subheader("Add Data to Current RAG Instance")
        with st.sidebar.expander("Add URL", expanded=False):
            url_in = st.text_input("Enter URL", key="url_input_ds")
            use_sitemap = st.checkbox("Process sitemap (finds and processes all pages)", value=True, 
                                     help="When checked, will find and process XML sitemap URLs for more comprehensive coverage")
            
            if st.button("Process URL"):
                if url_in:
                    with st.spinner(f"Processing URL: {url_in[:50]}..."):
                        MemoryManager.log_memory_usage("before URL processing")
                        # Use the enhanced URL processor with sitemap support
                        count = process_url(url_in, use_sitemap=use_sitemap)
                        MemoryManager.log_memory_usage("after URL processing")
                        if count > 0: st.success(f"URL processed into {count} chunks.")
                        # process_url handles its own errors/st.error
                else: st.error("Please enter a URL.")
        
        with st.sidebar.expander("Upload Files", expanded=False):
            # Allow multiple files
            uploaded_files = st.file_uploader(
                "Choose files (PDF, DOCX, TXT, MD, etc.)", 
                type=["pdf", "docx", "txt", "md", "py", "js", "html", "css"], 
                accept_multiple_files=True,
                key="file_uploader_ds"
            )
            if uploaded_files:
                for up_file in uploaded_files:
                    with st.spinner(f"Processing {up_file.name}..."):
                        MemoryManager.log_memory_usage(f"before file processing {up_file.name}")
                        count = process_uploaded_file(up_file)
                        MemoryManager.log_memory_usage(f"after file processing {up_file.name}")
                        if count > 0: st.success(f"Processed {up_file.name} into {count} chunks.")
                        # process_uploaded_file handles its own errors/st.error

    # User Context (CAG)
    st.sidebar.subheader("User Context (CAG)")
    with st.sidebar.expander("Manage User Context", expanded=False):
        ctx_key = st.text_input("Context Key (e.g., 'My Name', 'Preferred Language')", key="ctx_key_cag")
        ctx_val = st.text_area("Context Value", key="ctx_val_cag")
        if st.button("Save Context"):
            if ctx_key and ctx_val:
                CAGSystem.update_user_context(ctx_key, ctx_val)
                st.success(f"Context '{ctx_key}' saved.")
            else: st.error("Both key and value are required for context.")
        
        if st.session_state.user_context:
            st.write("Current Context:")
            for key, data in list(st.session_state.user_context.items()):  # Iterate on a copy for safe deletion
                col1, col2 = st.columns([4,1])
                col1.text(f"{key}: {data['value'][:30]}{'...' if len(data['value'])>30 else ''}")
                if col2.button("Del", key=f"del_ctx_{key}", help=f"Delete context '{key}'"):
                    del st.session_state.user_context[key]
                    st.session_state.update_ui = True  # Rerun to update display

    # Current RAG Instance Info
    if st.session_state.current_rag_instance:
        instance = get_rag_instance(st.session_state.current_rag_instance)
        if instance:
            st.sidebar.subheader(f"Info: '{instance.name}'")
            st.sidebar.caption(f"ID: {instance.id}")
            st.sidebar.write(f"Documents Tracked: {instance.get_document_count()}")
            st.sidebar.write(f"Total Vectors: {instance.get_vector_count()}")
            if instance.documents_info:
                with st.sidebar.expander("Tracked Document Details", expanded=False):
                    for doc_info in instance.documents_info:
                        title = doc_info.get('title', doc_info.get('filename', doc_info.get('url', 'Unknown')))
                        st.markdown(f"- **{title}**: {doc_info.get('chunks', 'N/A')} chunks "
                                    f"({doc_info.get('type', 'N/A')}, "
                                    f"{(doc_info.get('size', 0) / 1024):.1f}KB)")
    
    # Advanced Settings
    with st.sidebar.expander("Advanced Settings", expanded=False):
        st.subheader("Chunking (Applied to new data)")
        
        # Use session state to persist these settings across runs if desired, then update globals
        if 'adv_small_chunk' not in st.session_state: st.session_state.adv_small_chunk = runtime_config.small_chunk_size
        if 'adv_medium_chunk' not in st.session_state: st.session_state.adv_medium_chunk = runtime_config.medium_chunk_size
        if 'adv_large_chunk' not in st.session_state: st.session_state.adv_large_chunk = runtime_config.large_chunk_size
        if 'adv_max_sitemap_urls' not in st.session_state: st.session_state.adv_max_sitemap_urls = runtime_config.max_sitemap_urls

        st.session_state.adv_small_chunk = st.slider("Small Chunk Size", 100, 500, st.session_state.adv_small_chunk)
        st.session_state.adv_medium_chunk = st.slider("Medium Chunk Size", 300, 800, st.session_state.adv_medium_chunk)
        st.session_state.adv_large_chunk = st.slider("Large Chunk Size", 600, 1200, st.session_state.adv_large_chunk)
        st.session_state.adv_max_sitemap_urls = st.slider("Max Sitemap URLs", 10, 200, st.session_state.adv_max_sitemap_urls)
        
        if st.button("Apply Advanced Settings"):
            runtime_config.update_chunking_settings(
                small=st.session_state.adv_small_chunk,
                medium=st.session_state.adv_medium_chunk,
                large=st.session_state.adv_large_chunk,
                max_sitemap=st.session_state.adv_max_sitemap_urls
            )
            st.success("Advanced settings updated for future processing.")

        st.subheader("Ollama Configuration")
        ollama_url_input = st.text_input("Ollama Server URL", value=runtime_config.ollama_base_url, key="ollama_url_adv")
        if st.button("Update Ollama URL"):
            runtime_config.ollama_base_url = ollama_url_input
            # Re-fetch models with new URL
            try:
                response = requests.get(f"{runtime_config.ollama_base_url}/api/tags", timeout=REQUEST_TIMEOUT)
                if response.status_code == 200:
                    st.session_state.ollama_models = [model["name"] for model in response.json().get("models", [])]
                else: st.session_state.ollama_models = []
            except: st.session_state.ollama_models = []
            st.success(f"Ollama URL updated to: {runtime_config.ollama_base_url}. Model list refreshed.")
            st.session_state.update_ui = True

        model_to_pull = st.text_input("Pull Ollama Model (e.g., llama3:8b)", key="model_pull_adv")
        if st.button("Pull Model") and model_to_pull:
            with st.spinner(f"Pulling '{model_to_pull}' from Ollama... (this may take a while)"):
                try:
                    # Use streaming for better feedback if Ollama API supports it well
                    # For now, just a long timeout
                    pull_response = requests.post(
                        f"{runtime_config.ollama_base_url}/api/pull",
                        json={"name": model_to_pull, "stream": False},  # stream: False for simpler handling
                        timeout=1800  # 30 minutes timeout
                    )
                    pull_response.raise_for_status()
                    # Some Ollama versions return streaming JSON, others a simple status.
                    # For stream=False, it should wait until done.
                    st.success(f"Model '{model_to_pull}' pull initiated/completed. Check Ollama server logs.")
                    # Refresh model list
                    response = requests.get(f"{runtime_config.ollama_base_url}/api/tags", timeout=REQUEST_TIMEOUT)
                    if response.status_code == 200:
                        st.session_state.ollama_models = [model["name"] for model in response.json().get("models", [])]
                    st.session_state.update_ui = True
                except requests.exceptions.Timeout:
                    st.error(f"Timeout pulling model '{model_to_pull}'. It might still be downloading on the server.")
                except requests.exceptions.RequestException as e:
                    st.error(f"Error pulling model '{model_to_pull}': {e}")
                except Exception as e:
                    st.error(f"Unexpected error pulling model '{model_to_pull}': {e}")
        
        st.subheader("Memory")
        if MemoryManager.psutil_available:
            mem_info = MemoryManager.get_memory_usage()
            st.write(f"App RSS: {mem_info['rss']:.1f}MB")
            st.write(f"System Available: {mem_info['available_gb']:.2f}GB / {mem_info['total_gb']:.2f}GB")
            if st.button("Force Garbage Collect"):
                import gc
                gc.collect()
                st.success("Garbage collection triggered.")
                MemoryManager.log_memory_usage("after manual GC")
                st.session_state.update_ui = True
        else:
            st.caption("psutil not installed, memory monitoring disabled.")

    return model_name  # Return the selected model name