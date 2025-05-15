import uuid
import logging
from datetime import datetime
from typing import List, Dict, Any, Optional

import streamlit as st
from data_store.chroma_store import ChromaVectorStore

class RAGInstance:
    def __init__(self, name, description=""):
        self.id = str(uuid.uuid4())
        self.name = name
        self.description = description
        self.collection_name = f"rag_{self.name.lower().replace(' ', '_')}_{self.id[:8]}"  # More readable collection name
        self.vector_store = ChromaVectorStore(self.collection_name)
        self.documents_info: List[Dict[str, Any]] = []  # Stores metadata about processed source documents
        self.created_at = datetime.now().isoformat()

    def add_document(self, doc_info: Dict[str, Any]):
        # Avoid duplicate document entries if identified by URL or filename
        key_field = "url" if doc_info.get("type") == "url" else "filename"
        if key_field in doc_info:
            existing_doc = next((d for d in self.documents_info if d.get(key_field) == doc_info[key_field]), None)
            if existing_doc:  # Update existing document info
                existing_doc.update(doc_info)
                existing_doc["chunks"] = self.get_vector_count()  # Update chunk count from vector store
                existing_doc["date_updated"] = datetime.now().isoformat()
                return
        self.documents_info.append(doc_info)

    def add_texts(self, texts: List[str], metadatas: Optional[List[dict]] = None):
        return self.vector_store.add_texts(texts, metadatas)

    def search(self, query: str, k: int = 5):
        return self.vector_store.similarity_search(query, k)

    def get_document_count(self):
        return len(self.documents_info)

    def get_vector_count(self):
        return self.vector_store.get_count()

    def get_summary(self):
        return {
            "id": self.id, "name": self.name, "description": self.description,
            "document_count": self.get_document_count(), "vector_count": self.get_vector_count(),
            "created_at": self.created_at
        }

    def delete(self):
        return self.vector_store.delete_collection()


def create_rag_instance(name, description=""):
    instance = RAGInstance(name, description)
    st.session_state.rag_instances[instance.id] = instance
    return instance.id


def get_rag_instance(instance_id):
    return st.session_state.rag_instances.get(instance_id)


def switch_rag_instance(instance_id):
    if instance_id in st.session_state.rag_instances:
        st.session_state.current_rag_instance = instance_id
        st.session_state.messages = []  # Clear chat history on instance switch
        return True
    return False


def delete_rag_instance(instance_id):
    if instance_id in st.session_state.rag_instances:
        instance = st.session_state.rag_instances[instance_id]
        if instance.delete():
            del st.session_state.rag_instances[instance_id]
            if st.session_state.current_rag_instance == instance_id:
                st.session_state.current_rag_instance = None
                st.session_state.messages = []  # Clear chat history
            return True
    return False