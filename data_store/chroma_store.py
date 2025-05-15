import os
import uuid
import logging
import threading
import chromadb
from typing import List, Dict, Any, Optional
from chromadb.utils import embedding_functions

import streamlit as st
from config import CHROMA_PERSIST_DIRECTORY

# Global locks for thread safety
chroma_lock = threading.RLock()

class ChromaVectorStore:
    def __init__(self, collection_name):
        self.collection_name = collection_name
        self.client = self._get_chroma_client()
        self.embedding_function = self._get_embedding_function()
        self.collection = None
        self._initialize_collection()

    @staticmethod
    @st.cache_resource(ttl=3600)  # Cache for 1 hour
    def _get_chroma_client():
        try:
            os.makedirs(CHROMA_PERSIST_DIRECTORY, exist_ok=True)
            return chromadb.PersistentClient(path=CHROMA_PERSIST_DIRECTORY)
        except Exception as e:
            logging.error(f"Error initializing ChromaDB client: {str(e)}", exc_info=True)
            st.error(f"Could not initialize ChromaDB: {str(e)}")
            return None

    @staticmethod
    @st.cache_resource
    def _get_embedding_function():
        try:
            return embedding_functions.SentenceTransformerEmbeddingFunction(
                model_name="all-MiniLM-L6-v2"
            )
        except Exception as e:
            logging.error(f"Error initializing embedding function: {str(e)}", exc_info=True)
            st.error(f"Could not initialize embedding function: {str(e)}")
            return None

    def _initialize_collection(self):
        if not self.client or not self.embedding_function:
            st.error("ChromaDB client or embedding function not available.")
            logging.error("ChromaDB client or embedding function not available for collection init.")
            return
        try:
            with chroma_lock:
                self.collection = self.client.get_or_create_collection(
                    name=self.collection_name,
                    embedding_function=self.embedding_function
                )
            logging.info(f"Retrieved or created collection: {self.collection_name}")
        except Exception as e:
            logging.error(f"Error initializing ChromaDB collection '{self.collection_name}': {str(e)}", exc_info=True)
            st.error(f"Could not initialize ChromaDB collection '{self.collection_name}': {str(e)}")

    def add_texts(self, texts: List[str], metadatas: Optional[List[dict]] = None, ids: Optional[List[str]] = None):
        if not texts: return []
        if not self.collection:
            logging.error(f"Collection '{self.collection_name}' not initialized for adding texts.")
            return []

        if metadatas is None: metadatas = [{} for _ in texts]
        if ids is None: ids = [str(uuid.uuid4()) for _ in texts]
        
        # Ensure IDs are unique if provided by user, or generate if not
        if len(ids) != len(set(ids)):
            logging.warning("Duplicate IDs provided, generating new unique IDs.")
            ids = [str(uuid.uuid4()) for _ in texts]

        try:
            with chroma_lock:
                self.collection.add(documents=texts, metadatas=metadatas, ids=ids)
            return ids
        except Exception as e:
            logging.error(f"Error adding texts to ChromaDB collection '{self.collection_name}': {str(e)}", exc_info=True)
            return []

    def similarity_search(self, query: str, k: int = 5):
        if not self.collection:
            logging.warning(f"Collection '{self.collection_name}' not initialized for similarity search.")
            return []
        try:
            with chroma_lock:
                count = self.collection.count()
            if count == 0: return []

            with chroma_lock:
                results = self.collection.query(
                    query_texts=[query],
                    n_results=min(k, count),
                    include=["documents", "metadatas", "distances"]
                )
            
            formatted_results = []
            if results and results.get("documents") and results["documents"][0]:
                for i, (doc, metadata, distance) in enumerate(zip(
                    results["documents"][0],
                    results["metadatas"][0],
                    results["distances"][0]
                )):
                    formatted_results.append({
                        "id": results["ids"][0][i],
                        "content": doc,
                        "metadata": metadata,
                        "score": 1.0 - float(distance)  # Assumes cosine distance
                    })
            return formatted_results
        except Exception as e:
            logging.error(f"Error during similarity search in '{self.collection_name}': {str(e)}", exc_info=True)
            return []

    def get_count(self):
        if not self.collection: return 0
        try:
            with chroma_lock:
                return self.collection.count()
        except Exception as e:
            logging.error(f"Error getting count for collection '{self.collection_name}': {str(e)}", exc_info=True)
            return 0

    def delete_collection(self):
        if not self.client: return False
        try:
            with chroma_lock:
                self.client.delete_collection(name=self.collection_name)
            logging.info(f"Deleted collection: {self.collection_name}")
            return True
        except Exception as e:
            logging.error(f"Error deleting collection '{self.collection_name}': {str(e)}", exc_info=True)
            return False