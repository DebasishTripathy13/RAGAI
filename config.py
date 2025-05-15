import os
import logging
from datetime import datetime

# Configure logging
logging.basicConfig(
    filename='chatbot_app.log',
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

# Constants
MAX_CHUNK_SIZE = 500  # Default max chunk size, estimate_text_density can override
CHUNK_OVERLAP = 100
MAX_SOURCES_TO_RETRIEVE = 5
MAX_CONTENT_SIZE = 1024 * 1024 * 10  # 10MB limit for processing text content / prompts
REQUEST_TIMEOUT = 30  # Seconds

# Web processing constants
WEB_BATCH_SIZE = 100000  # 100KB batches for iter_content
SMALL_CHUNK_SIZE = 300  # For dense text
MEDIUM_CHUNK_SIZE = 500  # Default
LARGE_CHUNK_SIZE = 800  # For sparse text
MAX_SITEMAP_URLS = 50  # Maximum URLs to process from a sitemap

# ChromaDB Configuration
CHROMA_PERSIST_DIRECTORY = "./chroma_db"
DEFAULT_COLLECTION_NAME = "hybrid_chatbot_docs"

# Ollama Configuration
AVAILABLE_OLLAMA_MODELS = [
    "gemma:2b",
    "gemma:7b",
    "llama3:8b",
    "llama3:70b",
    "mistral:7b",
    "mixtral:8x7b",
    "phi3:mini",
    "phi3:14b"
]

# Runtime configuration (can be modified at runtime)
class RuntimeConfig:
    def __init__(self):
        self.ollama_base_url = os.environ.get("OLLAMA_BASE_URL", "http://localhost:11434")
        self.small_chunk_size = SMALL_CHUNK_SIZE
        self.medium_chunk_size = MEDIUM_CHUNK_SIZE
        self.large_chunk_size = LARGE_CHUNK_SIZE
        self.max_sitemap_urls = MAX_SITEMAP_URLS

    def update_chunking_settings(self, small=None, medium=None, large=None, max_sitemap=None):
        """Update chunking settings with new values"""
        if small is not None:
            self.small_chunk_size = small
        if medium is not None:
            self.medium_chunk_size = medium
        if large is not None:
            self.large_chunk_size = large
        if max_sitemap is not None:
            self.max_sitemap_urls = max_sitemap
        return True

# Instantiate runtime config
runtime_config = RuntimeConfig()