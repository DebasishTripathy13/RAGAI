import json
import logging
import threading
import requests

import streamlit as st
from config import REQUEST_TIMEOUT, runtime_config

# Global lock for thread safety
ollama_lock = threading.RLock()

class OllamaClient:
    def __init__(self, base_url=None):  # Accept base_url override
        self.base_url = base_url or runtime_config.ollama_base_url  # Use runtime_config

    def list_models(self):
        try:
            response = requests.get(f"{self.base_url}/api/tags", timeout=REQUEST_TIMEOUT)
            response.raise_for_status()  # Raise HTTPError for bad responses (4xx or 5xx)
            models_data = response.json().get("models", [])
            return [model["name"] for model in models_data]
        except requests.exceptions.RequestException as e:
            logging.error(f"Error connecting to Ollama ({self.base_url}) to list models: {e}", exc_info=True)
            return []
        except json.JSONDecodeError as e:
            logging.error(f"Failed to decode JSON response from Ollama ({self.base_url}) model endpoint: {e}", exc_info=True)
            return []

    def generate(self, model_name, prompt, system_prompt=None, temperature=0.7, max_tokens=2048):
        payload = {
            "model": model_name, "prompt": prompt, "stream": False,
            "options": {"temperature": temperature, "num_predict": max_tokens}
        }
        if system_prompt: payload["system"] = system_prompt

        try:
            with ollama_lock:
                response = requests.post(
                    f"{self.base_url}/api/generate", json=payload, timeout=120  # Increased timeout for generation
                )
            response.raise_for_status()
            result = response.json()
            return result.get("response", "").strip(), None
        except requests.exceptions.Timeout:
            error_msg = f"Request to Ollama model '{model_name}' timed out after 120s."
            logging.error(error_msg)
            return "The language model took too long to respond. Please try a simpler query or try again later.", error_msg
        except requests.exceptions.RequestException as e:
            error_msg = f"Ollama API error ({model_name}): {e}. Response: {e.response.text if e.response else 'No response'}"
            logging.error(error_msg, exc_info=True)
            return f"I encountered an error communicating with the language model ({e}). Please check your Ollama server.", error_msg
        except json.JSONDecodeError as e:
            error_msg = f"Failed to decode JSON response from Ollama generate endpoint ({model_name}): {e}"
            logging.error(error_msg, exc_info=True)
            return "Received an invalid response from the language model. Please try again.", error_msg