import re
from typing import Dict, Any
from datetime import datetime

import streamlit as st
from utils import extract_topics_and_entities

class CAGSystem:
    @staticmethod
    def update_user_context(key, value):
        if key and value:
            st.session_state.user_context[key.strip()] = {
                "value": value.strip(), "timestamp": datetime.now().isoformat()
            }
    
    @staticmethod
    def get_user_context(): return st.session_state.user_context
    
    @staticmethod
    def get_relevant_context(query: str) -> Dict[str, str]:
        if not st.session_state.user_context: return {}
        relevant_context = {}
        query_lower = query.lower()
        for key, data in st.session_state.user_context.items():
            if key.lower() in query_lower or any(word in key.lower() for word in query_lower.split()):
                relevant_context[key] = data["value"]
        return relevant_context

    @staticmethod
    def infer_preferences(query: str) -> Dict[str, Any]:
        """Infer user preferences from their query"""
        preferences = {}
        
        # Look for explicit preferences
        preference_patterns = [
            (r"(?:prefer|like|want)s?\s+([^.?!;]+)", "preference"),
            (r"interested in\s+([^.?!;]+)", "interest"),
            (r"don't (?:like|want|need)\s+([^.?!;]+)", "dislike"),
        ]
        
        for pattern, pref_type in preference_patterns:
            matches = re.findall(pattern, query, re.IGNORECASE)
            for match in matches:
                preferences[match.strip()] = pref_type
                
        return preferences

    @staticmethod
    def update_conversation_context(user_input, system_response=None):
        """Update the conversation context with new information from the exchange"""
        # Extract and add topics/entities
        new_topics, new_entities = extract_topics_and_entities(user_input)
        st.session_state.conversation_context["topics"].update(new_topics)
        st.session_state.conversation_context["entities"].update(new_entities)
        
        # Track recent queries (keep last 5)
        st.session_state.conversation_context["last_queries"].append(user_input)
        if len(st.session_state.conversation_context["last_queries"]) > 5:
            st.session_state.conversation_context["last_queries"].pop(0)
        
        # Check if query might be unresolved
        if system_response and any(phrase in system_response.lower() for phrase in 
                                ["don't know", "cannot answer", "no information", "not enough context"]):
            st.session_state.conversation_context["unresolved_queries"].append(user_input)
        
        # Identify potential user preferences
        preference_patterns = [
            (r"I (?:prefer|like|want) (.*)", "preference"),
            (r"I'm interested in (.*)", "interest"),
            (r"I don't (?:like|want|need) (.*)", "dislike")
        ]
        
        for pattern, pref_type in preference_patterns:
            matches = re.findall(pattern, user_input, re.IGNORECASE)
            for match in matches:
                st.session_state.conversation_context["user_preferences"][match.strip()] = pref_type