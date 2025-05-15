import re
import streamlit as st
from datetime import datetime
from dateutil import parser
from typing import List, Dict, Any

class DynamicPrompting:
    @staticmethod
    def get_adaptive_system_prompt(query: str, conversation_context: Dict = None):
        """Generate an adaptive system prompt based on the conversation context"""
        if not conversation_context:
            conversation_context = st.session_state.conversation_context
        
        # Start with a base system prompt
        system_prompt_parts = [
            "You are a helpful and knowledgeable AI assistant. Answer based on the provided context."
        ]
        
        # Add context-aware instructions
        topics = conversation_context.get("topics", set())
        if "technical" in topics:
            system_prompt_parts.append("Provide technically accurate and detailed explanations.")
        if "business" in topics:
            system_prompt_parts.append("Focus on business value and practical applications.")
        if "support" in topics:
            system_prompt_parts.append("Offer troubleshooting steps and direct solutions to problems.")
            
        # Handle unanswered queries
        unresolved = conversation_context.get("unresolved_queries", [])
        if unresolved and len(unresolved) > 0:
            system_prompt_parts.append("The user has previously asked questions I couldn't fully answer. "
                                      "If the current query relates to these topics, admit limitations clearly.")
            
        # Instruction on source citation
        system_prompt_parts.append("Cite sources using [Source X] notation, corresponding to the numbered sources.")
        
        # Instruction on unknown information
        system_prompt_parts.append("If information is not in the provided context, clearly state that.")
        
        return " ".join(system_prompt_parts)

    @staticmethod
    def get_enhanced_user_prompt(query: str, relevant_docs: List[Dict] = None, user_context: Dict = None):
        """Build an enhanced prompt with relevant docs and context"""
        prompt_parts = [f"User query: {query}\n"]
        
        if relevant_docs:
            prompt_parts.append("Relevant Information:")
            for i, doc in enumerate(relevant_docs):
                source_info = doc.get("metadata", {})
                source_type = source_info.get("source_type", "unknown")
                title = source_info.get('title', source_info.get('filename', source_info.get('url', f'Source {i+1}')))
                prompt_parts.append(f"[Source {i+1}: {title}]\n{doc['content']}\n")
        else:
            prompt_parts.append("Relevant Information: None provided.\n")

        if user_context:
            prompt_parts.append("User-specific Information:")
            for key, value in user_context.items():
                prompt_parts.append(f"- {key}: {value}")
            prompt_parts.append("\n")
        
        # Add conversation context
        import streamlit as st  # Import here to avoid circular imports
        conv_context = st.session_state.conversation_context
        recent_queries = conv_context.get("last_queries", [])
        if recent_queries and len(recent_queries) > 1:
            prompt_parts.append("Recent conversation context:")
            # Only include the last few queries to avoid context overload
            for i, recent_query in enumerate(recent_queries[-3:]):
                prompt_parts.append(f"- Previous query {i+1}: {recent_query}")
            prompt_parts.append("\n")
            
        prompt_parts.append("Based on all the above information, please answer the user's query. Remember to cite your sources and indicate if information is unavailable in the context.")
        
        return "\n".join(prompt_parts)

    @staticmethod
    def generate_follow_up_questions(query: str, response: str, relevant_docs: List[Dict]) -> List[str]:
        """Generate follow-up questions based on the query, response, and relevant documents"""
        # Extract topics from the documents
        topics = set()
        for doc in relevant_docs:
            doc_text = doc.get('content', '')
            # Simple keyword extraction - in production use proper NLP
            for word in re.findall(r'\b\w+\b', doc_text.lower()):
                if len(word) > 5 and word not in ['should', 'would', 'could', 'because', 'about']:
                    topics.add(word)
        
        # Generate follow-up questions based on common patterns and extracted topics
        follow_ups = []
        
        # Check if the response indicates uncertainty
        if any(phrase in response.lower() for phrase in ["i don't know", "not enough information", "can't determine"]):
            follow_ups.append("Would you like me to explain what information would help answer your question better?")
        
        # Check for "how to" questions
        if "how" in query.lower() and any(word in query.lower() for word in ["do", "can", "make", "use", "implement"]):
            follow_ups.append("Would you like me to provide more detailed step-by-step instructions?")
        
        # Check for comparative questions
        if any(word in query.lower() for word in ["compare", "difference", "versus", "vs"]):
            follow_ups.append("Would you like me to compare specific aspects in more detail?")

        # Add topic-specific follow-ups
        topic_list = list(topics)
        if topic_list and len(topic_list) >= 2:
            t1, t2 = topic_list[0], topic_list[1]
            follow_ups.append(f"Would you like to know more about the relationship between {t1} and {t2}?")
            
        # Limit to 3 follow-up questions
        return follow_ups[:3]