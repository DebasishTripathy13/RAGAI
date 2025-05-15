import re
from typing import List, Optional

from config import runtime_config

def clean_text(text: str) -> str:
    """Clean and normalize text."""
    text = re.sub(r'\s+', ' ', text)  # Remove extra whitespace
    text = re.sub(r'[^\w\s\.\,\?\!\:\;\(\)\[\]\{\}\-\–\—\'\"\`]', '', text)  # Keep specific punctuation
    return text.strip()


def estimate_text_density(text: str) -> int:
    """Estimate text density to determine appropriate chunk size."""
    if not text:
        return runtime_config.medium_chunk_size

    word_count = len(text.split())
    # More robust sentence count, though still an estimate
    sentence_count = len(re.findall(r'[.!?]+', text)) if re.search(r'[.!?]', text) else 1
    special_char_count = len(re.findall(r'[^\w\s]', text))

    avg_words_per_sentence = word_count / max(sentence_count, 1)
    special_char_ratio = special_char_count / max(len(text), 1)

    if avg_words_per_sentence > 25 or special_char_ratio < 0.05:
        return runtime_config.large_chunk_size
    elif avg_words_per_sentence < 10 or special_char_ratio > 0.15:
        return runtime_config.small_chunk_size
    else:
        return runtime_config.medium_chunk_size


def split_into_chunks(text: str, chunk_size: Optional[int] = None, chunk_overlap: int = 100) -> List[str]:
    """Split text into overlapping chunks with adaptive sizing."""
    from config import CHUNK_OVERLAP
    
    text = clean_text(text)
    chunks = []

    if chunk_size is None:
        chunk_size = estimate_text_density(text)

    if len(text) <= chunk_size:
        return [text] if text else []

    start = 0
    while start < len(text):
        end = min(start + chunk_size, len(text))

        current_chunk_text = text[start:end]

        if end < len(text):  # If not the last chunk, try to find a better split point
            # This means looking backward from 'end' but not too far back from 'start + chunk_size - chunk_overlap'
            # More simply, find a good split point within the current_chunk_text
            split_search_area = text[start: min(start + chunk_size, len(text))]
            
            # Try to find the last period, question mark, or exclamation point
            last_sentence_end_char = max(
                split_search_area.rfind('.'),
                split_search_area.rfind('?'),
                split_search_area.rfind('!')
            )
            last_newline = split_search_area.rfind('\n')
            
            # Prefer sentence boundaries, then newlines, if they are reasonably close to the desired end
            # and not too close to the start (e.g., > chunk_overlap distance from start)
            best_split_point = -1
            if last_sentence_end_char > chunk_overlap:  # Ensure the chunk is not too small
                 best_split_point = last_sentence_end_char
            
            if last_newline > chunk_overlap and last_newline > best_split_point:  # Prefer newline if it's later and valid
                 best_split_point = last_newline

            if best_split_point != -1:
                actual_end_in_text = start + best_split_point + 1
                chunks.append(text[start:actual_end_in_text].strip())
                start = actual_end_in_text - chunk_overlap
            else:  # No good split point found, take the chunk as is
                chunks.append(text[start:end].strip())
                start = end - chunk_overlap
        else:  # This is the last chunk or text is smaller than chunk_size
            chunks.append(text[start:end].strip())
            start = end  # Move to the end

        if start >= len(text):  # Ensure loop termination if overlap logic pushes start beyond text length
            break
            
    return [c for c in chunks if c]  # Filter out empty chunks


def extract_topics_and_entities(text: str) -> tuple:
    """
    Extract potential topics and entities from user input.
    This is a simple implementation - in production, use NLP libraries like spaCy.
    """
    # Simple topic extraction based on keywords
    topics = set()
    entities = set()
    
    # Common topics that might be discussed
    topic_keywords = {
        "technical": ["code", "programming", "debug", "error", "function", "api"],
        "business": ["company", "market", "strategy", "customer", "product"],
        "support": ["help", "issue", "problem", "ticket", "assistance"],
        "information": ["what is", "tell me about", "explain", "information", "details", "how to"]
    }
    
    text_lower = text.lower()
    
    # Check for topics
    for topic, keywords in topic_keywords.items():
        if any(keyword in text_lower for keyword in keywords):
            topics.add(topic)
    
    # Very basic entity extraction (could be replaced with NER from spaCy)
    # Look for capitalized words that might be entities
    potential_entities = re.findall(r'\b[A-Z][a-zA-Z]*\b', text)
    entities.update(potential_entities)
    
    return topics, entities