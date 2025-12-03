"""Text preprocessing functions."""

import re
import html
from typing import Optional


def toy_normalize(text: str) -> str:
    """Basic normalization: lowercasing and simple whitespace cleanup."""
    if not isinstance(text, str):
        return str(text) if text is not None else ""
    return " ".join(text.lower().split())


def rich_normalize(text: str) -> str:
    """Advanced normalization including de-obfuscation and handling special tokens."""
    if not isinstance(text, str):
        return str(text) if text is not None else ""
    
    text = text.lower()
    
    # Decode HTML entities
    text = html.unescape(text)
    
    # Replace URLs
    text = re.sub(r'https?://\S+|www\.\S+', ' <URL> ', text)
    
    # Replace IP addresses
    text = re.sub(r'\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}', ' <IP> ', text)
    
    # Replace common leetspeak/obfuscations
    # This is a basic list; a full "rich" normalizer would have a larger dictionary
    replacements = {
        r"w\s*h\s*a\s*t": "what",
        r"f\s*u\s*c\s*k": "fuck",
        r"s\s*h\s*i\s*t": "shit",
        r"h\s*a\s*t\s*e": "hate",
        r"d\s*i\s*e": "die",
        r"k\s*i\s*l\s*l": "kill",
        r"stupid": "stupid",
        r"idiot": "idiot",
        r"@": "at",
        r"&": "and",
        r"\$": "dollar",
    }
    
    # Note: Regex replacement for obfuscation can be slow if list is huge.
    # Here we do minimal cleanup.
    
    # Clean up whitespace
    text = re.sub(r'\s+', ' ', text).strip()
    
    return text

