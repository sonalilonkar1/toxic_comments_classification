"""Text cleaning utilities for preprocessing toxic comments."""

import re
from typing import Optional

import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize

# Download required NLTK data (will only download if not already present)
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt', quiet=True)

try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords', quiet=True)

try:
    nltk.data.find('corpora/wordnet')
except LookupError:
    nltk.download('wordnet', quiet=True)

# Contractions mapping
CONTRACTIONS = {
    "ain't": "am not",
    "aren't": "are not",
    "can't": "cannot",
    "can't've": "cannot have",
    "'cause": "because",
    "could've": "could have",
    "couldn't": "could not",
    "couldn't've": "could not have",
    "didn't": "did not",
    "doesn't": "does not",
    "don't": "do not",
    "hadn't": "had not",
    "hadn't've": "had not have",
    "hasn't": "has not",
    "haven't": "have not",
    "he'd": "he would",
    "he'd've": "he would have",
    "he'll": "he will",
    "he'll've": "he will have",
    "he's": "he is",
    "how'd": "how did",
    "how'd'y": "how do you",
    "how'll": "how will",
    "how's": "how is",
    "i'd": "i would",
    "i'd've": "i would have",
    "i'll": "i will",
    "i'll've": "i will have",
    "i'm": "i am",
    "i've": "i have",
    "isn't": "is not",
    "it'd": "it would",
    "it'd've": "it would have",
    "it'll": "it will",
    "it'll've": "it will have",
    "it's": "it is",
    "let's": "let us",
    "ma'am": "madam",
    "mayn't": "may not",
    "might've": "might have",
    "mightn't": "might not",
    "mightn't've": "might not have",
    "must've": "must have",
    "mustn't": "must not",
    "mustn't've": "must not have",
    "needn't": "need not",
    "needn't've": "need not have",
    "o'clock": "of the clock",
    "oughtn't": "ought not",
    "oughtn't've": "ought not have",
    "shan't": "shall not",
    "sha'n't": "shall not",
    "shan't've": "shall not have",
    "she'd": "she would",
    "she'd've": "she would have",
    "she'll": "she will",
    "she'll've": "she will have",
    "she's": "she is",
    "should've": "should have",
    "shouldn't": "should not",
    "shouldn't've": "should not have",
    "so've": "so have",
    "so's": "so is",
    "that'd": "that would",
    "that'd've": "that would have",
    "that's": "that is",
    "there'd": "there would",
    "there'd've": "there would have",
    "there's": "there is",
    "they'd": "they would",
    "they'd've": "they would have",
    "they'll": "they will",
    "they'll've": "they will have",
    "they're": "they are",
    "they've": "they have",
    "to've": "to have",
    "wasn't": "was not",
    "we'd": "we would",
    "we'd've": "we would have",
    "we'll": "we will",
    "we'll've": "we will have",
    "we're": "we are",
    "we've": "we have",
    "weren't": "were not",
    "what'll": "what will",
    "what'll've": "what will have",
    "what're": "what are",
    "what's": "what is",
    "what've": "what have",
    "when's": "when is",
    "when've": "when have",
    "where'd": "where did",
    "where's": "where is",
    "where've": "where have",
    "who'll": "who will",
    "who'll've": "who will have",
    "who's": "who is",
    "who've": "who have",
    "why's": "why is",
    "why've": "why have",
    "will've": "will have",
    "won't": "will not",
    "won't've": "will not have",
    "would've": "would have",
    "wouldn't": "would not",
    "wouldn't've": "would not have",
    "y'all": "you all",
    "y'all'd": "you all would",
    "y'all'd've": "you all would have",
    "y'all're": "you all are",
    "y'all've": "you all have",
    "you'd": "you would",
    "you'd've": "you would have",
    "you'll": "you will",
    "you'll've": "you will have",
    "you're": "you are",
    "you've": "you have",
}


def expand_contractions(text: str) -> str:
    """Expand contractions in text."""
    pattern = re.compile(
        r'\b(' + '|'.join(re.escape(k) for k in CONTRACTIONS.keys()) + r')\b',
        flags=re.IGNORECASE | re.DOTALL
    )
    return pattern.sub(lambda x: CONTRACTIONS[x.group().lower()], text)


def remove_urls(text: str) -> str:
    """Remove URLs from text."""
    url_pattern = re.compile(
        r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+'
    )
    return url_pattern.sub('', text)


def remove_emails(text: str) -> str:
    """Remove email addresses from text."""
    email_pattern = re.compile(r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b')
    return email_pattern.sub('', text)


def remove_special_chars(text: str, keep_punctuation: bool = True) -> str:
    """Remove special characters, optionally keeping punctuation."""
    if keep_punctuation:
        # Keep alphanumeric, spaces, and common punctuation
        pattern = re.compile(r'[^a-zA-Z0-9\s.,!?;:\'\"-]')
    else:
        # Keep only alphanumeric and spaces
        pattern = re.compile(r'[^a-zA-Z0-9\s]')
    return pattern.sub('', text)


def normalize_whitespace(text: str) -> str:
    """Normalize whitespace (multiple spaces to single space, trim)."""
    text = re.sub(r'\s+', ' ', text)
    return text.strip()


def remove_stopwords(text: str, language: str = 'english') -> str:
    """Remove stopwords from text."""
    try:
        stop_words = set(stopwords.words(language))
    except LookupError:
        nltk.download('stopwords', quiet=True)
        stop_words = set(stopwords.words(language))
    
    words = word_tokenize(text.lower())
    filtered_words = [word for word in words if word not in stop_words]
    return ' '.join(filtered_words)


def lemmatize_text(text: str) -> str:
    """Lemmatize text using WordNet lemmatizer."""
    lemmatizer = WordNetLemmatizer()
    words = word_tokenize(text)
    lemmatized_words = [lemmatizer.lemmatize(word) for word in words]
    return ' '.join(lemmatized_words)


def clean_text(
    text: str,
    lowercase: bool = True,
    remove_urls_flag: bool = True,
    remove_emails_flag: bool = True,
    expand_contractions_flag: bool = True,
    remove_special_chars_flag: bool = True,
    keep_punctuation: bool = True,
    remove_stopwords_flag: bool = False,
    lemmatize: bool = False,
) -> str:
    """
    Comprehensive text cleaning function.
    
    Args:
        text: Input text to clean
        lowercase: Convert to lowercase
        remove_urls_flag: Remove URLs
        remove_emails_flag: Remove email addresses
        expand_contractions_flag: Expand contractions
        remove_special_chars_flag: Remove special characters
        keep_punctuation: If True, keep punctuation when removing special chars
        remove_stopwords_flag: Remove stopwords (use for traditional ML)
        lemmatize: Apply lemmatization (use for traditional ML)
    
    Returns:
        Cleaned text string
    """
    if not isinstance(text, str):
        text = str(text)
    
    # Expand contractions first (before lowercasing to preserve case patterns)
    if expand_contractions_flag:
        text = expand_contractions(text)
    
    # Lowercase
    if lowercase:
        text = text.lower()
    
    # Remove URLs
    if remove_urls_flag:
        text = remove_urls(text)
    
    # Remove emails
    if remove_emails_flag:
        text = remove_emails(text)
    
    # Remove special characters
    if remove_special_chars_flag:
        text = remove_special_chars(text, keep_punctuation=keep_punctuation)
    
    # Normalize whitespace
    text = normalize_whitespace(text)
    
    # Remove stopwords (requires tokenization)
    if remove_stopwords_flag:
        text = remove_stopwords(text)
    
    # Lemmatize (requires tokenization)
    if lemmatize:
        text = lemmatize_text(text)
    
    # Final whitespace normalization
    text = normalize_whitespace(text)
    
    return text


def clean_text_minimal(text: str) -> str:
    """
    Minimal text cleaning for BERT/DistilBERT models.
    Preserves most structure while doing basic cleanup.
    
    Args:
        text: Input text to clean
    
    Returns:
        Minimally cleaned text
    """
    if not isinstance(text, str):
        text = str(text)
    
    # Only normalize whitespace and remove excessive newlines
    text = re.sub(r'\n+', ' ', text)
    text = normalize_whitespace(text)
    
    return text

