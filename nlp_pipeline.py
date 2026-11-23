"""Reusable NLP utilities for QuestionCraft AI."""
import re
from collections import Counter

import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import sent_tokenize, word_tokenize
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

_REQUIRED_NLTK = {
    "tokenizers/punkt": "punkt",
    "tokenizers/punkt_tab": "punkt_tab",
    "corpora/stopwords": "stopwords",
    "corpora/wordnet": "wordnet",
    "corpora/omw-1.4": "omw-1.4",
}

def ensure_nltk_data():
    for resource, package in _REQUIRED_NLTK.items():
        try:
            nltk.data.find(resource)
        except LookupError:
            nltk.download(package, quiet=True)

def clean_text(text: str) -> str:
    """Normalize whitespace and remove common extraction artifacts conservatively."""
    text = str(text or "").replace("\u00ad", "")
    text = re.sub(r"(?<=\w)-\s*\n\s*(?=\w)", "", text)
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    text = re.sub(r"[^\S\n]+", " ", text)
    return text.strip()

def preprocess_text(text: str):
    """Return cleaned text, sentences, tokens and lemmatized content words."""
    ensure_nltk_data()
    cleaned = clean_text(text)
    try:
        sentences = sent_tokenize(cleaned)
    except LookupError:
        nltk.download("punkt", quiet=True)
        nltk.download("punkt_tab", quiet=True)
        sentences = sent_tokenize(cleaned)
    try:
        stop_words = set(stopwords.words("english"))
    except LookupError:
        nltk.download("stopwords", quiet=True)
        stop_words = set(stopwords.words("english"))
    lemmatizer = WordNetLemmatizer()
    tokens = []
    lemmas = []
    for token in word_tokenize(cleaned.lower()):
        if re.fullmatch(r"[a-z][a-z'-]*", token):
            tokens.append(token)
            if token not in stop_words:
                try:
                    lemmas.append(lemmatizer.lemmatize(token))
                except LookupError:
                    nltk.download("wordnet", quiet=True)
                    lemmas.append(lemmatizer.lemmatize(token))
    return {
        "cleaned_text": cleaned,
        "sentences": sentences,
        "tokens": tokens,
        "lemmas": lemmas,
        "stopwords_removed": len(tokens) - len(lemmas),
    }

def split_into_chunks(text: str, max_chars: int = 1800):
    """Split text on paragraph/sentence boundaries into manageable chunks."""
    text = clean_text(text)
    if not text:
        return []
    paragraphs = [p.strip() for p in re.split(r"\n\s*\n", text) if p.strip()]
    chunks, current = [], ""
    for paragraph in paragraphs:
        pieces = [paragraph]
        if len(paragraph) > max_chars:
            pieces = re.split(r"(?<=[.!?])\s+", paragraph)
        for piece in pieces:
            if len(piece) > max_chars:
                for start in range(0, len(piece), max_chars):
                    sub = piece[start:start + max_chars]
                    if current:
                        chunks.append(current)
                        current = ""
                    chunks.append(sub)
                continue
            candidate = f"{current}\n{piece}".strip() if current else piece
            if len(candidate) > max_chars and current:
                chunks.append(current)
                current = piece
            else:
                current = candidate
    if current:
        chunks.append(current)
    return chunks

def tfidf_keywords(text: str, top_n: int = 15):
    chunks = split_into_chunks(text)
    if not chunks:
        return [], None
    vectorizer = TfidfVectorizer(max_features=1000, stop_words="english", ngram_range=(1, 2))
    try:
        matrix = vectorizer.fit_transform(chunks)
    except ValueError:
        return [], None
    scores = matrix.mean(axis=0).A1
    terms = vectorizer.get_feature_names_out()
    ranked = sorted(zip(terms, scores), key=lambda x: x[1], reverse=True)[:top_n]
    return [(term, float(score)) for term, score in ranked], matrix

def topic_model(text: str, n_topics: int = 4, top_words: int = 8):
    chunks = split_into_chunks(text)
    if len(chunks) < 2:
        return [], "Please provide at least two paragraphs or sections for topic modelling."
    vectorizer = CountVectorizer(stop_words="english", max_features=1500)
    try:
        matrix = vectorizer.fit_transform(chunks)
    except ValueError:
        return [], "Not enough meaningful vocabulary to detect topics."
    if matrix.shape[1] < 2:
        return [], "Not enough distinct terms to detect topics."
    n_topics = max(2, min(int(n_topics), len(chunks), 6))
    model = LatentDirichletAllocation(
        n_components=n_topics, random_state=42, learning_method="batch", max_iter=10
    )
    model.fit(matrix)
    terms = vectorizer.get_feature_names_out()
    topics = []
    for idx, weights in enumerate(model.components_):
        top_indices = weights.argsort()[-top_words:][::-1]
        topics.append({
            "topic": idx + 1,
            "keywords": [str(terms[i]) for i in top_indices],
        })
    return topics, None

def sentiment_analyze(text: str):
    analyzer = SentimentIntensityAnalyzer()
    scores = analyzer.polarity_scores(str(text or ""))
    compound = scores["compound"]
    label = "Positive" if compound >= 0.05 else "Negative" if compound <= -0.05 else "Neutral"
    return {"label": label, **scores}

def extract_information(text: str):
    """Use spaCy English NER when its small English model is installed."""
    try:
        import spacy
        nlp = spacy.load("en_core_web_sm")
    except (ImportError, OSError):
        return [], "spaCy English model is not installed. Run: python -m spacy download en_core_web_sm"
    doc = nlp(clean_text(text)[:100000])
    entities = []
    seen = set()
    for ent in doc.ents:
        key = (ent.text.strip(), ent.label_)
        if key not in seen:
            seen.add(key)
            entities.append({"text": ent.text.strip(), "label": ent.label_})
    noun_chunks = []
    try:
        noun_chunks = list(dict.fromkeys(chunk.text.strip() for chunk in doc.noun_chunks if len(chunk.text.strip()) > 2))
    except Exception:
        pass
    return {"entities": entities, "noun_chunks": noun_chunks[:30]}, None
