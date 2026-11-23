import json
import os
import re
import uuid
from io import BytesIO
from pathlib import Path
from xml.sax.saxutils import escape

import requests
import streamlit as st
import pymupdf
from reportlab.lib import colors
from reportlab.lib.enums import TA_LEFT
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import mm
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, HRFlowable, KeepTogether

from nlp_pipeline import (
    clean_text, preprocess_text, tfidf_keywords, topic_model,
    sentiment_analyze, extract_information,
)

BASE_DIR = Path(__file__).resolve().parent
HISTORY_FILE = BASE_DIR / "question_history.json"
OLLAMA_URL = os.getenv("OLLAMA_URL", "http://localhost:11434")
DEFAULT_MODEL = os.getenv("OLLAMA_MODEL", "llama3.1:8b")

st.set_page_config(page_title="QuestionCraft AI", page_icon="📝", layout="wide", initial_sidebar_state="expanded")

def load_css():
    css_path = BASE_DIR / "styles" / "style.css"
    if css_path.exists():
        st.markdown(f"<style>{css_path.read_text(encoding='utf-8')}</style>", unsafe_allow_html=True)

load_css()

st.markdown("""
<div class="hero">
  <div class="hero-kicker">AI-powered learning workspace</div>
  <h1>QuestionCraft AI</h1>
  <p>Turn your notes into structured question papers with NLP analysis, local AI-generated answers,
  Bloom's Taxonomy settings, and print-ready PDF downloads.</p>
</div>
""", unsafe_allow_html=True)

def load_history():
    try:
        data = json.loads(HISTORY_FILE.read_text(encoding="utf-8"))
        return data.get("papers", [])
    except (OSError, json.JSONDecodeError):
        return []

def save_history(papers):
    HISTORY_FILE.write_text(json.dumps({"papers": papers[-50:]}, indent=2, ensure_ascii=False), encoding="utf-8")

def safe_filename(value):
    return re.sub(r"[^A-Za-z0-9_-]+", "_", str(value).strip()).strip("_") or "question_paper"

def ollama_status():
    try:
        response = requests.get(f"{OLLAMA_URL}/api/tags", timeout=3)
        response.raise_for_status()
        return True, response.json().get("models", [])
    except requests.RequestException:
        return False, []

def get_openrouter_api_key():
    """Read the OpenRouter key from environment variables or Streamlit secrets."""
    api_key = os.getenv("OPENROUTER_API_KEY", "").strip()
    if not api_key:
        try:
            api_key = str(st.secrets.get("OPENROUTER_API_KEY", "")).strip()
        except Exception:
            api_key = ""
    return api_key


def call_ollama(prompt, model):
    response = requests.post(
        f"{OLLAMA_URL}/api/generate",
        json={
            "model": model,
            "prompt": prompt,
            "stream": False,
            "format": "json",
            "options": {"temperature": 0.3, "num_ctx": 4096},
        },
        timeout=600,
    )
    response.raise_for_status()
    content = response.json().get("response", "")
    if not content:
        raise ValueError("Ollama returned an empty response.")
    return json.loads(content)


def call_openrouter(prompt):
    api_key = get_openrouter_api_key()
    if not api_key:
        raise ValueError("OpenRouter API key is missing. Add OPENROUTER_API_KEY to Streamlit secrets.")

    response = requests.post(
        "https://openrouter.ai/api/v1/chat/completions",
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
            "X-Title": "QuestionCraft AI",
        },
        json={
            "model": "openrouter/free",
            "messages": [{"role": "user", "content": prompt}],
            "temperature": 0.3,
            "response_format": {"type": "json_object"},
        },
        timeout=180,
    )
    response.raise_for_status()
    data = response.json()
    choices = data.get("choices", [])
    if not choices:
        raise ValueError(f"OpenRouter returned no choices: {data}")
    content = choices[0].get("message", {}).get("content", "")
    if not content:
        raise ValueError("OpenRouter returned an empty response.")
    return json.loads(content)

def bloom_guidance(level):
    guidance = {
        "Remember": "Recall facts, define, list, identify, name.",
        "Understand": "Explain, summarize, describe, compare, classify.",
        "Apply": "Use concepts to solve a new problem, demonstrate, calculate.",
        "Analyze": "Examine relationships, differentiate, organize, infer.",
        "Evaluate": "Justify a decision, critique, assess, defend using evidence.",
        "Create": "Design, formulate, construct, develop an original solution.",
    }
    return guidance.get(level, "")
def pdf_safe(text):
    text = str(text)

    replacements = {
        "\u2010": "-",
        "\u2011": "-",
        "\u2012": "-",
        "\u2013": "-",
        "\u2014": "-",
        "\u2212": "-",
        "\u2192": "->",
        "\u2190": "<-",
        "\u00a0": " ",
        "\u2022": "-",
    }

    for old, new in replacements.items():
        text = text.replace(old, new)

    return text.encode("cp1252", errors="replace").decode("cp1252")

def make_pdf(title, subject, difficulty, question_type, bloom_level, questions, include_answers=False):
    buffer = BytesIO()
    doc = SimpleDocTemplate(buffer, pagesize=A4, rightMargin=19*mm, leftMargin=19*mm,
                            topMargin=19*mm, bottomMargin=18*mm, title=title, author="QuestionCraft AI")
    styles = getSampleStyleSheet()
    styles.add(ParagraphStyle(name="QC_Title", parent=styles["Title"], fontName="Helvetica-Bold",
                              fontSize=21, leading=26, textColor=colors.HexColor("#5142B8"),
                              alignment=TA_LEFT, spaceAfter=8))
    styles.add(ParagraphStyle(name="QC_Meta", parent=styles["Normal"], fontSize=9, leading=14,
                              textColor=colors.HexColor("#5D6478"), spaceAfter=3))
    styles.add(ParagraphStyle(name="QC_Question", parent=styles["Normal"], fontName="Helvetica-Bold",
                              fontSize=10.5, leading=15, textColor=colors.HexColor("#20243A"),
                              spaceBefore=9, spaceAfter=5))
    styles.add(ParagraphStyle(name="QC_Body", parent=styles["Normal"], fontSize=9.5, leading=14,
                              leftIndent=10, textColor=colors.HexColor("#41485B"), spaceAfter=3))
    story = [
        Paragraph(escape(title), styles["QC_Title"]),
        Paragraph(f"<b>Subject:</b> {escape(str(subject))}", styles["QC_Meta"]),
        Paragraph(f"<b>Difficulty:</b> {escape(str(difficulty))}", styles["QC_Meta"]),
        Paragraph(f"<b>Question Type:</b> {escape(str(question_type))}", styles["QC_Meta"]),
        Paragraph(f"<b>Bloom's Level:</b> {escape(str(bloom_level))}", styles["QC_Meta"]),
        Spacer(1, 6*mm),
        HRFlowable(width="100%", thickness=1, color=colors.HexColor("#DAD7FA")),
        Spacer(1, 3*mm),
    ]
    section_names = {"MCQ": "SECTION A – MULTIPLE CHOICE QUESTIONS", "Short Answer": "SECTION B – SHORT ANSWER QUESTIONS", "Long Answer": "SECTION C – LONG ANSWER QUESTIONS"}
    section_style = ParagraphStyle(name="QC_Section", parent=styles["Heading2"], textColor=colors.HexColor("#5142B8"), spaceBefore=14, spaceAfter=7)
    ordered = []
    for kind in ("MCQ", "Short Answer", "Long Answer"):
        ordered.extend([q for q in questions if q.get("question_type", kind) == kind])
    if not ordered:
        ordered = questions
    current_kind = None
    for i, item in enumerate(ordered, 1):
        kind = item.get("question_type", "MCQ")
        if kind != current_kind:
            story.append(Paragraph(escape(section_names.get(kind, kind.upper())), section_style))
            current_kind = kind
        block = [Paragraph(f"Q{i}. {escape(pdf_safe(item.get('question', '')))}", styles["QC_Question"])]
        opts = item.get("options", [])
        if isinstance(opts, list):
            block.extend(Paragraph(escape(pdf_safe(opt)), styles["QC_Body"]) for opt in opts)
        if include_answers:
            block.append(Paragraph(f"<b>Answer:</b> {escape(str(item.get('correct_answer', 'Not provided')))}", styles["QC_Body"]))
        story.append(KeepTogether(block))
    doc.build(story)
    buffer.seek(0)
    return buffer.getvalue()

# AI provider status
openrouter_api_key = get_openrouter_api_key()
ollama_ready, available_models = ollama_status()
with st.sidebar:
    st.header("AI provider status")
    if openrouter_api_key:
        st.success("OpenRouter API key detected")
        st.caption("Question generation will use OpenRouter free models.")
    else:
        st.warning("OpenRouter API key not found")
        st.caption("Add OPENROUTER_API_KEY in Streamlit secrets to use OpenRouter.")
    st.divider()
    st.caption("Optional local fallback")
    model_name = st.text_input("Ollama model", value=DEFAULT_MODEL)
    if ollama_ready:
        st.success("Ollama is running")
        if available_models:
            st.caption("Installed models: " + ", ".join(m.get("name", "") for m in available_models))
    else:
        st.caption("Ollama is not running. It is only needed if you want local fallback.")

# Input
st.markdown('<div class="section-title">01 · Add your study material</div>', unsafe_allow_html=True)
st.markdown('<div class="section-caption">Paste notes or upload a text-based PDF.</div>', unsafe_allow_html=True)
with st.container(border=True):
    input_method = st.radio("Choose your input method", ["Paste Text", "Upload PDF"], horizontal=True)
    raw_material = ""
    if input_method == "Paste Text":
        raw_material = st.text_area("Study notes", height=190, placeholder="Paste your chapter notes here...")
    else:
        uploaded = st.file_uploader("Upload study material", type=["pdf"])
        if uploaded:
            try:
                with pymupdf.open(stream=uploaded.getvalue(), filetype="pdf") as pdf:
                    raw_material = "\n\n".join(page.get_text() for page in pdf)
                if raw_material.strip():
                    st.success("PDF text extracted successfully.")
                    with st.expander("Preview extracted text"):
                        st.text_area("Extracted text", value=raw_material, height=160, disabled=True)
                else:
                    st.warning("No readable text found. Scanned PDFs require OCR before they can be processed.")
            except Exception as exc:
                st.error(f"Could not read the PDF: {exc}")

# NLP processing
cleaned_material = clean_text(raw_material)
analysis = None
keywords = []
topics = []
info = None
if cleaned_material:
    with st.spinner("Running NLP preprocessing..."):
        analysis = preprocess_text(cleaned_material)
        keywords, _ = tfidf_keywords(cleaned_material)
        topics, topic_error = topic_model(cleaned_material)
        info, info_error = extract_information(cleaned_material)
    st.caption(f"Study material ready · {len(cleaned_material.split())} words · {len(analysis['sentences'])} sentences")
    with st.expander("View NLP analysis · CO1–CO3", expanded=False):
        st.caption("These results are produced by NLTK, Scikit-learn and spaCy. OpenRouter (or optional Ollama fallback) is used for question generation.")
        tab_pre, tab_info, tab_topics = st.tabs(["Preprocessing & TF-IDF", "Information extraction", "Topic modelling"])
        with tab_pre:
            c1, c2, c3 = st.columns(3)
            c1.metric("Word tokens", len(analysis["tokens"]))
            c2.metric("Lemmas after stopword removal", len(analysis["lemmas"]))
            c3.metric("Sentences", len(analysis["sentences"]))
            with st.expander("View cleaned text"):
                st.text_area(
                    "Cleaned study material",
                    value=cleaned_material[:10000],
                    height=300,
                    disabled=True,
                    label_visibility="collapsed",
                )
            with st.expander("View tokenization and lemmatization"):
                st.write("**Tokens:**", ", ".join(analysis["tokens"][:60]) or "No tokens found.")
                st.write("**Lemmas:**", ", ".join(analysis["lemmas"][:60]) or "No lemmas found.")
                st.caption(f"Stopwords removed: {analysis['stopwords_removed']}")
            st.write("**Top TF-IDF terms**")
            if keywords:
                st.dataframe([{"Term": term, "TF-IDF score": round(score, 4)} for term, score in keywords], use_container_width=True, hide_index=True)
            else:
                st.info("More text is needed for TF-IDF analysis.")
        with tab_info:
            if info_error:
                st.warning(info_error)
            else:
                st.write("**Named entities (spaCy NER)**")
                if info["entities"]:
                    st.dataframe(info["entities"], use_container_width=True, hide_index=True)
                else:
                    st.info("No named entities were detected in this material.")
                st.write("**Noun phrases**")
                st.write(", ".join(info["noun_chunks"]) or "No noun phrases detected.")
        with tab_topics:
            if topic_error:
                st.info(topic_error)
            else:
                for topic in topics:
                    st.markdown(f"**Topic {topic['topic']}**")
                    st.write(", ".join(topic["keywords"]))

# Generator settings
st.markdown('<div class="section-title">02 · Customize your paper</div>', unsafe_allow_html=True)
st.markdown('<div class="section-caption">Choose how many questions of each type to include, then set difficulty and cognitive level.</div>', unsafe_allow_html=True)
with st.container(border=True):
    col_subject, col_difficulty, col_bloom = st.columns([2, 1, 1])
    with col_subject:
        subject = st.text_input("Subject name", placeholder="e.g., Machine Learning")
    with col_difficulty:
        difficulty = st.selectbox("Difficulty", ["Easy", "Medium", "Hard"], index=1)
    with col_bloom:
        bloom_level = st.selectbox("Bloom's Taxonomy", ["Remember", "Understand", "Apply", "Analyze", "Evaluate", "Create"], index=1)
    st.markdown("**Question distribution**")
    q1, q2, q3 = st.columns(3)
    with q1:
        mcq_count = st.number_input("MCQs", min_value=0, max_value=20, value=5, step=1)
    with q2:
        short_count = st.number_input("Short answers", min_value=0, max_value=20, value=3, step=1)
    with q3:
        long_count = st.number_input("Long answers", min_value=0, max_value=20, value=2, step=1)
    num_questions = int(mcq_count + short_count + long_count)
    st.info(f"Total questions in this paper: **{num_questions}**")
    generate_clicked = st.button("✨ Generate Question Paper", type="primary", use_container_width=True)

if generate_clicked:
    if not cleaned_material:
        st.warning("Please provide readable study material.")
    elif not subject.strip():
        st.warning("Please enter a subject name.")
    elif num_questions == 0:
        st.warning("Select at least one question in the distribution.")
    elif not openrouter_api_key and not ollama_ready:
        st.error("No AI provider is available. Add OPENROUTER_API_KEY to Streamlit secrets, or start Ollama for local fallback.")
    else:
        history = load_history()
        previous = [q.get("question", "") for p in history for q in p.get("questions", []) if isinstance(q, dict)]
        previous_text = "\n".join(f"- {q}" for q in previous[-20:])
        key_terms = ", ".join(term for term, _ in keywords[:12])
        # Keep prompt bounded while retaining representative chunks from throughout the notes.
        chunks = cleaned_material.split("\n\n")
        material_for_prompt = "\n\n".join(chunks[:12])[:14000]
        distribution = {"MCQ": int(mcq_count), "Short Answer": int(short_count), "Long Answer": int(long_count)}
        distribution_text = "\n".join(f"- {kind}: {count}" for kind, count in distribution.items())
       
        prompt = f"""
You are an expert university examination paper setter and answer writer.
Create a mixed-format question paper with complete model answers.
Return ONE valid JSON object only. Do not include Markdown outside JSON.

Required JSON structure:
{{
  "questions": [
    {{"question_type": "MCQ", "question": "Question text", "options": ["A. ...", "B. ...", "C. ...", "D. ..."], "correct_answer": "Correct option and brief explanation"}},
    {{"question_type": "Short Answer", "question": "Question text", "options": [], "correct_answer": "Concise, complete, exam-ready answer"}},
    {{"question_type": "Long Answer", "question": "Question text", "options": [], "correct_answer": "Detailed structured answer with headings and examples where supported"}}
  ]
}}

Subject: {subject}
Difficulty: {difficulty}
Bloom level: {bloom_level}
Bloom guidance: {bloom_guidance(bloom_level)}

Generate EXACTLY these quantities:
{distribution_text}

RULES:
1. Generate exactly the requested number of each question type.
2. Put MCQs first, then Short Answer, then Long Answer.
3. Every MCQ must have exactly four options and a non-empty correct answer.
4. Every Short Answer must have a concise but complete model answer with important keywords.
5. Every Long Answer must have a detailed answer with an introduction, relevant headings, explanations and examples where supported by the notes.
6. Long answers must be substantially more detailed than short answers.
7. Never leave correct_answer empty, null or omitted.
8. Use only information supported by the supplied study material; do not invent facts.
9. Make questions distinct and aligned with the selected difficulty and Bloom level.
10. Avoid repeating these previous questions: {previous_text}
11. Consider these extracted key terms where relevant: {key_terms}
12. Verify every answer is present and the requested distribution is exact before returning.

STUDY MATERIAL:
{material_for_prompt}
"""

        try:
            with st.spinner("Generating your question paper. This may take a little time..."):
                if openrouter_api_key:
                    result = call_openrouter(prompt)
                else:
                    result = call_ollama(prompt, model_name.strip())
            questions = result.get("questions", [])
            expected_total = num_questions
            if not isinstance(questions, list) or len(questions) != expected_total:
                st.error(f"The model returned {len(questions) if isinstance(questions, list) else 0} questions; {expected_total} were requested. Try fewer questions or run generation again.")
            else:
                valid = []
                for q in questions:
                    if isinstance(q, dict) and str(q.get("question", "")).strip():
                        kind = q.get("question_type", "")
                        if kind not in distribution:
                            continue
                        q["question_type"] = kind
                        q.setdefault("options", [])
                        q["options"] = q.get("options") if isinstance(q.get("options"), list) else []
                        answer = str(q.get("correct_answer", "")).strip()
                        if not answer:
                            continue
                        q["correct_answer"] = answer
                        if kind == "MCQ" and len(q["options"]) != 4:
                            continue
                        if kind != "MCQ":
                            q["options"] = []
                        valid.append(q)
                actual = {kind: sum(1 for q in valid if q["question_type"] == kind) for kind in distribution}
                if len(valid) != expected_total or actual != distribution:
                    st.error(f"The model did not follow the requested distribution. Requested: {distribution}. Received: {actual}. Please generate again, possibly with fewer questions.")
                else:
                    paper = {
                        "paper_id": str(uuid.uuid4()), "subject": subject.strip(),
                        "difficulty": difficulty, "question_type": "Mixed question paper",
                        "distribution": distribution, "bloom_level": bloom_level, "generated_at": __import__("datetime").datetime.now().strftime("%d %b %Y, %I:%M %p"), "questions": valid,
                    }
                    history.append(paper)
                    save_history(history)
                    st.session_state["current_paper"] = paper
                    st.success(f"Generated a mixed question paper with {len(valid)} questions successfully.")
        except requests.HTTPError as exc:
            details = ""
            try:
                details = exc.response.text[:1000]
            except Exception:
                pass
            st.error(f"AI provider request failed: {exc}. {details}")
        except requests.ConnectionError:
            st.error("Could not connect to the selected AI provider. Check your internet connection or local Ollama service.")
        except requests.Timeout:
            st.error("The AI provider took too long to respond. Try fewer questions and run generation again.")
        except json.JSONDecodeError:
            st.error("The AI provider did not return valid JSON. Please try again.")
        except Exception as exc:
            st.error(f"Question generation failed: {exc}")

# Generated paper display
st.markdown('<div class="section-title">03 · Your generated question paper</div>', unsafe_allow_html=True)
st.markdown('<div class="section-caption">Your generated questions and model answers will appear here.</div>', unsafe_allow_html=True)
paper = st.session_state.get("current_paper")
if paper:
    st.markdown(
        f'<span class="pill">{escape(paper["subject"])}</span>'
        f'<span class="pill">{escape(paper["question_type"])}</span>'
        f'<span class="pill">{escape(paper["bloom_level"])}</span>',
        unsafe_allow_html=True,
    )
    section_titles = {"MCQ": "SECTION A · MULTIPLE CHOICE QUESTIONS", "Short Answer": "SECTION B · SHORT ANSWER QUESTIONS", "Long Answer": "SECTION C · LONG ANSWER QUESTIONS"}
    ordered_questions = []
    for kind in ("MCQ", "Short Answer", "Long Answer"):
        ordered_questions.extend([q for q in paper["questions"] if q.get("question_type") == kind])
    with st.container(border=True):
        question_number = 0
        for kind in ("MCQ", "Short Answer", "Long Answer"):
            section_questions = [q for q in ordered_questions if q.get("question_type") == kind]
            if not section_questions:
                continue
            st.markdown(f"### {section_titles[kind]}")
            for item in section_questions:
                question_number += 1
                st.markdown(f'<div class="question-number">QUESTION {question_number:02d}</div><div class="question-text">{escape(str(item["question"]))}</div>', unsafe_allow_html=True)
                for option in item.get("options", []):
                    st.markdown(f"- {escape(str(option))}")
                with st.expander(f"View model answer · Question {question_number}"):
                    st.markdown(str(item.get("correct_answer", "Answer unavailable.")))
                if question_number < len(ordered_questions):
                    st.divider()
    try:
        distribution_label = ", ".join(f"{k}: {v}" for k, v in paper.get("distribution", {}).items() if v)
        question_pdf = make_pdf("Question Paper", paper["subject"], paper["difficulty"], distribution_label or paper["question_type"], paper["bloom_level"], paper["questions"], False)
        answer_pdf = make_pdf("Answer Key", paper["subject"], paper["difficulty"], distribution_label or paper["question_type"], paper["bloom_level"], paper["questions"], True)
        d1, d2 = st.columns(2)
        with d1:
            st.download_button("⬇️ Download Question Paper (PDF)", question_pdf, f"{safe_filename(paper['subject'])}_question_paper.pdf", "application/pdf", use_container_width=True)
        with d2:
            st.download_button("🔑 Download Answer Key (PDF)", answer_pdf, f"{safe_filename(paper['subject'])}_answer_key.pdf", "application/pdf", use_container_width=True)
    except Exception as exc:
        st.error(f"Could not create PDF files: {exc}")
    if st.button("Start a new paper"):
        st.session_state.pop("current_paper", None)
        st.rerun()
else:
    with st.container(border=True):
        st.info("Add your study material, choose your question distribution, and generate a paper. Your questions will appear in this section.")

# CO2 sentiment feature on reviewer/student feedback
st.markdown('<div class="section-title">04 · Feedback sentiment analysis (CO2)</div>', unsafe_allow_html=True)
st.markdown('<div class="section-caption">Analyse comments about a generated question paper.</div>', unsafe_allow_html=True)
with st.container(border=True):
    feedback = st.text_area("Enter feedback", placeholder="e.g., The questions are clear and cover the important concepts.")
    if st.button("Analyse feedback sentiment"):
        if not feedback.strip():
            st.warning("Enter feedback text first.")
        else:
            sentiment = sentiment_analyze(feedback)
            st.metric("Sentiment", sentiment["label"])
            st.write({
                "Positive score": round(sentiment["pos"], 3),
                "Neutral score": round(sentiment["neu"], 3),
                "Negative score": round(sentiment["neg"], 3),
                "Compound score": round(sentiment["compound"], 3),
            })

# Saved question paper history
st.markdown('<div class="section-title">05 · Question paper history</div>', unsafe_allow_html=True)
st.markdown('<div class="section-caption">Open a saved paper to revisit its questions and model answers.</div>', unsafe_allow_html=True)
with st.container(border=True):
    saved_papers = load_history()
    if not saved_papers:
        st.info("No generated papers saved yet. Your papers will be stored here automatically.")
    else:
        for index, saved in enumerate(reversed(saved_papers[-20:]), start=1):
            saved_questions = saved.get("questions", [])
            distribution = saved.get("distribution", {})
            dist_label = " · ".join(f"{k}: {v}" for k, v in distribution.items() if v) or saved.get("question_type", "Question paper")
            label = f"{saved.get('subject', 'Untitled')}  |  {len(saved_questions)} questions  |  {dist_label}"
            with st.expander(label):
                st.caption(f"Difficulty: {saved.get('difficulty', '—')}  ·  Bloom level: {saved.get('bloom_level', '—')}  ·  Created: {saved.get('generated_at', 'Previously saved')}")
                for q_index, item in enumerate(saved_questions, start=1):
                    st.markdown(f"**Q{q_index}. [{item.get('question_type', 'Question')}] {item.get('question', '')}**")
                    for option in item.get("options", []):
                        st.write(f"- {option}")
                    st.markdown(f"**Model answer:** {item.get('correct_answer', 'Answer unavailable.')}")
                    if q_index < len(saved_questions):
                        st.divider()

st.markdown("""
<div style="text-align:center; padding:24px 0 4px 0; color:#8A8FA3; font-size:12px;">
  QuestionCraft AI · Built with Streamlit, NLP libraries and OpenRouter
</div>
""", unsafe_allow_html=True)
