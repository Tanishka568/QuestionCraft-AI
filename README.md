# 🎓 QuestionCraft AI
### AI-Based Question Paper Generator using NLP and Large Language Models

QuestionCraft AI is an intelligent question paper generation application that combines **Natural Language Processing (NLP)** and **Generative AI** to automate the creation of question papers from academic study materials.

The application accepts text or PDF documents, processes the content using NLP techniques, extracts important information and generates MCQs, short-answer questions and long-answer questions with model answers using a locally running Large Language Model (LLM).

---

## 📌 1. Problem Statement

Preparing question papers manually is time-consuming, repetitive and requires considerable effort to maintain consistency in question difficulty, coverage and question formats.

QuestionCraft AI addresses this problem by automating question paper generation from academic study materials using NLP techniques and Generative AI.

## 🎯 2. Objectives

- Automate question paper generation from study materials.
- Apply NLP techniques for text preprocessing and analysis.
- Extract important keywords, entities and topics.
- Generate MCQs, short-answer and long-answer questions.
- Incorporate Bloom's Taxonomy and difficulty levels.
- Generate model answers for the questions.
- Provide downloadable question papers and separate answer keys.

## ✨ 3. Key Features

- **Multiple Input Options:** Paste study material or upload PDF documents.
- **PDF Text Extraction:** Extracts text from uploaded PDFs.
- **Text Preprocessing:** Performs cleaning, tokenization, stopword removal and lemmatization.
- **Keyword Extraction:** Identifies important keywords using TF-IDF.
- **Information Extraction:** Identifies named entities and noun phrases using spaCy.
- **Topic Modelling:** Discovers topics using LDA.
- **Sentiment Analysis:** Analyses student/reviewer feedback using VADER.
- **AI Question Generation:** Generates MCQs, short-answer and long-answer questions.
- **Bloom's Taxonomy:** Supports different cognitive learning levels.
- **Difficulty Selection:** Provides Easy, Medium and Hard levels.
- **Model Answers:** Generates answers along with questions.
- **PDF Generation:** Creates separate question paper and answer key PDFs.
- **Question Paper History:** Maintains records of previously generated papers.

## 🛠️ 4. Technologies and Libraries

| Technology | Purpose |
|---|---|
| Python | Core programming language |
| Streamlit | Web application interface |
| PyMuPDF | PDF text extraction |
| Regular Expressions | Text cleaning |
| NLTK | Text preprocessing |
| Scikit-learn | TF-IDF, CountVectorizer and LDA |
| spaCy | Named Entity Recognition and noun chunks |
| VADER | Sentiment analysis |
| Ollama | Local LLM execution |
| Llama 3.1 8B | AI-based question and answer generation |
| Requests | Communication with Ollama API |
| ReportLab | PDF generation |
| JSON | Structured output and question paper history |

## 🔄 5. Methodology

The application follows these steps:

1. **Input:** The user pastes study material or uploads a PDF.
2. **Text Extraction:** PyMuPDF extracts text from uploaded PDF documents.
3. **Text Cleaning:** Regular Expressions remove unwanted characters and extra spaces.
4. **NLP Preprocessing:** NLTK performs tokenization, stopword removal and lemmatization.
5. **NLP Analysis:**
   - TF-IDF extracts important keywords.
   - spaCy performs information extraction using NER and noun chunks.
   - CountVectorizer and LDA perform topic modelling.
   - VADER performs sentiment analysis on student/reviewer feedback.
6. **Prompt Engineering:** The application prepares a structured prompt using the study material, keywords, subject, difficulty, question distribution and Bloom's Taxonomy level.
7. **LLM Integration:** The prompt is sent to Llama 3.1 8B through Ollama.
8. **Question Generation:** The LLM generates questions and their model answers in JSON format.
9. **PDF Generation:** ReportLab creates the question paper and separate answer key.
10. **History Management:** Generated question paper details are stored in JSON.

## 🏗️ 6. System Architecture

```text
             USER INPUT
          (Text / PDF Upload)
                  |
                  v
          PDF Text Extraction
              (PyMuPDF)
                  |
                  v
            Text Cleaning
               (Regex)
                  |
                  v
          NLP Preprocessing
               (NLTK)
                  |
                  v
             NLP ANALYSIS
                  |
       +----------+----------+
       |          |          |
       v          v          v
     TF-IDF     spaCy       LDA
    Keywords   Information  Topic
               Extraction   Modelling
                  |
                  v
           PROMPT ENGINEERING
                  |
                  v
          OLLAMA + LLAMA 3.1 8B
                  |
                  v
          STRUCTURED JSON OUTPUT
                  |
                  v
          GENERATED QUESTIONS
            AND MODEL ANSWERS
                  |
                  v
           REPORTLAB PDF OUTPUT
              /          \
             v            v
       Question Paper   Answer Key
             |
             v
        JSON HISTORY
```

## 🧠 7. NLP Techniques Implemented

### Text Preprocessing – NLTK
- Sentence Tokenization
- Word Tokenization
- Stopword Removal
- Lemmatization

### Keyword Extraction – TF-IDF
TF-IDF (Term Frequency–Inverse Document Frequency) identifies important words based on their frequency and distribution across documents.

### Information Extraction – spaCy
- Named Entity Recognition (NER)
- Noun Chunk Extraction

### Topic Modelling – LDA
Latent Dirichlet Allocation identifies groups of related words representing topics in the study material.

CountVectorizer converts text into a numerical word-frequency matrix for LDA.

### Sentiment Analysis – VADER
VADER (Valence Aware Dictionary and sEntiment Reasoner) classifies feedback into positive, negative and neutral sentiments.

## 🤖 8. Generative AI Integration

QuestionCraft AI uses **Llama 3.1 8B**, executed locally through Ollama.

Ollama provides a local API that allows the Python application to communicate with the language model.

Prompt engineering is used to control:
- Subject name
- Question distribution
- Difficulty level
- Bloom's Taxonomy level
- Study material
- Important extracted keywords
- Expected JSON response structure

The LLM generates:
- Multiple Choice Questions (MCQs)
- Short-answer questions
- Long-answer questions
- Model answers

The generated JSON response is processed by Python before displaying questions and creating PDF documents.

## 📂 9. Project Structure

```text
QuestionCraft-AI/
│
├── app.py
├── nlp_pipeline.py
├── requirements.txt
├── README.md
├── .gitignore
│
├── styles/
│   └── style.css
│
├── assets/
│
└── question_history.json
```

*Note: The project structure may vary slightly depending on the final submitted version. The question history file may be created automatically when the application runs.*

## ⚙️ 10. Installation and Setup

### Prerequisites

Install the following:

- Python 3.12
- Git
- Ollama
- Llama 3.1 8B model

### Step 1: Clone the Repository

```bash
git clone https://github.com/YOUR_USERNAME/QuestionCraft-AI.git
```

Navigate to the project directory:

```bash
cd QuestionCraft-AI
```

### Step 2: Create a Virtual Environment

```bash
python -m venv .venv
```

Activate it on Windows:

```powershell
.venv\Scripts\activate
```

### Step 3: Install Dependencies

```bash
pip install -r requirements.txt
```

### Step 4: Install the spaCy English Model

```bash
python -m spacy download en_core_web_sm
```

### Step 5: Download NLTK Resources

Run:

```bash
python -c "import nltk; nltk.download('punkt'); nltk.download('punkt_tab'); nltk.download('stopwords'); nltk.download('wordnet'); nltk.download('omw-1.4')"
```

### Step 6: Install and Run Ollama

Download Ollama from:

https://ollama.com/

Pull the Llama 3.1 8B model:

```bash
ollama pull llama3.1:8b
```

Make sure the Ollama service is running.

### Step 7: Run the Application

```bash
streamlit run app.py
```

The application will open in your default browser.

## 💻 11. Hardware and Software Requirements

**Software Requirements**
- Operating System: Windows / Linux / macOS
- Python: 3.12
- Streamlit
- Ollama
- Llama 3.1 8B
- Required Python libraries

**Hardware Requirements**
- Minimum 16 GB RAM recommended for comfortable local execution of the 8B model.
- Sufficient storage for Python dependencies and the downloaded LLM.
- A modern multi-core processor.

Actual generation speed depends on available hardware and Ollama configuration.

## 📊 12. Results and Observations

The application demonstrates the following outcomes:

- Successfully extracts and preprocesses text from study materials.
- Performs keyword extraction and information extraction.
- Identifies relevant topics using LDA.
- Performs sentiment analysis on feedback.
- Generates different types of questions with model answers.
- Supports difficulty and Bloom's Taxonomy selection.
- Produces downloadable question papers and answer keys.
- Maintains question paper history.

## ⚠️ 13. Limitations

- Scanned PDFs containing images require OCR, which is not currently implemented.
- Generated questions and answers may require human verification.
- Local LLM generation time depends on system hardware.
- Named Entity Recognition may occasionally misclassify technical terms.
- VADER may not accurately understand sarcasm or complex contextual meaning.

## 🚀 14. Future Scope

- Integrate OCR to support scanned PDF documents.
- Introduce multilingual question generation.
- Improve answer validation and evaluation.
- Add evaluation metrics for NLP components.
- Deploy the application using a compatible hosted LLM inference service.
- Provide customizable question patterns and marking schemes.
- Add advanced question quality assessment.

## 🎓 15. Applications

- Educational institutions
- Colleges and universities
- Teachers and faculty members
- Online learning platforms
- Students preparing for examinations
- Automated assessment systems

## 👥 16. Project Team

**Project Title:** AI-Based Question Paper Generator using NLP and LLM

**Application Name:** QuestionCraft AI

**Submitted By:**
- Tanishka Tapkir
- Harsh Joshi

**Department:** Computer Science and Engineering – Artificial Intelligence and Machine Learning

**Institute:** MIT Academy of Engineering (MITAOE), Alandi

---

## 📌 Note

QuestionCraft AI combines traditional NLP techniques with Generative AI to simplify question paper preparation and support automated educational assessment.

**Developed as an academic NLP mini project.**
