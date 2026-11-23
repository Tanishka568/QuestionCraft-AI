# QuestionCraft AI – AI-Based Question Paper Generator using NLP and LLM

## 🚀 Live Application
**Try QuestionCraft AI:** https://questioncraft-ai-aqqdtmnxgqgnpjre6zy2df.streamlit.app/

## 📌 Project Overview
QuestionCraft AI is an AI-powered question paper generation application that uses Natural Language Processing (NLP) techniques and Large Language Models (LLMs) to automatically generate question papers from study materials.

Users can paste study notes or upload PDF documents, analyse the content using NLP techniques, and generate customized question papers with model answers.

## 🎯 Objectives
- Automate question paper generation from study materials.
- Apply NLP techniques for text preprocessing and analysis.
- Extract important keywords, entities and topics.
- Generate MCQs, short-answer and long-answer questions.
- Support different difficulty levels and Bloom's Taxonomy.
- Generate downloadable question papers and answer keys.

## ✨ Key Features

1. **Study Material Input:** Paste text or upload PDF documents.
2. **Text Preprocessing:** Cleaning, tokenization, stopword removal and lemmatization.
3. **Keyword Extraction:** Identify important terms using TF-IDF.
4. **Information Extraction:** Extract named entities and noun phrases.
5. **Topic Modelling:** Identify major topics using LDA.
6. **Sentiment Analysis:** Analyse user feedback using VADER.
7. **AI Question Generation:** Generate questions using an LLM.
8. **Bloom's Taxonomy:** Generate questions at different cognitive levels.
9. **Difficulty Selection:** Easy, Medium and Hard.
10. **PDF Generation:** Download separate question paper and answer key PDFs.
11. **Question History:** Access previously generated question papers.

## 🧠 NLP Techniques Used

| Technique | Technology |
|---|---|
| Text preprocessing | NLTK |
| Tokenization and lemmatization | NLTK |
| TF-IDF keyword extraction | Scikit-learn |
| Named Entity Recognition | spaCy |
| Noun phrase extraction | spaCy |
| Topic Modelling | LDA |
| Sentiment Analysis | VADER |

## 🤖 AI and LLM Integration

QuestionCraft AI supports two AI providers:

- **Local execution:** Ollama with Llama 3.1 (8B).
- **Cloud deployment:** OpenRouter API using the `openrouter/free` model route.

The LLM generates questions and model answers based on the supplied study material and selected settings.

## 🛠️ Technologies Used

- Python
- Streamlit
- NLTK
- Scikit-learn
- spaCy
- PyMuPDF
- Ollama
- OpenRouter API
- ReportLab
- HTML and CSS

## ⚙️ Local Installation and Setup

### 1. Clone the repository

```bash
git clone https://github.com/Tanishka568/QuestionCraft-AI.git
cd QuestionCraft-AI
```

### 2. Create a virtual environment

```bash
python -m venv venv
```

### 3. Activate the environment (Windows)

```powershell
.\venv\Scripts\Activate.ps1
```

### 4. Install dependencies

```bash
pip install -r requirements.txt
```

### 5. Download the spaCy model

```bash
python -m spacy download en_core_web_sm
```

### 6. Configure the AI provider

For local execution, install Ollama and download the model:

```bash
ollama pull llama3.1:8b
```

Alternatively, configure an OpenRouter API key using an environment variable or Streamlit secrets.

**Never upload API keys to GitHub.**

### 7. Run the application

```bash
streamlit run app.py
```

The application will open at:

`http://localhost:8501`

## 📂 Project Structure

```text
QuestionCraft-AI/
│
├── app.py
├── nlp_pipeline.py
├── requirements.txt
├── README.md
├── sentiment_dataset_template.csv
│
├── styles/
│   └── style.css
│
└── .streamlit/
    └── config.toml
```

## 🔄 Application Workflow

1. User enters study material or uploads a PDF.
2. Text is extracted and preprocessed.
3. NLP techniques analyse the supplied content.
4. Important keywords, entities and topics are identified.
5. User selects subject, difficulty, Bloom's level and question distribution.
6. The LLM generates questions and model answers.
7. The application displays the generated question paper.
8. User downloads the question paper and answer key as PDF files.

## 📊 Expected Outputs

- MCQ questions with four options and correct answers.
- Short-answer questions with model answers.
- Long-answer questions with detailed explanations.
- Question Paper PDF.
- Separate Answer Key PDF.
- NLP analysis results.
- Feedback sentiment analysis.

## ⚠️ Limitations

- Scanned PDFs require OCR and are not directly supported.
- Question quality depends on the quality and completeness of the supplied study material.
- Free LLM availability and request limits may vary.
- NLP analysis results depend on the input text.

## 🔮 Future Scope

- Multilingual question generation.
- OCR support for scanned documents.
- Advanced question evaluation and validation.
- Additional NLP algorithms.
- More LLM integration options.
- Improved question paper customization.

## 👩‍💻 Project Team

- **Tanishka Tapkir**
- **Harsh Joshi**

## 🔗 Project Links

- **Live Application:** https://questioncraft-ai-aqqdtmnxgqgnpjre6zy2df.streamlit.app/
- **GitHub Repository:** https://github.com/Tanishka568/QuestionCraft-AI

---

*QuestionCraft AI – Transforming Study Materials into Intelligent Question Papers.*
