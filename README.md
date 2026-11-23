# QuestionCraft AI — NLP Question Paper Generator

A Streamlit application that extracts study material from pasted text or text-based PDFs, demonstrates NLP processing, and generates question papers and answer keys with a locally running Ollama model.

## Features

- Paste study notes or upload a PDF using PyMuPDF.
- Text cleaning, sentence segmentation, tokenization, stopword removal and lemmatization using NLTK.
- TF-IDF keyword extraction using scikit-learn.
- Named-entity extraction and noun phrases using spaCy.
- Topic modelling with Latent Dirichlet Allocation (LDA).
- Feedback sentiment analysis using VADER.
- Local question generation through Ollama (default model: `llama3.1:8b`).
- MCQ, short-answer and long-answer generation; difficulty and Bloom's Taxonomy selection.
- Question history and separate Question Paper / Answer Key PDF downloads.

## Requirements

- Windows 10/11, Python 3.10–3.12 recommended.
- Ollama installed and running.
- 16 GB RAM is a reasonable starting point for a quantized 8B model; generation speed depends on CPU/GPU and available memory. If it is slow, use a smaller model such as `llama3.2:3b`.

## Setup (Windows PowerShell)

1. Extract the ZIP.
2. Open PowerShell in the extracted `QuestionCraft_AI_Upgraded` folder.
3. Create and activate a virtual environment:

```powershell
py -3.12 -m venv venv
.\venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
```

4. Download the spaCy English model:

```powershell
python -m spacy download en_core_web_sm
```

5. Install Ollama from https://ollama.com/download/windows, then open PowerShell and download a model:

```powershell
ollama pull llama3.1:8b
ollama run llama3.1:8b
```

When the model prompt appears, type `/bye` to exit. Keep Ollama running in the background.

6. Run the app:

```powershell
python -m streamlit run app.py
```

The first NLP run may download NLTK data packages automatically. An internet connection is needed for these first-time downloads.

## Important notes

- No Groq SDK or Groq API key is used by this version.
- Ollama serves the local model at `http://localhost:11434` by default. Set `OLLAMA_URL` or `OLLAMA_MODEL` environment variables only if your local setup differs.
- Scanned/image-only PDFs are not OCR processed. The app will report that no readable text was found.
- spaCy named-entity recognition requires the `en_core_web_sm` model.
- LDA topic modelling needs at least two meaningful text chunks. Results depend on the length and quality of the supplied material.
- VADER sentiment currently analyses feedback entered by the user. It is not a sentiment classifier for arbitrary academic notes.
- The NLP features are local algorithms. Your faculty rubric mentions cloud ML services in the CO2 course outcome and CO3 assessment row. Confirm with faculty whether local libraries satisfy those specific expectations before claiming full rubric coverage.
- The AWS NLP certificate is a separate 20-mark assessment component and is not replaced by this application.

## Project structure

```text
QuestionCraft_AI_Upgraded/
├── app.py
├── nlp_pipeline.py
├── requirements.txt
├── README.md
├── .gitignore
└── styles/
    └── style.css
```

## Suggested demonstration

1. Paste a multi-paragraph study material or upload a text-based PDF.
2. Open NLP preprocessing details and explain cleaned text, tokens, stopwords, lemmas and TF-IDF terms.
3. Open Information extraction and explain entities and noun phrases.
4. Open Detected topics and explain LDA topic keywords.
5. Generate a small paper (start with 3 questions) and show the local Ollama integration.
6. Download both PDFs.
7. Enter feedback and show VADER sentiment scores.

## Limitations / remaining validation

- This code has been syntax-checked, but it has not been run end-to-end on your laptop. Ollama model availability, hardware speed and package installation must be tested locally.
- CO2 evaluation metrics (precision, recall, F1 and confusion matrix on a labelled dataset) are not yet included. Do not claim these metrics have been evaluated until a labelled dataset and evaluation run are added.
- Cloud-service requirements for CO2/CO3 need confirmation from faculty.
- A deployed project link may not be possible on free hosting if the local Ollama model must be available to the hosted app. Confirm acceptable demo/deployment arrangements with faculty.

## Mixed-format question papers
The generator now accepts a quantity for each question type: MCQ, Short Answer, and Long Answer. Set any quantity to zero to omit that section. The app checks that the local model returns the exact requested distribution and that each MCQ has four options. Generated papers and PDFs are arranged into Section A (MCQ), Section B (Short Answer), and Section C (Long Answer), with a separate answer-key PDF.

## Cleaner NLP analysis interface
The NLP results are grouped inside one collapsed `View NLP analysis · CO1–CO3` panel with tabs for preprocessing/TF-IDF, information extraction, and topic modelling. The algorithms still run on the supplied study material; the tabs only organize how results are displayed. Ollama remains separate and is used for question generation.
