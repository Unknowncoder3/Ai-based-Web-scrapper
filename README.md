🤖 AI-Powered Web Scraper with FAISS & Local LLM (RAG)

An AI-powered web scraping and question-answering system built with Streamlit, FAISS, and local LLMs via Ollama.
The application scrapes website content, converts it into vector embeddings, stores it in a FAISS vector database, and allows users to ask natural-language questions using a Retrieval-Augmented Generation (RAG) approach.

🚀 Features

🌍 Scrape textual content from any public website

✂️ Automatic text chunking for efficient processing

🔢 Semantic embeddings using Hugging Face Sentence Transformers

⚡ Fast similarity search with FAISS

🤖 Local LLM inference using Ollama (Mistral, LLaMA, etc.)

🧠 Context-aware question answering (RAG)

🖥️ Interactive Streamlit web interface

🔒 Fully local & offline (after model download)

🧠 Architecture Overview
User → Streamlit UI
        ↓
   Website URL
        ↓
  Requests + BeautifulSoup
        ↓
   Clean Text Extraction
        ↓
 Character Text Splitter
        ↓
 HuggingFace Embeddings
        ↓
     FAISS Vector Store
        ↓
 Similarity Search (Query)
        ↓
 Retrieved Context
        ↓
     Ollama LLM
        ↓
   Final AI Answer

🛠️ Tech Stack

Python

Streamlit – Web UI

Requests – HTTP requests

BeautifulSoup4 – HTML parsing

FAISS – Vector similarity search

NumPy – Numerical operations

Hugging Face Sentence Transformers – Text embeddings

LangChain (Modular packages) – LLM & embeddings interface

Ollama – Local LLM execution (Mistral / LLaMA)
