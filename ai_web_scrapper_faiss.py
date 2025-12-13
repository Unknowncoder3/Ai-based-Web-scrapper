import requests
from bs4 import BeautifulSoup
import streamlit as st
import faiss
import numpy as np
from langchain_ollama import OllamaLLM
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import CharacterTextSplitter

# ======================
# Model & Embeddings
# ======================

# Load AI Model (change to any local Ollama model you have, e.g. "llama3")
llm = OllamaLLM(model="mistral")

# Load Hugging Face Embeddings
# all-MiniLM-L6-v2 has dimension 384
embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

# ======================
# FAISS Vector Index
# ======================

# Initialize FAISS index (L2 distance, dim=384)
embedding_dim = 384
index = faiss.IndexFlatL2(embedding_dim)

# This will map each FAISS vector row index -> {"url": ..., "text": ...}
# IMPORTANT: length of vector_store must always match index.ntotal
vector_store = []


# ======================
# Utils: Scraping
# ======================

def scrape_website(url: str) -> str:
    """Scrape a website and return cleaned text."""
    try:
        st.write(f"🌍 Scraping website: {url}")
        headers = {"User-Agent": "Mozilla/5.0"}
        response = requests.get(url, headers=headers, timeout=10)

        if response.status_code != 200:
            return f"⚠️ Failed to fetch {url} (status code: {response.status_code})"

        soup = BeautifulSoup(response.text, "html.parser")

        # Get all paragraph text
        paragraphs = soup.find_all("p")
        text = " ".join([p.get_text(separator=" ", strip=True) for p in paragraphs])

        if not text.strip():
            return "⚠️ No readable text content found on this page."

        # Limit to avoid too-large prompt later
        return text[:5000]
    except Exception as e:
        return f"❌ Error while scraping: {str(e)}"


# ======================
# Store in FAISS
# ======================

def store_in_faiss(text: str, url: str) -> str:
    """Split text, embed chunks, and store in FAISS + in-memory vector_store."""
    global index, vector_store

    st.write("📥 Storing data in FAISS...")

    # Split text into chunks
    splitter = CharacterTextSplitter(chunk_size=500, chunk_overlap=100)
    chunks = splitter.split_text(text)

    if not chunks:
        return "⚠️ Nothing to store (no chunks created)."

    # Convert chunks into embeddings (list of vectors)
    chunk_vectors = embeddings.embed_documents(chunks)
    vectors = np.array(chunk_vectors, dtype=np.float32)

    # Sanity check on dimensions
    if vectors.shape[1] != embedding_dim:
        return f"❌ Embedding dimension mismatch: expected {embedding_dim}, got {vectors.shape[1]}"

    # Add to FAISS index
    index.add(vectors)

    # Add to vector_store (order must match FAISS vectors)
    for chunk in chunks:
        vector_store.append({"url": url, "text": chunk})

    return f"✅ Stored {len(chunks)} chunks from {url} in FAISS!"


# ======================
# Retrieval + QA
# ======================

def retrieve_and_answer(query: str) -> str:
    """Retrieve relevant chunks from FAISS and ask LLM to answer."""
    global index, vector_store

    # Handle case when nothing is stored yet
    if index.ntotal == 0 or len(vector_store) == 0:
        return "🤖 No data stored yet. Please scrape and store a website first."

    # Convert query into embedding
    query_vector = np.array(
        embeddings.embed_query(query),
        dtype=np.float32
    ).reshape(1, -1)

    if query_vector.shape[1] != embedding_dim:
        return f"❌ Query embedding dimension mismatch."

    # Choose k <= number of vectors stored
    k = min(5, index.ntotal)

    # Search FAISS
    distances, indices = index.search(query_vector, k=k)

    context_chunks = []
    for idx in indices[0]:
        # idx is a row index in the FAISS index; map directly into vector_store
        if 0 <= idx < len(vector_store):
            context_chunks.append(vector_store[idx]["text"])

    if not context_chunks:
        return "🤖 No relevant data found."

    context = "\n\n".join(context_chunks)

    # Ask the LLM to answer based on retrieved context
    prompt = (
        "You are a helpful assistant. Use ONLY the context below to answer the question.\n"
        "If the answer is not in the context, say you don't know.\n\n"
        f"Context:\n{context}\n\n"
        f"Question: {query}\n"
        "Answer:"
    )

    return llm.invoke(prompt)


# ======================
# Streamlit Web UI
# ======================

st.title("🤖 AI-Powered Web Scraper with FAISS Storage")
st.write("🔗 Enter a website URL below and store its content for AI-based Q&A!")

# --- Website input & scraping ---
url = st.text_input("🔗 Enter Website URL:")

if url:
    content = scrape_website(url)

    if content.startswith("⚠️") or content.startswith("❌"):
        st.write(content)
    else:
        st.subheader("📄 Scraped Preview (first 1000 characters)")
        st.write(content[:1000] + ("..." if len(content) > 1000 else ""))

        store_message = store_in_faiss(content, url)
        st.write(store_message)

# --- Q&A section ---
st.markdown("---")
st.subheader("❓ Ask a question based on stored content:")

query = st.text_input("Your question:")

if query:
    answer = retrieve_and_answer(query)
    st.subheader("🤖 AI Answer:")
    st.write(answer)
