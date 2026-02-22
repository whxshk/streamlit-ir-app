import streamlit as st
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer

st.set_page_config(page_title="IR App", page_icon="🔎")

@st.cache_resource
def load_model():
    return SentenceTransformer("all-MiniLM-L6-v2")

@st.cache_data
def load_docs_and_embeddings():
    with open("documents.txt", "r", encoding="utf-8") as f:
        documents = [line.strip() for line in f.readlines() if line.strip()]
    embeddings = np.load("embeddings.npy")
    return documents, embeddings

def retrieve_top_k(query_embedding, embeddings, documents, k=10):
    sims = cosine_similarity(query_embedding.reshape(1, -1), embeddings)[0]
    top_idx = sims.argsort()[-k:][::-1]
    return [(documents[i], float(sims[i])) for i in top_idx]

# Load
model = load_model()
documents, embeddings = load_docs_and_embeddings()

# Validate
if len(documents) != embeddings.shape[0]:
    st.error(
        f"Mismatch: documents={len(documents)} but embeddings rows={embeddings.shape[0]}. "
        "Regenerate embeddings.npy using the same documents.txt."
    )
    st.stop()

# UI
st.title("Information Retrieval using Document Embeddings")
query = st.text_input("Enter your query:")
k = st.slider("Top K results", 1, min(20, len(documents)), min(10, len(documents)))

if st.button("Search"):
    q = query.strip()
    if not q:
        st.warning("Please enter a query.")
        st.stop()

    # ✅ Exact-match override (guarantees 1.0000 for exact same line)
    if q in documents:
        results = [(q, 1.0)]
        # fill rest with semantic results excluding the exact match
        query_emb = model.encode([q], convert_to_numpy=True)[0]
        semantic_results = retrieve_top_k(query_emb, embeddings, documents, k=len(documents))
        semantic_results = [(d, s) for d, s in semantic_results if d != q]
        results.extend(semantic_results[: max(0, k - 1)])
    else:
        # Semantic search
        query_emb = model.encode([q], convert_to_numpy=True)[0]
        results = retrieve_top_k(query_emb, embeddings, documents, k=k)

    st.write(f"### Top {k} Relevant Documents:")
    for doc, score in results[:k]:
        st.write(f"- **{doc}** (Score: {score:.4f})")
