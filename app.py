# app.py
import streamlit as st
import faiss, numpy as np, json
from sentence_transformers import SentenceTransformer
from rank_bm25 import BM25Okapi

@st.cache_resource
def load_resources():
    model = SentenceTransformer("all-MiniLM-L6-v2")
    index = faiss.read_index("index/faiss_hnsw.index")
    with open("corpus/abstracts.json") as f:
        corpus = json.load(f)
    tokenized = [doc["abstract"].lower().split() for doc in corpus]
    bm25 = BM25Okapi(tokenized)
    return model, index, corpus, bm25

model, index, corpus, bm25 = load_resources()

st.title("🔍 Recherche sémantique vs BM25")
query = st.text_input("Entrez votre requête :", "attention mechanism for sequences")

if query:
    col1, col2 = st.columns(2)
    
    # === Sémantique ===
    with col1:
        st.subheader("🧠 Sémantique (HNSW)")
        q_emb = model.encode([query]).astype("float32")
        faiss.normalize_L2(q_emb)
        distances, indices = index.search(q_emb, 5)
        for rank, (idx, score) in enumerate(zip(indices[0], distances[0])):
            doc = corpus[idx]
            st.markdown(f"**#{rank+1}** — Score: `{score:.3f}`")
            st.caption(doc["abstract"][:200] + "...")
            st.divider()
    
    # === BM25 ===
    with col2:
        st.subheader("📖 Lexical (BM25)")
        tokens = query.lower().split()
        scores = bm25.get_scores(tokens)
        top5 = sorted(range(len(scores)), key=lambda i: -scores[i])[:5]
        for rank, idx in enumerate(top5):
            doc = corpus[idx]
            st.markdown(f"**#{rank+1}** — Score: `{scores[idx]:.3f}`")
            st.caption(doc["abstract"][:200] + "...")
            st.divider()