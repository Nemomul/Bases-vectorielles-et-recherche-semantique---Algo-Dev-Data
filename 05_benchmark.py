# 05_benchmark.py
import faiss
import numpy as np
import json
from sentence_transformers import SentenceTransformer
from rank_bm25 import BM25Okapi

# ── Chargement des ressources ──────────────────────────────
print("Chargement des ressources...")

model = SentenceTransformer("all-MiniLM-L6-v2")
index = faiss.read_index("index/faiss_hnsw.index")

with open("corpus/abstracts.json") as f:
    corpus = json.load(f)

tokenized = [doc["abstract"].lower().split() for doc in corpus]
bm25 = BM25Okapi(tokenized)

print("✅ Ressources chargées\n")

# ── Fonction BM25 ──────────────────────────────────────────
def bm25_search(query, k=5):
    tokens = query.lower().split()
    scores = bm25.get_scores(tokens)
    top_k = sorted(range(len(scores)), key=lambda i: -scores[i])[:k]
    return [(idx, scores[idx]) for idx in top_k]

# ── Requêtes de test ───────────────────────────────────────
test_queries = [
    ("neural network learns from data",    "synonyme: 'model trains'"),
    ("computer sees images",               "paraphrase de 'visual recognition'"),
    ("making machines understand language","paraphrase de NLP"),
]

for query, cas in test_queries:
    print(f"=== Requête: '{query}' ({cas}) ===")

    # Sémantique
    q_emb = model.encode([query]).astype("float32")
    faiss.normalize_L2(q_emb)
    _, idx_sem = index.search(q_emb, 5)

    # BM25
    idx_bm25 = bm25_search(query, 5)

    sem_ids = set(idx_sem[0])
    lex_ids = set(i for i, _ in idx_bm25)

    print(f"  Sémantique uniquement : {sem_ids - lex_ids}")
    print(f"  BM25 uniquement       : {lex_ids - sem_ids}")
    print(f"  En commun             : {sem_ids & lex_ids}")

    # Afficher les textes trouvés
    print("\n  📘 Top 3 Sémantique :")
    for idx in list(idx_sem[0])[:3]:
        print(f"    → {corpus[idx]['abstract'][:100]}...")

    print("\n  📖 Top 3 BM25 :")
    for idx, score in idx_bm25[:3]:
        print(f"    → [{score:.2f}] {corpus[idx]['abstract'][:100]}...")

    print("\n" + "─"*60 + "\n")