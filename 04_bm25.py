# 04_bm25.py
from rank_bm25 import BM25Okapi
import json

with open("corpus/abstracts.json") as f:
    corpus = json.load(f)

# Tokenisation simple
tokenized = [doc["abstract"].lower().split() for doc in corpus]
bm25 = BM25Okapi(tokenized)

def bm25_search(query, k=5):
    tokens = query.lower().split()
    scores = bm25.get_scores(tokens)
    top_k = sorted(range(len(scores)), key=lambda i: -scores[i])[:k]
    return [(idx, scores[idx]) for idx in top_k]

# Test
results = bm25_search("attention mechanism transformer")
print("BM25 results:", results)