# 02_embeddings.py
from sentence_transformers import SentenceTransformer
import numpy as np, json

model = SentenceTransformer("all-MiniLM-L6-v2")  # rapide, 384 dims

with open("corpus/abstracts.json") as f:
    corpus = json.load(f)

texts = [doc["abstract"] for doc in corpus]
embeddings = model.encode(texts, batch_size=64, show_progress_bar=True)
# Shape: (2000, 384)

np.save("embeddings/embeddings.npy", embeddings)
print(f"Embeddings shape: {embeddings.shape}")

# Exemple de similarité cosinus à la main
from numpy.linalg import norm
u, v = embeddings[0], embeddings[1]
cosine_sim = np.dot(u, v) / (norm(u) * norm(v))
print(f"Similarité cosinus [0] vs [1]: {cosine_sim:.4f}")