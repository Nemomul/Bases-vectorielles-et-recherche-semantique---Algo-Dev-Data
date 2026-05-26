# 03_faiss_index.py
import faiss, numpy as np

embeddings = np.load("embeddings/embeddings.npy").astype("float32")
d = embeddings.shape[1]  # dimension = 384

# Normaliser pour que la recherche L2 == cosinus
faiss.normalize_L2(embeddings)

# Index HNSW : M=32 voisins par nœud, ef_construction=200
index = faiss.IndexHNSWFlat(d, 32)
index.hnsw.efConstruction = 200
index.add(embeddings)

faiss.write_index(index, "index/faiss_hnsw.index")
print(f"Index HNSW construit avec {index.ntotal} vecteurs")

# Recherche test
query = embeddings[0:1]
index.hnsw.efSearch = 50  # précision de recherche
distances, indices = index.search(query, k=5)
print(f"Top-5 voisins de [0]: {indices[0]}, distances: {distances[0]}")