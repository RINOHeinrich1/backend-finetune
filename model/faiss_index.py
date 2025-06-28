import faiss
from config import TOP_K

def build_faiss_index(embeddings):
    dim = embeddings.shape[1]
    index = faiss.IndexFlatIP(dim)
    index.add(embeddings)
    return index

def search(index, model, documents, question, k=TOP_K):
    q_emb = model.encode([question], convert_to_numpy=True, normalize_embeddings=True)
    distances, indices = index.search(q_emb, k)
    return [documents[i] for i in indices[0]], distances[0].tolist()
