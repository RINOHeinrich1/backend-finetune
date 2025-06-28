from sentence_transformers import SentenceTransformer
from config import EMBEDDING_MODEL_PATH, DEFAULT_MODEL_NAME, DEVICE
import os

def load_model(model=""):
    model_name = EMBEDDING_MODEL_PATH if os.path.exists(EMBEDDING_MODEL_PATH) else DEFAULT_MODEL_NAME
    print(f"🔁 Chargement modèle : {model_name}")
    return SentenceTransformer(model_name, device=DEVICE)
