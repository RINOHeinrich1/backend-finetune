from sentence_transformers import SentenceTransformer
import os
import torch
import numpy as np
import re

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

DEFAULT_MODEL_NAME = 'sentence-transformers/all-MiniLM-L6-v2' #"dangvantuan/sentence-camembert-large"
MODELS_DIR = "./models"

def get_latest_model_path():
    """
    Renvoie le chemin du modèle versionné le plus récent dans ./models,
    ou DEFAULT_MODEL_NAME si aucun modèle local n’est trouvé.
    """
    if not os.path.exists(MODELS_DIR):
        return DEFAULT_MODEL_NAME

    versions = [d for d in os.listdir(MODELS_DIR) if os.path.isdir(os.path.join(MODELS_DIR, d)) and d.startswith("esti-rag-ft-v")]
    if not versions:
        return DEFAULT_MODEL_NAME

    # Trier par numéro de version décroissant
    versions.sort(key=lambda v: int(v.split("-v")[-1]), reverse=True)
    return os.path.join(MODELS_DIR, versions[0])

# Chargement du modèle SentenceTransformer
MODEL_PATH = get_latest_model_path()
print(f"📦 Utilisation du modèle : {MODEL_PATH}")
model = SentenceTransformer(MODEL_PATH, device=DEVICE)

def get_embedding(texts, model=""):
    model_path = model
    if model == "":
        model_path = get_latest_model_path()
        print(f"🧠 Chargement SentenceTransformer depuis : {model_path}")
    model = SentenceTransformer(model_path, device=DEVICE)
    return model.encode(texts, convert_to_numpy=True, normalize_embeddings=True)

# --- Chunker heuristique ---

MOTS_TITRES = {"Objectifs", "Missions", "Présentation", "Vision", "Valeurs", "Produits", "Services", "Contact"}

def contains_conjugated_verb(line: str) -> bool:
    patterns = [
        r"\b\w+(e|es|ons|ez|ent)\b",
        r"\b\w+(ai|as|a|âmes|âtes|èrent)\b",
        r"\b(étais|était|étions|étaient|suis|es|est|sommes|êtes|sont)\b",
        r"\b(ai|as|a|avons|avez|ont|avais|avait|avaient|aurai|auras|auront)\b",
    ]
    for pattern in patterns:
        if re.search(pattern, line.lower()):
            return True
    return False

def is_short_line(line: str, max_words=10) -> bool:
    return len(line.split()) <= max_words

def starts_with_capital(line: str) -> bool:
    return line and line[0].isupper()

def ends_with_period(line: str) -> bool:
    return line.strip().endswith(".")

def is_uppercase(line: str) -> bool:
    return line.isupper()

def contains_known_title_keyword(line: str) -> bool:
    return any(keyword.lower() in line.lower() for keyword in MOTS_TITRES)

def starts_with_number_or_symbol(line: str) -> bool:
    return bool(re.match(r"^(\d+[\.\)]?|[-*•])", line.strip()))

def score_title_likelihood(line: str) -> float:
    score = 0
    if not contains_conjugated_verb(line): score += 1
    if is_short_line(line): score += 1
    if starts_with_capital(line): score += 0.5
    if not ends_with_period(line): score += 0.5
    if is_uppercase(line): score += 0.5
    if contains_known_title_keyword(line): score += 1
    if starts_with_number_or_symbol(line): score += 1
    return score

def is_probable_title(line: str, threshold=2.5) -> bool:
    return score_title_likelihood(line) >= threshold

def chunk_text_heuristique(text: str):
    lines = text.splitlines()
    chunks = []
    current_title = None
    current_content = []

    def is_list_line(line: str) -> bool:
        # Retourne True si la ligne commence par un caractère non alphanumérique suivi d'un espace
        return bool(re.match(r"^[^a-zA-Z0-9\s]\s+", line.strip()))


    for i, line in enumerate(lines):
        stripped = line.strip()
        if stripped == "":
            continue

        # Gestion liste par lookahead : si ligne i et i+1 commencent par un tiret, c’est une liste
        next_line = lines[i + 1].strip() if i + 1 < len(lines) else ""

        if is_list_line(stripped) and is_list_line(next_line):
            # Liste détectée, on considère cette ligne comme contenu
            current_content.append(stripped)
            continue

        # Si c’est une ligne listée mais la suivante n’est pas liste, on peut quand même considérer contenu
        if is_list_line(stripped) and not is_list_line(next_line):
            current_content.append(stripped)
            continue

        # Sinon, on teste si titre
        if is_probable_title(stripped):
            if current_title and current_content:
                chunks.append(f"{current_title}\n{' '.join(current_content)}")
            current_title = stripped
            current_content = []
        else:
            current_content.append(stripped)

    if current_title and current_content:
        chunks.append(f"{current_title}\n{' '.join(current_content)}")

    return chunks

