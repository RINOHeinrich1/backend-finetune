import os
import faiss
import torch
import numpy as np
from fastapi import FastAPI, HTTPException
from sentence_transformers import SentenceTransformer, InputExample, losses
from torch.utils.data import DataLoader


# --- Configuration ---
EMBEDDING_MODEL_NAME = "models/esti-rag-ft"
DEFAULT_EMBEDDING_MODEL_NAME='sentence-transformers/all-MiniLM-L6-v2'
if os.path.exists(EMBEDDING_MODEL_NAME):
    print(f"📂 Chargement du modèle fine-tuné depuis {EMBEDDING_MODEL_NAME}")
else:
    print(f"🌐 Aucun modèle fine-tuné trouvé. Chargement du modèle de base : {DEFAULT_EMBEDDING_MODEL_NAME}")
    EMBEDDING_MODEL_NAME = DEFAULT_EMBEDDING_MODEL_NAME

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
TOP_K = 3
BATCH_SIZE = 4
EPOCHS = 2
WARMUP_STEPS = 10
model = SentenceTransformer(EMBEDDING_MODEL_NAME, device=DEVICE)


def fine_tune_model(question, positive_docs, negative_docs):
    train_examples = []
    for doc in positive_docs:
        train_examples.append(InputExample(texts=[question, doc], label=1.0))
    for doc in negative_docs:
        train_examples.append(InputExample(texts=[question, doc], label=0.0))

    if not train_examples:
        print("⚠️ Aucun exemple pour fine-tuning.")
        return

    train_dataloader = DataLoader(train_examples, shuffle=True, batch_size=BATCH_SIZE)
    train_loss = losses.CosineSimilarityLoss(model)

    # 🔍 Vérifier les paramètres entraînables
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"🧠 Nombre de paramètres entraînables : {trainable_params}")
    if trainable_params == 0:
        print("❌ Aucun paramètre à entraîner. Abandon.")
        return

    print(f"🏋️ Fine-tuning sur {len(train_examples)} exemples...")
    model.train()  # Assure le mode training
    model.fit(
        train_objectives=[(train_dataloader, train_loss)],
        epochs=EPOCHS,
        warmup_steps=WARMUP_STEPS,
        show_progress_bar=True
    )
    print("✅ Fine-tuning terminé.")

    # 💾 Sauvegarde temporaire et rechargement du modèle pour rafraîchir les poids
    model.save("models/esti-rag-ft")
    return SentenceTransformer("models/esti-rag-ft", device=DEVICE)

def evaluate_score_margin(model, question, positive_docs, negative_docs, device):
    model.eval()
    with torch.no_grad():
        q_emb = model.encode(question, convert_to_tensor=True, device=device)

        pos_scores = []
        for doc in positive_docs:
            d_emb = model.encode(doc, convert_to_tensor=True, device=device)
            pos_scores.append(torch.nn.functional.cosine_similarity(q_emb, d_emb, dim=0).item())

        neg_scores = []
        for doc in negative_docs:
            d_emb = model.encode(doc, convert_to_tensor=True, device=device)
            neg_scores.append(torch.nn.functional.cosine_similarity(q_emb, d_emb, dim=0).item())

    min_pos = min(pos_scores)
    max_neg = max(neg_scores)
    return min_pos, max_neg, pos_scores, neg_scores

def fine_tune_until_margin_respected(question, positive_docs, negative_docs,
                                     model, batch_size, epochs, warmup_steps, device,
                                     max_iterations=10):
    iteration = 0
    current_model = model

    while iteration < max_iterations:
        iteration += 1
        print(f"\n🔄 Itération #{iteration} de fine-tuning...")

        train_examples = [
            InputExample(texts=[question, doc], label=1.0) for doc in positive_docs
        ] + [
            InputExample(texts=[question, doc], label=0.0) for doc in negative_docs
        ]

        if not train_examples:
            print("⚠️ Aucun exemple pour fine-tuning.")
            break

        train_dataloader = DataLoader(train_examples, shuffle=True, batch_size=batch_size)
        train_loss = losses.CosineSimilarityLoss(current_model)

        current_model.train()
        current_model.fit(
            train_objectives=[(train_dataloader, train_loss)],
            epochs=epochs,
            warmup_steps=warmup_steps,
            show_progress_bar=True
        )
        print("✅ Fine-tuning terminé pour cette itération.")

        min_pos, max_neg, pos_scores, neg_scores = evaluate_score_margin(
            current_model, question, positive_docs, negative_docs, device
        )

        print(f"📈 Scores positifs : {['%.4f' % s for s in pos_scores]}")
        print(f"📉 Scores négatifs : {['%.4f' % s for s in neg_scores]}")
        print(f"✅ min(score_positif) = {min_pos:.4f}")
        print(f"❌ max(score_négatif) = {max_neg:.4f}")

        if min_pos > max_neg:
            print("🎯 Condition atteinte : tous les positifs sont mieux scorés que tous les négatifs.")
            break
        else:
            print("🔁 Encore des négatifs mieux scorés que des positifs. On continue...")

    current_model.save("models/esti-rag-ft")
    return SentenceTransformer("models/esti-rag-ft", device=device)

def fine_tune_with_multiple_negatives(question, positive_docs, model, batch_size, epochs, warmup_steps, device):
    # Créer des paires question <-> positive
    train_examples = [InputExample(texts=[question, doc]) for doc in positive_docs]

    if not train_examples:
        print("⚠️ Aucun exemple pour fine-tuning.")
        return None

    train_dataloader = DataLoader(train_examples, shuffle=True, batch_size=batch_size)
    train_loss = losses.MultipleNegativesRankingLoss(model)

    print(f"🏋️ Fine-tuning avec MultipleNegativesRankingLoss sur {len(train_examples)} exemples...")

    model.train()
    model.fit(
        train_objectives=[(train_dataloader, train_loss)],
        epochs=epochs,
        warmup_steps=warmup_steps,
        show_progress_bar=True
    )
    print("✅ Fine-tuning terminé.")

    model.save("models/esti-rag-ft")
    return SentenceTransformer("models/esti-rag-ft", device=device)

