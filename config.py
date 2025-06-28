import os
import torch

EMBEDDING_MODEL_PATH = "models/esti-rag-ft"
DEFAULT_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
TOP_K = 3
BATCH_SIZE = 4
EPOCHS = 2
WARMUP_STEPS = 10
