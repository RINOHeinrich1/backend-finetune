#!/bin/bash

# Répertoire source (le dossier courant)
SOURCE_DIR="$(pwd)"

# Répertoire cible sur le serveur
REMOTE_USER="rino"
REMOTE_HOST="onirtech.com"
REMOTE_DIR="/home/rino/deploy-finetune"  # à adapter selon l’endroit voulu

echo "📤 Déploiement en cours depuis $SOURCE_DIR vers $REMOTE_USER@$REMOTE_HOST:$REMOTE_DIR"

# Lancement du transfert avec rsync
rsync -avz --exclude 'node_modules' \
           --exclude 'lib' \
           --exclude 'lib64' \
           --exclude '__pycache__' \
           --exclude '.venv' \
           --exclude '.git' \
           --exclude '*.pyc' \
           "$SOURCE_DIR/" "$REMOTE_USER@$REMOTE_HOST:$REMOTE_DIR"

echo "✅ Déploiement terminé."
