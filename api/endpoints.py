from fastapi import APIRouter, HTTPException,UploadFile,Depends, File
from .schemas import QuestionRequest, FeedbackRequest
from middlewares.auth import get_current_user 
from model.fine_tuning import fine_tune_until_margin_respected
from qdrant_client import QdrantClient
from qdrant_client.models import PointStruct, VectorParams,ScrollRequest
from qdrant_client.models import Filter, FieldCondition, Range
from qdrant_client.http.models import PayloadSchemaType
from model.document_parser import extract_text
from model.embedding import get_embedding, chunk_text_heuristique,get_latest_model_path
import numpy as np
from sentence_transformers import SentenceTransformer
from config import BATCH_SIZE, EPOCHS, WARMUP_STEPS, DEVICE
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from fastapi import Query
import requests
import os
import uuid
import hashlib
import shutil
from fastapi.responses import FileResponse
from dotenv import load_dotenv
from model.embedding import get_embedding
from supabase import create_client
import re
import gc
import unicodedata

def sanitize_filename(filename: str) -> str:
    # Supprimer les accents
    name = unicodedata.normalize('NFKD', filename).encode('ascii', 'ignore').decode('ascii')
    # Remplacer les espaces par des underscores
    name = name.replace(" ", "_")
    # Supprimer les caractères non alphanumériques ou d’extension
    name = re.sub(r"[^a-zA-Z0-9._-]", "", name)
    return name

load_dotenv()

SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_SERVICE_ROLE_KEY = os.getenv("SUPABASE_SERVICE_ROLE_KEY")

supabase = create_client(SUPABASE_URL, SUPABASE_SERVICE_ROLE_KEY)
router = APIRouter()

CHATBOT_RELOAD_URL = os.getenv("CHATBOT_RELOAD_URL", "https://madachat-embedder.hf.space/reload-model")  # À adapter selon ton infra


QDRANT_URL = os.getenv("QDRANT_URL")
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY")
COLLECTION = os.getenv("COLLECTION_NAME")
POSTGRES_COLLECTION = os.getenv("POSTGRES_COLLECTION_NAME")
VECTOR_SIZE = int(os.getenv("VECTOR_SIZE", "384"))

class DeployRequest(BaseModel):
    version: str = "esti-rag-ft-v7"

def get_next_model_version(base_name="esti-rag-ft", models_dir="./models") -> str:
    existing_versions = []
    for name in os.listdir(models_dir):
        if name.startswith(base_name + "-v"):
            try:
                version_num = int(name.replace(base_name + "-v", ""))
                existing_versions.append(version_num)
            except ValueError:
                continue
    next_version = max(existing_versions + [0]) + 1
    return f"{base_name}-v{next_version}"

@router.get("/")
def root():
    return {"message": "✅ RAG Webservice is running."}


@router.get("/documents")
def list_documents(user=Depends(get_current_user)):
    try:
        user_id = user.get("sub")
        results = []
        scroll_offset = None

        # Appliquer un filtre pour récupérer uniquement les documents de l'utilisateur
        qdrant_filter = Filter(
            must=[
                FieldCondition(key="owner_id", match={"value": user_id})
            ]
        )

        while True:
            scroll_result = client.scroll(
                collection_name=COLLECTION,
                scroll_filter=qdrant_filter,  # filtre activé ici
                with_payload=True,
                limit=100,
                offset=scroll_offset
            )
            points, scroll_offset = scroll_result
            results.extend([
                {
                    "text": point.payload.get("text", ""),
                    "source": point.payload.get("source", "inconnu")
                }
                for point in points if "text" in point.payload
            ])


            if scroll_offset is None:
                break

        return {"documents": results}

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/ask")
def ask(request: QuestionRequest, user=Depends(get_current_user)):
    try:
        model_path = "models/esti-rag-ft"

        if os.path.exists(model_path) and os.path.isdir(model_path):
            query_vector = get_embedding(request.question, model=model_path)
        else:
            query_vector = get_embedding(request.question, model="")  # modèle par défaut

        user_id = user.get("sub")

        # Filtre pour ne récupérer que les documents de l'utilisateur connecté
        qdrant_filter = Filter(
            must=[
                FieldCondition(key="owner_id", match={"value": user_id})
            ]
        )

        results = client.search(
            collection_name=COLLECTION,
            query_vector=query_vector,
            limit=request.top_k,
            with_payload=True,
            query_filter=qdrant_filter
        )

        return {
            "question": request.question,
            "results": [{"doc": r.payload["text"],"source":r.payload["source"], "score": r.score} for r in results]
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

def deterministic_id(text: str) -> str:
    hash_bytes = hashlib.sha256(text.encode('utf-8')).digest()
    return str(uuid.UUID(bytes=hash_bytes[:16]))

@router.post("/feedback")
def feedback(request: FeedbackRequest, user=Depends(get_current_user)):
    try:
        user_id = user.get("sub")
        model = SentenceTransformer(get_latest_model_path(), device=DEVICE)

        # Fine-tuning
        model = fine_tune_until_margin_respected(
            request.question,
            [doc.text for doc in request.positive_docs],
            [doc.text for doc in request.negative_docs],
            model,
            BATCH_SIZE,
            EPOCHS,
            WARMUP_STEPS,
            DEVICE,
            30,
        )

        # Réinjection dans Qdrant
        points = []
        for doc in request.positive_docs + request.negative_docs:
            embedding = model.encode(doc.text, normalize_embeddings=True).tolist()
            points.append(PointStruct(
                id=deterministic_id(doc.text),
                vector=embedding,
                payload={
                    "text": doc.text,
                    "source": doc.source,
                    "owner_id": user_id,
                }
            ))

        client.upsert(collection_name=COLLECTION, points=points)
        return {
            "message": "✅ Fine-tuning terminé et documents mis à jour dans Qdrant.",
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/deploy")
def deploy_model(user=Depends(get_current_user)):
    try:
        # 1. Générer une version
        version_name = get_next_model_version()

        model_src_dir = f"./models/esti-rag-ft"
        model_version_dir = f"./models/{version_name}"
        zip_path = f"./exported/{version_name}.zip"

        # 2. Copier le dossier du modèle vers une nouvelle version
        shutil.copytree(model_src_dir, model_version_dir)

        # 3. Créer dossier exporté s’il n’existe pas
        os.makedirs("./exported", exist_ok=True)

        # 4. Zipper la nouvelle version
        if os.path.exists(zip_path):
            os.remove(zip_path)
        shutil.make_archive(f"./exported/{version_name}", 'zip', model_version_dir)

        # 5. Construire l’URL de téléchargement
        model_url = f"https://madaTuneApi.onirtech.com/download-model?version={version_name}"

        # 6. Notifier le chatbot-service
        response = requests.get(CHATBOT_RELOAD_URL, params={
            "version": version_name,
            "url": model_url
        })

        if response.status_code != 200:
            raise Exception(f"Erreur de notification: {response.text}")

        return {
            "message": f"✅ Modèle '{version_name}' exporté et notification envoyée",
            "chatbot_response": response.json()
        }

    except Exception as e:
        print(str(e))
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/download-model")
def download_model(version: str):
    zip_path = f"./exported/{version}.zip"
    if not os.path.isfile(zip_path):
        raise HTTPException(status_code=404, detail="Modèle non trouvé")
    return FileResponse(path=zip_path, filename=f"{version}.zip", media_type="application/zip")

client = QdrantClient(
    url=QDRANT_URL,
    api_key=QDRANT_API_KEY,
)

# Initialiser la collection
""" client.recreate_collection(
    COLLECTION,
    vectors_config=VectorParams(size=VECTOR_SIZE, distance="Cosine"),
) """
if not client.collection_exists(COLLECTION):
    client.create_collection(
    collection_name=COLLECTION,
    vectors_config=VectorParams(size=VECTOR_SIZE, distance="Cosine")
)
    client.create_payload_index(
    collection_name=COLLECTION,
    field_name="source",
    field_schema="keyword"
)

    client.create_payload_index(
    collection_name=COLLECTION,
    field_name="owner_id",
    field_schema="keyword"
)



@router.post("/upload-file")
async def upload(file: UploadFile = File(...), user=Depends(get_current_user)):
    user_id = user.get("sub")
    filename = file.filename
    # 1. Vérifier doublon dans la table
    existing = supabase.table("documents").select("id") \
        .eq("name", filename).eq("owner_id", user_id).execute()
    if existing.data and len(existing.data) > 0:
        raise HTTPException(status_code=400, detail="Fichier déjà existant")
    # 2. Lire contenu du fichier
    contents = await file.read()
    # 3. Upload dans Supabase Storage (bucket = "documents")
    try:
        storage_path = f"{user_id}/{filename}"  # Structure: documents/user_id/filename.pdf
        safe_filename = sanitize_filename(filename)
        storage_path = f"{user_id}/{safe_filename}"
        supabase.storage.from_("documents").upload(
    path=storage_path,
    file=contents,
    file_options={"content-type": file.content_type}
)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Échec upload storage Supabase: {str(e)}")
    # 4. Sauvegarde temporaire pour extraction texte
    filepath = f"/tmp/{filename}"
    with open(filepath, "wb") as f:
        f.write(contents)
    text = extract_text(filepath)
    os.remove(filepath)
    # 5. Insertion dans la table "documents"
    document_data = {
        "name": filename,
        "owner_id": user_id,
        "url": storage_path  # ← optionnel mais utile pour l'accès plus tard
    }
    res = supabase.table("documents").insert(document_data).execute()
    if not res.data:
        raise HTTPException(status_code=500, detail="Échec insertion dans la table documents")
    # 6. Chunking + Qdrant
    points = []
    for chunk in chunk_text_heuristique(text):
        embedding = get_embedding(chunk)
        points.append(PointStruct(
            id=deterministic_id(chunk),
            vector=embedding,
            payload={
                "text": chunk,
                "source": filename,
                "owner_id": user_id
            }
        ))
    client.upsert(collection_name=COLLECTION, points=points)
    return {"status": "ok", "chunks": len(points)}


@router.delete("/delete-document")
async def delete_document(
    filename: str = Query(..., description="Nom exact du fichier à supprimer"),
    user=Depends(get_current_user)
):
    user_id = user.get("sub")
    # 1. Récupérer le chemin du fichier dans Supabase avant suppression
    existing = supabase.table("documents") \
        .select("url") \
        .eq("name", filename) \
        .eq("owner_id", user_id) \
        .execute()
    if not existing.data:
        raise HTTPException(status_code=404, detail="Document introuvable")
    # En déduire le chemin relatif dans le bucket (supprime le domaine de l'URL)
    # Exemple : https://xyz.supabase.co/storage/v1/object/public/documents/USER_ID/fichier.pdf
    key = existing.data[0]["url"]
    # 2. Supprimer l’entrée dans Supabase table "documents"
    delete_result = supabase.table("documents") \
        .delete() \
        .eq("name", filename) \
        .eq("owner_id", user_id) \
        .execute()
    print(f"delete_result: {delete_result}")
    if delete_result.data is  None:
        raise HTTPException(status_code=500, detail="Échec de la suppression dans Supabase")
    # 3. Supprimer le fichier du storage Supabase
    try:
        print(f"key: {key}")
        supabase.storage.from_("documents").remove([key])
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Erreur suppression storage: {str(e)}")
    # 4. Supprimer les vecteurs liés dans Qdrant
    # Supprimer les vecteurs liés dans Qdrant
    try:
        qdrant_filter = Filter(
            must=[
                FieldCondition(key="source", match={"value": filename}),
                FieldCondition(key="owner_id", match={"value": user_id})
            ]
        )
        client.delete(
    collection_name=COLLECTION,
    points_selector=Filter(
        must=[
            FieldCondition(key="source", match={"value": filename}),
            FieldCondition(key="owner_id", match={"value": user_id})
        ]
    ))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Erreur lors de la suppression dans Qdrant: {str(e)}")

    return {"status": "ok", "message": f"Document '{filename}' supprimé avec succès"}

@router.get("/search-docs")
def searchDocs(q: str, user=Depends(get_current_user)):
    user_id = user.get("sub")
    vector = get_embedding(q)
    # Filtrer les documents appartenant à l'utilisateur
    qdrant_filter = Filter(
        must=[
            FieldCondition(key="owner_id", match={"value": user_id})
        ]
    )
    results = client.search(
        collection_name=COLLECTION,
        query_vector=vector,
        limit=5,
        with_payload=True,
        query_filter=qdrant_filter
    )
    return [{"text": r.payload["text"], "score": r.score} for r in results]
