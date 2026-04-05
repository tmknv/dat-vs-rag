'''
Файл для генерации эмбедингов бертом
'''


from sentence_transformers import SentenceTransformer
import chromadb
import numpy as np
from dotenv import load_dotenv
import os

from dat_vs_rag.utils.load_params import get_params

PARAMS = get_params()

load_dotenv()


def load_model():
    """Загружает ModernBERT через SentenceTransformer"""
    save_path = PARAMS["paths"]["models"]["ModernBert_path"]
    model_name = PARAMS["model"]["name"]
    token = os.getenv("HF_TOKEN")

    # SentenceTransformer умеет загружать из локальной папки
    if os.path.exists(save_path) and os.listdir(save_path):
        return 
    os.makedirs(save_path, exist_ok=True)

    model = SentenceTransformer(model_name_or_path=model_name, token=token)
    # Сохраняем локально
    model.save(save_path)
    print(f"{model_name} loaded!")




def generate_embeddings(documents: list[str]) ->list[list[int]]:
    
    '''
    инициализирует модель берт и создает эмбединги для документов
    '''

    BERT = SentenceTransformer(PARAMS["paths"]["models"]["ModernBert_path"])

    doc_embeddings = BERT.encode(
        documents,
        normalize_embeddings=True 
    )

    return doc_embeddings


def generate_query_embedding(query: str) ->list[int]:

    '''
    создает эмбединги для запросов
    '''
    
    BERT = SentenceTransformer(PARAMS["paths"]["models"]["ModernBert_path"])

    return BERT.encode(
        [query],
        normalize_embeddings=True 
    )[0]




def semantic_scores(query: str) ->dict:
    """
    Возвращает описание документов с косинусными расстояниями (скорами).
    """
    client = chromadb.PersistentClient(path=PARAMS["paths"]["chroma_db"]["chroma_db_path"])
    collection = client.get_collection("semantic_collection")
    
    query_emb = generate_query_embedding(query)

    data = collection.get(include=["documents", "embeddings"])
    
    scores = {}
    for i in range(len(data["documents"])):
        doc = data["documents"][i]
        doc_emb = data["embeddings"][i]
        scores[doc] = 1 + np.dot(query_emb, doc_emb)
    
    return scores

