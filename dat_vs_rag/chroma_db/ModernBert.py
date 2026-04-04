'''
Файл для генерации эмбедингов бертом
'''


from sentence_transformers import SentenceTransformer
import chromadb
import numpy as np
from dotenv import load_dotenv
import os
from huggingface_hub import login

from dat_vs_rag.utils.load_params import get_params

PARAMS = get_params()

load_dotenv()
login(token=os.getenv("HF_TOKEN"))



def generate_embeddings(documents: list[str]) ->list[list[int]]:
    
    '''
    инициализирует модель берт и создает эмбединги для документов
    '''

    BERT = SentenceTransformer("nickprock/ModernBERT-base-sts")

    doc_embeddings = BERT.encode(
        documents,
        normalize_embeddings=True 
    )

    return doc_embeddings


def generate_query_embedding(query: str) ->list[int]:

    '''
    создает эмбединги для запросов
    '''
    
    BERT = SentenceTransformer("nickprock/ModernBERT-base-sts")

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

