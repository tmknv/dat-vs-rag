'''
Файл для генерации эмбедингов бертом
'''


from sentence_transformers import SentenceTransformer
import chromadb
import numpy as np
from dotenv import load_dotenv
import os
from huggingface_hub import login

load_dotenv()
login(token=os.getenv("HF_TOKEN"))

'''глобальная модель берта, чтобы при каждом запросе не подгружать'''
BERT = None # вынести в config.yaml

def generate_embeddings(documents: list[str]) ->list[list[int]]:
    
    '''
    инициализирует модель берт и создает эмбединги для документов
    '''

    global BERT

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

    global BERT
    if BERT is None:
        BERT = SentenceTransformer("nickprock/ModernBERT-base-sts")


    return BERT.encode(
        [query],
        normalize_embeddings=True 
    )[0]

def semantic_scores(query: str):
    """
    Возвращает описание документов с косинусными расстояниями (скорами).
    """
    client = chromadb.PersistentClient(path="./dat_vs_rag/chroma_db/data")
    collection = client.get_collection("semantic_collection")
    
    query_emb = generate_query_embedding(query)

    data = collection.get(include=["documents", "embeddings"])
    
    scores = {}
    for i in range(len(data["documents"])):
        doc = data["documents"][i]
        doc_emb = data["embeddings"][i]
        scores[doc] = 1 + np.dot(query_emb, doc_emb)
    
    return scores


# print(len(generate_embeddings(["привет"])[0]))

# print(len(generate_embeddings(["привет, как дела?"])[0]))