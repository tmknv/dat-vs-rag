'''
Инициализация и заполнения Chroma db
'''


import chromadb 
import os

from dat_vs_rag.chroma_db.create_chunks import get_chunks_with_embedding, get_dataset, get_chunks
from dat_vs_rag.chroma_db.BM25 import train_bm25
from dat_vs_rag.chroma_db.ModernBert import load_model
from dat_vs_rag.utils.load_params import get_params


PARAMS = get_params()


def init_chroma_db():

    '''
    инициализирует chroma db:    
    создает клиент и две коллекции - для лексического и семантического поисков
    '''

    #проверка на наличие бд. если есть - выходим из функции
    if os.path.exists(PARAMS["paths"]["chroma_db"]["chroma_sqlite_path"]):
        return
    
    chromadb_path = PARAMS["paths"]["chroma_db"]["chroma_db_path"]
    os.makedirs(chromadb_path, exist_ok=True)



    client = chromadb.PersistentClient(path=chromadb_path)

    #коллекция семантического поиска
    semantic_collection = client.create_collection(
        name="semantic_collection",
        metadata={"type": "semantic", "description": "Поиск по смыслу"}
    )

    #коллекция лексического поиска
    lexical_collection = client.create_collection(
        name="lexical_collection",
        metadata={"type": "lexical", "description": "Поиск по ключевым словам"}
    )
    
    print("Chroma db initialised!")
    load_chroma_db(lexical_collection, semantic_collection)


def load_chroma_db(lexical_collection, semantic_collection):

    '''
    заполняет базу данных
    '''


    DATASET = get_dataset(dataset_name="natural_questions", limit=65)

    total_chunks = []
    for sample in DATASET:
        chunks = get_chunks(sample)
        total_chunks += chunks

    load_model()
    train_bm25(total_chunks)

    total_chunks_with_embedings = get_chunks_with_embedding(total_chunks)

    lexical_collection.add(
        ids=[f"id{i}" for i in range(len(total_chunks_with_embedings["documents"]))],
        embeddings=total_chunks_with_embedings["sparse_vectors"],
        documents=total_chunks_with_embedings["documents"]
    )
    semantic_collection.add(
        ids=[f"id{i}" for i in range(len(total_chunks_with_embedings["documents"]))],
        embeddings=total_chunks_with_embedings["embeddings"],
        documents=total_chunks_with_embedings["documents"]
    )

    print("Chroma db loaded!")

