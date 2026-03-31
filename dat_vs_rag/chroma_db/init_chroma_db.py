'''
Инициализация и заполнения Chroma db
'''


import chromadb 
import os

from .create_chunks import get_chunks_with_embedding, get_dataset, get_chunks
from .BM25 import train_bm25


def init_chroma_db():

    '''
    инициализирует chroma db:    
    создает клиент и две коллекции - для лексического и семантического поисков
    '''


    #проверка на наличие бд. если есть - выходим из функции
    if os.path.exists("./dat_vs_rag/chroma_db/data/chroma.sqlite3"):
        return

    client = chromadb.PersistentClient(path="./dat_vs_rag/chroma_db/data")

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


    DATASET = get_dataset(dataset_name="natural_questions", limit=10)

    total_chunks = []
    for sample in DATASET:
        chunks = get_chunks(sample)
        total_chunks += chunks

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
