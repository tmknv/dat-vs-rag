import chromadb 
import os

from create_chunks import get_chunks_with_embedding, get_dataset



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


    #заполнение базы данных 
    DATASET = get_dataset()
    for filename in DATASET:
        chunks = get_chunks_with_embedding(filename)
        lexical_collection.add(
            ids=[f"id{i}" for i in range(len(chunks["documents"]))],
            embeddings=chunks["sparse_vectors"],
            documents=chunks["documents"]
        )
        semantic_collection.add(
            ids=[f"id{i}" for i in range(len(chunks["documents"]))],
            embeddings=chunks["embeddings"],
            documents=chunks["documents"]
        )
    
    print("Chroma db initialised!")


init_chroma_db()