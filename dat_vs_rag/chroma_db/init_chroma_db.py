import chromadb 
import os

from .create_chunks import get_chunks_with_embedding, get_dataset


# слишком много логики в 1 фцнкции, разбить (касается всего проекта)
# Если написано init_chroma_db, то это обычно инициализация, создание, а у вас еще и заполнение
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


    DATASET = get_dataset()
    for filename in DATASET:
        chunks = get_chunks_with_embedding(filename)
        lexical_collection.add(
            ids=[f"id{i}" for i in range(len(chunks["documents"]))],
            embeddings=[vec[len(vec)//2:] for vec in chunks["sparse_vectors"]],
            documents=chunks["documents"],
            metadatas=[{"indices": vec[:len(vec)//2]} for vec in chunks["sparse_vectors"]]
        )
        semantic_collection.add(
            ids=[f"id{i}" for i in range(len(chunks["documents"]))],
            embeddings=chunks["embeddings"],
            documents=chunks["documents"]
        )
