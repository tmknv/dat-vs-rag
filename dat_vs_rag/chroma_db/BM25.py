#написать шапку
from pinecone_text.sparse import BM25Encoder
import chromadb 

import os


# хранить в отдельном месте
documents = [
    "there are 10 apples in the box",
    "apple is red",
    "box is large",
    "womans play games bad"
]



'''Глобальная обученая модель bm25'''
BM25 = None

'''
Глобальный словарь с индексами BM25 их местами в разреженом векторе
индексы - ключи, значения - место в векторе    
'''
KEYS = {} 

#докстринги не в гугл формате, но это уже в несколько раз лучше чем у лехи с савой
def generate_query_sparse_vector(query: str) ->list[float]:

    '''Генерирует разреженый вектор для запроса

    Аргументы:
        Запрос пользователя
    
    Возвращает:
        Разреженый вектор запроса
    '''

    #если модель не загружена - загружаем
    global BM25
    if BM25 is None:
        BM25 = BM25Encoder()
        BM25.load("./dat_vs_rag/chroma_db/data/bm25_param.json")


    indices_values = BM25.encode_queries(query)
    indices = indices_values["indices"]
    values = indices_values["values"]

    sparse_vector = [0]*len(KEYS)
    for i in range(len(indices)):
        if int(indices[i]) in KEYS:
            sparse_vector[KEYS[int(indices[i])]] = values[i]


    return sparse_vector


def genetate_sparse_vectors(documents: list[str]) -> list[float]:


    '''Генерирует разреженые вектора для документов

    Аргументы:
        Список документов (чанков)

    Возвращает:
        Список разреженых векторов каждого документа
    '''

    #если модель не загружена - загружаем   
    global BM25
    if BM25 is None:
        BM25 = BM25Encoder()
        BM25.load("./dat_vs_rag/chroma_db/data/bm25_param.json")


    sparse_vectors = []
    
    for document in documents:
        vector = BM25.encode_documents(document)

        sparse_vector = [0]*len(KEYS)
        for i in range(len(vector["indices"])):
            sparse_vector[KEYS[int(vector["indices"][i])]] = vector["values"][i]

        sparse_vectors.append(sparse_vector)
    
    return sparse_vectors


def train_bm25():

    '''
    Тренирует bm25 на документах
    '''

    global BM25

    if os.path.exists("./dat_vs_rag/chroma_db/data/bm25_param.json"):
        return

    BM25 = BM25Encoder()
    BM25.fit(documents)
    BM25.dump("./dat_vs_rag/chroma_db/data/bm25_param.json")
    
    keys = list(BM25.doc_freq.keys())
    for i in range(len(keys)):
        KEYS[keys[i]] = i

    print("bm25 trained!")








def BM25_score(query: str,doc_vector: list[float]) ->int:

    '''Рассчитывает score между запросом и документом

    Аргументы:
        Запрос пользователя, разреженый вектор документа 

    Возвращает:
        Score между запросом и документом
    '''

    query_vector = generate_query_sparse_vector(query)
    score = sum(a*b for a, b in zip(query_vector, doc_vector))
    return score

def get_BM25_scores(query: str) -> dict[str, float]: 

    '''Рассчитывает скоры между запросом и каждым документом

    Аргументы:
        Запрос полльзователя

    Возвращает:
        Score змежду запросом и каждым документом
    '''

    client = chromadb.PersistentClient(path="./dat_vs_rag/chroma_db/data")
    collection = client.get_collection(name="lexical_collection")
    data = collection.get(include=["documents", "embeddings"])

    documents = data["documents"]
    doc_vectors = data["embeddings"]

    sqores = {}

    for i in range(len(documents)):
        sqores[documents[i]] = BM25_score(query, doc_vectors[i])

    return sqores



