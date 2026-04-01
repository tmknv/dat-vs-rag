'''
Файл для обучения BM25, преобразования запросов и документов в разреженые векторы, расчета score между запросом пользователя и всеми документами 
'''


from pinecone_text.sparse import BM25Encoder
import chromadb 
from scipy.special import expit
import numpy as np

import os


#докстринги не в гугл формате, но это уже в несколько раз лучше чем у лехи с савой
def generate_query_sparse_vector(query: str) ->list[float]:

    '''Генерирует разреженый вектор для запроса

    Аргументы:
        Запрос пользователя
    
    Возвращает:
        Разреженый вектор запроса
    '''

    BM25 = BM25Encoder()
    BM25.load("./dat_vs_rag/chroma_db/data/bm25_param.json")

    KEYS = {}
    keys = list(BM25.doc_freq.keys())
    for i in range(len(keys)):
        KEYS[keys[i]] = i


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

    BM25 = BM25Encoder()
    BM25.load("./dat_vs_rag/chroma_db/data/bm25_param.json")
    
    KEYS = {}
    keys = list(BM25.doc_freq.keys())
    for i in range(len(keys)):
        KEYS[keys[i]] = i
    

    sparse_vectors = []
    
    for document in documents:
        vector = BM25.encode_documents(document)

        sparse_vector = [0]*len(KEYS)
        for i in range(len(vector["indices"])):
            sparse_vector[KEYS[int(vector["indices"][i])]] = vector["values"][i]

        sparse_vectors.append(sparse_vector)
    
    return sparse_vectors


def train_bm25(documents: list[str]):

    '''
    Тренирует bm25 на документах
    '''

    if os.path.exists("./dat_vs_rag/chroma_db/data/bm25_param.json"):
        return

    BM25 = BM25Encoder()
    BM25.fit(documents)
    BM25.dump("./dat_vs_rag/chroma_db/data/bm25_param.json")

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

    scores = [0]*len(documents)

    for i in range(len(documents)):
        scores[i] = BM25_score(query, doc_vectors[i])
    
    mean = np.average(scores)
    std = np.std(scores)

    scores_with_docs = {}

    for i in range(len(doc_vectors)):
        scores_with_docs[documents[i]] = expit((scores[i] - mean)/std)

    return scores_with_docs

