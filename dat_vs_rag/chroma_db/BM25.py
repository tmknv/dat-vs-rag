#написать шапку
from pinecone_text.sparse import BM25Encoder
import chromadb 

import os


# хранить в отдельном месте
documents = [
    "cat eat mouse",
    "dog play cat",
    "man eat apple"
]

def make_dict_sparse_vector(indices: list[int], embeddings: list[float]) ->dict[int, float]:

    '''
    превращает неудобное представление sparse vector в виде двух списков в удобный словарь
    '''

    dict = {}
    for i in range(len(indices)):
        dict[indices[i]] = embeddings[i]

    return dict




'''Глобальная обученая модель bm25'''
BM25 = None

#докстринги не в гугл формате, но это уже в несколько раз лучше чем у лехи с савой
def generate_query_sparse_vector(query: str) ->dict[int, float]:

    '''
    генерирует разреженый вектор для запроса
    '''

    #если модель не загружена - загружаем
    global BM25
    if BM25 is None:
        BM25 = BM25Encoder()
        BM25.load("./dat_vs_rag/chroma_db/data/bm25_param.json")


    vector = BM25.encode_queries([query])

    return make_dict_sparse_vector(vector[0]['indices'], vector[0]['values'])


def genetate_sparse_vectors(documents: list[str]) -> dict:


    '''
    генерирует разреженые вектора для документов
    структура вектора: первая половина - индексы, вторая половина - веса
    '''

    #если модель не загружена - загружаем   
    global BM25
    if BM25 is None:
        BM25 = BM25Encoder()
        BM25.load("./dat_vs_rag/chroma_db/data/bm25_param.json")


    sparse_vectors = []
    
    for document in documents:
        vector = BM25.encode_documents(document)
        sparse_vectors.append(vector["indices"] + vector["values"])
    
    return sparse_vectors


def train_bm25():

    '''
    тренирует bm25 на документах
    '''

    global BM25

    if os.path.exists("./dat_vs_rag/chroma_db/data/bm25_param.json"):
        return

    BM25 = BM25Encoder()
    BM25.fit(documents)
    BM25.dump("./dat_vs_rag/chroma_db/data/bm25_param.json")

    print("bm25 trained!")








def BM25_score(query: str, indices: list[int], embeddings: list[float]) ->int:

    '''
    возвращает score между запросов и документом
    '''


    query_vector = generate_query_sparse_vector(query)
    document_vector = make_dict_sparse_vector(indices, embeddings)


    sqore = 0

    for index in query_vector:
        if index in document_vector.keys():
            sqore += query_vector[index]*document_vector[index]

    return sqore


def get_BM25_scores(query: str) -> dict[str, str]: 

    '''
    возвращает score змежду запросом и каждым документом
    '''

    client = chromadb.PersistentClient(path="./dat_vs_rag/chroma_db/data")
    collection = client.get_collection(name="lexical_collection")
    data = collection.get(include=["documents", "embeddings", "metadatas"])

    documents = data["documents"]
    embeddings = data["embeddings"]
    indices = [dir["indices"] for dir in data["metadatas"]]

    sqores = {}

    for i in range(len(documents)):
        sqores[documents[i]] = BM25_score(query, indices[i], embeddings[i])

    return sqores


