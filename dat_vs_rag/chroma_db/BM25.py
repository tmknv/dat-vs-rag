from pinecone_text.sparse import BM25Encoder
import chromadb 

import os

documents = [
    "cat eat mouse",
    "dog play cat",
    "man eat apple"
]



'''Глобальная обученая модель bm25'''
BM25 = None


def generate_query_sparse_vector(query: str) ->list[int]:

    '''
    генерирует разреженый вектор для запроса
    структура вектора: первая половина - индексы, вторая половина - веса
    '''

    vector = BM25.encode_queries([query])
    return vector[0]['indices'] + vector[0]['values']


def genetate_sparse_vectors(documents: list[str]) -> dict:


    '''
    генерирует разреженые вектора для документов
    структура вектора: первая половина - индексы, вторая половина - веса
    '''

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





'''
все что ниже - недоделано!
'''

def make_dict_sparse_vector(vector: list[int]) ->dict[int, int]:

    '''
    превращает неудобное представление sparse vector в виде списка в удобный словарь
    '''

    dict = {}
    for i in range(len(vector)//2):
        dict[int(vector[i])] = vector[i+len(vector)//2]

    return dict


def BM25_score(query: str, document: list[int]) ->int:

    '''
    возвращает score между запросов и документом
    '''


    query_vector = make_dict_sparse_vector(generate_query_sparse_vector(query))
    document_vector = make_dict_sparse_vector(document)

    print("\ndocs", document)
    print("\nvecs", document_vector)
    print("\nquer", query_vector, "\n")

    sqore = 0

    for index in query_vector:
        if index in document_vector.keys():
            print(index, "is yes")
            sqore += query_vector[index]*document_vector[index]

    return sqore


def get_BM25_scores(query: str) -> dict[str, str]: 

    '''
    возвращает score змежду запросом и каждым документом
    '''

    client = chromadb.PersistentClient(path="./dat_vs_rag/chroma_db/data")
    collection = client.get_collection(name="lexical_collection")
    documents_with_embeddings = collection.get(include=["documents", "embeddings"])

    print(documents_with_embeddings)

    scores = []

    for i in range(len(documents_with_embeddings["documents"])):
        score = BM25_score(query, documents_with_embeddings["embeddings"][i])
        scores.append(score)

    return {
        "documents": documents_with_embeddings["documents"],
        "scores": scores
    }




