from pinecone_text.sparse import BM25Encoder

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
    
    return vector["indices"] + vector["values"]



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

