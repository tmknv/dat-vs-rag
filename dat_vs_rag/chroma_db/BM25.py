from pinecone_text.sparse import BM25Encoder
import json

import os

documents = [
    "cat eat mouse",
    "dog play cat",
    "man eat apple"
]

def genetate_sparse_vectors(documents) -> dict:
    bm25 = BM25Encoder()
    bm25.load("./dat_vs_rag/chroma_db/data/bm25_param.json")

    sparse_vectors = []
    
    for document in documents:
        vector = bm25.encode_documents(document)
        sparse_vectors.append(vector["indices"] + vector["values"])
    
    return sparse_vectors


def train_bm25():
    if os.path.exists("./dat_vs_rag/chroma_db/data/bm25_param.json"):
        return

    bm25 = BM25Encoder()
    bm25.fit(documents)
    bm25.dump("./dat_vs_rag/chroma_db/data/bm25_param.json")

    print("bm25 trained!")

