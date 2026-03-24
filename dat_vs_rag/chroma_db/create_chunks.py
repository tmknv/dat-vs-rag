from BM25 import train_bm25, genetate_sparse_vectors



def get_dataset():
    return [""]

def get_chunks(filename: str):
    return [
    "cat eat mouse",
    "dog play cat",
    "man eat apple"
]

def get_chunks_with_embedding(filename: str) ->dict: #dict{"chunks", "dense_embeddings", "sparse_vectors"}
    train_bm25()

    documents = get_chunks(filename)

    sparse_vectors = genetate_sparse_vectors(documents)

    dict = {"documents": documents, "sparse_vectors": sparse_vectors}
    return dict


get_chunks_with_embedding("file_name")

    