from BM25 import train_bm25, genetate_sparse_vectors
from ModernBert import generate_embeddings



def get_dataset() ->list[str]:
    
    '''
    Возвращает список названий файлов датасета
    '''

    return [""]

def get_chunks(filename: str) ->list[str]:

    '''
    Разбивает текст одного файла на чанки
    '''

    return [
        "cat eat mouse",
        "dog play cat",
        "man eat apple"
    ]   

def get_chunks_with_embedding(filename: str) ->dict: #dict{"chunks", "sparse_vectors", "embeddings"}

    '''
    возвращает чанки файла в виде словаря со структурой: [чанки, их разряженые вектора, их эмбеддинги]
    '''

    train_bm25()

    documents = get_chunks(filename)

    sparse_vectors = genetate_sparse_vectors(documents)
    embeddings = generate_embeddings(documents)

    dict = {"documents": documents, "sparse_vectors": sparse_vectors, "embeddings": embeddings}
    return dict


    