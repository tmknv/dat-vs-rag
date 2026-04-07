'''
Файл для разбивания датасета на чанки
'''
from dat_vs_rag.chroma_db.making_NQjsonl import load_NQjsonl
from dat_vs_rag.chroma_db.BM25 import genetate_sparse_vectors
from dat_vs_rag.chroma_db.ModernBert import generate_embeddings 
from dat_vs_rag.utils.load_params import get_params

from chonkie import TokenChunker
import json
import os


PARAMS = get_params()


chunker = TokenChunker(
    tokenizer="answerdotai/ModernBERT-base",
    chunk_size=256, 
    chunk_overlap=20
)

def load_local_nq(path: str, limit: int = 1000) -> list[dict]:

    if not os.path.exists(PARAMS["paths"]["datasets"]["natural_questions_path"]):
        load_NQjsonl()

    result = []

    with open(path, "r", encoding="utf-8") as f:
        for i, line in enumerate(f):
            if i >= limit:
                break

            sample = json.loads(line)

            if not sample["text"].strip():
                continue

            result.append(sample)

    return result


def get_dataset(dataset_name: str = "natural_questions", limit: int = 10) -> list[dict]:
    '''
    Универсальная точка входа для загрузки датасета.
    Все датасеты приводятся к формату:
    {
        "id": "...",
        "question": "...",
        "text": "..."
    }
    '''

    os.makedirs(PARAMS["paths"]["datasets"]["datasets_path"], exist_ok=True)

    if dataset_name == "natural_questions":
        return load_local_nq(PARAMS["paths"]["datasets"]["natural_questions_path"], limit)

    raise ValueError(f"Unsupported dataset: {dataset_name}")


def get_chunks(sample: dict) -> list[str]:
    text = sample["text"]

    context_chunks = chunker.chunk(text)

    return [
        chunk.text
        for chunk in context_chunks
    ]


def get_chunks_with_embedding(documents: list[str]):

    sparse_vectors = genetate_sparse_vectors(documents)
    embeddings = generate_embeddings(documents)

    return {
        "documents": documents,
        "sparse_vectors": sparse_vectors,
        "embeddings": embeddings
    }

