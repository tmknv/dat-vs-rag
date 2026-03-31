'''
Файл для разбивания датасета на чанки
'''

from itertools import islice
from .BM25 import train_bm25, genetate_sparse_vectors
from .ModernBert import generate_embeddings 

from datasets import load_dataset
from chonkie import TokenChunker



chunker = TokenChunker(
    tokenizer="answerdotai/ModernBERT-base",
    chunk_size=128, 
    chunk_overlap=20
)

def load_natural_questions(limit: int = 1) -> list[dict]:
    print("load")
    ds = load_dataset("natural_questions", split="train", streaming=True)
    result = []
    print("loaded")
    for i, row in enumerate(islice(ds, limit)):
        question = row["question"]["text"]

        tokens = row["document"]["tokens"]["token"]
        is_html = row["document"]["tokens"]["is_html"]

        clean_tokens = [tok for tok, html_flag in zip(tokens, is_html) if not html_flag]
        text = " ".join(clean_tokens).strip()

        if not text:
            continue

        result.append({
            "id": f"nq_{i}",
            "question": question,
            "text": text
        })
    print("ds len:", len(result))
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
    if dataset_name == "natural_questions":
        return load_natural_questions(limit)

    raise ValueError(f"Unsupported dataset: {dataset_name}")


def get_chunks(sample: dict) -> list[str]:
    question = sample["question"]
    text = sample["text"]

    context_chunks = chunker.chunk(text)

    return [
        f"{question}\n{chunk.text}"
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

