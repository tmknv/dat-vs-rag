from sentence_transformers import SentenceTransformer

BERT = None

def generate_embeddings(documents):
    global BERT

    BERT = SentenceTransformer("nickprock/ModernBERT-base-sts")

    doc_embeddings = BERT.encode(
        documents,
        normalize_embeddings=True 
    )

    return doc_embeddings


def generate_query_embedding(query: str):
    return BERT.encode(
        [query],
        normalize_embeddings=True 
    )[0]