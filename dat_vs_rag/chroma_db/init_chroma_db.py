import chromadb 
from chromadb.utils.embedding_functions import ChromaBm25EmbeddingFunction


client = chromadb.PersistentClient(path="./dat_vs_rag/chroma_db")

bm25_ef = ChromaBm25EmbeddingFunction(
    k=1.2,
    b=0.75,
    avg_doc_length=256.0,
    token_max_length=40
)
    
lexical_collection = client.create_collection(
    name="lexical_collection",
    embedding_function=bm25_ef,  # BM25 для лексического поиска
    metadata={"type": "lexical", "description": "Поиск по ключевым словам"}
)

semantic_collection = client.create_collection(
    name="semantic_collection",
    metadata={"type": "semantic", "description": "Поиск по смыслу"}
)

