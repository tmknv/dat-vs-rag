from sentence_transformers import SentenceTransformer


'''глобальная модель берта, чтобы при каждом запросе не подгружать'''
BERT = None # вынести в config.yaml

def generate_embeddings(documents: list[str]) ->list[list[int]]:
    
    '''
    инициализирует модель берт и создает эмбединги для документов
    '''

    global BERT

    BERT = SentenceTransformer("nickprock/ModernBERT-base-sts")

    doc_embeddings = BERT.encode(
        documents,
        normalize_embeddings=True 
    )

    return doc_embeddings


def generate_query_embedding(query: str) ->list[int]:

    '''
    создает эмбединги для запросов
    '''

    return BERT.encode(
        [query],
        normalize_embeddings=True 
    )[0]