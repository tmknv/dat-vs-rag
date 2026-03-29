'''
Файл с реализацией RAG
'''


from dat_vs_rag.chroma_db.BM25 import get_BM25_scores
from dat_vs_rag.chroma_db.ModernBert import semantic_scores


def get_hibrid_scores(query: str, alpha: float) ->dict:

    '''расчет гибридного скора между запросом и каждым документом
    
    Аргументы:
        Запрос пользователя

    Возвращает:
        Гибридный скор между запросом и каждым документом
    '''

    BM25_scores = get_BM25_scores(query)
    sem_scores = semantic_scores(query)

    hibrid_scores = {}

    for document in BM25_scores:
        hibrid_scores[document] = alpha * sem_scores[document] + (1-alpha) * BM25_scores[document]

    return hibrid_scores

def get_top3_docs(scores: dict[str, float]) ->list[str]:

    '''ищет топ 3 в словаре со всеми документами и их скорами 
    
    Аргументы:
        Словарь гибридных скоров между запросом и каждым документом

    Возвращает:
        Топ 3 документа по гибридному скору
    '''

    top3 = [{"doc1": -10.0}, {"doc2": -10.0}, {"doc3": -10.0}]

    for doc in scores:
        score = scores[doc]

        if score > list(top3[2].values())[0]:
            if score > list(top3[1].values())[0]:
                if score > list(top3[0].values())[0]:
                    top3 = [{doc: score}] + top3[:2]
                else:
                    top3[2] = top3[1]
                    top3[1] = {doc: score}
            else:
                top3[2] = {doc: score}
    
    return [list(top.keys())[0] for top in top3]

def get_RAG_context(query: str) ->list[str]:

    '''возвращает контекст по запросу

    Аргументы:
        Запрос пользователя
    
    Возвращает:
        контекст, найденный алгоритмом DAT
    '''

    docs_with_hibrid_score = get_hibrid_scores(query, alpha=0.5)

    top3 = get_top3_docs(docs_with_hibrid_score)

    return top3
