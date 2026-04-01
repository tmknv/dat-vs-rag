'''
Файл генерации ответа DAT SLM
'''

from .DAT import get_DAT_context

from .models import Gemma_3_4B



def DAT_SLM_response(query: str) ->str:

    '''генерирует ответ SLM на основе запроса и контекста, найденного алгоритмом DAT

    Аргументы:
        Запрос пользователя
    
    Возврацает:
        Ответ модели
    '''

    context = get_DAT_context(query)
    
    request_with_context = f"""
        You are a helpful assistant that answers questions based ONLY on the provided contexts.

        Contexts:
        1. {context[0]}
        2. {context[1]}
        3. {context[2]}

        Question: {query}

        Instructions:
        - Answer ONLY using information from the contexts above
        - If the answer cannot be found in the contexts, say "I don't have enough information to answer this question"
        - Do not use your own knowledge or external information
        - Be concise and direct
        - If the contexts conflict, explain the discrepancy

        Answer:
    """

    response = Gemma_3_4B(request_with_context)

    return response




