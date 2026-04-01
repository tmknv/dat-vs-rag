'''
Файл генерации ответа RAG LLM
'''

from .RAG import get_RAG_context

from .models import Gemma_3_27B



def RAG_LLM_response(query: str) ->str:
  
    '''генерирует ответ LLM на основе запроса и контекста, найденного алгоритмом RAG

    Аргументы:
        Запрос пользователя
    
    Возврацает:
        Ответ модели
    '''

    context = get_RAG_context(query)
    
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

    response = Gemma_3_27B(request_with_context)

    return response
