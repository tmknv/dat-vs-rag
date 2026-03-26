from DAT import get_DAT_context

from models import Arcee_Mini

def create_request_with_context(query: str, context: list[str]):
    return ""


def DAT_SLM_response(query: str) ->str:

    '''
    генерирует ответ на основе запроса и контекста, найденного алгоритмом DAT
    '''

    context = get_DAT_context(query)
    
    request_with_context = create_request_with_context(query, context)

    response = Arcee_Mini(request_with_context)

    return response




