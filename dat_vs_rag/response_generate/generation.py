from dat_vs_rag.response_generate.qr_process import query_process
from .agents.DAT_SLM import DAT_SLM_response

def get_responses(query: str):
    response = "response"

    processed_query = query_process(query)

    resonse = DAT_SLM_response(processed_query)

    return resonse
    