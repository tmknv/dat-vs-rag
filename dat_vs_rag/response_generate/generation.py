from dat_vs_rag.response_generate.qr_process import query_process

def get_responses(query: str):
    response = "response"

    processed_query = query_process(query)
    return processed_query
    