def query_process(query: str):
    return query.lower()

def get_responses(query: str):
    response = "response"

    processed_query = query_process(query)

    return response