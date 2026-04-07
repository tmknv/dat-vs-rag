from dat_vs_rag.response_generate.qr_process import query_process
from .agents.DAT_SLM import DAT_SLM_response
from .agents.RAG_LLM import RAG_LLM_response

def get_responses(query: str, RAG_retriever_type: str, alpha_coefficient: float):
    processed_query = query_process(query)

    DAT_response = DAT_SLM_response(processed_query)
    RAG_response = RAG_LLM_response(processed_query, RAG_retriever_type, alpha_coefficient)

    total_response = f"DAT SLM response:\n{DAT_response}\n\n{RAG_retriever_type} RAG LLM response:\n{RAG_response}"

    return total_response
    