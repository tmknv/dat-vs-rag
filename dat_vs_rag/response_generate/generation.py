from dat_vs_rag.response_generate.qr_process import query_process
from dat_vs_rag.response_generate.agents.DAT_SLM import DAT_SLM_response
from dat_vs_rag.response_generate.agents.RAG_LLM import RAG_LLM_response
from dat_vs_rag.utils.logger import setup_logging
from dat_vs_rag.utils.logger import get_logger

import time

setup_logging()
logger = get_logger(__name__)

def get_responses(query: str, RAG_retriever_type: str, alpha_coefficient: float):

    logger.info(f"Запрос пользователя: {query}")

    processed_query = query_process(query)

    logger.info(f"Обработанй запрос: {processed_query}")

    DAT_time = 0
    RAG_time = 0

    start = time.time()
    DAT_response = DAT_SLM_response(processed_query)
    end = time.time()
    DAT_time = end-start

    start = time.time()
    RAG_response = RAG_LLM_response(processed_query, RAG_retriever_type, alpha_coefficient)
    end = time.time()
    RAG_time = end - start

    logger.info(f"DAT SLM response:\n{DAT_response}\nTIME: {DAT_time} seconds\n")
    logger.info(f"RAG LLM response:\n{RAG_response}\nTIME: {RAG_time} seconds\n")

    total_response = f"**DAT SLM response:**\n{DAT_response}\n\n**{RAG_retriever_type} RAG LLM response:**\n{RAG_response}"

    return total_response
    