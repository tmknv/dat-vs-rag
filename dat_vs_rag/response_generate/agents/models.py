'''
Файл с языковыми моделями
'''

import requests
import json
import time
from dotenv import load_dotenv
import os

load_dotenv()

# api_key = os.getenv("OPENROUTER_API_KEY")

# ip всегда разный (яндекс меняет его при каждом новом запуске сервака), вводим вручную каждый раз
SLM_SERVER_URL = "http://111.88.153.175:8081"
LLM_SERVER_URL = "http://111.88.153.175:8082"

def Gemma_3_4B(query: str, max_retries: int = 3) -> str:
  
    '''Запрос к Gemma_3_4B

    Аргументы:
        Запрос пользователя, максимальное колличество повторения запроса при ошибке
    
    Возвращает:
        Ответ модели
    '''
        
    # headers = {
    #     "Authorization": f"Bearer {api_key}",
    #     "Content-Type": "application/json",
    #     "HTTP-Referer": "https://t.me/your_bot",
    #     "X-Title": "DAT-vs-RAG Bot",
    # }

    payload = {
        "prompt": query,
        "temperature": 0.7,
        "max_tokens": 50,
    }
    
    for attempt in range(max_retries):
        try:
            response = requests.post(
                url=f"{SLM_SERVER_URL}/completion",
                headers={"Content-Type": "application/json"},
                json=payload,
                timeout=60  
            )
            
            result = response.json()

            if 'error' not in result and result:
                return result['content']
            
            # Если ошибка, пробуем снова
            print(f"SLM:Attempt {attempt + 1} failed, error: {result['error']['message']}, retrying...")
            time.sleep(2 * (attempt + 1))
            
        except Exception as e:
            print(f"SLM:Attempt {attempt + 1} error: {e}, retrying...")
            time.sleep(2 * (attempt + 1))
    
    # Если все попытки провалились, возвращаем сообщение об ошибке
    return "Извините, не удалось получить ответ от модели. Попробуйте позже."


def Gemma_3_27B(query: str, max_retries: int = 3) -> str:
 
    '''Запрос к Gemma_3_27B

    Аргументы:
        Запрос пользователя, максимальное колличество повторения запроса при ошибке
    
    Возвращает:
        Ответ модели
    '''
        
    # headers = {
    #     "Authorization": f"Bearer {api_key}",
    #     "Content-Type": "application/json",
    #     "HTTP-Referer": "https://t.me/your_bot",
    #     "X-Title": "DAT-vs-RAG Bot",
    # }
    
    # payload = {
    #     "model": "google/gemma-3-27b-it:free",
    #     "messages": [{"role": "user", "content": query}]
    # }

    payload = {
        "prompt": query,
        "temperature": 0.7,
        "max_tokens": 50,
        "stop": ["\n\nВопрос:", "\n\nUser:"]
    }
    
    for attempt in range(max_retries):
        try:
            response = requests.post(
                url=f"{LLM_SERVER_URL}/completion",
                headers={"Content-Type": "application/json"},
                json=payload,
                timeout=60  
            )
            
            result = response.json()
            
            
            if 'error' not in result and result:
                return result['content']
            
            # Если ошибка, пробуем снова
            print(f"LLM: Attempt {attempt + 1} failed, error: {result['error']['message']}, retrying...")
            time.sleep(2 * (attempt + 1))
            
        except Exception as e:
            print(f"LLM:Attempt {attempt + 1} error: {e}, retrying...")
            time.sleep(2 * (attempt + 1))
    
    # Если все попытки провалились, возвращаем сообщение об ошибке
    return "Извините, не удалось получить ответ от модели. Попробуйте позже."
