'''
Файл с языковыми моделями
'''

import requests
import json
import time
from dotenv import load_dotenv
import os

load_dotenv()

api_key = os.getenv("OPENROUTER_API_KEY")


def Gemma_3_4B(query: str, max_retries: int = 3) -> str:
  
    '''Запрос к Gemma_3_4B

    Аргументы:
        Запрос пользователя, максимальное колличество повторения запроса при ошибке
    
    Возвращает:
        Ответ модели
    '''
        
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
        "HTTP-Referer": "https://t.me/your_bot",
        "X-Title": "DAT-vs-RAG Bot",
    }
    
    payload = {
        "model": "mistralai/ministral-3b-2512",
        "messages": [{"role": "user", "content": query}]
    }
    
    for attempt in range(max_retries):
        try:
            response = requests.post(
                url="https://openrouter.ai/api/v1/chat/completions",
                headers=headers,
                data=json.dumps(payload),
                timeout=30
            )
            
            result = response.json()

            # Если есть 'choices', возвращаем ответ
            if 'error' not in result and result:
                return result['choices'][0]['message']['content']
            
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
        
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
        "HTTP-Referer": "https://t.me/your_bot",
        "X-Title": "DAT-vs-RAG Bot",
    }
    
    payload = {
        "model": "mistralai/ministral-14b-2512",
        "messages": [{"role": "user", "content": query}]
    }
    
    for attempt in range(max_retries):
        try:
            response = requests.post(
                url="https://openrouter.ai/api/v1/chat/completions",
                headers=headers,
                data=json.dumps(payload),
                timeout=30
            )
            
            result = response.json()
            
            # Если есть 'choices', возвращаем ответ
            if 'error' not in result and result:
                return result['choices'][0]['message']['content']
            
            # Если ошибка, пробуем снова
            print(f"LLM: Attempt {attempt + 1} failed, error: {result['error']['message']}, retrying...")
            time.sleep(2 * (attempt + 1))
            
        except Exception as e:
            print(f"LLM:Attempt {attempt + 1} error: {e}, retrying...")
            time.sleep(2 * (attempt + 1))
    
    # Если все попытки провалились, возвращаем сообщение об ошибке
    return "Извините, не удалось получить ответ от модели. Попробуйте позже."