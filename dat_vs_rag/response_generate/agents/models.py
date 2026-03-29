'''
Файл с языковыми моделями
'''

import requests
import json
import time


def Gemma_3_4B(query: str, max_retries: int = 3) -> str:
  
    '''Запрос к Gemma_3_4B

    Аргументы:
        Запрос пользователя, максимальное колличество повторения запроса при ошибке
    
    Возвращает:
        Ответ модели
    '''
    
    api_key = "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"
    
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
        "HTTP-Referer": "https://t.me/your_bot",
        "X-Title": "DAT-vs-RAG Bot",
    }
    
    payload = {
        "model": "google/gemma-3-4b-it:free",
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
            if 'choices' in result and result['choices']:
                return result['choices'][0]['message']['content']
            
            # Если ошибка, пробуем снова
            print(f"Attempt {attempt + 1} failed, retrying...")
            time.sleep(2 * (attempt + 1))
            
        except Exception as e:
            print(f"Attempt {attempt + 1} error: {e}, retrying...")
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
    
    api_key = "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"
    
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
        "HTTP-Referer": "https://t.me/your_bot",
        "X-Title": "DAT-vs-RAG Bot",
    }
    
    payload = {
        "model": "google/gemma-3-27b-it:free",
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
            if 'choices' in result and result['choices']:
                return result['choices'][0]['message']['content']
            
            # Если ошибка, пробуем снова
            print(f"Attempt {attempt + 1} failed, retrying...")
            time.sleep(2 * (attempt + 1))
            
        except Exception as e:
            print(f"Attempt {attempt + 1} error: {e}, retrying...")
            time.sleep(2 * (attempt + 1))
    
    # Если все попытки провалились, возвращаем сообщение об ошибке
    return "Извините, не удалось получить ответ от модели. Попробуйте позже."
