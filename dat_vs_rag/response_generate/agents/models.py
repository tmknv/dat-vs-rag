import requests
import json
import time


def Gemma_3_4B(query: str, max_retries: int = 5) -> str:
    """
    Отправляет запрос к Gemma 3 4B с автоматическими повторными попытками.
    """
    
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


print(Gemma_3_4B("Hello! How many days in year?"))



def check_credits():
    api_key = "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"
    response = requests.get(
        "https://openrouter.ai/api/v1/auth/key",
        headers={"Authorization": f"Bearer {api_key}"}
    )
    print(response.json())

# check_credits()