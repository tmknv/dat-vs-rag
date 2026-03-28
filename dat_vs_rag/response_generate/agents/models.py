import requests
import json


def Arcee_Mini(query: str) ->str:
    """
    Отправляет запрос к Arcee_Mini и возвращает ответ
    """
    
    # Получаем API ключ из переменной окружения
    api_key = "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"
    

    # Формируем запрос
    payload = {
        "model": "arcee-ai/trinity-mini:free",  # бесплатная модель
        "messages": [
            {"role": "user", "content": query}
        ]
    }
    
    try:
        # Отправляем запрос
        response = requests.post(
            url="https://openrouter.ai/api/v1/chat/completions",
            headers={
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json",
            },
            data=json.dumps(payload)
        )
        
        # Получаем ответ
        result = response.json()
        answer = result['choices'][0]['message']['content']
        
        return answer
        
    except Exception as e:
        return f"Ошибка: {e}"


print(Arcee_Mini("Hello! How many days in year?"))