from langdetect import detect, DetectorFactory
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
from pathlib import Path
import nltk
import re

from mawo_pymorphy3 import create_analyzer
_russian_morph = create_analyzer()


"""следующие 3 строчки при первом запуске раскомментить"""
#nltk.download('wordnet')
#nltk.download('stopwords')
#nltk.download('omw-1.4')

IMPORTANT_RU = {'не', 'нет'}
IMPORTANT_EN = {'not', 'no', 'none', 'never', 'without'}


ENGLISH_STOP_WORDS = set(stopwords.words('english'))
RUSSIAN_STOP_WORDS = set(stopwords.words('russian'))


DetectorFactory.seed = 0
_english_lemmatizer = WordNetLemmatizer()

lang = detect


# можете тут использовать на фаст тексте ml model для определения языка. Могу скинуть если не найдете
def detect_language(query: str):
    sample = query[:100].strip()
    if not sample:
        return 'unknown'
    lang = detect(sample)
    return lang if lang in ['ru', 'en'] else 'unknown'

def lemmatize_russian(text: str) -> str:
    # Извлекаем слова только из кириллицы (включая ё)
    words = re.findall(r'\b[а-яё]+\b', text)
    
    lemmatized = []
    for word in words:
        try:
            parse = _russian_morph.parse(word)[0]
            lemma = parse.normal_form
            lemmatized.append(lemma)
        except Exception:
            lemmatized.append(word)
    
    return ' '.join(lemmatized)

def lemmatize_english(text: str):
    words = text.split()
    
    lemmatized = []
    for word in words:
        if word: 
            lemma = _english_lemmatizer.lemmatize(word)
            lemmatized.append(lemma)
    
    return ' '.join(lemmatized)

def remove_stopwords(text: str, lang_code: str):
    words = text.split()
    
    if lang_code == 'en':
        words = [w for w in words if w not in ENGLISH_STOP_WORDS or w in IMPORTANT_EN]
    else:  
        words = [w for w in words if w not in RUSSIAN_STOP_WORDS or w in IMPORTANT_RU]
    
    return ' '.join(words)

def query_process(query: str):
    # очистка
    query = re.sub(r'@\w+', '', query) #собака
    query = re.sub(r'/\w+', '', query) # /
    query = re.sub(r'[^\w\s\+\-#]', '', query) #знаки препинания
    query = re.sub(r'(.)\1{2,}', r'\1', query) # повторные !!!! и тп
    
    # нижний регистр
    query = query.lower()
    
    lang = detect_language(query)

    # лемматизация
    if lang == 'ru':
        query = lemmatize_russian(query)
    elif lang == 'en':
        query = lemmatize_english(query)

    # удаляем стоп-слова, если слов больше 3
    words = query.split()
    if len(words) > 2:
        if lang in ['ru', 'en']:
            query = remove_stopwords(query, lang)
    
    return query.strip()