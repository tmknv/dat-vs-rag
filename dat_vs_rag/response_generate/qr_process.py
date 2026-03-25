from langdetect import detect, DetectorFactory
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
from pathlib import Path
import nltk
import re
import yaml

#from mawo_pymorphy3 import create_analyzer
#import emoji
#_morph = create_analyzer()

#nltk.download('wordnet')
#nltk.download('stopwords')
#nltk.download('omw-1.4')

def load_russian_stop_words(yaml_path='dat-vs-rag/dat_vs_rag/response_generate/ru_stop_voc.yaml'):
    try:
        with open(yaml_path, 'r', encoding='utf-8') as f:
            data = yaml.safe_load(f)
        return set(data.get('stop_words', []))
    except FileNotFoundError:
        print(f"Файл {yaml_path} не найден, используются пустые стоп-слова")
        return set()

RUSSIAN_STOP_WORDS = load_russian_stop_words()

DetectorFactory.seed = 0
_lemmatizer = WordNetLemmatizer()
ENGLISH_STOP_WORDS = set(stopwords.words('english'))

lang = detect


# можете тут использовать на фаст тексте ml model для определения языка. Могу скинуть если не найдете
def detect_language(query: str):
    sample = query[:100].strip()
    if not sample:
        return 'unknown'
    lang = detect(sample)
    return lang if lang in ['ru', 'en'] else 'unknown'

def lemmatize_english(query: str):
    words = query.split()
    lemmatized = [_lemmatizer.lemmatize(w.lower()) for w in words]
    return ' '.join(lemmatized)

def remove_english_stopwords(query: str):
    words = query.split()
    important = {'not', 'no', 'none', 'never', 'without'}
    words = [w for w in words if w not in ENGLISH_STOP_WORDS or w in important]
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
    if lang == 'en':
        query = lemmatize_english(query)

    # удаляем стоп-слова, если слов больше 3
    words = query.split()
    if len(words) > 3:
        if lang == 'en':
            query = remove_english_stopwords(query)
        else:
            important = {'не'}
            words = [w for w in words if w not in RUSSIAN_STOP_WORDS or w in important]
            query = ' '.join(words)
    
    return query.strip()