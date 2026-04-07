from aiogram.types import ReplyKeyboardMarkup, KeyboardButton

RAG_retriever_type_keyboard = ReplyKeyboardMarkup(keyboard=[
    [KeyboardButton(text='Hybrid', )],
    [KeyboardButton(text='Lexical')],
    [KeyboardButton(text='Semantic')]
],
        resize_keyboard=True)