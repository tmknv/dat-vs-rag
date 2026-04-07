from aiogram.types import ReplyKeyboardMarkup, KeyboardButton

RAG_retriever_type_keyboard = ReplyKeyboardMarkup(keyboard=[
    [KeyboardButton(text='hybrid', )],
    [KeyboardButton(text='lexical')],
    [KeyboardButton(text='semantic')]
],
        resize_keyboard=True)