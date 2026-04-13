import asyncio
from aiogram import Bot, Dispatcher

from handlers import router
# from dat_vs_rag.SQL_DB.connect import connect_DB
from dat_vs_rag.chroma_db.init_chroma_db import init_chroma_db

import os
from dotenv import load_dotenv
load_dotenv()

bot = Bot(token=os.getenv("TELEGRAM_BOT_TOKEN"))
dp = Dispatcher()

async def main():
    # try:
    #     connect_DB()
    #     print("Database connected!")
    # except Exception as e:
    #     print(f"Database connection error: {e}")
    #     return
    
    try:
        init_chroma_db()
        print("Chroma db is ready!")
    except Exception as e:
        print(f"Chroma init error: {e}") 

    dp.include_router(router)
   
    print("bot started!")
    await dp.start_polling(bot)

if __name__ == '__main__':
    asyncio.run(main())
