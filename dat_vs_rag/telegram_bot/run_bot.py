import asyncio
from aiogram import Bot, Dispatcher

from config import TOKEN
from handlers import router
from dat_vs_rag.SQL_DB.connect import connect_DB

bot = Bot(token=TOKEN)
dp = Dispatcher()

async def main():
    try:
        connect_DB()
        print("Database connected!")
    except:
        print("Database connection error!")
        return

    dp.include_router(router)
   
    print("bot started!")
    await dp.start_polling(bot)

if __name__ == '__main__':
    asyncio.run(main())