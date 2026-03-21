import asyncio
from aiogram import Bot, Dispatcher

from config import TOKEN
from handlers import router

bot = Bot(token=TOKEN)
dp = Dispatcher()

async def main():
    dp.include_router(router)
   
    print("bot started!")
    await dp.start_polling(bot)

if __name__ == '__main__':
    asyncio.run(main())