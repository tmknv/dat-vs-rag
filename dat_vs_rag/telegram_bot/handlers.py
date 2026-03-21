from aiogram import F, Router
from aiogram.filters import CommandStart, Command
from aiogram.types import Message

from dat_vs_rag.SQL_DB.users import add_user
from dat_vs_rag.response_generate.generation import get_responses 

router = Router()

@router.message(CommandStart())
async def command_start(msg: Message):
    add_user(msg.from_user.id, msg.from_user.username)
    await msg.answer("Hello!")


@router.message(F.text)
async def generate_answer(msg: Message):
    await msg.answer(get_responses(msg.text))