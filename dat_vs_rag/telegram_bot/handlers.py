from aiogram import F, Router
from aiogram.filters import CommandStart, Command
from aiogram.fsm.state import State, StatesGroup
from aiogram.fsm.context import FSMContext
from aiogram.types import Message
from aiogram.types import ReplyKeyboardRemove

# from dat_vs_rag.SQL_DB.users import add_user
from dat_vs_rag.response_generate.generation import get_responses 
import dat_vs_rag.telegram_bot.keyboards as kb

from dat_vs_rag.utils.load_params import get_params

PARAMS = get_params()

router = Router()

class Form_response_generate(StatesGroup):
    query = State()
    RAG_retriever_type = State()
    hybrid_RAG_alpha = State()

@router.message(CommandStart())
async def command_start(msg: Message, state: FSMContext):
    """
    ответ на команду /start
    """
    
    await state.set_state(Form_response_generate.query)
  
    # add_user(msg.from_user.id, msg.from_user.username)

    await msg.answer("Hello!")



@router.message(Form_response_generate.query, F.text)
async def get_query(msg: Message, state: FSMContext):

    """
    реакция на запрос, запускает цепочку генерации: спрашивает про тип посика RAG
    """

    await state.update_data(query=msg.text)
    await state.set_state(Form_response_generate.RAG_retriever_type)

    await msg.answer(
        "Choose RAG retriever type", 
        reply_markup=kb.RAG_retriever_type_keyboard
    )


@router.message(Form_response_generate.RAG_retriever_type, F.text)
async def get_RAG_retriever_type(msg: Message, state: FSMContext):

    """
    реакция на выбор типа поиска RAG, спрашивает про альфа коэффициент, если был выбран гибридный поиск
    """
    
    #если типа не существует
    if msg.text not in PARAMS["generation"]["RAG_retriever_types"]:
        await msg.answer("Invalid type, try again")
        return

    #если тип гибридный, то нужно узнать альфа параметр
    if(msg.text == "hybrid"):
        await state.update_data(RAG_retriever_type=msg.text)
        await state.set_state(Form_response_generate.hybrid_RAG_alpha)
        await msg.answer("Enter hybrid RAG alpha coefficient", reply_markup=ReplyKeyboardRemove())

    else:
        wait_message = await msg.answer("Generation is in progress, please wait", reply_markup=ReplyKeyboardRemove())
        data = await state.get_data()

        response = get_responses(query=data["query"], RAG_retriever_type=msg.text, alpha_coefficient=-1)
        await wait_message.delete()
        await msg.answer(response)

        await state.set_state(Form_response_generate.query)


@router.message(Form_response_generate.hybrid_RAG_alpha, F.text)
async def get_hybrid_RAG_alpha(msg: Message, state: FSMContext):

    """
    реакция на сообщение со значением альфа коэффициента
    """

    try:
        alpha = float(msg.text.replace(',', '.'))

        #альфа должен быть от 0 до 1
        if alpha<0 or alpha>1:
            await msg.answer("Alpha lies in the range from 0 to 1")
            return
        
        wait_message = await msg.answer("Generation is in progress, please wait", reply_markup=ReplyKeyboardRemove())

        data = await state.get_data()
        response = get_responses(query=data["query"], RAG_retriever_type=data["RAG_retriever_type"], alpha_coefficient=alpha)

        await wait_message.delete()
        await msg.answer(response)

        await state.set_state(Form_response_generate.query)

    except ValueError:
        await msg.answer("Its not a number")
        return