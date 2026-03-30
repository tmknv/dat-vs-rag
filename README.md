telegram_bot - каталог реализации телеграм бота на aiogram
    run-bot.py - зпускает бота, psql db, chroma db
    handlers.py - обработчики команд и сообщений пользователя

SQL_DB - каталог запуска psql db и запросов к ней
    connect.py - конект с бд
    user.py - команды к таблице users

response_generate - каталог генерации ответа пользователю
    agents - каталог с агентами RAG LLM и DAT SLM
        DAT - реализация алгоритма DAT
        DAT_SLM - реализаця генерации SLM с DAT
        RAG - реализация алгоритма RAG
        RAG_LLM - реализвация LLM с RAG
        models - файл с языковыми моделями

    generation.py - финальное составление сообщения пользователю
    qr_processing.py - нормализация запроса пользователя

chroma_db - каталог для создания и работы с chroma db
    BM25.py - обучение и запросы к модели Bbm25
    create_chunks.py - генерация чанков датасета, их эмбедингов и разреженых векторов
    init_chroma_db.py - инициализации и заполнение chroma db
    ModernBert.py - работа с моделью ModernBert