import psycopg2
from psycopg2 import OperationalError

conn_params = {
    "host": "localhost",      # или IP вашего сервера
    "port": "5432",           # порт PostgreSQL по умолчанию
    "database": "postgres",
    "user": "postgres",
    "password": "postgres_pas"
}

conn = None

def connect_DB():
    global conn
    conn = psycopg2.connect(**conn_params)
    