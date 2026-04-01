import psycopg2
from psycopg2 import OperationalError
from dotenv import load_dotenv
import json
import os

load_dotenv()

conn_params = json.loads(os.getenv("SQL_CONNETCT_PARAMS"))

conn = None

def connect_DB():
    global conn
    conn = psycopg2.connect(**conn_params)
    