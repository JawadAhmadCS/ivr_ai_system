
import os
from dotenv import load_dotenv
from sqlalchemy.engine import URL
load_dotenv()

DB_NAME = os.getenv("DB_NAME", "ivr_ai")
DB_USER = os.getenv("DB_USER", "root")
DB_PASS = os.getenv("DB_PASS", "")
DB_HOST = os.getenv("DB_HOST", "localhost")
DB_PORT = os.getenv("DB_PORT", "3306")

DB_PORT_INT = int(DB_PORT)

ROOT_DATABASE_URL = URL.create(
    "mysql+pymysql",
    username=DB_USER,
    password=DB_PASS,
    host=DB_HOST,
    port=DB_PORT_INT,
)

DATABASE_URL = URL.create(
    "mysql+pymysql",
    username=DB_USER,
    password=DB_PASS,
    host=DB_HOST,
    port=DB_PORT_INT,
    database=DB_NAME,
)
