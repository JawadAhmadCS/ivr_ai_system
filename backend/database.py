
from sqlalchemy import create_engine, text
from sqlalchemy.orm import sessionmaker, declarative_base
from config import DB_NAME, ROOT_DATABASE_URL, DATABASE_URL

root_engine = create_engine(
    ROOT_DATABASE_URL,
    isolation_level="AUTOCOMMIT"
)

with root_engine.connect() as conn:
    conn.execute(text(f"CREATE DATABASE IF NOT EXISTS {DB_NAME}"))

engine = create_engine(DATABASE_URL, echo=False)
SessionLocal = sessionmaker(bind=engine)
Base = declarative_base()


def _column_exists(conn, table_name: str, column_name: str) -> bool:
    q = text(
        "SELECT COUNT(*) FROM information_schema.columns "
        "WHERE table_schema = :schema AND table_name = :table AND column_name = :col"
    )
    return bool(conn.execute(q, {"schema": DB_NAME, "table": table_name, "col": column_name}).scalar())


def ensure_schema():
    with engine.connect() as conn:
        try:
            if not _column_exists(conn, "users", "restaurant_id"):
                conn.execute(text("ALTER TABLE users ADD COLUMN restaurant_id INT NULL"))
        except Exception:
            pass
        try:
            if not _column_exists(conn, "call_logs", "restaurant_id"):
                conn.execute(text("ALTER TABLE call_logs ADD COLUMN restaurant_id INT NULL"))
        except Exception:
            pass
