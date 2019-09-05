import sqlite3
from contextlib import closing
import pandas as pd


class Database(object):
    SQL_FRIEND_TABLE = """
    CREATE TABLE IF NOT EXISTS twitter_friend (
    id TEXT PRIMARY KEY,
    friends TEXT
);
"""

    def __init__(self, db_file):
        self.db_file = db_file
        self.init_db()

    def connect_db(self):
        return sqlite3.connect(self.db_file)

    def create_table(self, create_table_sql):
        with closing(self.connect_db()) as db:
            db.cursor().execute(create_table_sql)
            db.commit()

    def init_db(self):
        print("initialising .....")

        self.create_table(self.SQL_FRIEND_TABLE)

    def select_one(self, sql, params=None):
        with closing(self.connect_db()) as db:
            c = db.cursor()
            if params:
                c.execute(sql, params)
            else:
                c.execute(sql)
            return c.fetchone()

    def select_many(self, sql, params=None):
        with closing(self.connect_db()) as db:
            c = db.cursor()
            if params:
                c.execute(sql, params)
            else:
                c.execute(sql)
            return c.fetchall()

    def select_to_dataframe(self, sql, params=None):
        with closing(self.connect_db()) as db:
            if params:
                return pd.read_sql_query(sql, db, params=params)
            else:
                return pd.read_sql_query(sql, db)

    def insert_one(self, sql, params):
        self.execute(sql, params)

    def execute(self, sql, params):
        with closing(self.connect_db()) as db:
            db.cursor().execute(sql, params)
            db.commit()

    def insert_many(self, sql, params):
        with closing(self.connect_db()) as db:
            db.cursor().executemany(sql, params)
            db.commit()
