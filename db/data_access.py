import json
from util.paths import ensure_output_path, get_output_file_path


class DataAccess(object):
    SQL_INSERT_USER = "INSERT INTO twitter_friend (id, friends) VALUES (?, ?)"
    SQL_ALL_USER = "SELECT * FROM twitter_friend"
    SQL_FIND_USER = "SELECT id FROM twitter_friend WHERE id = ?"

    def __init__(self, db):
        self.db = db

    def is_user_added(self, user_id):
        user = self.db.select_one(self.SQL_FIND_USER, (user_id,))
        return user is not None

    def add_user_friends(self, user_id, friend_ids):
        self.db.insert_one(self.SQL_INSERT_USER, (user_id, json.dumps(friend_ids)))

    def generate_dataset(self, out_file):
        df = self.db.select_to_dataframe(self.SQL_ALL_USER)
        ensure_output_path()
        df.to_csv(get_output_file_path(out_file), index=False)
