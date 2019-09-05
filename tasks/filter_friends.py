from db.database import Database
from db.data_access import DataAccess
import pandas as pd
from util.paths import get_output_file_path, get_input_file_path
import json


def generate():
    _db = Database("app.db")
    data_access = DataAccess(db=_db)
    data_access.generate_dataset("all_friends.csv")
    friends_list = pd.read_csv(get_output_file_path("all_friends.csv"))
    users = pd.read_csv(get_input_file_path("all_infected.csv"))
    user_ids = set(users['id'])
    friends_list['friends'] = friends_list.apply(lambda x: list(set(json.loads(x['friends'])) & user_ids), axis=1)
    friends_list.to_csv(get_output_file_path("filtered_friends.csv"), index=False, header=False, sep="\t")


if __name__ == '__main__':
    generate()