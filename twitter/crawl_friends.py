import pandas as pd
from util.paths import get_input_file_path
import roundrobin


class FriendsCrawler(object):
    def __init__(self, crawl, data_access):
        self._crawl = crawl
        self.round_robin = roundrobin.basic(range(crawl.get_number_of_apis()))
        self._data_access = data_access

    def extract_friends(self, in_file):
        ds = pd.read_csv(get_input_file_path(in_file))
        for index, row in ds.iterrows():
            user_id = str(row.id)
            if self._data_access.is_user_added(user_id):
                print(f"User {user_id} already processed")
            else:
                print(f"Processing : {user_id}")
                api = self.get_next_api()
                friend_ids = api.crawl_friends(user_id)
                self._data_access.add_user_friends(user_id, friend_ids)

    def get_next_api(self):
        return self._crawl.get_api_pool()[self.round_robin()]
