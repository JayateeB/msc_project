import pandas as pd
from util.paths import get_input_file_path
import queue
from concurrent.futures import ThreadPoolExecutor


class FriendsCrawler(object):
    def __init__(self, crawl, data_access):
        self.crawl = crawl
        self.queue = queue.Queue()
        self.data_access = data_access
        self.start_workers()

    def produce_work(self, in_file):
        ds = pd.read_csv(get_input_file_path(in_file))
        for index, row in ds.iterrows():
            user_id = str(row.id)
            if self.data_access.is_user_added(user_id):
                print(f"User {user_id} already processed")
            else:
                print(f"Queuing user : {user_id}")
                self.queue.put(user_id)

    def start_workers(self):
        num_apis = self.crawl.get_number_of_apis()
        api_range = range(num_apis)
        with ThreadPoolExecutor(max_workers=num_apis) as executor:
            for i in api_range:
                api = self.crawl.get_api_pool()[i]
                worker = Worker(api=api,
                                data_access=self.data_access,
                                work_queue=self.queue)
                executor.submit(worker.work)


class Worker(object):

    def __init__(self, api, data_access, work_queue):
        self.api = api
        self.data_access = data_access
        self.work_queue = work_queue

    def work(self):
        while True:
            user_id = self.work_queue.get()
            if user_id is None:
                break
            print(f"Processing : {user_id}")
            #self.do_work(user_id)
            #self.work_queue.task_done()

    def do_work(self, user_id):
        friend_ids = self.api.crawl_friends(user_id)
        self.data_access.add_user_friends(user_id, friend_ids)
