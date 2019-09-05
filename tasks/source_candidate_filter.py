import pickle

import pandas as pd

from util.paths import get_output_file_path


class SrcCandidateFilter(object):

    def __init__(self, users_file):
        self.users_file = users_file

    @staticmethod
    def load_pickle_file(pickled_file):
        print(f'Loading data file from {pickled_file}')
        infile = open(pickled_file, 'rb')
        unpickled_file = pickle.load(infile)
        print(f'Loaded {len(unpickled_file)} entries')
        infile.close()
        return unpickled_file

    @staticmethod
    def save_pickle_file(path, data):
        print('Dumping data to path {}'.format(path))
        with open(path, 'wb') as file:
            pickle.dump(data, file)
        print('Finished dumping data to path {}'.format(path))

    def filter_indices(self, candidate_idx_list, idx_out_of_range_set):
        if len(candidate_idx_list) > 0:
            return list(set(candidate_idx_list).difference(idx_out_of_range_set))
        return list()

    def index_to_id(self, df, idx):
        if idx:
            return int(df.loc[int(idx), 'id'])
        return None

    def prepare(self, followers_file):
        users_df = self.load_pickle_file(self.users_file)
        index_list = users_df[users_df['time_lapsed'] > 540].index.tolist()

        idx_out_of_range_set = set(index_list)
        users_df["source_candidates"] = users_df["source_candidates"].map(
            lambda x: self.filter_indices(x, idx_out_of_range_set))
        users_df = users_df[users_df['time_lapsed'] <= 540]
        followers = pd.read_csv(get_output_file_path(followers_file),
                                sep='\t',
                                header=None,
                                names=["id", "followers_list"])

        users_df = users_df.join(followers.set_index('id'),
                                 on='id',
                                 how="left")

        self.save_pickle_file(get_output_file_path("nyc_users_6_9_infected.dat"), users_df)
        print(users_df.tail(1)["id"])

    def merge(self, ext_followers_df):
        pass


#m = SrcCandidateFilter("/Users/syamantak/JayateeB/new_files/data/nyc/nyc_users.dat")
#m.prepare("nyc_users_6_9_followers.tsv")
