import json
from datetime import datetime

import pandas as pd
from tqdm import tqdm

from util.paths import get_output_file_path

columns = ["id", "created_at", "favourites_count", "followers_count", "friends_count", "listed_count", "statuses_count",
           "user_created_days", "normalized_statuses_count", "normalized_followers_count", "normalized_favourites_count",
           "normalized_listed_count", "normalized_friends_count"]


def create_filter_set(filter_file):

    filter_df = pd.read_csv(get_output_file_path(filter_file),
                            sep='\t',
                            names=["id", "followers"])

    id_set = set()
    for _, row in filter_df.iterrows():
        followers = set(map(lambda x: int(x), json.loads(row["followers"])))
        id_set.update(followers)
    return id_set


def json_to_csv(in_file, filter_file, out_file, event_day):

    filter_ids = create_filter_set(filter_file)

    with open(in_file, "r") as json_in:
        with open(get_output_file_path(out_file), "w") as csv_out:
            header = "\t".join(columns)
            csv_out.write(f"{header}\n")

            for line in tqdm(json_in, total=4417245):
                row = {}
                user = json.loads(line)
                if int(user["id"]) in filter_ids:

                    created_at = datetime.strptime(user["created_at"], "%a %b %d %H:%M:%S %z %Y")
                    ucd = event_day - created_at
                    user_created_days = ucd.days if ucd.days > 0 else 1
                    row["id"] = user["id"]
                    row["created_at"] = created_at.strftime("%Y-%m-%d %H:%M:%S")
                    row["favourites_count"] = user["favourites_count"]
                    row["followers_count"] = user["followers_count"]
                    row["friends_count"] = user["friends_count"]
                    row["listed_count"] = user["listed_count"]
                    row["statuses_count"] = user["statuses_count"]
                    row["user_created_days"] = user_created_days
                    row['normalized_statuses_count'] = row['statuses_count'] / row['user_created_days']
                    row['normalized_followers_count'] = row['followers_count'] / row['user_created_days']
                    row['normalized_favourites_count'] = row['favourites_count'] / row['user_created_days']
                    row['normalized_listed_count'] = row['listed_count'] / row['user_created_days']
                    row['normalized_friends_count'] = row['friends_count'] / row['user_created_days']

                    row_collector = []
                    for col in columns:
                        row_collector.append(str(row[col]))
                    row = "\t".join(row_collector)
                    csv_out.write(f"{row}\n")


def merge(ext_followers, source_candidates, out_file):
    print(f"Loading dataframe {ext_followers}")
    ext_followers_df = pd.read_csv(get_output_file_path(ext_followers), sep='\t')
    print(f"Loading dataframe {source_candidates}")
    source_candidates_df = pd.read_csv(get_output_file_path(source_candidates),
                                       sep='\t',
                                       names=["id", "source_candidates"])

    df = ext_followers_df.join(source_candidates_df.set_index('id'), on='id')
    df["followers_list"] = None
    print(f"saving dataframe as {out_file}")
    df.to_csv(get_output_file_path(out_file), index=False)


event_day = datetime.strptime('2018-12-11 12:00:00 +0000', "%Y-%m-%d %H:%M:%S %z")
json_to_csv(in_file="/Users/syamantak/JayateeB/new_files/data/users_for_jay.txt",
            filter_file="nyc_users_6_followers.tsv",
            out_file="nyc_ext_followers.tsv",
            event_day=event_day)

merge("nyc_ext_followers.tsv", "nyc_users_source_candidates.tsv", "nyc_6_7_ext_followers.csv")
