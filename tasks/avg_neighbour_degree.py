import csv
import json

import pandas as pd

from util.paths import get_output_file_path, get_input_file_path


class AvgNeighbourDegree(object):
    def __init__(self, adj_list_file):
        self.adj_list_file = adj_list_file
        self.adj_list = self.build_adj_list()

    def build_adj_list(self):
        print("building adj_list_dict...")
        adj_list_dict = {}
        with open(get_input_file_path(self.adj_list_file), 'r') as tsvfile:
            reader = csv.reader(tsvfile, delimiter='\t')
            for row in reader:
                source = row[0]
                targets = json.loads(row[1])
                adj_list_dict[source] = targets
        return adj_list_dict

    def generate(self, user_file, out_file):
        avg_file = self._compute_adv_degree()
        self._merge(user_file=user_file,
                    avg_file=avg_file,
                    out_file=out_file)

    def _compute_adv_degree(self):
        print("starting computation for avg neighbours degree")
        avg_file = f"avg_neighbour_degree_{self.adj_list_file}"
        with open(get_output_file_path(avg_file), 'w') as out:
            with open(get_input_file_path(self.adj_list_file), 'r') as tsvfile:
                reader = csv.reader(tsvfile, delimiter='\t')
                for row in reader:
                    source = row[0]
                    targets = json.loads(row[1])
                    neighbour_degree = 0
                    neighbour_count = 0
                    for t in targets:
                        if t in self.adj_list:
                            neighbour_degree += len(self.adj_list[t])
                            neighbour_count += 1

                    #neighbour_degree = neighbour_degree / len(targets)
                    if neighbour_count == 0:
                        neighbour_count = 1

                    avg_neighbour_degree = neighbour_degree / neighbour_count
                    out.write(f"{source}\t{avg_neighbour_degree}\n")

        return avg_file

    @staticmethod
    def _merge(user_file, avg_file, out_file):
        user_df = pd.read_csv(get_input_file_path(user_file))
        nd_df = pd.read_csv(get_output_file_path(avg_file),
                            sep='\t',
                            names=["user_id", "avg_neighbour_degree"])

        df = user_df.join(nd_df.set_index('user_id'), on='user_id')
        print(f"saving dataframe as {out_file}")
        df.to_csv(get_output_file_path(out_file), index=False)
        df_filter = df.drop_duplicates(subset='avg_neighbour_degree', keep="last")
        df_filter = df_filter.sort_values(by=['avg_neighbour_degree'], ascending=False)
        df_filter.to_csv(get_output_file_path(f"filtered_{out_file}"), index=False)

    def size(self):
        return len(self.adj_list)


avg_neighbours_degree = AvgNeighbourDegree("stanford_adj_list.tsv")
avg_neighbours_degree.generate("70013users.csv", "70013users.csv")