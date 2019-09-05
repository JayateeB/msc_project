import pandas as pd
import numpy as np
import json
from util.paths import get_output_file_path, get_input_file_path


class SimGraphDataGenerator:
    """
    A class to generate data needed to create the network simulation graph with time lapse
    """

    @staticmethod
    def __find(df, arr):
        if isinstance(arr, str):
            candidate_list = json.loads(arr)
            if len(candidate_list) > 0:
                idx = candidate_list[-1] # last element of the list is the index of the infection source
                return df.iloc[idx]['id']
        return np.nan

    def generate(self, in_file, out_file, start_time):
        sim_data = pd.read_csv(get_input_file_path(in_file))
        sim_data['infection_source'] = sim_data['source_candidates'].map(lambda x: self.__find(sim_data, x))
        out_columns = ['id', 'time_lapsed', 'infection_source']
        temp_df = sim_data[out_columns]
        sorted_df = temp_df.sort_values(by=['time_lapsed'])

        initial_nodes = set()
        initial_links = list()

        dynamic_nodes = set()
        dynamic_links = list()
        for _, row in sorted_df.iterrows():
            if row['time_lapsed'] <= start_time:
                initial_nodes.add(row['id'])
                if not np.isnan(row['infection_source']):
                    initial_nodes.add(row['infection_source'])
                    if not np.isnan(row['time_lapsed']):
                        initial_links.append({"source": row['infection_source'], "target": row['id']})

            else:
                dynamic_nodes.add(row['id'])
                if not np.isnan(row['infection_source']):
                    dynamic_nodes.add(row['infection_source'])
                    if not np.isnan(row['time_lapsed']):
                        dynamic_links.append({"source": row['infection_source'],
                                              "target": row['id'],
                                              "timeLapsed": row['time_lapsed']})

        data = {
            "initialData": {
                "nodes": list(map(lambda x: {"id": x, "group": 1}, initial_nodes)),
                "links": initial_links
            },
            "dynamicData": {
                "nodes": list(map(lambda x: {"id": x}, dynamic_nodes)),
                "links": dynamic_links
            }
        }

        with open(get_output_file_path(out_file), 'w') as fp:
            json.dump(data, fp)


#generator = SimGraphDataGenerator()
#generator.generate("givenchy_simulation_result_6hrs_6_hrs_model_retrained.csv", "6hrs_sim_graph_retrained.json", 360.0)