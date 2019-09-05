import pandas as pd
import pickle
import numpy as np
import multiprocessing
from concurrent.futures.thread import ThreadPoolExecutor

from sklearn.externals import joblib
import xgboost as xgb
import time


def load_pickle_file(pickled_file):
    print(f'Loading data file from {pickled_file}')
    infile = open(pickled_file, 'rb')
    unpickled_file = pickle.load(infile)
    print(f'Loaded {len(unpickled_file)} entries')
    infile.close()
    return unpickled_file


def save_pickle_file(file_path, data):
    print('Dumping data to path {}'.format(file_path))
    with open(file_path, 'wb') as file:
        pickle.dump(data, file)
    print('Finished dumping data to path {}'.format(file_path))


def mean(numbers):
    return float(sum(numbers)) / max(len(numbers), 1)


def safe_division(x, y):
    if y == 0:
        return 0
    else:
        return x/y


def seed_initialization(init_features, init_dataset, init_seeds):
    seed_user_ids = list(init_seeds['user_id'])
    for seed_id in seed_user_ids:
        init_features.loc[init_features['user_id'] == seed_id, 'infected_status'] = True
        init_features.loc[init_features['user_id'] == seed_id, 'infection_time'] = 0
        init_features.loc[init_features['user_id'] == seed_id, 'infected_status'] = True
        index = init_features.index[init_features['user_id'] == seed_id]
        init_features.loc[init_features['user_id'] == seed_id, 'seed_index'] = index
        init_features.loc[init_features['user_id'] == seed_id, 'generation'] = 0
        init_features.loc[init_features['user_id'] == seed_id, 'time_lapsed'] = 0

        index = init_dataset.index[init_dataset['id'] == seed_id]
        init_dataset.loc[init_dataset['id'] == seed_id, 'seed_index'] = index
        init_dataset.loc[init_dataset['id'] == seed_id, 'generation'] = 0
        init_dataset.loc[init_dataset['id'] == seed_id, 'time_lapsed'] = 0
        init_dataset.loc[init_dataset['id'] == seed_id, 'time_since_seed'] = 0


def generate_features(degree, in_degree, out_degree, target_id, sim_features, net_sim, curr_time):

    if isinstance(net_sim.loc[target_id, 'exposed_source_candidates'], list):

        source_candidates = sorted(net_sim.loc[target_id, 'exposed_source_candidates'])
        sources = source_candidates

        first_source_index = source_candidates[0]
        first_source_row = net_sim.loc[first_source_index]
        first_source_seed_row = net_sim.loc[first_source_row['seed_index']]

        sources_df = net_sim.loc[sources]
        degree_list = list(net_sim.loc[i, 'followers_count'] + net_sim.loc[i, 'friends_count'] for i in sources)
        time_list = [curr_time - net_sim.loc[x, 'time_lapsed'] for x in sources]

        last_source_index = sources[-1]
        try:
            last_source_row = net_sim.loc[last_source_index]
            last_source_seed_row = net_sim.loc[last_source_row['seed_index']]
        except:
            print(f"target_index:{target_id}")
            print(f"last_source_index:{last_source_index}")

        user_row = net_sim.loc[target_id]

        sim_features.loc[target_id, 'UsM_deltaDays0'] = first_source_row.user_created_days
        sim_features.loc[target_id, 'UsM_statusesCount0'] = first_source_row.statuses_count
        sim_features.loc[target_id, 'UsM_followersCount0'] = first_source_row.followers_count
        sim_features.loc[target_id, 'UsM_favouritesCount0'] = first_source_row.favourites_count
        sim_features.loc[target_id, 'UsM_friendsCount0'] = first_source_row.friends_count
        sim_features.loc[target_id, 'UsM_listedCount0'] = first_source_row.listed_count
        sim_features.loc[target_id, 'UsM_normalizedUserStatusesCount0'] = first_source_row.normalized_statuses_count
        sim_features.loc[target_id, 'UsM_normalizedUserFollowersCount0'] = first_source_row.normalized_followers_count
        sim_features.loc[target_id, 'UsM_normalizedUserFavouritesCount0'] = first_source_row.normalized_favourites_count
        sim_features.loc[target_id, 'UsM_normalizedUserListedCount0'] = first_source_row.normalized_listed_count
        sim_features.loc[target_id, 'UsM_normalizedUserFriendsCount0'] = first_source_row.normalized_friends_count
        sim_features.loc[target_id, 'UsM_deltaDays-1'] = last_source_row.user_created_days
        sim_features.loc[target_id, 'UsM_statusesCount-1'] = last_source_row.statuses_count
        sim_features.loc[target_id, 'UsM_followersCount-1'] = last_source_row.followers_count
        sim_features.loc[target_id, 'UsM_favouritesCount-1'] = last_source_row.favourites_count
        sim_features.loc[target_id, 'UsM_friendsCount-1'] = last_source_row.friends_count
        sim_features.loc[target_id, 'UsM_listedCount-1'] = last_source_row.listed_count
        sim_features.loc[target_id, 'UsM_normalizedUserStatusesCount-1'] = last_source_row.normalized_statuses_count
        sim_features.loc[target_id, 'UsM_normalizedUserFollowersCount-1'] = last_source_row.normalized_followers_count
        sim_features.loc[target_id, 'UsM_normalizedUserFavouritesCount-1'] = last_source_row.normalized_favourites_count
        sim_features.loc[target_id, 'UsM_normalizedUserListedCount-1'] = last_source_row.normalized_listed_count
        sim_features.loc[target_id, 'UsM_normalizedUserFriendsCount-1'] = last_source_row.normalized_friends_count
        # TwM: Tweet metadata
        sim_features.loc[target_id, 'TwM_t0'] = round(time_list[0], 1)
        sim_features.loc[target_id, 'TwM_tSeed0'] = round(curr_time - first_source_seed_row['time_lapsed'], 1)
        sim_features.loc[target_id, 'TwM_t-1'] = round(time_list[-1], 1)
        sim_features.loc[target_id, 'TwM_tSeed-1'] = round(curr_time - last_source_seed_row['time_lapsed'], 1)
        sim_features.loc[target_id, 'TwM_tCurrent'] = curr_time
        # Nw: Network
        sim_features.loc[target_id, 'Nw_degree'] = degree[target_id]
        sim_features.loc[target_id, 'Nw_inDegree'] = in_degree[target_id]
        sim_features.loc[target_id, 'Nw_outDegree'] = out_degree[target_id]
        sim_features.loc[target_id, 'Nw_degree0'] = degree[first_source_index]
        sim_features.loc[target_id, 'Nw_inDegree0'] = in_degree[first_source_index]
        sim_features.loc[target_id, 'Nw_outDegree0'] = out_degree[first_source_index]
        sim_features.loc[target_id, 'Nw_degree-1'] = degree[last_source_index]
        sim_features.loc[target_id, 'Nw_inDegree-1'] = in_degree[last_source_index]
        sim_features.loc[target_id, 'Nw_outDegree-1'] = out_degree[last_source_index]
        sim_features.loc[target_id, 'Nw_degreeSeed0'] = degree[int(first_source_row['seed_index'])]
        sim_features.loc[target_id, 'Nw_inDegreeSeed0'] = in_degree[int(first_source_row['seed_index'])]
        sim_features.loc[target_id, 'Nw_outDegreeSeed0'] = out_degree[int(first_source_row['seed_index'])]
        sim_features.loc[target_id, 'Nw_degreeSeed-1'] = degree[int(last_source_row['seed_index'])]
        sim_features.loc[target_id, 'Nw_inDegreeSeed-1'] = in_degree[int(last_source_row['seed_index'])]
        sim_features.loc[target_id, 'Nw_outDegreeSeed-1'] = out_degree[int(last_source_row['seed_index'])]
        # SNw: Spreading Network
        sim_features.loc[target_id, 'SNw_nFriendsInfected'] = len(sources)
        sim_features.loc[target_id, 'SNw_friendsInfectedRatio'] = safe_division(len(sources), user_row['friends_count'])
        sim_features.loc[target_id, 'SNw_generation0'] = first_source_row['generation']
        sim_features.loc[target_id, 'SNw_generation-1'] = last_source_row['generation']
        sim_features.loc[target_id, 'SNw_timeSinceSeed0'] = first_source_row['time_since_seed']
        sim_features.loc[target_id, 'SNw_timeSinceSeed-1'] = last_source_row['time_since_seed']

        infected_dataframe = net_sim[net_sim.time_lapsed <= curr_time]
        total_nodes_infected = infected_dataframe.shape[0]
        total_in_degree = sum(infected_dataframe.friends_count)
        total_out_degree = sum(infected_dataframe.followers_count)

        sim_features.loc[target_id, 'SNw_totalNodesInfected'] = total_nodes_infected
        sim_features.loc[target_id, 'SNw_nodeInfectedCentrality'] = len(sources) / total_nodes_infected
        sim_features.loc[target_id, 'SNw_totalInDegree'] = total_in_degree
        sim_features.loc[target_id, 'SNw_totalOutDegree'] = total_out_degree
        sim_features.loc[target_id, 'SNw_inDegreeCentrality'] = in_degree[target_id] / total_in_degree
        sim_features.loc[target_id, 'SNw_inDegreeCentrality0'] = in_degree[first_source_index] / total_in_degree
        sim_features.loc[target_id, 'SNw_inDegreeCentrality-1'] = in_degree[last_source_index] / total_in_degree
        sim_features.loc[target_id, 'SNw_outDegreeCentrality'] = out_degree[target_id] / total_out_degree
        sim_features.loc[target_id, 'SNw_outDegreeCentrality0'] = out_degree[first_source_index] / total_out_degree
        sim_features.loc[target_id, 'SNw_outDegreeCentrality-1'] = out_degree[last_source_index] / total_out_degree
        sim_features.loc[target_id, 'SNw_inDegreeCentralitySeed0'] = in_degree[int(first_source_row['seed_index'])] / total_in_degree
        sim_features.loc[target_id, 'SNw_outDegreeCentralitySeed0'] = out_degree[int(first_source_row['seed_index'])] / total_out_degree
        sim_features.loc[target_id, 'SNw_inDegreeCentralitySeed-1'] = in_degree[int(last_source_row['seed_index'])] / total_in_degree
        sim_features.loc[target_id, 'SNw_outDegreeCentralitySeed-1'] = out_degree[int(last_source_row['seed_index'])] / total_out_degree
        # Stat: Statistical
        sim_features.loc[target_id, 'Stat_average_kOut'] = round(mean(degree_list), 1)
        sim_features.loc[target_id, 'Stat_average_t'] = round(mean(time_list), 1)
        sim_features.loc[target_id, 'Stat_average_deltaDays'] = sources_df.user_created_days.mean()
        sim_features.loc[target_id, 'Stat_average_statusesCount'] = sources_df.statuses_count.mean()
        sim_features.loc[target_id, 'Stat_average_followersCount'] = sources_df.followers_count.mean()
        sim_features.loc[target_id, 'Stat_average_favouritesCount'] = sources_df.favourites_count.mean()
        sim_features.loc[target_id, 'Stat_average_friendsCount'] = sources_df.friends_count.mean()
        sim_features.loc[target_id, 'Stat_average_listedCount'] = sources_df.listed_count.mean()
        sim_features.loc[target_id, 'Stat_average_normalizedUserStatusesCount'] = sources_df.normalized_statuses_count.mean()
        sim_features.loc[target_id, 'Stat_average_normalizedUserFollowersCount'] = sources_df.normalized_followers_count.mean()
        sim_features.loc[target_id, 'Stat_average_normalizedUserFavouritesCount'] = sources_df.normalized_favourites_count.mean()
        sim_features.loc[target_id, 'Stat_average_normalizedUserListedCount'] = sources_df.normalized_listed_count.mean()
        sim_features.loc[target_id, 'Stat_average_normalizedUserFriendsCount'] = sources_df.normalized_friends_count.mean()
        sim_features.loc[target_id, 'Stat_max_kOut'] = max(degree_list)
        sim_features.loc[target_id, 'Stat_min_kOut'] = min(degree_list)


def chunkify(uninfected_followers_indices, num_chunks):

    indices_nparr = np.array(uninfected_followers_indices)
    chunks = np.array_split(indices_nparr, num_chunks)

    return list(map(lambda l: l.tolist(), chunks))


class SimulationResult(object):

    def __init__(self, source_index, target_index):
        self.source_index = source_index
        self.target_index = target_index

    def get_source_index(self):
        return self.source_index

    def get_target_index(self):
        return self.target_index


def run_simulation_on_chunk(degree,
                            in_degree,
                            out_degree,
                            infected_user_index,
                            uninfected_followers_indices,
                            sim_features,
                            net_sim,
                            curr_time):
    result = []
    source_index = infected_user_index
    for j in uninfected_followers_indices:
        target_index = j
        generate_features(degree, in_degree, out_degree, target_index, sim_features, net_sim, curr_time)

    valid = sim_features.drop(['user_id','infected_status','infection_time','followers_list',
                               'Nw_inDegree', 'generation', 'time_lapsed', 'seed_index', 'Nw_outDegree'],axis=1)

    valid_value = valid.astype('float64')

    pre_data = xgb.DMatrix(valid_value)
    score_vector = model.predict(pre_data)
    for index in range(len(score_vector)):
        if score_vector[index] > 0.6:
            target_index = uninfected_followers_indices[index]
            print(f"Infected source_index: {source_index}, target_index: {target_index}")
            result.append(SimulationResult(source_index=source_index, target_index=target_index))
    return result


def update_simulation_state(net_sim, simulation_result, curr_time):
    net_sim.loc[simulation_result.target_index, 'time_lapsed'] = curr_time
    net_sim.loc[simulation_result.target_index, 'source_index'] = simulation_result.source_index
    if np.isnan(net_sim.loc[simulation_result.target_index, 'seed_index']):
        net_sim.loc[simulation_result.target_index, 'seed_index'] = simulation_result.source_index
    net_sim.loc[simulation_result.target_index, 'generation'] = net_sim.loc[simulation_result.source_index, 'generation'] + 1
    seed_index = net_sim.loc[simulation_result.target_index, 'seed_index']
    net_sim.loc[simulation_result.target_index, 'time_since_seed'] = curr_time - net_sim.loc[seed_index, 'time_lapsed']
    followers_of_node = net_sim.loc[simulation_result.target_index, 'followers_list']
    if isinstance(followers_of_node, list):
        for f in followers_of_node:
            if np.isnan(net_sim[net_sim['id'] == f]['time_lapsed'].values[0]):
                follower_index = net_sim[net_sim['id'] == f].index.values
                try:
                    list(net_sim.loc[follower_index, 'exposed_source_candidates'].values).append(f)
                except:
                    print(f"exposed_source_candidates:{net_sim.loc[follower_index,'exposed_source_candidates'].values}")
                    print(f"f:{f}")


def parallel_simulation(degree, in_degree, out_degree, sim_features, net_sim, curr_time):
    infected_users_indices = net_sim[net_sim['time_lapsed'].isnull() == False].index.values
    for i in infected_users_indices:
        if isinstance(net_sim.loc[i, 'followers_list'], list):
            followers_indices = []
            for x in net_sim.loc[i].followers_list:
                followers_indices.append(net_sim[net_sim['id'] == x].index.values.item())

            uninfected_followers_indices = [y for y in followers_indices if np.isnan(net_sim.loc[y, 'time_lapsed']) == True]

            if len(uninfected_followers_indices) > 0:
                num_cpu = multiprocessing.cpu_count()
                # Divide the work into number of chunks, equal to number of cpus available
                chunks = chunkify(uninfected_followers_indices, num_cpu)
                simulation_results = []
                for chunk in chunks:
                    with ThreadPoolExecutor(max_workers=num_cpu) as executor:
                        future = executor.submit(run_simulation_on_chunk,
                                                 degree,
                                                 in_degree,
                                                 out_degree,
                                                 i,
                                                 chunk,
                                                 sim_features,
                                                 net_sim,
                                                 curr_time)

                        simulation_results.append(future)

                for future in simulation_results:
                    try:
                        result = future.result()

                        for sim_res in result:
                            update_simulation_state(net_sim=net_sim,
                                                    simulation_result=sim_res,
                                                    curr_time=curr_time)
                    except Exception as e:
                        print(f'Simulation generated an exception: {e}')

    return net_sim


path = "/Users/jay/MSC_WSBDA/MSc_Thesis/Msc_project/Data/"
model = joblib.load(path+"/xgb_model_GIVENCHY (2).dat")


def main():

    initial_features = load_pickle_file(path+'keynode_initial_features.pkl')
    initial_dataset = load_pickle_file(path+'network_simulation_keynode_initial.pkl')
    users = load_pickle_file(path+"80000users_3.pkl")
    users.reset_index(drop =True , inplace =True)
    seeds_10 = pd.read_csv(path+'top_10_avg_neigh_degree_seeds.csv')
    seeds_5 = seeds_10.loc[0:5, :]
    seeds = seeds_5

    in_degree = list(initial_dataset.friends_count)
    out_degree = list(initial_dataset.followers_count)
    degree = in_degree + out_degree

    total_time_duration = 24*60
    interval = 30

    current_time = 0

    seed_initialization(initial_features, initial_dataset, seeds)
    features = initial_features
    network_simulation = initial_dataset

    print("Simulation started")
    start_time = time.time()

    while current_time < total_time_duration:
        print(f"current_time:{current_time}")
        network_simulation = parallel_simulation(degree,
                                                 in_degree,
                                                 out_degree,
                                                 features,
                                                 network_simulation,
                                                 current_time)
        current_time += interval

    print(f"Simulation finished after {round((time.time() - start_time)/60,2)} minutes")

    network_simulation.to_csv(path+'simulation_result_key_node_avg_neighbour_degree_top_5.csv')


if __name__ == '__main__':
    main()
