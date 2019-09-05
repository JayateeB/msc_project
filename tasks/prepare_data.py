import pandas as pd
import csv
import pickle
import numpy as np
import multiprocessing
from multiprocessing import Pool
import math
from tqdm import tqdm
import time
import traceback
import json
import numpy as np

path = '/Users/syamantak/JayateeB/new_files/data/'
event = 'nyc'
interval = 30
current_time = 420
start_hour = 7

def load_pickle_file(pickled_file):
    print(f'Loading data file from {pickled_file}')
    infile = open(pickled_file,'rb')
    unpickled_file = pickle.load(infile)
    print(f'Loaded {len(unpickled_file)} entries')
    infile.close()
    return unpickled_file


def save_pickle_file(path, data):
    print('Dumping data to path {}'.format(path))
    with open(path, 'wb') as file:
        pickle.dump(data, file)
    print('Finished dumping data to path {}'.format(path))


def mean(numbers):
    return float(sum(numbers)) / max(len(numbers), 1)


def safe_division(x, y):
    if y == 0:
        return 0
    else:
        return x/y

def mean_of(rows, column):
    result = list(map(lambda row: row[column], rows))
    return mean(result)



users = load_pickle_file(path + "nyc_users_6_9_infected.dat")
ext_followers = pd.read_csv(path + "nyc_6_7_ext_followers.csv")

def id_to_index_list(idx_lookup, src_candidate_ids):
    return list(filter(lambda x: x is not None, map(lambda x: idx_lookup.get(x, None), src_candidate_ids)))


network_simulation = pd.DataFrame(columns=['id', 'time_lapsed', 'favourites_count', 'followers_count', 'friends_count',
                                           'listed_count', 'statuses_count', 'source_candidates', 'source_index',
                                           'seed_index', 'generation', 'time_since_seed', 'user_created_days',
                                           'normalized_statuses_count', 'normalized_followers_count',
                                           'normalized_favourites_count', 'normalized_listed_count',
                                           'normalized_friends_count'])

network_simulation['id']=users['id'].append(ext_followers['id'],ignore_index = True)
network_simulation['favourites_count']=users['favourites_count'].append(ext_followers['favourites_count'],ignore_index = True)
network_simulation['followers_count']=users['followers_count'].append(ext_followers['followers_count'],ignore_index = True)
network_simulation['friends_count']=users['friends_count'].append(ext_followers['friends_count'],ignore_index = True)
network_simulation['listed_count']=users['listed_count'].append(ext_followers['listed_count'],ignore_index = True)
network_simulation['statuses_count']=users['statuses_count'].append(ext_followers['statuses_count'],ignore_index = True)
network_simulation['user_created_days']=users['user_created_days'].append(ext_followers['user_created_days'],ignore_index = True)
network_simulation['normalized_statuses_count']=users['normalized_statuses_count'].append(ext_followers['normalized_statuses_count'],ignore_index = True)
network_simulation['normalized_followers_count']=users['normalized_followers_count'].append(ext_followers['normalized_followers_count'],ignore_index = True)
network_simulation['normalized_favourites_count']=users['normalized_favourites_count'].append(ext_followers['normalized_favourites_count'],ignore_index = True)
network_simulation['normalized_listed_count']=users['normalized_listed_count'].append(ext_followers['normalized_listed_count'],ignore_index = True)
network_simulation['normalized_friends_count']=users['normalized_friends_count'].append(ext_followers['normalized_friends_count'],ignore_index = True)
network_simulation['source_candidates']=users['source_candidates'].append(ext_followers['source_candidates'],ignore_index = True)
network_simulation['time_lapsed'] = users['time_lapsed'].apply(lambda x: x if x <= current_time else None)
network_simulation['source_index'] = users.apply(lambda x: x['source_index'] if x['time_lapsed'] <= current_time else None,axis=1)
network_simulation['seed_index'] = users.apply(lambda x: x['seed_index'] if x['time_lapsed'] <= current_time else None,axis=1)
network_simulation['generation'] = users.apply(lambda x: x['generation'] if x['time_lapsed'] <= current_time else None,axis=1)
network_simulation['time_since_seed'] = users.apply(lambda x: x['time_since_seed'] if x['time_lapsed'] <= current_time else None,axis=1)
network_simulation['followers_list'] = users["followers_list"]

id_list = network_simulation['id'].tolist()
index_lookup = {k: v for v, k in enumerate(network_simulation['id'].tolist())}

for i in range(88266, len(id_list)):
    source_candidate_ids = network_simulation.loc[i, "source_candidates"]
    if isinstance(source_candidate_ids, list):
        network_simulation.loc[i, "source_candidates"] = id_to_index_list(index_lookup, source_candidate_ids)


def process_data(start_index, end_index,current_time):

    in_degree = list(network_simulation.friends_count)
    out_degree = list(network_simulation.followers_count)
    degree = in_degree + out_degree

    features = {
        #Columns which are added for simulation, but they are not used as features for model prediction
        'user_id':[],
        'infected_status':[],
        'infection_time':[],
        'followers_list':[],

        #Columns used as features for model prediction
        'UsM_deltaDays': [],
        'UsM_statusesCount': [],
        'UsM_followersCount': [],
        'UsM_favouritesCount': [],
        'UsM_friendsCount': [],
        'UsM_listedCount': [],
        'UsM_normalizedUserStatusesCount': [],
        'UsM_normalizedUserFollowersCount': [],
        'UsM_normalizedUserFavouritesCount': [],
        'UsM_normalizedUserListedCount': [],
        'UsM_normalizedUserFriendsCount': [],
        'UsM_deltaDays0': [],
        'UsM_statusesCount0': [],
        'UsM_followersCount0': [],
        'UsM_favouritesCount0': [],
        'UsM_friendsCount0': [],
        'UsM_listedCount0': [],
        'UsM_normalizedUserStatusesCount0': [],
        'UsM_normalizedUserFollowersCount0': [],
        'UsM_normalizedUserFavouritesCount0': [],
        'UsM_normalizedUserListedCount0': [],
        'UsM_normalizedUserFriendsCount0': [],
        'UsM_deltaDays-1': [],
        'UsM_statusesCount-1': [],
        'UsM_followersCount-1': [],
        'UsM_favouritesCount-1': [],
        'UsM_friendsCount-1': [],
        'UsM_listedCount-1': [],
        'UsM_normalizedUserStatusesCount-1': [],
        'UsM_normalizedUserFollowersCount-1': [],
        'UsM_normalizedUserFavouritesCount-1': [],
        'UsM_normalizedUserListedCount-1': [],
        'UsM_normalizedUserFriendsCount-1': [],
        # TwM: Tweet metadata
        'TwM_t0': [],
        'TwM_tSeed0': [],
        'TwM_t-1': [],
        'TwM_tSeed-1': [],
        'TwM_tCurrent': [],
        # Nw: Network
        'Nw_degree': [],
        'Nw_inDegree': [],
        'Nw_outDegree': [],
        'Nw_degree0': [],
        'Nw_inDegree0': [],
        'Nw_outDegree0': [],
        'Nw_degree-1': [],
        'Nw_inDegree-1': [],
        'Nw_outDegree-1': [],
        'Nw_degreeSeed0': [],
        'Nw_inDegreeSeed0': [],
        'Nw_outDegreeSeed0': [],
        'Nw_degreeSeed-1': [],
        'Nw_inDegreeSeed-1': [],
        'Nw_outDegreeSeed-1': [],
        # SNw: Spreading Network
        'SNw_nFriendsInfected': [],
        'SNw_friendsInfectedRatio': [],
        'SNw_generation0': [],
        'SNw_generation-1': [],
        'SNw_timeSinceSeed0': [],
        'SNw_timeSinceSeed-1': [],
        'SNw_totalNodesInfected': [],
        'SNw_nodeInfectedCentrality': [],
        'SNw_totalInDegree': [],
        'SNw_totalOutDegree': [],
        'SNw_inDegreeCentrality': [],
        'SNw_inDegreeCentrality0': [],
        'SNw_inDegreeCentrality-1': [],
        'SNw_outDegreeCentrality': [],
        'SNw_outDegreeCentrality0': [],
        'SNw_outDegreeCentrality-1': [],
        'SNw_inDegreeCentralitySeed0':[],
        'SNw_outDegreeCentralitySeed0':[],
        'SNw_inDegreeCentralitySeed-1':[],
        'SNw_outDegreeCentralitySeed-1':[],
        # Stat: Statistical
        'Stat_average_kOut': [],
        'Stat_average_t': [],
        'Stat_average_deltaDays': [],
        'Stat_average_statusesCount': [],
        'Stat_average_followersCount': [],
        'Stat_average_favouritesCount': [],
        'Stat_average_friendsCount': [],
        'Stat_average_listedCount': [],
        'Stat_average_normalizedUserStatusesCount': [],
        'Stat_average_normalizedUserFollowersCount': [],
        'Stat_average_normalizedUserFavouritesCount': [],
        'Stat_average_normalizedUserListedCount': [],
        'Stat_average_normalizedUserFriendsCount': [],
        'Stat_max_kOut': [],
        'Stat_min_kOut': []

    }

    with tqdm(total=end_index - start_index) as pbar:
        #for index, user_row in network_simulation[start_index: end_index].iterrows():
        for index in range(start_index,end_index):
            user_row = network_simulation.loc[index]
            source_candidates = sorted(user_row['source_candidates'])
            features['user_id'].append(user_row['id'])
            features['infected_status'].append(False)
            features['infection_time'].append(None)
            #print(f"user_row['followers_list']:{user_row['followers_list']}")
            features['followers_list'].append(user_row['followers_list'])
            #print("b")
            features['UsM_deltaDays'].append(user_row['user_created_days'])
            features['UsM_statusesCount'].append(user_row['statuses_count'])
            features['UsM_followersCount'].append(user_row['followers_count'])
            features['UsM_favouritesCount'].append(user_row['favourites_count'])
            features['UsM_friendsCount'].append(user_row['friends_count'])
            features['UsM_listedCount'].append(user_row['listed_count'])
            features['UsM_normalizedUserStatusesCount'].append(user_row['normalized_statuses_count'])
            features['UsM_normalizedUserFollowersCount'].append(user_row['normalized_followers_count'])
            features['UsM_normalizedUserFavouritesCount'].append(user_row['normalized_favourites_count'])
            features['UsM_normalizedUserListedCount'].append(user_row['normalized_listed_count'])
            features['UsM_normalizedUserFriendsCount'].append(user_row['normalized_friends_count'])

            try:
                sources = []
                for x in source_candidates:

                    #network_simulation.loc[x,'time_lapsed'].isnull() == False &
                    if network_simulation.loc[x,'time_lapsed'] <= current_time:
                        sources.append(x)

            except:
                print(f"x:{x}")
                print(f"source_candidates:{source_candidates}")




            #sources = [x for x in source_candidates if users.loc[x,'time_lapsed'] <= current_time]
            sources = []

            if len(sources) > 0:

                # Assign the values here to save computation
                first_source_index = source_candidates[0]
                first_source_row = network_simulation.loc[first_source_index]
                first_source_seed_row = network_simulation.loc[first_source_row['seed_index']]

                sources_dataframe = network_simulation.loc[sources]
                degreeList = list(degree[i] for i in sources)
                inDegreeList = list(in_degree[i] for i in sources)
                outDegreeList = list(out_degree[i] for i in sources)
                degreeList = list(network_simulation.loc[i, 'followers_count'] + network_simulation.loc[i, 'friends_count']  for i in sources)
                timeList = [current_time - network_simulation.loc[x].time_lapsed for x in sources]


                last_source_index = sources[-1]
                last_source_row = network_simulation.loc[last_source_index]
                last_source_seed_row = network_simulation.loc[last_source_row['seed_index']]

                usr_index = index

                if user_row['time_lapsed'] <= current_time:

                    features['infected_status'][-1] = True
                    features['infection_time'][-1] = user_row['time_lapsed']

                    network_simulation.loc[usr_index,'time_lapsed'] = user_row['time_lapsed']

                    network_simulation.loc[usr_index,'source_index'] = user_row['source_index']
                    network_simulation.loc[usr_index,'seed_index'] = user_row['seed_index']
                    network_simulation.loc[usr_index, 'generation'] = user_row['generation']
                    network_simulation.loc[usr_index, 'time_since_seed'] = user_row['time_since_seed']


                network_simulation.at[usr_index,'source_candidates'] = sources

                # UsM: User metadata

                features['UsM_deltaDays0'].append(first_source_row.user_created_days)
                features['UsM_statusesCount0'].append(first_source_row.statuses_count)
                features['UsM_followersCount0'].append(first_source_row.followers_count)
                features['UsM_favouritesCount0'].append(first_source_row.favourites_count)
                features['UsM_friendsCount0'].append(first_source_row.friends_count)
                features['UsM_listedCount0'].append(first_source_row.listed_count)
                features['UsM_normalizedUserStatusesCount0'].append(first_source_row.normalized_statuses_count)
                features['UsM_normalizedUserFollowersCount0'].append(first_source_row.normalized_followers_count)
                features['UsM_normalizedUserFavouritesCount0'].append(first_source_row.normalized_favourites_count)
                features['UsM_normalizedUserListedCount0'].append(first_source_row.normalized_listed_count)
                features['UsM_normalizedUserFriendsCount0'].append(first_source_row.normalized_friends_count)
                features['UsM_deltaDays-1'].append(last_source_row.user_created_days)
                features['UsM_statusesCount-1'].append(last_source_row.statuses_count)
                features['UsM_followersCount-1'].append(last_source_row.followers_count)
                features['UsM_favouritesCount-1'].append(last_source_row.favourites_count)
                features['UsM_friendsCount-1'].append(last_source_row.friends_count)
                features['UsM_listedCount-1'].append(last_source_row.listed_count)
                features['UsM_normalizedUserStatusesCount-1'].append(last_source_row.normalized_statuses_count)
                features['UsM_normalizedUserFollowersCount-1'].append(last_source_row.normalized_followers_count)
                features['UsM_normalizedUserFavouritesCount-1'].append(last_source_row.normalized_favourites_count)
                features['UsM_normalizedUserListedCount-1'].append(last_source_row.normalized_listed_count)
                features['UsM_normalizedUserFriendsCount-1'].append(last_source_row.normalized_friends_count)
                # TwM: Tweet metadata
                features['TwM_t0'].append(round(timeList[0], 1))
                features['TwM_tSeed0'].append(round(current_time - first_source_seed_row['time_lapsed'], 1))
                features['TwM_t-1'].append(round(timeList[-1], 1))
                features['TwM_tSeed-1'].append(round(current_time - last_source_seed_row['time_lapsed'], 1))
                features['TwM_tCurrent'].append(current_time)
                # Nw: Network
                features['Nw_degree'].append(degree[index])
                features['Nw_inDegree'].append(in_degree[index])
                features['Nw_outDegree'].append(out_degree[index])
                features['Nw_degree0'].append(degree[first_source_index])
                features['Nw_inDegree0'].append(in_degree[first_source_index])
                features['Nw_outDegree0'].append(out_degree[first_source_index])
                features['Nw_degree-1'].append(degree[last_source_index])
                features['Nw_inDegree-1'].append(in_degree[last_source_index])
                features['Nw_outDegree-1'].append(out_degree[last_source_index])
                features['Nw_degreeSeed0'].append(degree[int(first_source_row['seed_index'])])
                features['Nw_inDegreeSeed0'].append(in_degree[int(first_source_row['seed_index'])])
                features['Nw_outDegreeSeed0'].append(out_degree[int(first_source_row['seed_index'])])
                features['Nw_degreeSeed-1'].append(degree[int(last_source_row['seed_index'])])
                features['Nw_inDegreeSeed-1'].append(in_degree[int(last_source_row['seed_index'])])
                features['Nw_outDegreeSeed-1'].append(out_degree[int(last_source_row['seed_index'])])
                # SNw: Spreading Network
                features['SNw_nFriendsInfected'].append(len(sources))
                features['SNw_friendsInfectedRatio'].append(safe_division(len(sources), user_row['friends_count']))
                features['SNw_generation0'].append(first_source_row['generation'])
                features['SNw_generation-1'].append(last_source_row['generation'])
                features['SNw_timeSinceSeed0'].append(first_source_row['time_since_seed'])
                features['SNw_timeSinceSeed-1'].append(last_source_row['time_since_seed'])

                features['SNw_totalNodesInfected'].append(total_nodes_infected)
                features['SNw_nodeInfectedCentrality'].append(len(sources)/total_nodes_infected)
                features['SNw_totalInDegree'].append(total_in_degree)
                features['SNw_totalOutDegree'].append(total_out_degree)
                features['SNw_inDegreeCentrality'].append(in_degree[index]/total_in_degree)
                features['SNw_inDegreeCentrality0'].append(in_degree[first_source_index]/total_in_degree)
                features['SNw_inDegreeCentrality-1'].append(in_degree[last_source_index]/total_in_degree)
                features['SNw_outDegreeCentrality'].append(out_degree[index]/total_out_degree)
                features['SNw_outDegreeCentrality0'].append(out_degree[first_source_index]/total_out_degree)
                features['SNw_outDegreeCentrality-1'].append(out_degree[last_source_index]/total_out_degree)
                features['SNw_inDegreeCentralitySeed0'].append(in_degree[int(first_source_row['seed_index'])]/total_in_degree)
                features['SNw_outDegreeCentralitySeed0'].append(out_degree[int(first_source_row['seed_index'])]/total_out_degree)
                features['SNw_inDegreeCentralitySeed-1'].append(in_degree[int(last_source_row['seed_index'])]/total_in_degree)
                features['SNw_outDegreeCentralitySeed-1'].append(out_degree[int(last_source_row['seed_index'])]/total_out_degree)
                # Stat: Statistical
                features['Stat_average_kOut'].append(round(mean(degreeList), 1))
                features['Stat_average_t'].append(round(mean(timeList), 1))
                features['Stat_average_deltaDays'].append(sources_dataframe.user_created_days.mean())
                features['Stat_average_statusesCount'].append(sources_dataframe.statuses_count.mean())
                features['Stat_average_followersCount'].append(sources_dataframe.followers_count.mean())
                features['Stat_average_favouritesCount'].append(sources_dataframe.favourites_count.mean())
                features['Stat_average_friendsCount'].append(sources_dataframe.friends_count.mean())
                features['Stat_average_listedCount'].append(sources_dataframe.listed_count.mean())
                features['Stat_average_normalizedUserStatusesCount'].append(sources_dataframe.normalized_statuses_count.mean())
                features['Stat_average_normalizedUserFollowersCount'].append(sources_dataframe.normalized_followers_count.mean())
                features['Stat_average_normalizedUserFavouritesCount'].append(sources_dataframe.normalized_favourites_count.mean())
                features['Stat_average_normalizedUserListedCount'].append(sources_dataframe.normalized_listed_count.mean())
                features['Stat_average_normalizedUserFriendsCount'].append(sources_dataframe.normalized_friends_count.mean())
                features['Stat_max_kOut'].append(max(degreeList))
                features['Stat_min_kOut'].append(min(degreeList))
            else:
                features['UsM_deltaDays0'].append(None)
                features['UsM_statusesCount0'].append(None)
                features['UsM_followersCount0'].append(None)
                features['UsM_favouritesCount0'].append(None)
                features['UsM_friendsCount0'].append(None)
                features['UsM_listedCount0'].append(None)
                features['UsM_normalizedUserStatusesCount0'].append(None)
                features['UsM_normalizedUserFollowersCount0'].append(None)
                features['UsM_normalizedUserFavouritesCount0'].append(None)
                features['UsM_normalizedUserListedCount0'].append(None)
                features['UsM_normalizedUserFriendsCount0'].append(None)
                features['UsM_deltaDays-1'].append(None)
                features['UsM_statusesCount-1'].append(None)
                features['UsM_followersCount-1'].append(None)
                features['UsM_favouritesCount-1'].append(None)
                features['UsM_friendsCount-1'].append(None)
                features['UsM_listedCount-1'].append(None)
                features['UsM_normalizedUserStatusesCount-1'].append(None)
                features['UsM_normalizedUserFollowersCount-1'].append(None)
                features['UsM_normalizedUserFavouritesCount-1'].append(None)
                features['UsM_normalizedUserListedCount-1'].append(None)
                features['UsM_normalizedUserFriendsCount-1'].append(None)
                # TwM: Tweet metadata
                features['TwM_t0'].append(None)
                features['TwM_tSeed0'].append(None)
                features['TwM_t-1'].append(None)
                features['TwM_tSeed-1'].append(None)
                features['TwM_tCurrent'].append(None)
                # Nw: Network
                features['Nw_degree'].append(None)
                features['Nw_inDegree'].append(None)
                features['Nw_outDegree'].append(None)
                features['Nw_degree0'].append(None)
                features['Nw_inDegree0'].append(None)
                features['Nw_outDegree0'].append(None)
                features['Nw_degree-1'].append(None)
                features['Nw_inDegree-1'].append(None)
                features['Nw_outDegree-1'].append(None)
                features['Nw_degreeSeed0'].append(None)
                features['Nw_inDegreeSeed0'].append(None)
                features['Nw_outDegreeSeed0'].append(None)
                features['Nw_degreeSeed-1'].append(None)
                features['Nw_inDegreeSeed-1'].append(None)
                features['Nw_outDegreeSeed-1'].append(None)
                # SNw: Spreading Network
                features['SNw_nFriendsInfected'].append(0)
                features['SNw_friendsInfectedRatio'].append(None)
                features['SNw_generation0'].append(None)
                features['SNw_generation-1'].append(None)
                features['SNw_timeSinceSeed0'].append(None)
                features['SNw_timeSinceSeed-1'].append(None)
                features['SNw_totalNodesInfected'].append(None)
                features['SNw_nodeInfectedCentrality'].append(None)
                features['SNw_totalInDegree'].append(None)
                features['SNw_totalOutDegree'].append(None)
                features['SNw_inDegreeCentrality'].append(None)
                features['SNw_inDegreeCentrality0'].append(None)
                features['SNw_inDegreeCentrality-1'].append(None)
                features['SNw_outDegreeCentrality'].append(None)
                features['SNw_outDegreeCentrality0'].append(None)
                features['SNw_outDegreeCentrality-1'].append(None)
                features['SNw_inDegreeCentralitySeed0'].append(None)
                features['SNw_outDegreeCentralitySeed0'].append(None)
                features['SNw_inDegreeCentralitySeed-1'].append(None)
                features['SNw_outDegreeCentralitySeed-1'].append(None)
                # Stat: Statistical
                features['Stat_average_kOut'].append(None)
                features['Stat_average_t'].append(None)
                features['Stat_average_deltaDays'].append(None)
                features['Stat_average_statusesCount'].append(None)
                features['Stat_average_followersCount'].append(None)
                features['Stat_average_favouritesCount'].append(None)
                features['Stat_average_friendsCount'].append(None)
                features['Stat_average_listedCount'].append(None)
                features['Stat_average_normalizedUserStatusesCount'].append(None)
                features['Stat_average_normalizedUserFollowersCount'].append(None)
                features['Stat_average_normalizedUserFavouritesCount'].append(None)
                features['Stat_average_normalizedUserListedCount'].append(None)
                features['Stat_average_normalizedUserFriendsCount'].append(None)
                features['Stat_max_kOut'].append(None)
                features['Stat_min_kOut'].append(None)

            pbar.update(1)
    processed_dataframe = pd.DataFrame(features)
    return processed_dataframe


infected_dataframe = network_simulation[network_simulation.time_lapsed <= current_time]
total_nodes_infected = infected_dataframe.shape[0]
total_in_degree = sum(infected_dataframe.friends_count)
total_out_degree = sum(infected_dataframe.followers_count)

def run_dataset_preparation():
    number_of_processes = multiprocessing.cpu_count()
    print('Will start {} processes'.format(number_of_processes))
    with Pool(number_of_processes) as pool:
        parameters = []
        number_of_users = len(network_simulation.index)
        task_size = math.ceil(number_of_users/number_of_processes)
        for i in range(number_of_processes):
            start_index = i * task_size
            end_index = min((i + 1) * task_size, number_of_users)
            parameters.append((start_index, end_index,current_time))
        dataframe_results = pool.starmap(process_data, parameters)

    result = pd.DataFrame()
    result = result.append(dataframe_results)
    #start_hour = math.trunc(current_time / 60)
    save_pickle_file(path+event+str(start_hour)+"_hrs_data.pkl",result)
    print('extracted {} of rows'.format(len(result.index)))

run_dataset_preparation()