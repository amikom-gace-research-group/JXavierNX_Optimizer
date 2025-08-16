import numpy as np
import sys
import time
import os
import csv
import requests
import math
from pyDOE import lhs
from scipy.spatial.distance import pdist, squareform
from sklearn.preprocessing import MinMaxScaler
from scipy.spatial import distance

eps = 1
th_target = int(sys.argv[7])
best_idx = -1

def get_result():
    headers = {
        'Authorization': sys.argv[2],
        'Content-Type': 'application/json'
    }
    try:
        response = requests.get(f"{sys.argv[1]}/api/output", headers=headers)
        if response.status_code == 200:
            return response.json()
    except requests.RequestException as e:
        print(f"Error fetching result: {e}")
    return False

# Execute the configuration on the system
def execute_config(cpu_cores, cpu_freq, gpu_freq, memory_freq, cl):
    url = f"{sys.argv[1]}/api/cfg"
    data = {
        "cpu_cores": int(cpu_cores),
        "cpu_freq": int(cpu_freq),
        "gpu_freq": int(gpu_freq),
        "mem_freq": int(memory_freq),
        "cl": int(cl)
    }
    headers = {'Authorization': sys.argv[3], 'Content-Type': 'application/json'}
    
    try:
        response = requests.post(url, json=data, headers=headers)
        if response.status_code == 201:
            av_dev = 0
            while True:
                t1 = time.time()
                metrics = get_result()
                elapsed = round(time.time() - t1, 3)
                if metrics:
                    metrics = [metrics[-1]]
                    requests.delete(f"{sys.argv[1]}/api/output", headers=headers)
                    return metrics, elapsed
                else:
                    av_dev += 1
                    print("Waiting for device...")
                    if av_dev == 60:
                        return "No Device", None
                    time.sleep(10)
        else:
            print(f"Error executing config: {response.status_code}")
    except requests.RequestException as e:
        print(f"Error executing config: {e}")
    return None, None

# def avg(val):
#     return sum(val)/len(val)

locked = False

def increment_target(outcome, target):
    global locked
    if not locked:
        if outcome == 'positive':
            target += 5
        else:
            target -= 5
    if target == 0 and not locked:
        target += 3
        locked = True
    return target

# Calculate reward with shaping
def calculate_reward(measured_metrics, target):
    power = measured_metrics[0]["power_cons"]
    throughput = measured_metrics[0]["throughput"]
    
    if int(throughput) <= target:
        return -(power/throughput * 1e6)
    
    return (throughput/power * 1e-6)

def rounded(x):
    integer_part = math.floor(x)
    fractional_part = x - integer_part
    return integer_part + 1 if fractional_part >= 0.5 else integer_part

# exploitation
def generate_neighbor(exist_configs, neighbor_configs, th_corr_conf, pwr_corr_conf, th, pwr, aside=False):
    global th_target, best_idx
    new_neighbors = []
    for exist_config, neighbor_config, range, th_conf, pwr_conf in zip(exist_configs, neighbor_configs, (CPU_CORES_RANGE, CPU_FREQ_RANGE, GPU_FREQ_RANGE, MEMORY_FREQ_RANGE, CL_RANGE), th_corr_conf, pwr_corr_conf):
        if th_conf > pwr_conf:
            corr_conf = th_conf
        else:
            corr_conf = pwr_conf
        if aside:
            home_l = neighbor_config
            home_h = exist_config
        else:
            home_l = exist_config
            home_h = neighbor_config
        if int(th[-1]) > th_target:
            new_neighbor = minmax(rounded(home_l - (abs(exist_config - neighbor_config) / 2) * corr_conf), range)
        else:
            new_neighbor = minmax(rounded(home_h + (abs(exist_config - neighbor_config) / 2) * corr_conf), range)
        new_neighbors.append(new_neighbor)
    if pwr[0] > min(pwr) and th[0] > th_target:
        new_neighbors[0] = 1
        new_neighbors[-1] = 3
    return tuple(new_neighbors)

# CSV saving optimization
def save_csv(dict_list, filename):
    with open(filename, 'a', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=['api_time','id', 'reward', 'phase', 'episode', 'xaviernx_time_elapsed', 'proposed_time_elapsed', 'cpu_cores', 'cpu_freq', 'gpu_freq', 'memory_freq', 'cl', 'elapsed', 'time_load', 'time_warm', 'time_c', 'th_target', 'throughput', 'power_cons', 'cpu_percent', 'gpu_percent', 'mem_percent'])
        if os.path.getsize(filename) == 0:
            writer.writeheader()
        for d in dict_list:
            writer.writerow(d)

def minmax(values, range):
    values = min(values, max(range))
    values = max(min(range), values)
    return int(values)

def distance_correlation(x, y):
    x, y = np.array(x), np.array(y)
    n = len(x)
    a = squareform(pdist(x.reshape(-1, 1)))
    b = squareform(pdist(y.reshape(-1, 1)))
    A = a - a.mean(axis=0)[None, :] - a.mean(axis=1)[:, None] + a.mean()
    B = b - b.mean(axis=0)[None, :] - b.mean(axis=1)[:, None] + b.mean()
    dcov = np.sqrt((A * B).sum() / (n**2))
    dvar_x = np.sqrt((A**2).sum() / (n**2))
    dvar_y = np.sqrt((B**2).sum() / (n**2))
    return dcov / np.sqrt(dvar_x * dvar_y)

print("PID", os.getpid())

# Configuration ranges for CPU, GPU, and memory
if sys.argv[5] == 'jxavier':
    CPU_CORES_RANGE = range(1, 6)
    CPU_FREQ_RANGE = range(1190, 1909)
    GPU_FREQ_RANGE = range(510, 1111)
    MEMORY_FREQ_RANGE = range(1500, 1867)
    CL_RANGE = range(1, 4)
    ranges = [
        (1, 6),
        (1190, 1909),
        (510, 1111),
        (1500, 1867),
        (1, 4)
    ]
elif sys.argv[5] == 'jorin-nano':
    CPU_CORES_RANGE = [5]
    CPU_FREQ_RANGE = range(806, 1511)
    GPU_FREQ_RANGE = range(306, 625)
    MEMORY_FREQ_RANGE = [2133]
    CL_RANGE = range(1, 4)
    ranges = [
        (5, 6),
        (806, 1511),
        (306, 625),
        (2133, 2134),
        (1, 4)
    ]

prohibited_configs = set()

sampled_configs = []

# Latin Hypercube Sampling to explore new states
def lhs_sampling(num_samples, ranges):
    lhs_samples = lhs(len(ranges), samples=num_samples)
    sampled_values = []
    for i, r in enumerate(ranges):
        sampled_values.append(lhs_samples[:, i] * (r[-1] - r[0]) + r[0])
    return np.array(sampled_values).T

def calculate_diversity(lhs_samples, config=None, condition=0):
    configurations = list(lhs_samples)

    if condition:
        target_idx = 0
    else:
        target_idx = -1
        if config:
            configurations.append(config)

    # Normalize configurations to [0, 1]
    scaler = MinMaxScaler()
    normalized_configs = scaler.fit_transform(np.array(configurations))

    target = normalized_configs[target_idx].reshape(1, -1)

    # Compute Manhattan distances between target and all configurations
    manhattan_distances = distance.cdist(target, normalized_configs, 'cityblock')[0]

    # Mask the self-distance (distance to itself) to avoid selecting it
    manhattan_distances[target_idx] = -np.inf

    # Find the index of the configuration with the maximum Manhattan distance
    max_distance_idx = np.argmax(manhattan_distances)

    # Get the corresponding configuration (original or normalized)
    max_distance_config = configurations[max_distance_idx]  # Original scale    
    return max_distance_config

# Generate LHS samples for the exploration phase
def generate_lhs_samples():
    num_samples = 9  # Number of samples per episode
    samples = lhs_sampling(num_samples, ranges)
    return [tuple(map(int, sample)) for sample in samples]

def count_trend(lst):
    increasing = 0
    decreasing = 0
    stable = 0
    prohibited = 0
    positive = 0
    
    for i in range(1, len(lst)):
        if lst[i] > lst[i-1]:
            increasing += 1
        elif lst[i] < lst[i-1]:
            decreasing += 1
        else:
            stable += 1
        if lst[i] < 1:
            prohibited += 1
        else:
            positive += 1
    
    return {"INC": increasing, "DEC": decreasing, "ST": stable, "-": prohibited, "+": positive}

stuck_count = 0
max_stuck_count = 5

def sampling(condition):
    global eps, stuck_count, sampled_configs, prohibited_configs, max_stuck_count
    t2 = time.time()
    lhs_samples = generate_lhs_samples()
    if condition:
        config = {"cpu_cores": max(CPU_CORES_RANGE), "cpu_freq": max(CPU_FREQ_RANGE), "gpu_freq": max(GPU_FREQ_RANGE), "memory_freq": max(MEMORY_FREQ_RANGE), "cl": max(CL_RANGE), "reward":0, "throughput":0, 'power_cons':-1, 'th_target': th_target}
        sampled_configs.append(config)
    else: # random hypercube
        home_dict = {k: v for k, v in sampled_configs[-1].items() if k != 'reward' and k != 'throughput' and k != 'power_cons' and k != 'th_target'}
        home_conf = tuple(home_dict.values())
        configs = calculate_diversity(lhs_samples, config=home_conf)
        config = {"cpu_cores": int(configs[0]), "cpu_freq": int(configs[1]), "gpu_freq": int(configs[2]), "memory_freq": int(configs[3]), "cl": int(configs[4]), "reward":0, "throughput":0, 'power_cons':-1, 'th_target': th_target}
        if config in sampled_configs:
            stuck_count += 1
            return "stuck"
        sampled_configs.append(config)

    cpu_cores, cpu_freq, gpu_freq, memory_freq, cl, _, _, _, _ = tuple(sampled_configs[-1].values())
    sampled_configs_checker = {k: v for k, v in sampled_configs[-1].items() if k != 'reward' and k != 'throughput' and k != 'power_cons'}
    
    if tuple(sampled_configs_checker.values()) in prohibited_configs or sampled_configs[-1]["power_cons"] != -1:
        return "stuck"

    t1 = time.time()
    measured_metrics, api_time = execute_config(cpu_cores, cpu_freq, gpu_freq, memory_freq, cl)
    elapsed_exec = round(time.time() - t1, 3)

    if isinstance(measured_metrics, list) or not measured_metrics:
        if not measured_metrics:
            print("EXECUTION PROBLEM!")
            return "stuck"
        elif measured_metrics[0]['power_cons'] == 0:
            print("EXECUTION PROBLEM!")
            return "stuck"
    if measured_metrics == "No Device":
        print("No Device/No Inference Runtime")
        return "stuck"
    
    sampled_configs[-1]["throughput"] = measured_metrics[0]["throughput"]
    sampled_configs[-1]["power_cons"] = measured_metrics[0]["power_cons"]

    reward = calculate_reward(measured_metrics, th_target)
    sampled_configs[-1]["reward"] = reward

    if reward < 0:
        print("PROHIBITED CONFIG!")
        prohibited_configs.add(tuple(sampled_configs_checker.values()))

    elapsed = round(((time.time() - t2) - elapsed_exec) * 1000, 3)

    configs = {
        "api_time": api_time,
        "reward": reward,
        "phase":"exploration",
        "episode": eps,
        "th_target": th_target,
        "xaviernx_time_elapsed": elapsed_exec,
        "proposed_time_elapsed": elapsed,
        "cpu_cores": cpu_cores+1,
        "cpu_freq": cpu_freq,
        "gpu_freq": gpu_freq,
        "memory_freq": memory_freq,
        "cl": cl
    }

    dict_record = [{**configs, **measured_metrics[0]}]
    save_csv(dict_record, f"proposed-{sys.argv[6]}_{sys.argv[5]}_{sys.argv[4]}.csv")
    rewards = [reward for reward in (sampled_config['reward'] for sampled_config in sampled_configs)]

    print(f"Episode: {eps}, Reward: {reward}, Max Reward: {max(rewards) if rewards else None}")
    eps += 1
    
sampling(1)
sampling(0)
max_trends_record = 5
visited = False
th_corr_conf_list = [1, 1, 1, 1, 1]
pwr_corr_conf_list = [1, 1, 1, 1, 1]
# backup_th_corr = th_corr_conf_list
# backup_pwr_corr = pwr_corr_conf_list

while eps <= (int(sys.argv[6])):
    t2 = time.time()
    rewards_dicts = [{idx:sampled_config['reward']} for idx, sampled_config in enumerate(sampled_configs) if sampled_config['reward'] != 0]
    items = sorted(rewards_dicts, key=lambda d: list(d.values())[0], reverse=True)
    filtered_configs = [sampled_config for sampled_config in sampled_configs if sampled_config["throughput"] != 0 and sampled_config["power_cons"] != -1]
    sorted_configs = sorted(filtered_configs, key=lambda x: x["reward"], reverse=True)
    th = [sampled_config["throughput"] for sampled_config in sorted_configs]
    pwr = [sampled_config["power_cons"] for sampled_config in sorted_configs]
    if len(items) > 1:
        second_best_item = items[1]
        second_best_idx = list(second_best_item.keys())[0]
        best_item = items[0]
        best_idx = list(best_item.keys())[0]
        home_dict = {k: v for k, v in sampled_configs[best_idx].items() if k != 'reward' and k != 'throughput' and k != 'power_cons' and k != 'th_target'}
        home_conf = tuple(home_dict.values())
        neig_dict = {k: v for k, v in sampled_configs[second_best_idx].items() if k != 'reward' and k != 'throughput' and k != 'power_cons' and k != 'th_target'}
        neig_conf = tuple(neig_dict.values())
        new_configs = generate_neighbor(home_conf, neig_conf, th_corr_conf_list, pwr_corr_conf_list, th, pwr)
        av_configs = [(sampled_config['cpu_cores'], sampled_config['cpu_freq'], sampled_config['gpu_freq'], sampled_config['memory_freq'], sampled_config['cl'], sampled_config['reward']) for sampled_config in sampled_configs]
        rewards = [list(reward)[0] for reward in (rewards_dict.values() for rewards_dict in rewards_dicts)]
        current_thtarget = th_target
        sorted_rewards = sorted(rewards, reverse=True)
        if (count_trend(rewards)['INC'] > count_trend(rewards)['DEC'] and count_trend(rewards)['-'] < count_trend(rewards)['+'] if len(rewards) >= max_trends_record else True):
            if len(rewards) >= max_trends_record+2:
                cores = [sampled_config["cpu_cores"] for sampled_config in sampled_configs if sampled_config["throughput"] != 0 and sampled_config["power_cons"] != -1]
                cpus = [sampled_config["cpu_freq"] for sampled_config in sampled_configs if sampled_config["throughput"] != 0 and sampled_config["power_cons"] != -1]
                gpus = [sampled_config["gpu_freq"] for sampled_config in sampled_configs if sampled_config["throughput"] != 0 and sampled_config["power_cons"] != -1]
                mems = [sampled_config["memory_freq"] for sampled_config in sampled_configs if sampled_config["throughput"] != 0 and sampled_config["power_cons"] != -1]
                _cls = [sampled_config["cl"] for sampled_config in sampled_configs if sampled_config["throughput"] != 0 and sampled_config["power_cons"] != -1]
                for i, x in enumerate([cpus, gpus, mems], start=1):
                    a = distance_correlation(x, th)
                    b = distance_correlation(x, pwr)
                    if not np.isnan(a) or not np.isnan(b):
                        th_corr_conf_list[i] = a
                        pwr_corr_conf_list[i] = b
                    else:
                        th_corr_conf_list[i] = 0
                        pwr_corr_conf_list[i] = 0
                # backup_th_corr = th_corr_conf_list
                # backup_pwr_corr = pwr_corr_conf_list
            if (count_trend(rewards)['ST'] > count_trend(rewards)['INC'] and len(rewards) >= max_trends_record+2 if len(rewards) >= max_trends_record else False):
                stuck_count += 1
                new_configs = generate_neighbor(home_conf, neig_conf, th_corr_conf_list, pwr_corr_conf_list, th, pwr, aside=True)
                check_config = [config[:-1] for config in av_configs if config[:-1] == new_configs]
                if (*new_configs, current_thtarget) in prohibited_configs or check_config:
                    print("Searching has no idea to search again, early stopping executed")
                    break
                    # max_trends_record = 5
                    # backup_sampled_configs = sampled_configs
                    # sampled_configs = [d for d in sampled_configs if d.get("reward") not in sorted_rewards[2:]]
                    # if not sampled_configs:
                    #     sampled_configs = backup_sampled_configs
                    # out = sampling(0)
                    # if out == 'stuck':
                    #     if stuck_count >= max_stuck_count:
                    #         print("Searching has no idea to search again, early stopping executed")
                    #         sampled_configs = backup_sampled_configs
                    #         break
                    # continue
            elif (count_trend(rewards)['ST'] < count_trend(rewards)['INC'] and len(rewards) >= max_trends_record+2 if len(rewards) >= max_trends_record else False):
                visited = True
                stuck_count = 0
                max_stuck_count*=2
                max_trends_record *= 2
            
            check_config = [config[:-1] for config in av_configs if config[:-1] == new_configs]

            if (*new_configs, current_thtarget) in prohibited_configs or check_config:
                stuck_count += 1
                new_configs = generate_neighbor(home_conf, neig_conf, th_corr_conf_list, pwr_corr_conf_list, th, pwr, aside=True)
                check_config = [config[:-1] for config in av_configs if config[:-1] == new_configs]
                if (*new_configs, current_thtarget) in prohibited_configs or check_config:
                    print("Searching has visited the prohibited/last config after sampling again, early stopping executed")
                    break
                    # max_trends_record = 5
                    # backup_sampled_configs = sampled_configs
                    # sampled_configs = [d for d in sampled_configs if d.get("reward") not in sorted_rewards[2:]]
                    # if not sampled_configs:
                    #     sampled_configs = backup_sampled_configs
                    # out = sampling(0)
                    # if out == 'stuck':
                    #     if stuck_count >= max_stuck_count:
                    #         print("Searching has visited the prohibited/last config after sampling again, early stopping executed")
                    #         sampled_configs = backup_sampled_configs
                    #         break
                    # continue
            elif not visited:
                stuck_count = 0
            else:
                visited = False
            
            cpu_cores, cpu_freq, gpu_freq, memory_freq, cl = tuple(new_configs)

            # Execute the chosen configuration and get metrics
            t1 = time.time()
            measured_metrics, api_time = execute_config(cpu_cores, cpu_freq, gpu_freq, memory_freq, cl)
            elapsed_exec = round(time.time() - t1, 3)

            if isinstance(measured_metrics, list) or not measured_metrics:
                if not measured_metrics:
                    print("EXECUTION PROBLEM!")
                    continue
                elif measured_metrics[0]['power_cons'] == 0:
                    print("EXECUTION PROBLEM!")
                    continue
            if measured_metrics == "No Device":
                print("No Device/No Inference Runtime")
                break

            # if int(measured_metrics[0]['throughput']) < int(sys.argv[7]) and cl != max(CL_RANGE):
            #     for i in range(4):
            #         th_corr_conf_list[i] = 0
            #         pwr_corr_conf_list[i] = 0
            #     th_corr_conf_list[-1] = 1
            #     pwr_corr_conf_list[-1] = 1
            # else:
            #     th_corr_conf_list = backup_th_corr
            #     pwr_corr_conf_list = backup_pwr_corr
            
            reward = calculate_reward(measured_metrics, current_thtarget)
            if (eps % 5) == 0:
                outcome = 'positive' if reward > 0 else 'negative'
                th_target = increment_target(outcome, th_target)
            dict_new_configs = {"cpu_cores": int(new_configs[0]), "cpu_freq": int(new_configs[1]), "gpu_freq": int(new_configs[2]), "memory_freq": int(new_configs[3]), "cl": new_configs[4], "reward":reward, "throughput":measured_metrics[0]["throughput"], "power_cons":measured_metrics[0]["power_cons"], "th_target": current_thtarget}
            if new_configs in av_configs[:-1]:
                target_item = next((d for d in rewards_dicts if list(d.values())[0] == av_configs[-1]), None)
                target_idx = list(target_item.keys())[0]
                sampled_configs[target_idx] = dict_new_configs
            else:
                sampled_configs.append(dict_new_configs)
            
            new_checker = {k: v for k, v in dict_new_configs.items() if k != 'reward' and k != 'throughput' and k != 'power_cons'}
            if reward < 0:
                print("PROHIBITED CONFIG!")
                prohibited_configs.add(tuple(new_checker.values()))

            elapsed = round(((time.time() - t2) - elapsed_exec) * 1000, 3)

            configs = {
                "api_time": api_time,
                "reward": reward,
                "phase":"exploitation",
                "episode": eps,
                "th_target": current_thtarget,
                "xaviernx_time_elapsed": elapsed_exec,
                "proposed_time_elapsed": elapsed,
                "cpu_cores": cpu_cores+1,
                "cpu_freq": cpu_freq,
                "gpu_freq": gpu_freq,
                "memory_freq": memory_freq,
                "cl": cl
            }

            dict_record = [{**configs, **measured_metrics[0]}]
            save_csv(dict_record, f"proposed-{sys.argv[6]}_{sys.argv[5]}_{sys.argv[4]}.csv")

            print(f"Episode: {eps}, Reward: {reward}, Max Reward: {max(rewards) if rewards else None}")
            eps += 1
        else:
            stuck_count += 1
            new_configs = generate_neighbor(home_conf, neig_conf, th_corr_conf_list, pwr_corr_conf_list, th, pwr, aside=True)
            check_config = [config[:-1] for config in av_configs if config[:-1] == new_configs]
            
            if (*new_configs, current_thtarget) in prohibited_configs or check_config:
                print("Searching has no idea to search again, early stopping executed")
                break
                # max_trends_record = 5
                # backup_sampled_configs = sampled_configs
                # sampled_configs = [d for d in sampled_configs if d.get("reward") not in sorted_rewards[2:]]
                # if not sampled_configs:
                #     sampled_configs = backup_sampled_configs
                # out = sampling(0)
                # if out == 'stuck':
                #     if stuck_count >= max_stuck_count:
                #         print("Searching has no idea to search again, early stopping executed")
                #         sampled_configs = backup_sampled_configs
                #         break
                # continue
            else:
                cpu_cores, cpu_freq, gpu_freq, memory_freq, cl = tuple(new_configs)

                # Execute the chosen configuration and get metrics
                t1 = time.time()
                measured_metrics, api_time = execute_config(cpu_cores, cpu_freq, gpu_freq, memory_freq, cl)
                elapsed_exec = round(time.time() - t1, 3)

                if isinstance(measured_metrics, list) or not measured_metrics:
                    if not measured_metrics:
                        print("EXECUTION PROBLEM!")
                        continue
                    elif measured_metrics[0]['power_cons'] == 0:
                        print("EXECUTION PROBLEM!")
                        continue
                if measured_metrics == "No Device":
                    print("No Device/No Inference Runtime")
                    break
                reward = calculate_reward(measured_metrics, current_thtarget)
                outcome = 'positive' if reward > 0 else 'negative'
                if (eps % 5) == 0:
                    th_target = increment_target(outcome, th_target)
                dict_new_configs = {"cpu_cores": int(new_configs[0]), "cpu_freq": int(new_configs[1]), "gpu_freq": int(new_configs[2]), "memory_freq": int(new_configs[3]), "cl": new_configs[4], "reward":reward, "throughput":measured_metrics[0]["throughput"], "power_cons":measured_metrics[0]["power_cons"], "th_target": current_thtarget}
                if new_configs in av_configs[:-1]:
                    target_item = next((d for d in rewards_dicts if list(d.values())[0] == av_configs[-1]), None)
                    target_idx = list(target_item.keys())[0]
                    sampled_configs[target_idx] = dict_new_configs
                else:
                    sampled_configs.append(dict_new_configs)
                
                new_checker = {k: v for k, v in dict_new_configs.items() if k != 'reward' and k != 'throughput' and k != 'power_cons'}
                if reward < 0:
                    print("PROHIBITED CONFIG!")
                    prohibited_configs.add(tuple(new_checker.values()))

                elapsed = round(((time.time() - t2) - elapsed_exec) * 1000, 3)

                configs = {
                    "api_time": api_time,
                    "reward": reward,
                    "phase":"exploitation",
                    "episode": eps,
                    "th_target": current_thtarget,
                    "xaviernx_time_elapsed": elapsed_exec,
                    "proposed_time_elapsed": elapsed,
                    "cpu_cores": cpu_cores+1,
                    "cpu_freq": cpu_freq,
                    "gpu_freq": gpu_freq,
                    "memory_freq": memory_freq,
                    "cl": cl
                }

                dict_record = [{**configs, **measured_metrics[0]}]
                save_csv(dict_record, f"proposed-{sys.argv[6]}_{sys.argv[5]}_{sys.argv[4]}.csv")

                print(f"Episode: {eps}, Reward: {reward}, Max Reward: {max(rewards) if rewards else None}")
                eps += 1
    else:
        sampling(0)

i = eps
set_target = 1

outcome = 'positive' if reward > 0 else 'negative'
th_target = increment_target(outcome, th_target)

#test 5 times
while i<30:
    t2 = time.time()
    filtered_configs = [sampled_config for sampled_config in sampled_configs if sampled_config["throughput"] != 0 and sampled_config["power_cons"] != -1]
    sorted_configs = sorted(filtered_configs, key=lambda x: x["reward"], reverse=True)
    th = [sampled_config["throughput"] for sampled_config in sorted_configs]
    pwr = [sampled_config["power_cons"] for sampled_config in sorted_configs]
    rewards_dicts = [{idx:sampled_config['reward']} for idx, sampled_config in enumerate(sampled_configs)]
    rewards = [list(reward)[0] for reward in (rewards_dict.values() for rewards_dict in rewards_dicts)]
    items = sorted(rewards_dicts, key=lambda d: list(d.values())[0], reverse=True)
    current_thtarget = th_target
    best_item = items[0]
    best_idx = list(best_item.keys())[0]
    configs = tuple(sampled_configs[best_idx].values())
    cpu_cores, cpu_freq, gpu_freq, memory_freq, cl, _, _, _, _ = configs
    new_configs = (cpu_cores, cpu_freq, gpu_freq, memory_freq, cl, th_target)

    if new_configs in prohibited_configs:
        second_best_item = items[1]
        second_best_idx = list(second_best_item.keys())[0]
        home_dict = {k: v for k, v in sampled_configs[best_idx].items() if k != 'reward' and k != 'throughput' and k != 'power_cons' and k != 'th_target'}
        home_conf = tuple(home_dict.values())
        neig_dict = {k: v for k, v in sampled_configs[second_best_idx].items() if k != 'reward' and k != 'throughput' and k != 'power_cons' and k != 'th_target'}
        neig_conf = tuple(neig_dict.values())
        new_configs = generate_neighbor(home_conf, neig_conf, th_corr_conf_list, pwr_corr_conf_list, th, pwr)
        cpu_cores, cpu_freq, gpu_freq, memory_freq, cl = tuple(new_configs)

        av_configs = [(sampled_config['cpu_cores'], sampled_config['cpu_freq'], sampled_config['gpu_freq'], sampled_config['memory_freq'], sampled_config['cl'], sampled_config['reward']) for sampled_config in sampled_configs]
        check_config = [config[:-1] for config in av_configs]
        if (*new_configs, th_target) in prohibited_configs and new_configs in check_config:
            target_idx = check_config.index(new_configs)
            sampled_configs[target_idx] = dict_new_configs
            new_configs = generate_neighbor(home_conf, neig_conf, th_corr_conf_list, pwr_corr_conf_list, th, pwr, aside=True)
            dict_new_configs = {"cpu_cores": int(new_configs[0]), "cpu_freq": int(new_configs[1]), "gpu_freq": int(new_configs[2]), "memory_freq": int(new_configs[3]), "cl": new_configs[4], "reward":reward, "throughput":measured_metrics[0]["throughput"], "power_cons":measured_metrics[0]["power_cons"], 'th_target': current_thtarget}
            cpu_cores, cpu_freq, gpu_freq, memory_freq, cl = tuple(new_configs)
            av_configs = [(sampled_config['cpu_cores'], sampled_config['cpu_freq'], sampled_config['gpu_freq'], sampled_config['memory_freq'], sampled_config['cl'], sampled_config['reward']) for sampled_config in sampled_configs]
            check_config = [config[:-1] for config in av_configs]
            if (*new_configs, th_target) in prohibited_configs and new_configs in check_config:
                target_idx = check_config.index(new_configs)
                sampled_configs[target_idx] = dict_new_configs
            else:
                sampled_configs.append(dict_new_configs)
        else:
            sampled_configs.append(dict_new_configs)

    # Execute the chosen configuration and get metrics
    t1 = time.time()
    measured_metrics, api_time = execute_config(cpu_cores, cpu_freq, gpu_freq, memory_freq, cl)
    elapsed_exec = round(time.time() - t1, 3)

    if isinstance(measured_metrics, list) or not measured_metrics:
        if not measured_metrics:
            print("EXECUTION PROBLEM!")
            continue
        elif measured_metrics[0]['power_cons'] == 0:
            print("EXECUTION PROBLEM!")
            continue
    if measured_metrics == "No Device":
        print("No Device/No Inference Runtime")
        break

    reward = calculate_reward(measured_metrics, current_thtarget)
    if set_target <= 2:
        outcome = 'positive' if reward > 0 else 'negative'
        phase = "exploitation"
        if set_target == 2:
            if outcome == 'positive':
                th_target = current_thtarget
            else:
                th_target = increment_target(outcome, th_target)
        else:
            th_target = increment_target(outcome, th_target)
        set_target += 1
    else:
        phase = "testing"
    dict_new_configs = {"cpu_cores": int(new_configs[0]), "cpu_freq": int(new_configs[1]), "gpu_freq": int(new_configs[2]), "memory_freq": int(new_configs[3]), "cl": new_configs[4], "reward":reward, "throughput":measured_metrics[0]["throughput"], "power_cons":measured_metrics[0]["power_cons"], "th_target": current_thtarget}
    sampled_configs[best_idx] = dict_new_configs

    new_checker = {k: v for k, v in dict_new_configs.items() if k != 'reward' and k != 'throughput' and k != 'power_cons'}
    if reward < 0:
        print("PROHIBITED CONFIG!")
        prohibited_configs.add(tuple(new_checker.values()))

    elapsed = round(((time.time() - t2) - elapsed_exec) * 1000, 3)

    configs = {
        "api_time": api_time,
        "th_target":current_thtarget,
        "reward": reward,
        "phase":phase,
        "episode": eps,
        "xaviernx_time_elapsed": elapsed_exec,
        "proposed_time_elapsed": elapsed,
        "cpu_cores": cpu_cores+1,
        "cpu_freq": cpu_freq,
        "gpu_freq": gpu_freq,
        "memory_freq": memory_freq,
        "cl": cl
    }

    dict_record = [{**configs, **measured_metrics[0]}]
    save_csv(dict_record, f"proposed-{sys.argv[6]}_{sys.argv[5]}_{sys.argv[4]}.csv")

    print(f"Episode: {eps}, Reward: {reward}, Max Reward: {max(rewards) if rewards else None}")
    eps += 1
    i+=1
else:
    pass
