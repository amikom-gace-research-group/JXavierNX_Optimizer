import numpy as np
import sys
import time
import os
import csv
import requests
import random
import yaml
from pyDOE import lhs

eps = 1

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

# Calculate reward with shaping
def calculate_reward(measured_metrics, power_budget):
    power = measured_metrics[0]["power_cons"]
    throughput = measured_metrics[0]["throughput"]
    
    if power > power_budget or throughput < int(sys.argv[7]):
        return (throughput / power) * 1e-6
    
    return (throughput / power) * 1e6

# exploitation
def generate_neighbor(exist_configs, neighbor_configs, th_corr_conf, pwr_corr_conf):
    new_neighbors = []
    for exist_config, neighbor_config, range, th_conf, pwr_conf in zip(exist_configs, neighbor_configs, (CPU_CORES_RANGE, CPU_FREQ_RANGE, GPU_FREQ_RANGE, MEMORY_FREQ_RANGE, CL_RANGE), th_corr_conf, pwr_corr_conf):
        if th_conf > pwr_conf:
            corr_conf = th_conf
        else:
            corr_conf = pwr_conf
        if exist_config > neighbor_config:
            new_neighbor = minmax(round(exist_config - ((exist_config - neighbor_config) / 2) * corr_conf), range) 
        elif exist_config < neighbor_config:
            new_neighbor = minmax(round(exist_config + (abs(exist_config - neighbor_config) / 2) * corr_conf), range)
        elif exist_config == neighbor_config:
            new_neighbor = exist_config
        new_neighbors.append(new_neighbor)
    return tuple(new_neighbors)

# CSV saving optimization
def save_csv(dict_list, filename):
    with open(filename, 'a', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=['api_time','id', 'reward', 'phase', 'power_budget', 'episode', 'xaviernx_time_elapsed', 'cpu_cores', 'cpu_freq', 'gpu_freq', 'memory_freq', 'cl', 'throughput', 'power_cons', 'cpu_percent', 'gpu_percent', 'mem_percent'])
        if os.path.getsize(filename) == 0:
            writer.writeheader()
        for d in dict_list:
            writer.writerow(d)

def minmax(values, range):
    values = min(values, max(range))
    values = max(min(range), values)
    return int(values)

def pearson_correlation(x, y):
    # Check if the input lists are of the same length
    if len(x) != len(y):
        raise ValueError("The input lists must be of the same length.")
    
    n = len(x)
    # Check if the input lists are empty
    if n == 0:
        raise ValueError("The input lists cannot be empty.")
    
    # Calculate the means of x and y
    mean_x = sum(x) / n
    mean_y = sum(y) / n
    
    # Initialize variables to accumulate the sums
    covariance = 0.0
    variance_x = 0.0
    variance_y = 0.0
    
    # Iterate through the data points to compute the necessary sums
    for xi, yi in zip(x, y):
        diff_x = xi - mean_x
        diff_y = yi - mean_y
        covariance += diff_x * diff_y
        variance_x += diff_x ** 2
        variance_y += diff_y ** 2
    
    # Check for zero variance to avoid division by zero
    if variance_x == 0 or variance_y == 0:
        return float('nan')
    
    # Calculate the Pearson correlation coefficient
    denominator = (variance_x * variance_y) ** 0.5
    return covariance / denominator

def apply_configs(id):
    return minmax(sampled_configs[id]["cpu_cores"], CPU_CORES_RANGE), minmax(sampled_configs[id]["cpu_freq"], CPU_FREQ_RANGE), minmax(sampled_configs[id]["gpu_freq"], GPU_FREQ_RANGE), minmax(sampled_configs[id]["memory_freq"], MEMORY_FREQ_RANGE), minmax(sampled_configs[id]["cl"], CL_RANGE)

print("PID", os.getpid())

with open(f'{sys.argv[5]}_{sys.argv[4]}.yml', 'r') as file:
    corr_conf_dict = yaml.safe_load(file)

th_corr_conf_list = corr_conf_dict['th_conf']
pwr_corr_conf_list = corr_conf_dict['pwr_conf']

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
    low_pwr = corr_conf_dict['low_pwr']
    high_pwr = corr_conf_dict['high_pwr']
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
    low_pwr = corr_conf_dict['low_pwr']
    high_pwr = corr_conf_dict['high_pwr']

POWER_BUDGET = int(sys.argv[6]) if low_pwr <= int(sys.argv[6]) <= high_pwr else high_pwr

prohibited_configs = set()

sampled_configs = []

# Latin Hypercube Sampling to explore new states
def lhs_sampling(num_samples, ranges):
    lhs_samples = lhs(len(ranges), samples=num_samples)
    sampled_values = []
    for i, r in enumerate(ranges):
        sampled_values.append(lhs_samples[:, i] * (r[-1] - r[0]) + r[0])
    return np.array(sampled_values).T

def calculate_diversity(lhs_samples, state_key, tau=1.0, max_diversity_score=500):
    """
    Select a configuration for exploration based on diversity scores.

    Args:
        lhs_samples: List of LHS-sampled configurations.
        state_key: Current state (for reference, though not directly used here).
        tau: Temperature parameter for softmax scaling (higher = more uniform).
        max_diversity_score: Maximum value for diversity scores to prevent overflow.

    Returns:
        selected_action: The selected action for exploration.
    """
    # Calculate diversity scores: Higher score for configurations far from each other
    diversity_scores = []
    for sample in lhs_samples:
        diversity_score = sum(abs(np.array(sample) - np.array(state_key))) if state_key else 1
        diversity_scores.append(diversity_score)

    # Clip diversity scores to prevent overflow
    diversity_scores = np.array(diversity_scores)
    diversity_scores = np.clip(diversity_scores, None, max_diversity_score)

    # Convert diversity scores to probabilities using softmax
    exp_scores = np.exp(diversity_scores / tau)

    # Prevent division by zero by adding a small epsilon value
    exp_scores_sum = np.sum(exp_scores)
    if exp_scores_sum == 0:
        exp_scores_sum = 1e-6  # Small value to avoid division by zero

    probabilities = exp_scores / exp_scores_sum

    # Handle any potential NaN values in probabilities
    if np.any(np.isnan(probabilities)):
        probabilities = np.ones_like(probabilities) / len(lhs_samples)  # Default uniform distribution

    # Select an action based on probabilities
    selected_action = lhs_samples[np.random.choice(len(lhs_samples), p=probabilities)]
    
    return selected_action

# Generate LHS samples for the exploration phase
def generate_lhs_samples():
    num_samples = 9  # Number of samples per episode
    samples = lhs_sampling(num_samples, ranges)
    return [tuple(map(int, sample)) for sample in samples]

def count_trend(lst):
    increasing = 0
    decreasing = 0
    stable = 0
    
    for i in range(1, len(lst)):
        if lst[i] > lst[i-1]:
            increasing += 1
        elif lst[i] < lst[i-1]:
            decreasing += 1
        else:
            stable += 1
    
    return {"INC": increasing, "DEC": decreasing, "ST": stable}

def sampling(condition):
    global eps, sampled_configs, prohibited_configs, POWER_BUDGET
    if condition:
        for cpu_cores, cpu_freq, gpu_freq, memory_freq, cl in [(min(CPU_CORES_RANGE), min(CPU_FREQ_RANGE), min(GPU_FREQ_RANGE), min(MEMORY_FREQ_RANGE), min(CL_RANGE)), (max(CPU_CORES_RANGE), max(CPU_FREQ_RANGE), max(GPU_FREQ_RANGE), max(MEMORY_FREQ_RANGE), max(CL_RANGE))]:
            config = {"cpu_cores": int(cpu_cores), "cpu_freq": int(cpu_freq), "gpu_freq": int(gpu_freq), "memory_freq": int(memory_freq), "cl": cl, "reward":0, "power_budget":POWER_BUDGET, "throughput":0, 'power_cons':-1}
            if config in sampled_configs:
                return "stuck"
            sampled_configs.append(config)
    else: # random hypercube
        lhs_samples = generate_lhs_samples()
        power_budget = random.choice(POWER_BUDGET)
        rewards_dicts = [{idx:sampled_config['reward']} for idx, sampled_config in enumerate(sampled_configs) if sampled_config['reward'] != 0]
        items = sorted(rewards_dicts, key=lambda d: list(d.values())[0], reverse=True)
        best_item = items[0]
        best_idx = list(best_item.keys())[0]
        home_dict = {k: v for k, v in sampled_configs[best_idx].items() if k != 'reward' and k != 'throughput' and k != 'power_cons' and k != 'power_budget'}
        home_conf = tuple(home_dict.values())
        configs = calculate_diversity(lhs_samples, home_conf)
        config = {"cpu_cores": int(configs[0]), "cpu_freq": int(configs[1]), "gpu_freq": int(configs[2]), "memory_freq": int(configs[3]), "cl": int(configs[4]), "reward":0, "power_budget":power_budget, "throughput":0, 'power_cons':-1}
        if config in sampled_configs:
            stuck_count += 1
            return "stuck"
        sampled_configs.append(config)

    for ids in sampled_configs:
        
        cpu_cores, cpu_freq, gpu_freq, memory_freq, cl, _, _, _, _ = tuple(ids.values())
        ids_checker = {k: v for k, v in ids.items() if k != 'reward' and k != 'throughput' and k != 'power_cons'}
        
        if tuple(ids_checker.values()) in prohibited_configs or ids["power_cons"] != -1:
            continue

        t1 = time.time()
        measured_metrics, api_time = execute_config(cpu_cores, cpu_freq, gpu_freq, memory_freq, cl)
        elapsed_exec = round(time.time() - t1, 3)

        ids["throughput"] = measured_metrics[0]["throughput"]
        ids["power_cons"] = measured_metrics[0]["power_cons"]

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

        reward = calculate_reward(measured_metrics, ids["power_budget"])
        ids["reward"] = reward

        if reward < 1:
            print("PROHIBITED CONFIG!")
            prohibited_configs.add(tuple(ids_checker.values()))

        reward = calculate_reward(measured_metrics, ids["power_budget"])

        configs = {
            "api_time": api_time,
            "reward": reward,
            "phase":"exploration",
            "power_budget": ids["power_budget"],
            "episode": eps,
            "xaviernx_time_elapsed": elapsed_exec,
            "cpu_cores": cpu_cores+1,
            "cpu_freq": cpu_freq,
            "gpu_freq": gpu_freq,
            "memory_freq": memory_freq,
            "cl": cl
        }

        dict_record = [{**configs, **measured_metrics[0]}]
        save_csv(dict_record, f"test-proposed_{sys.argv[5]}_{sys.argv[4]}.csv")
        rewards = [reward for reward in (sampled_config['reward'] for sampled_config in sampled_configs)]

        print(f"Episode: {eps}, Reward: {reward}, Max Reward: {max(rewards) if rewards else None}")
        eps += 1
    

sampling(1)

while eps <= 5:
    rewards_dicts = [{idx:sampled_config['reward']} for idx, sampled_config in enumerate(sampled_configs) if sampled_config['reward'] != 0]
    rewards = [list(reward)[0] for reward in (rewards_dict.values() for rewards_dict in rewards_dicts)]
    items = sorted(rewards_dicts, key=lambda d: list(d.values())[0], reverse=True)
    if len(items) > 1:
        second_best_item = items[1]
        second_best_idx = list(second_best_item.keys())[0]
    best_item = items[0]
    best_idx = list(best_item.keys())[0]
    home_dict = {k: v for k, v in sampled_configs[best_idx].items() if k != 'reward' and k != 'throughput' and k != 'power_cons'}
    home_conf = tuple(home_dict.values())
    neig_dict = {k: v for k, v in sampled_configs[second_best_idx].items() if k != 'reward' and k != 'throughput' and k != 'power_cons'}
    neig_conf = tuple(neig_dict.values())
    new_configs = generate_neighbor(home_conf[:-1], neig_conf[:-1], th_corr_conf_list, pwr_corr_conf_list)
    home_checker = tuple([v for k, v in sampled_configs[best_idx].items() if k != 'reward' and k != 'throughput' and k != 'power_cons'])
    neig_checker = tuple([v for k, v in sampled_configs[second_best_idx].items() if k != 'reward' and k != 'throughput' and k != 'power_cons'])

    if home_checker in prohibited_configs and neig_checker in prohibited_configs:
        out = sampling(0)
        if out == 'stuck':
            print("Home and neighbor has visited the prohibited config after sampling again, early stopping executed")
            break
        continue
    elif (*new_configs, POWER_BUDGET) in prohibited_configs:
        out = sampling(0)
        if out == 'stuck':
            print("Searching has visited the prohibited config after sampling again, early stopping executed")
            break
        continue
    
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
    
    reward = calculate_reward(measured_metrics, POWER_BUDGET)
    dict_new_configs = {"cpu_cores": int(new_configs[0]), "cpu_freq": int(new_configs[1]), "gpu_freq": int(new_configs[2]), "memory_freq": int(new_configs[3]), "cl": new_configs[4], "reward":reward, "power_budget": POWER_BUDGET, "throughput":measured_metrics[0]["throughput"], "power_cons":measured_metrics[0]["power_cons"]}
    av_configs = [(sampled_config['cpu_cores'], sampled_config['cpu_freq'], sampled_config['gpu_freq'], sampled_config['memory_freq'], sampled_config['cl'], sampled_config['reward'], sampled_config['power_budget']) for sampled_config in sampled_configs]
    if new_configs in av_configs[:-2] and av_configs[-1] == POWER_BUDGET:
        target_item = next((d for d in rewards_dicts if list(d.values())[0] == av_configs[-2]), None)
        target_idx = list(target_item.keys())[0]
        sampled_configs[target_idx] = dict_new_configs
    else:
        sampled_configs.append(dict_new_configs)
    
    new_checker = {k: v for k, v in dict_new_configs.items() if k != 'reward' and k != 'throughput' and k != 'power_cons'}
    if reward < 1:
        print("PROHIBITED CONFIG!")
        prohibited_configs.add(tuple(new_checker.values()))

    configs = {
        "api_time": api_time,
        "reward": reward,
        "phase":"exploitation",
        "power_budget": POWER_BUDGET,
        "episode": eps,
        "xaviernx_time_elapsed": elapsed_exec,
        "cpu_cores": cpu_cores+1,
        "cpu_freq": cpu_freq,
        "gpu_freq": gpu_freq,
        "memory_freq": memory_freq,
        "cl": cl
    }

    dict_record = [{**configs, **measured_metrics[0]}]
    save_csv(dict_record, f"test-proposed_{sys.argv[5]}_{sys.argv[4]}.csv")

    print(f"Episode: {eps}, Reward: {reward}, Max Reward: {max(rewards) if rewards else None}")
    eps += 1

i = 0
#test 5 times
while i<6:
    rewards_dicts = [{idx:sampled_config['reward']} for idx, sampled_config in enumerate(sampled_configs) if sampled_config['power_budget'] == POWER_BUDGET and sampled_config['reward'] > 1]
    if not rewards_dicts:
        continue
    rewards = [list(reward)[0] for reward in (rewards_dict.values() for rewards_dict in rewards_dicts)]
    items = sorted(rewards_dicts, key=lambda d: list(d.values())[0], reverse=True)
    best_item = items[0]
    best_idx = list(best_item.keys())[0]
    configs = tuple(sampled_configs[best_idx].values())
    cpu_cores, cpu_freq, gpu_freq, memory_freq, cl, _, _, _, _ = configs
    new_configs = (cpu_cores, cpu_freq, gpu_freq, memory_freq, cl, POWER_BUDGET)

    if new_configs in prohibited_configs:
        print("optimization searcher failed to search the best configuration :(")
        break

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

    reward = calculate_reward(measured_metrics, POWER_BUDGET)
    dict_new_configs = {"cpu_cores": int(new_configs[0]), "cpu_freq": int(new_configs[1]), "gpu_freq": int(new_configs[2]), "memory_freq": int(new_configs[3]), "cl": new_configs[4], "reward":reward, "power_budget": POWER_BUDGET, "throughput":measured_metrics[0]["throughput"], "power_cons":measured_metrics[0]["power_cons"]}
    sampled_configs[best_idx] = dict_new_configs

    new_checker = {k: v for k, v in dict_new_configs.items() if k != 'reward' and k != 'throughput' and k != 'power_cons'}
    if reward < 1:
        print("PROHIBITED CONFIG!")
        prohibited_configs.add(tuple(new_checker.values()))

    configs = {
        "api_time": api_time,
        "reward": reward,
        "phase":"post-training",
        "power_budget": POWER_BUDGET,
        "episode": eps,
        "xaviernx_time_elapsed": elapsed_exec,
        "cpu_cores": cpu_cores+1,
        "cpu_freq": cpu_freq,
        "gpu_freq": gpu_freq,
        "memory_freq": memory_freq,
        "cl": cl
    }

    dict_record = [{**configs, **measured_metrics[0]}]
    save_csv(dict_record, f"test-proposed_{sys.argv[5]}_{sys.argv[4]}.csv")

    print(f"Episode: {eps}, Reward: {reward}, Max Reward: {max(rewards) if rewards else None}")
    eps += 1
    i+=1