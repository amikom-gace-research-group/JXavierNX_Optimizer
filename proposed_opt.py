import numpy as np
import sys
import time
import os
import csv
import requests

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
def calculate_reward(measured_metrics):
    power = measured_metrics[0]["power_cons"]
    throughput = measured_metrics[0]["throughput"]
    
    if power > POWER_BUDGET:
        return 1e-6
    
    return (throughput / power) * 1e6

def generate_neighbor(exist_configs, neighbor_configs):
    new_neighbor = []
    for exist_config, neighbor_config, range in zip(exist_configs, neighbor_configs, (CPU_CORES_RANGE, CPU_FREQ_RANGE, GPU_FREQ_RANGE, MEMORY_FREQ_RANGE, CL_RANGE)):
        new_neighbor.append(minmax(round(exist_config - abs(exist_config - neighbor_config) / 2), range))
    return tuple(new_neighbor)

# CSV saving optimization
def save_csv(dict_list, filename):
    with open(filename, 'a', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=['api_time','id', 'reward', 'phase', 'episode', 'xaviernx_time_elapsed', 'cpu_cores', 'cpu_freq', 'gpu_freq', 'memory_freq', 'cl', 'throughput', 'power_cons', 'cpu_percent', 'gpu_percent', 'mem_percent'])
        if os.path.getsize(filename) == 0:
            writer.writeheader()
        for d in dict_list:
            writer.writerow(d)

def minmax(values, range):
    values = min(values, max(range))
    values = max(min(range), values)
    return int(values)

def apply_configs(id):
    return minmax(sampled_configs[id]["cpu_cores"], CPU_CORES_RANGE), minmax(sampled_configs[id]["cpu_freq"], CPU_FREQ_RANGE), minmax(sampled_configs[id]["gpu_freq"], GPU_FREQ_RANGE), minmax(sampled_configs[id]["memory_freq"], MEMORY_FREQ_RANGE), minmax(sampled_configs[id]["cl"], CL_RANGE)

print("PID", os.getpid())

# Configuration ranges for CPU, GPU, and memory
if sys.argv[5] == 'jxavier':
    CPU_CORES_RANGE = range(1, 6)
    CPU_FREQ_RANGE = range(1190, 1909)
    GPU_FREQ_RANGE = range(510, 1111)
    MEMORY_FREQ_RANGE = range(1500, 1867)
    CL_RANGE = range(1, 4)
elif sys.argv[5] == 'jorin-nano':
    CPU_CORES_RANGE = [5]
    CPU_FREQ_RANGE = range(806, 1510)
    GPU_FREQ_RANGE = range(306, 624)
    MEMORY_FREQ_RANGE = range(1500, 2133)
    CL_RANGE = range(1, 3)

POWER_BUDGET = int(sys.argv[6])

prohibited_configs = set()

sampled_configs = []

# Stratified sampling: Select a subset of configurations
for cpu_cores in np.linspace(min(CPU_CORES_RANGE), max(CPU_CORES_RANGE), 3):
    for cpu_freq in np.linspace(min(CPU_FREQ_RANGE), max(CPU_FREQ_RANGE), 3):
        for gpu_freq in np.linspace(min(GPU_FREQ_RANGE), max(GPU_FREQ_RANGE), 3):
            for memory_freq in np.linspace(min(MEMORY_FREQ_RANGE), max(MEMORY_FREQ_RANGE), 3):
                for cl in CL_RANGE:
                    config = {"cpu_cores": int(cpu_cores), "cpu_freq": int(cpu_freq), "gpu_freq": int(gpu_freq), "memory_freq": int(memory_freq), "cl": cl}
                    sampled_configs.append(config)

# Calculate the indices for the quartiles
indices = np.percentile(range(len(sampled_configs)), [0, 25, 50, 75, 100])

# Convert indices to integers (since they represent positions in the list)
quartile_indices = [int(idx) for idx in indices]

initial_config_id = []

# filter for initial config
for i, idx in enumerate(quartile_indices):
    if i < 2:
        for k in [0, 1, 2]:
            initial_config_id.append(idx+k)
    elif 2 <= i <= 3:
        for k in [-1, 0, 1]:
            initial_config_id.append(idx+k)
    else:
        for k in [-2, -1, 0]:
            initial_config_id.append(idx+k)

rewards = []

for episode, ids in enumerate(initial_config_id):
    cpu_cores, cpu_freq, gpu_freq, memory_freq, cl = apply_configs(ids)

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

    reward = calculate_reward(measured_metrics)
    rewards.append(reward)

    if reward == 1e-6:
        print("PROHIBITED CONFIG!")
        prohibited_configs.add(apply_configs(ids))

    configs = {
        "api_time": api_time,
        "reward": reward,
        "phase":"exploration",
        "episode": episode+1,
        "xaviernx_time_elapsed": elapsed_exec,
        "cpu_cores": cpu_cores+1,
        "cpu_freq": cpu_freq,
        "gpu_freq": gpu_freq,
        "memory_freq": memory_freq,
        "cl": cl
    }

    dict_record = [{**configs, **measured_metrics[0]}]
    save_csv(dict_record, f"proposed_{sys.argv[5]}_{sys.argv[4]}.csv")

    print(f"Episode: {episode+1}, Reward: {reward}, Max Reward: {max(rewards) if rewards else None}")

final_reward = []
final_configs_id = []

for episode in range(len(initial_config_id), 100):
    if max(rewards) != 1e-6 or len(rewards) < 2:
        sorted_rewards = sorted(rewards, reverse=True)
        second_best_id = initial_config_id[rewards.index(sorted_rewards[1])]
        best_id = initial_config_id[rewards.index(max(rewards))]
        sorted_reward_id = sorted([rewards.index(max(rewards)), rewards.index(sorted_rewards[1])], reverse=True)
        sorted_neighbor_id = sorted([best_id, second_best_id], reverse=True)

        new_configs = generate_neighbor(apply_configs(sorted_neighbor_id[0]), apply_configs(sorted_neighbor_id[1]))

        if new_configs != apply_configs(best_id):
            if new_configs in prohibited_configs:
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

            reward = calculate_reward(measured_metrics)
            rewards.insert(sorted_reward_id[0], reward)

            if reward == 1e-6:
                print("PROHIBITED CONFIG!")
                prohibited_configs.add(new_configs)

            configs = {
                "api_time": api_time,
                "reward": reward,
                "phase":"exploitation",
                "episode": episode+1,
                "xaviernx_time_elapsed": elapsed_exec,
                "cpu_cores": cpu_cores+1,
                "cpu_freq": cpu_freq,
                "gpu_freq": gpu_freq,
                "memory_freq": memory_freq,
                "cl": cl
            }

            dict_record = [{**configs, **measured_metrics[0]}]
            save_csv(dict_record, f"proposed_{sys.argv[5]}_{sys.argv[4]}.csv")

            print(f"Episode: {episode+1}, Reward: {reward}, Max Reward: {max(rewards) if rewards else None}")
        else:
            final_reward.append(max(rewards))
            final_configs_id.append(best_id)
            rewards.pop(rewards.index(max(rewards)))
            initial_config_id.pop(rewards.index(max(rewards)))
        sampled_configs.insert(sorted_neighbor_id[0], {"cpu_cores": int(new_configs[0]), "cpu_freq": int(new_configs[1]), "gpu_freq": int(new_configs[2]), "memory_freq": int(new_configs[3]), "cl": new_configs[4]})
    else:
        if final_reward:
            best_id = final_configs_id[final_reward.index(max(final_reward))]
            configs = apply_configs(best_id)
            cpu_cores, cpu_freq, gpu_freq, memory_freq, cl = tuple(configs)

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

            reward = calculate_reward(measured_metrics)
            final_reward.insert(final_reward.index(max(final_reward)), reward)

            configs = {
                "api_time": api_time,
                "reward": reward,
                "phase":"exploitation",
                "episode": episode+1,
                "xaviernx_time_elapsed": elapsed_exec,
                "cpu_cores": cpu_cores+1,
                "cpu_freq": cpu_freq,
                "gpu_freq": gpu_freq,
                "memory_freq": memory_freq,
                "cl": cl
            }

            dict_record = [{**configs, **measured_metrics[0]}]
            save_csv(dict_record, f"proposed_{sys.argv[5]}_{sys.argv[4]}.csv")

            print(f"Episode: {episode+1}, Reward: {reward}, Max Reward: {max(final_reward) if final_reward else None}")