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

# Calculate reward with shaping
def calculate_reward(measured_metrics, balanced=1):
    power = measured_metrics[0]["power_cons"]
    throughput = measured_metrics[0]["throughput"]
    
    if power > POWER_BUDGET:
        return (power * 1e-6) / throughput
    
    return (throughput / (power if balanced else 1)) * 1e6

# exploitation
def generate_neighbor(exist_configs, neighbor_configs):
    new_neighbors = []
    for exist_config, neighbor_config, range in zip(exist_configs, neighbor_configs, (CPU_CORES_RANGE, CPU_FREQ_RANGE, GPU_FREQ_RANGE, MEMORY_FREQ_RANGE, CL_RANGE)):
        if exist_config > neighbor_config:
            new_neighbor = minmax(round(exist_config - abs(exist_config - neighbor_config) / 2), range)
        else:
            new_neighbor = minmax(round(exist_config + abs(exist_config - neighbor_config) / 2), range)
        new_neighbors.append(new_neighbor)
    return tuple(new_neighbors)

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
    CPU_CORES_RANGE = np.linspace(1, 6, 3)
    CPU_FREQ_RANGE = range(1190, 1909)
    GPU_FREQ_RANGE = range(510, 1111)
    MEMORY_FREQ_RANGE = range(1500, 1867)
    CL_RANGE = range(1, 4)
elif sys.argv[5] == 'jorin-nano':
    CPU_CORES_RANGE = [5]
    CPU_FREQ_RANGE = range(806, 1510)
    GPU_FREQ_RANGE = range(306, 624)
    MEMORY_FREQ_RANGE = range(1500, 2133)
    CL_RANGE = range(1, 4)

POWER_BUDGET = int(sys.argv[6])

prohibited_configs = set()

sampled_configs = []

# Stratified sampling: Select a subset of configurations
for cpu_cores in CPU_CORES_RANGE:
    for cpu_freq in np.linspace(min(CPU_FREQ_RANGE), max(CPU_FREQ_RANGE), 3):
        for gpu_freq in np.linspace(min(GPU_FREQ_RANGE), max(GPU_FREQ_RANGE), 3):
            for memory_freq in np.linspace(min(MEMORY_FREQ_RANGE), max(MEMORY_FREQ_RANGE), 3):
                for cl in CL_RANGE:
                    config = {"cpu_cores": int(cpu_cores), "cpu_freq": int(cpu_freq), "gpu_freq": int(gpu_freq), "memory_freq": int(memory_freq), "cl": cl}
                    sampled_configs.append(config)

initial_config_id = []

for cpu_cores in CPU_CORES_RANGE:
    for cpu_freq, gpu_freq, memory_freq in zip(np.linspace(min(CPU_FREQ_RANGE), max(CPU_FREQ_RANGE), 3), np.linspace(min(GPU_FREQ_RANGE), max(GPU_FREQ_RANGE), 3),  np.linspace(min(MEMORY_FREQ_RANGE), max(MEMORY_FREQ_RANGE), 3)):
        for cl in CL_RANGE:
            config = {"cpu_cores": int(cpu_cores), "cpu_freq": int(cpu_freq), "gpu_freq": int(gpu_freq), "memory_freq": int(memory_freq), "cl": cl}
            idx = sampled_configs.index(config)
            initial_config_id.append({str(idx):[]})

for episode, ids in enumerate(initial_config_id):
    cpu_cores, cpu_freq, gpu_freq, memory_freq, cl = apply_configs(int(list(ids.keys())[0]))

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

    reward = calculate_reward(measured_metrics, balanced=int(sys.argv[7]))
    ids[list(ids.keys())[0]].append(reward)

    if reward < 1:
        print("PROHIBITED CONFIG!")
        prohibited_configs.add(apply_configs(int(list(ids.keys())[0])))

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

    if int(sys.argv[7]):
        mode = "balanced"
    else:
        mode = "max"

    dict_record = [{**configs, **measured_metrics[0]}]
    save_csv(dict_record, f"proposed-{mode}_{sys.argv[5]}_{sys.argv[4]}.csv")
    rewards = [list(initial_config_id[i].values())[0][0] for i in range(len(initial_config_id)) if list(initial_config_id[i].values())[0]]

    print(f"Episode: {episode+1}, Reward: {reward}, Max Reward: {max(rewards) if rewards else None}")

final_configs_id = []

max_episode = 100
exploration_eps = len(initial_config_id) + 1

while exploration_eps <= max_episode:
    rewards = [list(initial_config_id[i].values())[0][0] for i in range(len(initial_config_id)) if list(initial_config_id[i].values())[0]]
    best_id = int(next((key for d in initial_config_id for key, value in d.items() if value[0] == max(rewards)), '0'))
    if max(rewards) > 1 and len(initial_config_id) >= 2 and exploration_eps < 75:
        sorted_rewards = sorted(rewards, reverse=True)
        if exploration_eps < 50:
            second_best_id = int(next((key for d in initial_config_id for key, value in d.items() if value[0] == min(rewards)), '0'))
        else:
            second_best_id = int(next((key for d in initial_config_id for key, value in d.items() if value[0] == sorted_rewards[1]), '0'))

        new_configs = generate_neighbor(apply_configs(best_id), apply_configs(second_best_id))
        dict_new_configs = {"cpu_cores": int(new_configs[0]), "cpu_freq": int(new_configs[1]), "gpu_freq": int(new_configs[2]), "memory_freq": int(new_configs[3]), "cl": new_configs[4]}
        if dict_new_configs not in sampled_configs:
            sampled_configs.append(dict_new_configs)

        if str(sampled_configs.index(dict_new_configs)) not in [list(initial_config_id[i].keys())[0] for i in range(len(initial_config_id))] and str(sampled_configs.index(dict_new_configs)) not in [list(final_configs_id[i].keys())[0] for i in range(len(final_configs_id)) if final_configs_id]:
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

            reward = calculate_reward(measured_metrics, balanced=int(sys.argv[7]))
            initial_config_id.append({str(sampled_configs.index(dict_new_configs)):[reward]})

            if reward < 1:
                print("PROHIBITED CONFIG!")
                prohibited_configs.add(new_configs)

            configs = {
                "api_time": api_time,
                "reward": reward,
                "phase":"exploitation",
                "episode": exploration_eps,
                "xaviernx_time_elapsed": elapsed_exec,
                "cpu_cores": cpu_cores+1,
                "cpu_freq": cpu_freq,
                "gpu_freq": gpu_freq,
                "memory_freq": memory_freq,
                "cl": cl
            }

            if int(sys.argv[7]):
                mode = "balanced"
            else:
                mode = "max"

            dict_record = [{**configs, **measured_metrics[0]}]
            save_csv(dict_record, f"proposed-{mode}_{sys.argv[5]}_{sys.argv[4]}.csv")

            print(f"Episode: {exploration_eps}, Reward: {reward}, Max Reward: {max(rewards) if rewards else None}")
            exploration_eps += 1
        else:
            if exploration_eps < 50:
                initial_config_id = [d for d in initial_config_id if str(second_best_id) not in d]
            else:
                if str(best_id) not in [list(final_configs_id[i].keys())[0] for i in range(len(final_configs_id))]:
                    final_configs_id.append({str(best_id):[max(rewards)]})
                initial_config_id = [d for d in initial_config_id if str(best_id) not in d]

    else:
        if str(best_id) not in [list(final_configs_id[i].keys())[0] for i in range(len(final_configs_id))]:
            final_configs_id.append({str(best_id):[max(rewards)]})
        rewards = [list(final_configs_id[i].values())[0][0] for i in range(len(final_configs_id)) if list(final_configs_id[i].values())[0]]
        if max(rewards) > 1:
            best_id = int(next((key for d in final_configs_id for key, value in d.items() if value[0] == max(rewards)), '0'))
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

            reward = calculate_reward(measured_metrics, balanced=int(sys.argv[7]))
            next((d.update({str(best_id): [reward]}) for d in final_configs_id if str(best_id) in d), None)

            configs = {
                "api_time": api_time,
                "reward": reward,
                "phase":"post-training",
                "episode": exploration_eps,
                "xaviernx_time_elapsed": elapsed_exec,
                "cpu_cores": cpu_cores+1,
                "cpu_freq": cpu_freq,
                "gpu_freq": gpu_freq,
                "memory_freq": memory_freq,
                "cl": cl
            }

            if int(sys.argv[7]):
                mode = "balanced"
            else:
                mode = "max"

            dict_record = [{**configs, **measured_metrics[0]}]
            save_csv(dict_record, f"proposed-{mode}_{sys.argv[5]}_{sys.argv[4]}.csv")

            print(f"Episode: {exploration_eps}, Reward: {reward}, Max Reward: {max(rewards) if rewards else None}")
            exploration_eps += 1