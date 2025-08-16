import numpy as np
import pandas as pd
import os
import time
import requests
import sys
import csv

# Configuration ranges for CPU, GPU, and memory
if sys.argv[5] == 'jxavier':
    CPU_CORES_RANGE = range(1, 6)
    CPU_FREQ_RANGE = range(1190, 1909)
    GPU_FREQ_RANGE = range(510, 1111)
    MEMORY_FREQ_RANGE = range(1500, 1867)
    CL_RANGE = range(1, 4)
elif sys.argv[5] == 'jorin-nano':
    CPU_CORES_RANGE = [5]
    CPU_FREQ_RANGE = range(806, 1511)
    GPU_FREQ_RANGE = range(306, 625)
    MEMORY_FREQ_RANGE = [2133]
    CL_RANGE = range(1, 4)

sampled_configs = []

# Stratified sampling: Select a subset of configurations
for cpu_cores in CPU_CORES_RANGE:
    for cpu_freq in np.linspace(min(CPU_FREQ_RANGE), max(CPU_FREQ_RANGE), 3):  # Example: 3 CPU frequency strata
        for gpu_freq in np.linspace(min(GPU_FREQ_RANGE), max(GPU_FREQ_RANGE), 3):
            for memory_freq in np.linspace(min(MEMORY_FREQ_RANGE), max(MEMORY_FREQ_RANGE), 3):
                for cl in CL_RANGE:
                    config = {"cpu_cores": int(cpu_cores), "cpu_freq": int(cpu_freq), "gpu_freq": int(gpu_freq), "memory_freq": int(memory_freq), "cl": cl, 
                              "throughput": 0, "power": 1e+10, "cpu_percent": 0, "gpu_percent": 0, "mem_percent": 0}
                    sampled_configs.append(config)

chosen_dvfs = {'1': [], 'D':[]}

POWER_BUDGET = int(sys.argv[6])

def minmax(values, range):
    values = min(values, max(range))
    values = max(min(range), values)
    return int(values)

def generate_neighbor(exist_configs, neighbor_configs):
    new_neighbors = []
    for exist_config, neighbor_config, range in zip(exist_configs, neighbor_configs, (CPU_CORES_RANGE, CPU_FREQ_RANGE, GPU_FREQ_RANGE, MEMORY_FREQ_RANGE, CL_RANGE)):
        if exist_config > neighbor_config:
            new_neighbor = minmax(round(exist_config - (exist_config - neighbor_config) / 2), range)
        else:
            new_neighbor = minmax(round(exist_config + abs(exist_config - neighbor_config) / 2), range)
        new_neighbors.append(new_neighbor)
    return new_neighbors

def select_dvfs(df_prof, episode, proposed=0):
    baseline_dvfs = df_prof["cpu_cores"].min(), df_prof["cpu_freq"].min(), df_prof["gpu_freq"].min(), df_prof["memory_freq"].min(), df_prof["cl"].min()
    chosen_dvfs['1'] = list(baseline_dvfs)
    baseline = df_prof[(df_prof["cpu_cores"] == baseline_dvfs[0]) & (df_prof["cpu_freq"] == baseline_dvfs[1]) & (df_prof["gpu_freq"] == baseline_dvfs[2]) & (df_prof["memory_freq"] == baseline_dvfs[3]) & (df_prof["cl"] == baseline_dvfs[4])]
    baseline_power = round(baseline["power"].iloc[0])
    dynamic = df_prof[(df_prof["power"] < (baseline_power * dynamic_powerup(baseline_power))) & (df_prof["power"] > baseline_power)]
    if not dynamic.empty:
        throughput_max = dynamic["throughput"].max()
        dynamic_best = dynamic[dynamic["throughput"] >= throughput_max]
        dynamic_dvfs = dynamic_best["cpu_cores"].min(), dynamic_best["cpu_freq"].min(), dynamic_best["gpu_freq"].min(), dynamic_best["memory_freq"].min(), dynamic_best["cl"].min()
        chosen_dvfs['D'] = list(dynamic_dvfs)
        if proposed and episode < 75:
            if episode < 50:
                dynamic_sec = df_prof[df_prof["power"] >= (baseline_power * dynamic_powerup(baseline_power))]
                if not dynamic_sec.empty:
                    throughput_sec = dynamic_sec["throughput"].max()
                    dynamic_sec = dynamic_sec[dynamic_sec["throughput"] >= throughput_sec]
                    dynamic_sec_dvfs = dynamic_sec["cpu_cores"].min(), dynamic_sec["cpu_freq"].min(), dynamic_sec["gpu_freq"].min(), dynamic_sec["memory_freq"].min(), dynamic_sec["cl"].min()
                    new_dynamic = generate_neighbor(dynamic_dvfs, dynamic_sec_dvfs)
                    chosen_dvfs['D'] = new_dynamic
            else:
                throughput_sec = dynamic['throughput'].nlargest(2).iloc[-1]
                dynamic_sec = dynamic[dynamic["throughput"] >= throughput_sec]
                dynamic_sec_dvfs = dynamic_sec["cpu_cores"].min(), dynamic_sec["cpu_freq"].min(), dynamic_sec["gpu_freq"].min(), dynamic_sec["memory_freq"].min(), dynamic_sec["cl"].min()
                new_dynamic = generate_neighbor(dynamic_dvfs, dynamic_sec_dvfs)
                chosen_dvfs['D'] = new_dynamic

def check_config(configs, df_prof):
    cpu_cores = configs[0] in df_prof["cpu_cores"].tolist()
    cpu_freq = configs[1] in df_prof["cpu_freq"].tolist()
    gpu_freq = configs[2] in df_prof["gpu_freq"].tolist()
    memory_freq = configs[3] in df_prof["memory_freq"].tolist()
    cl = configs[4] in df_prof["cl"].tolist()
    return (cpu_cores and cpu_freq and gpu_freq and memory_freq and cl)

def update_output(throughput, power, configs, df_prof, proposed=0):
    if proposed and not check_config(configs, df_prof):
        df_prof["cpu_cores"] = configs[0]
        df_prof["cpu_freq"] = configs[1]
        df_prof["gpu_freq"] = configs[2]
        df_prof["memory_freq"] = configs[3]
        df_prof["cl"] = configs[4]
    df_prof["throughput"][(df_prof["cpu_cores"] == configs[0]) & (df_prof["cpu_freq"] == configs[1]) & (df_prof["gpu_freq"] == configs[2]) & (df_prof["memory_freq"] == configs[3]) & (df_prof["cl"] == configs[4])] = throughput
    df_prof["power"][(df_prof["cpu_cores"] == configs[0]) & (df_prof["cpu_freq"] == configs[1]) & (df_prof["gpu_freq"] == configs[2]) & (df_prof["memory_freq"] == configs[3]) & (df_prof["cl"] == configs[4])] = power

def dynamic_powerup(power_consumed):
    return POWER_BUDGET / power_consumed

# Retrieve the result from the system API
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
                    if av_dev == 30:
                        return "No Device", None
                    time.sleep(10)
        else:
            print(f"Error executing config: {response.status_code}")
    except requests.RequestException as e:
        print(f"Error executing config: {e}")
    return None, None

# CSV saving optimization
def save_csv(dict_list, filename):
    with open(filename, 'a', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=['api_time','episode', 'infer_overhead (sec)', 'neuos_overhead (ms)', 'lag', 'power_budget', 'cpu_cores', 'cpu_freq', 'gpu_freq', 'memory_freq', 'cl', 'throughput', 'power', 'cpu_percent', 'gpu_percent', 'mem_percent'])
        if os.path.getsize(filename) == 0:
            writer.writeheader()
        for d in dict_list:
            writer.writerow(d)

def execute_runtime(num_episodes=100):
    df_prof = pd.DataFrame(sampled_configs)
    select_dvfs(df_prof, 0, proposed=int(sys.argv[8]))
   
    cpu_cores, cpu_freq, gpu_freq, memory_freq, cl = chosen_dvfs['1']

    best_power = POWER_BUDGET
    time_got = []
    best_config = None
    conf = 0

    for episode in range(num_episodes):  # Example: run for 100 episodes
        # Get current metrics (throughput and power)
        t1 = time.time()
        measured_metrics, api_time = execute_config(cpu_cores, cpu_freq, gpu_freq, memory_freq, cl)

        elapsed_exec = round(time.time() - t1, 3)
        if isinstance(measured_metrics, list) or not measured_metrics:
            if not measured_metrics:
                print("EXECUTION PROBLEM!")
                conf += 1
                config = sampled_configs[conf]
                cpu_cores, cpu_freq, gpu_freq, memory_freq, cl = config["cpu_cores"], config["cpu_freq"], config["gpu_freq"], config["memory_freq"], config["cl"]
                continue
            elif measured_metrics[0]['power_cons'] == 0:
                print("EXECUTION PROBLEM!")
                conf += 1
                config = sampled_configs[conf]
                cpu_cores, cpu_freq, gpu_freq, memory_freq, cl = config["cpu_cores"], config["cpu_freq"], config["gpu_freq"], config["memory_freq"], config["cl"]
                continue
        if measured_metrics == "No Device":
            print("No Device/No Inference Runtime")
            break
        
        # Update Kalman Filters with current metrics
        throughput = measured_metrics[0]['throughput']
        power_consumed = measured_metrics[0]['power_cons']

        confs = cpu_cores, cpu_freq, gpu_freq, memory_freq, cl
        update_output(throughput, power_consumed, confs, df_prof)
        select_dvfs(df_prof, episode, proposed=int(sys.argv[8]))

        elapsed = round(((time.time() - elapsed_exec) - t1) * 1000, 3)
        time_got.append(elapsed+elapsed_exec)

        configs = {
            "api_time": api_time,
            "episode": episode,
            "infer_overhead (sec)" : elapsed_exec,
            "neuos_overhead (ms)" : elapsed,
            "power_budget": POWER_BUDGET,
            "cpu_cores": cpu_cores+1,
            "cpu_freq": cpu_freq,
            "gpu_freq": gpu_freq,
            "memory_freq": memory_freq,
            "cl": cl,
            "throughput": throughput,
            "power": power_consumed,
            "cpu_percent": measured_metrics[0]["cpu_percent"],
            "gpu_percent": measured_metrics[0]["gpu_percent"],
            "mem_percent": measured_metrics[0]["mem_percent"]
        }

        if episode < round((int(sys.argv[7])/100)*len(sampled_configs)):
            conf += 1
            config = sampled_configs[conf]
            cpu_cores, cpu_freq, gpu_freq, memory_freq, cl = config["cpu_cores"], config["cpu_freq"], config["gpu_freq"], config["memory_freq"], config["cl"]
        else:
            if chosen_dvfs["D"] and power_consumed <= POWER_BUDGET:
                cpu_cores, cpu_freq, gpu_freq, memory_freq, cl = chosen_dvfs["D"]
            else:
                cpu_cores, cpu_freq, gpu_freq, memory_freq, cl = chosen_dvfs['1']

        if int(sys.argv[8]):
            file = f"neuos-online-proposed-{sys.argv[7]}_scaled_{sys.argv[5]}_{sys.argv[4]}.csv"
        else:
            file = f"neuos-online-{sys.argv[7]}_scaled_{sys.argv[5]}_{sys.argv[4]}.csv"
        
        save_csv([configs], file)
        # Log the results
        print(f"Configs: {configs}")

        # Check if the system meets the target and stop if it stabilizes
        if power_consumed <= best_power:
            best_power = power_consumed
            best_config = configs

    print(f"Best Config: {best_config} in {sum(time_got)} sec")

if __name__ == "__main__":
    execute_runtime()