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
    CPU_FREQ_RANGE = range(806, 1510)
    GPU_FREQ_RANGE = range(306, 624)
    MEMORY_FREQ_RANGE = range(1500, 2133)
    CL_RANGE = range(1, 3)

sampled_configs = []

# Stratified sampling: Select a subset of configurations
for cpu_cores in np.linspace(min(CPU_CORES_RANGE), max(CPU_CORES_RANGE), 3):
    for cpu_freq in np.linspace(min(CPU_FREQ_RANGE), max(CPU_FREQ_RANGE), 3):  # Example: 3 CPU frequency strata
        for gpu_freq in np.linspace(min(GPU_FREQ_RANGE), max(GPU_FREQ_RANGE), 3):
            for memory_freq in np.linspace(min(MEMORY_FREQ_RANGE), max(MEMORY_FREQ_RANGE), 3):
                for cl in CL_RANGE:
                    config = {"cpu_cores": int(cpu_cores), "cpu_freq": int(cpu_freq), "gpu_freq": int(gpu_freq), "memory_freq": int(memory_freq), "cl": cl}
                    sampled_configs.append(config)

sampled_configs = pd.DataFrame(sampled_configs)

sampled_configs = sampled_configs.sort_values(
    by=["cpu_cores", "cpu_freq", "gpu_freq", "memory_freq", "cl"]
).reset_index(drop=True)

POWER_BUDGET = int(sys.argv[6])
best_throughput = -float('inf')

# SpeedUp and PowerUp table from the NeuOS algorithm
SpeedUp_PowerUp = [
    {"SpeedUp": 1.0, "PowerUp": 1.0},  # Baseline (default DVFS configuration)
    {"SpeedUp": 2.1, "PowerUp": 2},  # Moderate performance boost
    {"SpeedUp": 2.8, "PowerUp": 1.5}   # High performance, higher power consumption
]

# Function to find the row ID based on configuration values
def get_row_id(cpu_cores, cpu_freq, gpu_freq, memory_freq, cl):
    row = sampled_configs[
        (sampled_configs['cpu_cores'] == cpu_cores) &
        (sampled_configs['cpu_freq'] == cpu_freq) &
        (sampled_configs['gpu_freq'] == gpu_freq) &
        (sampled_configs['memory_freq'] == memory_freq) &
        (sampled_configs['cl'] == cl)
    ]
    return row.index[0] if not row.empty else None

def speedup_powerup_dvfs_selector(value, sampled_configs, configs):
    if lag < 0:
        configs_id = int(get_row_id(configs) * value)
        configs_id = min(configs_id, len(sampled_configs)-1)
        updated_configs = sampled_configs.iloc[configs_id]
        return updated_configs['cpu_cores'], updated_configs['cpu_freq'], updated_configs['gpu_freq'], updated_configs['memory_freq'], updated_configs['cl']
    else:
        configs_id = int(get_row_id(configs) * value)
        configs_id = max(get_row_id(configs) - abs(get_row_id(configs) - configs_id), 0)
        updated_configs = sampled_configs.iloc[configs_id]
        return updated_configs['cpu_cores'], updated_configs['cpu_freq'], updated_configs['gpu_freq'], updated_configs['memory_freq'], updated_configs['cl']
    
def calculate_lag(power, power_budget):
    return (power - power_budget) / power_budget

def delta_calculator(lag, power_consumed):
    required_speedup = 1 / (1 - lag)  # Calculate the required speedup to bring throughput closer to the target
    
    if lag < 0:
        # Loop through configurations, looking for the one that provides at least the required speedup
        for config in SpeedUp_PowerUp:
            if config["SpeedUp"] >= required_speedup and config["PowerUp"] * power_consumed <= POWER_BUDGET:
                return config
        return SpeedUp_PowerUp[-1]  # If no match, default to the highest performance config
    else:
        # Loop through configurations in reverse to find the one that slows down performance but stays within the budget
        for config in reversed(SpeedUp_PowerUp):
            if config["SpeedUp"] <= required_speedup and config["PowerUp"] * power_consumed <= POWER_BUDGET:
                return config
        return SpeedUp_PowerUp[0]  # If no match, default to the lowest performance config

def minmax(values, range):
    values = min(values, max(range))
    values = max(min(range), values)
    return values

# Function to apply the chosen DVFS configuration (adjust CPU, GPU, memory)
def apply_dvfs(config, throughput):
    global cpu_cores, cpu_freq, gpu_freq, memory_freq, cl, best_throughput
    # Dynamically adjust CPU cores and concurrency level (CL) based on throughput and power
    if lag < 0:
        if throughput < best_throughput:
            # Update frequencies based on SpeedUp configuration
            configs = cpu_cores, cpu_freq, gpu_freq, memory_freq, cl
            cpu_cores, cpu_freq, gpu_freq, memory_freq, cl = speedup_powerup_dvfs_selector(config["SpeedUp"], sampled_configs, *configs)
        else:
            best_throughput = throughput

    elif lag > 0:  # If system is ahead of throughput target, decrease resources to save power
        configs = cpu_cores, cpu_freq, gpu_freq, memory_freq, cl
        cpu_cores, cpu_freq, gpu_freq, memory_freq, cl = speedup_powerup_dvfs_selector(config["SpeedUp"], sampled_configs, *configs)
        
    return cpu_cores, cpu_freq, gpu_freq, memory_freq, cl

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
        "cpu_cores": minmax(cpu_cores, sampled_configs['cpu_cores']),
        "cpu_freq": minmax(cpu_freq, sampled_configs['cpu_freq']),
        "gpu_freq": minmax(gpu_freq, sampled_configs['gpu_freq']),
        "mem_freq": minmax(memory_freq, sampled_configs['memory_freq']),
        "cl": minmax(cl, sampled_configs['cl'])
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

# Main loop for NeuOS-based optimization (focus on throughput and power)
cpu_cores, cpu_freq, gpu_freq, memory_freq, cl = max(sampled_configs['cpu_cores']), max(sampled_configs['cpu_freq']), max(sampled_configs['gpu_freq']), max(sampled_configs['memory_freq']), max(sampled_configs['cl'])

last_lag = 0
best_lag = 0
best_power = POWER_BUDGET
last_power = 0
time_got = []
best_config = None

for episode in range(100):  # Example: run for 100 episodes
    # Get current metrics (throughput and power)
    t1 = time.time()
    measured_metrics, api_time = execute_config(cpu_cores, cpu_freq, gpu_freq, memory_freq, cl)

    elapsed_exec = round(time.time() - t1, 3)
    if not measured_metrics:
        print("EXECUTION PROBLEM!")
        continue
    if measured_metrics == "No Device":
        print("No Device/No Inference Runtime")
        break
    
    # Update Kalman Filters with current metrics
    throughput = measured_metrics[0]['throughput']
    power_consumed = measured_metrics[0]['power_cons']

    # Calculate LAG based on throughput
    lag = calculate_lag(power_consumed, POWER_BUDGET)
    
    # Choose DVFS configuration based on LAG and power consumption
    dvfs_config = delta_calculator(lag, power_consumed)

    elapsed = round(((time.time() - elapsed_exec) - t1) * 1000, 3)
    time_got.append(elapsed+elapsed_exec)

    configs = {
	    "api_time": api_time,
        "episode": episode,
        "infer_overhead (sec)" : elapsed_exec,
        "neuos_overhead (ms)" : elapsed,
        "lag": lag,
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
    
    # Apply the DVFS configuration
    cpu_cores, cpu_freq, gpu_freq, memory_freq, cl = apply_dvfs(dvfs_config, throughput)
    
    save_csv([configs], f"neuos_jxavier_{sys.argv[4]}.csv")
    # Log the results
    print(f"Configs: {configs}")

    # Check if the system meets the target and stop if it stabilizes
    if lag < best_lag and power_consumed <= best_power:
        best_lag = lag
        best_power = power_consumed
        best_config = configs

    last_lag = lag
    last_power = power_consumed

print(f"Best Config: {best_config} in {sum(time_got)} sec")
