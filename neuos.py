import numpy as np
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

POWER_BUDGET = int(sys.argv[6])
THROUGHPUT_TARGET = int(sys.argv[7])

# SpeedUp and PowerUp table from the NeuOS algorithm
SpeedUp_PowerUp = [
    {"SpeedUp": 0.8, "PowerUp": 0.9},  # Speed down, less power
    {"SpeedUp": 1.0, "PowerUp": 1.0},  # Baseline (default DVFS configuration)
    {"SpeedUp": 1.5, "PowerUp": 1.2},  # Moderate performance boost
    {"SpeedUp": 2.0, "PowerUp": 1.5}   # High performance, higher power consumption
]

# Function to calculate LAG (how much the system is ahead/behind the throughput target)
def calculate_lag(throughput, throughput_target):
    return (throughput_target - throughput) / throughput_target

def delta_calculator(lag, power_consumed):
    required_speedup = 1 / (1 - lag)  # Calculate the required speedup to bring throughput closer to the target
    
    if lag < 0:  # System is behind throughput target (speed up)
        # Loop through configurations, looking for the one that provides at least the required speedup
        for config in SpeedUp_PowerUp:
            if config["SpeedUp"] >= required_speedup and config["PowerUp"] * power_consumed <= POWER_BUDGET:
                return config
        return SpeedUp_PowerUp[-1]  # If no match, default to the highest performance config
    else:  # System is ahead of throughput target (slow down)
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
def apply_dvfs(config):
    global cpu_cores, cpu_freq, gpu_freq, memory_freq, cl
    # Dynamically adjust CPU cores and concurrency level (CL) based on throughput and power
    if lag < 0:  # If system is behind throughput target, increase resources if within power budget
        if power_consumed <= POWER_BUDGET:
            # Update frequencies based on SpeedUp configuration
            cpu_freq = min(int(cpu_freq * config["SpeedUp"]), max(CPU_FREQ_RANGE))
            gpu_freq = min(int(gpu_freq * config["SpeedUp"]), max(GPU_FREQ_RANGE))
            memory_freq = min(int(memory_freq * config["SpeedUp"]), max(MEMORY_FREQ_RANGE))
            if cpu_cores < max(CPU_CORES_RANGE):
                cpu_cores += 1
            if cl < max(CL_RANGE):
                cl += 1
        else:  # Exceeding power budget, lower frequencies
            cpu_freq = max((int(cpu_freq) - abs(int(cpu_freq) - int(cpu_freq * config["PowerUp"]))), min(CPU_FREQ_RANGE))
            gpu_freq = max((int(gpu_freq) - abs(int(gpu_freq) - int(gpu_freq * config["PowerUp"]))), min(GPU_FREQ_RANGE))
            memory_freq = max((int(memory_freq) - abs(int(memory_freq) - int(memory_freq * config["PowerUp"]))), min(MEMORY_FREQ_RANGE))
            if cpu_cores > min(CPU_CORES_RANGE):
                cpu_cores -= 1
            if cl > min(CL_RANGE):
                cl -= 1
    elif lag > 0:  # If system is ahead of throughput target, decrease resources to save power
        if power_consumed > POWER_BUDGET:
            cpu_freq = max((int(cpu_freq) - abs(int(cpu_freq) - int(cpu_freq * config["PowerUp"]))), min(CPU_FREQ_RANGE))
            gpu_freq = max((int(gpu_freq) - abs(int(gpu_freq) - int(gpu_freq * config["PowerUp"]))), min(GPU_FREQ_RANGE))
            memory_freq = max((int(memory_freq) - abs(int(memory_freq) - int(memory_freq * config["PowerUp"]))), min(MEMORY_FREQ_RANGE))
            if cpu_cores > min(CPU_CORES_RANGE):
                cpu_cores -= 1
            if cl > min(CL_RANGE):
                cl -= 1
        else:
            cpu_freq = min(int(cpu_freq * config["SpeedUp"]), max(CPU_FREQ_RANGE))
            gpu_freq = min(int(gpu_freq * config["SpeedUp"]), max(GPU_FREQ_RANGE))
            memory_freq = min(int(memory_freq * config["SpeedUp"]), max(MEMORY_FREQ_RANGE))
            if cpu_cores < max(CPU_CORES_RANGE):
                cpu_cores += 1
            if cl < max(CL_RANGE):
                cl += 1
        
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
        "cpu_cores": minmax(cpu_cores, CPU_CORES_RANGE),
        "cpu_freq": minmax(cpu_freq, CPU_FREQ_RANGE),
        "gpu_freq": minmax(gpu_freq, GPU_FREQ_RANGE),
        "mem_freq": minmax(memory_freq, MEMORY_FREQ_RANGE),
        "cl": minmax(cl, CL_RANGE)
    }
    headers = {'Authorization': sys.argv[3], 'Content-Type': 'application/json'}
    
    try:
        response = requests.post(url, json=data, headers=headers)
        if response.status_code == 201:
            av_dev = 0
            while True:
                metrics = get_result()
                if metrics:
                    metrics = [metrics[-1]]
                    requests.delete(f"{sys.argv[1]}/api/output", headers=headers)
                    return metrics
                else:
                    av_dev += 1
                    print("Waiting for device...")
                    if av_dev == 30:
                        return "No Device"
                    time.sleep(10)
        else:
            print(f"Error executing config: {response.status_code}")
    except requests.RequestException as e:
        print(f"Error executing config: {e}")
    return None

# CSV saving optimization
def save_csv(dict_list, filename):
    with open(filename, 'a', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=['episode', 'infer_overhead (sec)', 'neuos_overhead (ms)', 'lag', 'throughput_target', 'power_budget', 'cpu_cores', 'cpu_freq', 'gpu_freq', 'memory_freq', 'cl', 'throughput', 'power'])
        if os.path.getsize(filename) == 0:
            writer.writeheader()
        for d in dict_list:
            writer.writerow(d)

# Main loop for NeuOS-based optimization (focus on throughput and power)
cpu_cores, cpu_freq, gpu_freq, memory_freq, cl = max(CPU_CORES_RANGE), max(CPU_FREQ_RANGE), max(GPU_FREQ_RANGE), max(MEMORY_FREQ_RANGE), max(CL_RANGE)

last_lag = 0
best_lag = 0
best_power = POWER_BUDGET
last_power = 0
time_got = []
best_config = None
max_saturated_count = 10

for episode in range(100):  # Example: run for 100 episodes
    # Get current metrics (throughput and power)
    t1 = time.time()
    measured_metrics = execute_config(cpu_cores, cpu_freq, gpu_freq, memory_freq, cl)

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
    lag = calculate_lag(throughput, THROUGHPUT_TARGET)
    
    # Choose DVFS configuration based on LAG and power consumption
    dvfs_config = delta_calculator(lag, power_consumed)

    elapsed = round(((time.time() - elapsed_exec) - t1) * 1000, 3)
    time_got.append(elapsed+elapsed_exec)

    configs = {
        "episode": episode,
        "infer_overhead (sec)" : elapsed_exec,
        "neuos_overhead (ms)" : elapsed,
        "lag": lag,
        "throughput_target": THROUGHPUT_TARGET,
        "power_budget": POWER_BUDGET,
        "cpu_cores": cpu_cores+1,
        "cpu_freq": cpu_freq,
        "gpu_freq": gpu_freq,
        "memory_freq": memory_freq,
        "cl": cl,
        "throughput": throughput,
        "power": power_consumed
    }
    
    # Apply the DVFS configuration
    cpu_cores, cpu_freq, gpu_freq, memory_freq, cl = apply_dvfs(dvfs_config)
    
    save_csv([configs], f"neuos_jxavier_{sys.argv[4]}.csv")
    # Log the results
    print(f"Configs: {configs}")

    if abs(last_lag - lag) < 0.01 and abs(power_consumed - last_power) < 500:
        max_saturated_count -= 1
        if max_saturated_count == 0:
            print("NeuOS is saturated")
            break
    else:
        max_saturated_count = 5
        
    # Check if the system meets the target and stop if it stabilizes
    if lag < best_lag and power_consumed <= best_power:
        best_lag = lag
        best_power = power_consumed
        best_config = configs

    last_lag = lag
    last_power = power_consumed

print(f"Best Config: {best_config} in {sum(time_got)} sec")
