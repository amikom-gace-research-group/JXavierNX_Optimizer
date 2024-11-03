import numpy as np
import random
import sys
import time
import os
import csv
import requests
from skopt import gp_minimize
from skopt.space import Integer, Real, Categorical
from skopt.utils import use_named_args

print("PID", os.getpid())

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

importance_power = 1
importance_throughput = 1

# Hyperparameters for Bayesian Optimization
n_calls = 100  # Number of iterations for Bayesian Optimization
n_initial_points = 10

time_got = []

last_rewards = []  # To store recent rewards for saturation check
MAX_SATURATION_CALLS = 5  # Number of calls to check for saturation
episode_counter = 0

cores_space = (Categorical(CPU_CORES_RANGE, name='cpu_cores') if len(CPU_CORES_RANGE) == 1 else Integer(min(CPU_CORES_RANGE), max(CPU_CORES_RANGE), name='cpu_cores'))

# Define the parameter space for Bayesian Optimization
space = [
    cores_space,
    Integer(min(CPU_FREQ_RANGE), max(CPU_FREQ_RANGE), name='cpu_freq'),
    Integer(min(GPU_FREQ_RANGE), max(GPU_FREQ_RANGE), name='gpu_freq'),
    Integer(min(MEMORY_FREQ_RANGE), max(MEMORY_FREQ_RANGE), name='mem_freq'),
    Integer(min(CL_RANGE), max(CL_RANGE), name='cl')
]

# Function to get the result from the external system
def get_result():
    headers = {
        'Authorization': sys.argv[2],  # Use 'APIKey' if your service requires this
        'Content-Type': 'application/json'  # Set content type to JSON
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
        "cpu_cores": str(cpu_cores),
        "cpu_freq": str(cpu_freq),
        "gpu_freq": str(gpu_freq),
        "mem_freq": str(memory_freq),
        "cl": str(cl)
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

# Reward function based on power and throughput metrics
# Efficient reward calculation
def calculate_reward(measured_metrics):
    power = measured_metrics[0]["power_cons"]
    throughput = measured_metrics[0]["throughput"]
    
    if power > POWER_BUDGET or throughput < THROUGHPUT_TARGET:
        return 1e6
    
    return (importance_throughput * (throughput / THROUGHPUT_TARGET) +
            importance_power * (POWER_BUDGET / power))

# CSV saving optimization
def save_csv(dict_list, filename):
    with open(filename, 'a', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=['id', 'episode', 'infer_time', 'cpu_cores', 'cpu_freq', 'gpu_freq', 'mem_freq', 'cl', 'throughput_target', 'power_budget', 'throughput', 'power_cons'])
        if os.path.getsize(filename) == 0:
            writer.writeheader()
        for d in dict_list:
            writer.writerow(d)

# The objective function for Bayesian Optimization
@use_named_args(space)
def objective(cpu_cores, cpu_freq, gpu_freq, mem_freq, cl):
    global episode_counter
    print(f"Testing configuration: CPU Cores={cpu_cores+1}, CPU Freq={cpu_freq}, GPU Freq={gpu_freq}, Mem Freq={mem_freq}, CL={cl}")
    
    t1 = time.time()
    measured_metrics = execute_config(cpu_cores, cpu_freq, gpu_freq, mem_freq, cl)

    elapsed = round(time.time() - t1, 3)
    time_got.append(elapsed)
    
    if not measured_metrics or measured_metrics == "No Device":
        print("No device detected. Raising an exception to stop optimization.")
        raise RuntimeError("No device detected. Stopping optimization.")  # Raise exception to stop the optimizer

    configs = {
        "episode" : episode_counter,
        "infer_time": elapsed,
        "cpu_cores": int(cpu_cores) + 1,
        "cpu_freq": int(cpu_freq),
        "gpu_freq": int(gpu_freq),
        "mem_freq": int(mem_freq),
        "cl": int(cl),
        "throughput_target": THROUGHPUT_TARGET,
        "power_budget": POWER_BUDGET,
    }
    result = {**configs, **measured_metrics[0]}
    save_csv([result], f"bo_{sys.argv[5]}_{sys.argv[4]}.csv")
    
    reward = calculate_reward(measured_metrics)
    print(f"Configuration reward: {reward}")
    
    if reward == 1e6:
        return reward  # Return penalty for invalid config

    episode_counter += 1
    last_rewards.append(reward)
    
    # Check if optimization is saturated
    if len(last_rewards) > MAX_SATURATION_CALLS and all(r == last_rewards[-1] for r in last_rewards[-MAX_SATURATION_CALLS:]):
        print("Optimization is saturated. Stopping further iterations.")
        raise RuntimeError("Optimization saturated.")  # Raising an exception to stop optimization

    return -reward  # Minimize the negative reward to maximize reward

# Main Optimization Loop
try:
    t2 = time.time()
    res = gp_minimize(objective, space, n_calls=n_calls, random_state=42, n_initial_points=n_initial_points)
    # Run Bayesian Optimization
    elapsed = round(((time.time() - sum(time_got)) - t2) * 1000, 3)
    elapsed_total = round(time.time() - t2, 3)

    # Output the best found configuration
    best_params = dict(zip(['cpu_cores', 'cpu_freq', 'gpu_freq', 'mem_freq', 'cl'], res.x))
    print(f"Best configuration found: {best_params} in {elapsed} ms for BO and total time is took {elapsed_total}")
except RuntimeError as e:
    print(e)  # Handle exception messages
