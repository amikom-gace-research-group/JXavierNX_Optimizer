import numpy as np
import random
import sys
import time
import os
import csv
import requests
from skopt import gp_minimize
from skopt.space import Integer
from skopt.utils import use_named_args

# Define configuration ranges
CPU_CORES_RANGE = range(1, 6)
CPU_FREQ_RANGE = range(1190, 1909)
GPU_FREQ_RANGE = range(510, 1111)
MEMORY_FREQ_RANGE = range(1500, 1867)
CL_RANGE = range(1, 4)

# Constants and thresholds
POWER_BUDGET = 5000
THROUGHPUT_TARGET = 30
importance_power = 1
importance_throughput = 1

# Hyperparameters for Bayesian Optimization
n_calls = 10  # First use BO for global exploration

# SARSA Hyperparameters
alpha = 0.1
gamma = 0.9
epsilon = 0.6
epsilon_min = 0.01
epsilon_decay = 0.995
num_episodes = 10
reward_threshold = 0.01

# Define the action space for SARSA
ACTIONS = [0, 1, 2, 3, 4, 5, 6]  # No change, small/medium/large increase/decrease

# Action mapping
ACTION_MAPPING = ['cpu_cores', 'cpu_freq', 'gpu_freq', 'memory_freq', 'cl']
action_shape = [len(ACTIONS)] * len(ACTION_MAPPING)

last_rewards = []  # To store recent rewards for saturation check
MAX_SATURATION_CALLS = 5  # Number of calls to check for saturation

# Step sizes for adjustment
STEP_SIZES = {
    'cpu_cores': (1, 3, 5),
    'cpu_freq': (1, 10, 50),
    'gpu_freq': (1, 10, 50),
    'memory_freq': (1, 10, 50),
    'cl': (1, 2, 0)
}

# SARSA Q-table initialization
Q_table = {}
prohibited_configs = set()

last_reward = 0
max_reward = float('inf')
best_config = None

# Define the parameter space for Bayesian Optimization
space = [
    Integer(min(CPU_CORES_RANGE), max(CPU_CORES_RANGE), name='cpu_cores'),
    Integer(min(CPU_FREQ_RANGE), max(CPU_FREQ_RANGE), name='cpu_freq'),
    Integer(min(GPU_FREQ_RANGE), max(GPU_FREQ_RANGE), name='gpu_freq'),
    Integer(min(MEMORY_FREQ_RANGE), max(MEMORY_FREQ_RANGE), name='mem_freq'),
    Integer(min(CL_RANGE), max(CL_RANGE), name='cl')
]

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

# Retrieve Q-value for a state-action pair
def get_q_value(state_index, action_index):
    state_key = tuple(state_index)
    if state_key not in Q_table:
        Q_table[state_key] = np.zeros(np.prod(action_shape))  # Initialize if not present
    return Q_table[state_key][np.ravel_multi_index(action_index, action_shape)]

# Update Q-value for a state-action pair
def update_q_value(state_index, action_index, new_value):
    state_key = tuple(state_index)
    if state_key not in Q_table:
        Q_table[state_key] = np.zeros(np.prod(action_shape))  # Initialize if not present
    Q_table[state_key][np.ravel_multi_index(action_index, action_shape)] = new_value

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

# Adjust value based on action
def adjust_value(value, action, steps, min_val, max_val):
    small_step, medium_step, large_step = steps
    if action == 1:
        return min(value + small_step, max_val)
    elif action == 2:
        return max(value - small_step, min_val)
    elif action == 3:
        return min(value + medium_step, max_val)
    elif action == 4:
        return max(value - medium_step, min_val)
    elif action == 5:
        return min(value + large_step, max_val)
    elif action == 6:
        return max(value - large_step, min_val)
    return value  # No change

# SARSA Epsilon-greedy action selection
def choose_action(state_index):
    if random.uniform(0, 1) < epsilon:
        return [random.choice(ACTIONS) for _ in range(len(ACTION_MAPPING))]  # Explore
    else:
        state_key = tuple(state_index)
        if state_key not in Q_table:
            return [random.choice(ACTIONS) for _ in range(len(ACTION_MAPPING))]  # Explore if unseen state
        return np.unravel_index(np.argmax(Q_table[state_key]), action_shape)  # Exploit

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

# Reward function
def calculate_reward(measured_metrics):
    power = measured_metrics[0]["power_cons"]
    throughput = measured_metrics[0]["throughput"]
    
    if power > POWER_BUDGET or throughput < THROUGHPUT_TARGET:
        return 1e6  # Large penalty

    return (importance_throughput * (throughput / THROUGHPUT_TARGET) +
            importance_power * (POWER_BUDGET / power))

# CSV saving optimization
def save_csv(dict_list, filename):
    with open(filename, 'a', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=['id', 'reward', 'xaviernx_time_elapsed', 'bosarsa_time_elapsed', 'cpu_cores', 'cpu_freq', 'gpu_freq', 'mem_freq', 'cl', 'throughput', 'power_cons'])
        if os.path.getsize(filename) == 0:
            writer.writeheader()
        for d in dict_list:
            writer.writerow(d)

# The objective function for Bayesian Optimization
@use_named_args(space)
def objective(cpu_cores, cpu_freq, gpu_freq, mem_freq, cl):
    print(f"BO testing configuration: CPU Cores={cpu_cores+1}, CPU Freq={cpu_freq}, GPU Freq={gpu_freq}, Mem Freq={mem_freq}, CL={cl}")
    
    t1 = time.time()
    measured_metrics = execute_config(cpu_cores, cpu_freq, gpu_freq, mem_freq, cl)
    elapsed = round(time.time() - t1, 3)

    if not measured_metrics or measured_metrics == "No Device":
        print("No device detected. Raising an exception to stop optimization.")
        raise RuntimeError("No device detected. Stopping optimization.")  # Raise exception to stop the optimizer
    
    if not measured_metrics:
        return 1e6  # Large penalty if no valid result

    reward = calculate_reward(measured_metrics)
    print(f"BO Configuration reward: {reward}")
    
    configs = {
        "reward": reward,
        "xaviernx_time_elapsed": elapsed,
        "bosarsa_time_elapsed": 0,
        "cpu_cores": int(cpu_cores) + 1,
        "cpu_freq": int(cpu_freq),
        "gpu_freq": int(gpu_freq),
        "mem_freq": int(mem_freq),
        "cl": int(cl)
    }
    result = {**configs, **measured_metrics[0]}
    save_csv([result], f"bosarsa_jxavier_{sys.argv[4]}.csv")

    state_index = [
        np.searchsorted(CPU_CORES_RANGE, cpu_cores),
        np.searchsorted(CPU_FREQ_RANGE, cpu_freq),
        np.searchsorted(GPU_FREQ_RANGE, gpu_freq),
        np.searchsorted(MEMORY_FREQ_RANGE, mem_freq),
        np.searchsorted(CL_RANGE, cl)
    ]
    if reward == 1e6:
        print("PROHIBITED CONFIG")
        prohibited_configs.add(state_index)
        return 1e6
    
    last_rewards.append(reward)
    # Check if optimization is saturated
    if len(last_rewards) > MAX_SATURATION_CALLS and all(r == last_rewards[-1] for r in last_rewards[-MAX_SATURATION_CALLS:]):
        print("Optimization is saturated. Stopping further iterations.")
        raise RuntimeError("Optimization saturated.")  # Raising an exception to stop optimization
    
    return -reward  # Minimize negative reward

# Step 1: Global search with Bayesian Optimization
def global_search_bo():
    res = gp_minimize(objective, space, n_calls=n_calls, random_state=42)
    best_params = dict(zip(['cpu_cores', 'cpu_freq', 'gpu_freq', 'mem_freq', 'cl'], res.x))
    print(f"Best configuration from BO: {best_params}")
    return best_params

# Step 2: Local exploration and exploitation with SARSA
def local_search_sarsa(best_params):
    cpu_cores, cpu_freq, gpu_freq, mem_freq, cl = best_params.values()
    state_index = [
        np.searchsorted(CPU_CORES_RANGE, cpu_cores),
        np.searchsorted(CPU_FREQ_RANGE, cpu_freq),
        np.searchsorted(GPU_FREQ_RANGE, gpu_freq),
        np.searchsorted(MEMORY_FREQ_RANGE, mem_freq),
        np.searchsorted(CL_RANGE, cl)
    ]

    for episode in range(num_episodes):
         # Check for prohibited configurations
        if (state_index in prohibited_configs):
            print("PROHIBITED CONFIG!")
            continue

        action = choose_action(state_index)
        new_cpu_cores = adjust_value(cpu_cores, action[0], STEP_SIZES['cpu_cores'], min(CPU_CORES_RANGE), max(CPU_CORES_RANGE))
        new_cpu_freq = adjust_value(cpu_freq, action[1], STEP_SIZES['cpu_freq'], min(CPU_FREQ_RANGE), max(CPU_FREQ_RANGE))
        new_gpu_freq = adjust_value(gpu_freq, action[2], STEP_SIZES['gpu_freq'], min(GPU_FREQ_RANGE), max(GPU_FREQ_RANGE))
        new_mem_freq = adjust_value(mem_freq, action[3], STEP_SIZES['memory_freq'], min(MEMORY_FREQ_RANGE), max(MEMORY_FREQ_RANGE))
        new_cl = adjust_value(cl, action[4], STEP_SIZES['cl'], min(CL_RANGE), max(CL_RANGE))
        
        t1 = time.time()
        # Execute the new configuration
        measured_metrics = execute_config(new_cpu_cores, new_cpu_freq, new_gpu_freq, new_mem_freq, new_cl)
        elapsed_exec = round(time.time() - t1, 3)

        if not measured_metrics:
            print("EXECUTION PROBLEM!")
            continue
        if measured_metrics == "No Device":
            print("No Device/Inference Runtime")
            break

        # SARSA updates
        new_state_index = [
            np.searchsorted(CPU_CORES_RANGE, new_cpu_cores),
            np.searchsorted(CPU_FREQ_RANGE, new_cpu_freq),
            np.searchsorted(GPU_FREQ_RANGE, new_gpu_freq),
            np.searchsorted(MEMORY_FREQ_RANGE, new_mem_freq),
            np.searchsorted(CL_RANGE, new_cl)
        ]
        reward = calculate_reward(measured_metrics)
        print(f"SARSA episode {episode + 1}: reward = {reward}")
        if reward == 1e6:
            print("PROHIBITED CONFIG")
            prohibited_configs.add(new_state_index)
            continue
        
        # Q-value update
        old_q_value = get_q_value(state_index, action)
        next_action = choose_action(new_state_index)  # SARSA uses next action
        next_q_value = get_q_value(new_state_index, next_action)
        updated_q_value = old_q_value + alpha * (reward + gamma * next_q_value - old_q_value)
        update_q_value(state_index, action, updated_q_value)

        state_index = new_state_index
        elapsed = round(((time.time() - t1) - elapsed_exec)*1000, 3)

        configs = {
            "reward": reward,
            "xaviernx_time_elapsed": elapsed_exec,
            "bosarsa_time_elapsed": elapsed,
            "cpu_cores": new_cpu_cores + 1,
            "cpu_freq": new_cpu_freq,
            "gpu_freq": new_gpu_freq,
            "memory_freq": new_mem_freq,
            "cl": new_cl
        }
        dict_record = [{**configs, **measured_metrics[0]}]
        save_csv(dict_record, f"bosarsa_jxavier_{sys.argv[4]}.csv")

        # Track max reward and configurations
        if reward > max_reward:
            max_reward = reward
            best_config = dict_record

        if reward > last_reward - reward_threshold:
            max_saturated_count -= 1
            epsilon = 0.5
            if max_saturated_count == 0:
                print("SARSA is saturated")
                break
        
        last_reward = reward

        # Epsilon decay for exploration
        if epsilon > epsilon_min:
            epsilon *= epsilon_decay
    return best_config

# Main Optimization Process
best_params_bo = global_search_bo()  # Step 1: Global Search with BO
best_config = local_search_sarsa(best_params_bo)    # Step 2: Local refinement with SARSA
print(f"Best Config: {best_config}")