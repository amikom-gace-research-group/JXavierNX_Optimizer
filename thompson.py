import numpy as np
import sys
import time
import os
import csv
import requests
from collections import defaultdict

print("PID", os.getpid())

# Device-specific configurations
if sys.argv[5] == 'jxavier':
    print("xavier")
    CPU_CORES_RANGE = range(1, 6)
    CPU_FREQ_RANGE = range(1190, 1909)
    GPU_FREQ_RANGE = range(510, 1111)
    MEMORY_FREQ_RANGE = range(1500, 1867)
elif sys.argv[5] == 'jorin-nano':
    print("orin")
    CPU_CORES_RANGE = range(1, 6)
    CPU_FREQ_RANGE = range(806, 1511)
    GPU_FREQ_RANGE = range(306, 625)
    MEMORY_FREQ_RANGE = range(1500, 2134)

CL_RANGE = range(1, 4)
POWER_BUDGET = 5000
THROUGHPUT_TARGET = 30
importance_power = 1
importance_throughput = 1

# Hyperparameters
alpha = 0.1
gamma = 0.9
num_episodes = 20
reward_threshold = 0.01
max_saturated_count = 5

# Define actions and step sizes
ACTIONS = [0, 1, 2, 3, 4, 5, 6]
ACTION_MAPPING = ['cpu_cores', 'cpu_freq', 'gpu_freq', 'memory_freq', 'cl']
action_shape = [len(ACTIONS)] * len(ACTION_MAPPING)
STEP_SIZES = {
    'cpu_cores': (1, 3, 5),
    'cpu_freq': (1, 10, 50),
    'gpu_freq': (1, 10, 50),
    'memory_freq': (1, 10, 50),
    'cl': (1, 0, 2)
}

prohibited_configs = set()

# Define initial beta parameters for Thompson Sampling (successes and failures for each action per state)
beta_params = defaultdict(lambda: {i: [(1, 1) for _ in ACTIONS] for i in range(len(ACTION_MAPPING))})

def state_to_index(cpu_cores, cpu_freq, gpu_freq, memory_freq, cl):
    return (
        np.searchsorted(CPU_CORES_RANGE, cpu_cores),
        np.searchsorted(CPU_FREQ_RANGE, cpu_freq),
        np.searchsorted(GPU_FREQ_RANGE, gpu_freq),
        np.searchsorted(MEMORY_FREQ_RANGE, memory_freq),
        np.searchsorted(CL_RANGE, cl)
    )

# Adjust configuration values based on action
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
        "cpu_cores": cpu_cores,
        "cpu_freq": cpu_freq,
        "gpu_freq": gpu_freq,
        "mem_freq": memory_freq,
        "cl": cl
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

def update_beta_params(state_index, actions, reward):
    state_key = tuple(state_index)
    
    # Scale success and failure counts based on the reward's magnitude
    for i, action in enumerate(actions):
        success, failure = beta_params[state_key][i][action]
        
        # Scale the update based on reward value
        if reward > 0:
            success_update = max(1, int(reward * 10))  # Scale success proportional to reward
            beta_params[state_key][i][action] = (success + success_update, failure)
        else:
            failure_update = max(1, int(abs(reward) * 10))  # Scale failure if reward is negative
            beta_params[state_key][i][action] = (success, failure + failure_update)

def choose_action_thompson(state_index):
    chosen_actions = []
    state_key = tuple(state_index)
    
    # Loop through each parameter/action dimension (e.g., 'cpu_cores', 'cpu_freq', etc.)
    for dimension in range(len(ACTION_MAPPING)):
        action_probabilities = []
        
        # Sample the probability for each action in this dimension
        for action in range(len(ACTIONS)):
            action_successes, action_failures = beta_params[state_key][dimension][action]
            sampled_prob = np.random.beta(action_successes, action_failures)
            action_probabilities.append(sampled_prob)
        
        # Choose the action with the highest sampled probability for this dimension
        best_action = np.argmax(action_probabilities)
        chosen_actions.append(best_action)
    
    return tuple(chosen_actions)

# Calculate reward with shaping
def calculate_reward(measured_metrics):
    power = measured_metrics[0]["power_cons"]
    throughput = measured_metrics[0]["throughput"]
    
    if power > POWER_BUDGET or throughput < THROUGHPUT_TARGET:
        return -1
    
    return (importance_throughput * (throughput / THROUGHPUT_TARGET) +
            importance_power * (POWER_BUDGET / power))

# CSV saving optimization
def save_csv(dict_list, filename):
    with open(filename, 'a', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=['id', 'episode', 'reward', 'xaviernx_time_elapsed', 'thompson_time_elapsed', 'cpu_cores', 'cpu_freq', 'gpu_freq', 'memory_freq', 'cl', 'throughput', 'power_cons'])
        if os.path.getsize(filename) == 0:
            writer.writeheader()
        for d in dict_list:
            writer.writerow(d)

last_reward = 0
max_reward = -float('inf')
best_config = None

# Initial state configuration
cpu_cores, cpu_freq, gpu_freq, memory_freq, cl = max(CPU_CORES_RANGE), max(CPU_FREQ_RANGE), max(GPU_FREQ_RANGE), max(MEMORY_FREQ_RANGE), max(CL_RANGE)
state_index = state_to_index(cpu_cores, cpu_freq, gpu_freq, memory_freq, cl)

for episode in range(num_episodes):
    actions = choose_action_thompson(state_index)
    
    # Adjust values for each action
    cpu_cores = adjust_value(cpu_cores, actions[0], STEP_SIZES['cpu_cores'], min(CPU_CORES_RANGE), max(CPU_CORES_RANGE))
    cpu_freq = adjust_value(cpu_freq, actions[1], STEP_SIZES['cpu_freq'], min(CPU_FREQ_RANGE), max(CPU_FREQ_RANGE))
    gpu_freq = adjust_value(gpu_freq, actions[2], STEP_SIZES['gpu_freq'], min(GPU_FREQ_RANGE), max(GPU_FREQ_RANGE))
    memory_freq = adjust_value(memory_freq, actions[3], STEP_SIZES['memory_freq'], min(MEMORY_FREQ_RANGE), max(MEMORY_FREQ_RANGE))
    cl = adjust_value(cl, actions[4], STEP_SIZES['cl'], min(CL_RANGE), max(CL_RANGE))
    print({"cpu_cores": cpu_cores+1, "cpu_freq": cpu_freq, "gpu_freq": gpu_freq, "memory_freq": memory_freq, "cl": cl})
    state_index = state_to_index(cpu_cores, cpu_freq, gpu_freq, memory_freq, cl)

    # Check for prohibited configurations
    if state_index in prohibited_configs:
        print("PROHIBITED CONFIG!")
        continue
    
    # Execution, measurement, and reward
    t1 = time.time()
    measured_metrics = execute_config(cpu_cores, cpu_freq, gpu_freq, memory_freq, cl)
    elapsed_exec = round(time.time() - t1, 3)
    if not measured_metrics:
        print("EXECUTION PROBLEM!")
        continue
    if measured_metrics == "No Device":
        print("No Device/Inference Runtime")
        break
    reward = calculate_reward(measured_metrics)

    if reward == -1:
        print("PROHIBITED CONFIG")
        prohibited_configs.add(state_index)

    # Update Thompson Sampling beta parameters based on reward feedback
    update_beta_params(state_index, actions, reward)

    # Track the best configuration
    if reward > max_reward:
        max_reward = reward
        best_config = {"cpu_cores": cpu_cores+1, "cpu_freq": cpu_freq, "gpu_freq": gpu_freq, "memory_freq": memory_freq, "cl": cl}

    if abs(reward - last_reward) < reward_threshold:
        max_saturated_count -= 1
        if max_saturated_count == 0:
                print("Thompson is saturated")
                break

    elapsed = round(((time.time() - t1) - elapsed_exec)*1000, 3)

    configs = {
        "episode": episode,
        "reward": reward,
        "xaviernx_time_elapsed": elapsed_exec,
        "thompson_time_elapsed": elapsed,
        "cpu_cores": cpu_cores + 1,
        "cpu_freq": cpu_freq,
        "gpu_freq": gpu_freq,
        "memory_freq": memory_freq,
        "cl": cl
    }
    dict_record = [{**configs, **measured_metrics[0]}]
    save_csv(dict_record, f"thompson_{sys.argv[5]}_{sys.argv[4]}.csv")
    print(f"Episode: {episode}, Reward: {reward}, Max Reward: {max_reward}")

print(f"Best Config: {best_config} with Reward: {max_reward}")
