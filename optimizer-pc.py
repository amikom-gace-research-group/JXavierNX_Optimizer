import numpy as np
import random
import sys
import time
import os
import csv
import requests
from statistics import median

print("PID", os.getpid())

# Define configuration ranges
CPU_CORES_RANGE = range(0, 6)  # Number of CPU cores (1 to 6)
CPU_FREQ_RANGE = range(1190, 1909)  # CPU frequency in MHz (1190 to 1908)
GPU_FREQ_RANGE = range(510, 1111)  # GPU frequency in MHz (510 to 1110)
MEMORY_FREQ_RANGE = range(1500, 1867)  # Memory frequency in MHz (1500 to 1866)
CL_RANGE = range(1, 4)  # Concurrency level (1 to 3)

# Constants and thresholds
POWER_BUDGET = 10000
THROUGHPUT_TARGET = 30
importance_power = 0
importance_throughput = 1

# Hyperparameters
alpha = 0.1
gamma = 0.9
epsilon = 0.6
epsilon_min = 0.01
epsilon_decay = 0.995
num_episodes = 100
reward_threshold = 0.01
max_saturated_count = 5

# Define action constants
ACTIONS = [0, 1, 2, 3, 4, 5, 6]  # No change, Small/Medium/Large Increase/Decrease

# Action mapping for each configuration
ACTION_MAPPING = ['cpu_cores', 'cpu_freq', 'gpu_freq', 'memory_freq', 'cl']
action_shape = [len(ACTIONS)] * len(ACTION_MAPPING)

# Step sizes for adjustment
STEP_SIZES = {
    'cpu_cores': (1, 3, 5),
    'cpu_freq': (1, 10, 50),
    'gpu_freq': (1, 10, 50),
    'memory_freq': (1, 10, 50),
    'cl': (1, 2, 0)  # Concurrency level doesn't need medium step
}

# Prohibited configs
prohibited_configs = set()

# Initialize an empty Q-table as a dictionary
Q_table = {}

# Efficient state to index mapping
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

# Epsilon-greedy action selection with state-based exploration
def choose_action(state_index):
    if random.uniform(0, 1) < epsilon:
        return [random.choice(ACTIONS) for _ in range(len(ACTION_MAPPING))]  # Explore
    else:
        state_key = tuple(state_index)
        if state_key not in Q_table:
            return [random.choice(ACTIONS) for _ in range(len(ACTION_MAPPING))]  # Explore if unseen state
        return np.unravel_index(np.argmax(Q_table[state_key]), action_shape)  # Exploit

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

# Efficient config execution
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
                if metrics and len(metrics) == 1:
                    requests.delete(f"{sys.argv[1]}/api/output", headers=headers)
                    return metrics
                else:
                    av_dev += 1
                    print("Waiting JXavier....")
                    if av_dev == 50:
                        return "No Device"
                    time.sleep(5)
        else:
            print(f"Error executing config: {response.status_code}")
    except requests.RequestException as e:
        print(f"Error executing config: {e}")
    return None

def close_freq_prohibited(state_index):
    for prohibited in prohibited_configs:
        for i in range(1, 4):
            if abs(state_index[i] - prohibited[i]) < 10:
                return True
            else:
                break
    return False

# Efficient reward calculation
def calculate_reward(measured_metrics):
    power = median([m["power_cons"] for m in measured_metrics])
    throughput = median([m["throughput"] for m in measured_metrics])
    
    if power > POWER_BUDGET or throughput < THROUGHPUT_TARGET:
        return -1
    
    return (importance_throughput * (throughput / THROUGHPUT_TARGET) +
            importance_power * (POWER_BUDGET / power))

# CSV saving optimization
def save_csv(dict_list, filename):
    with open(filename, 'a', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=['id', 'reward', 'xaviernx_time_elapsed', 'q-learning_time_elapsed', 'cpu_cores', 'cpu_freq', 'gpu_freq', 'memory_freq', 'cl', 'throughput', 'power_cons'])
        if os.path.getsize(filename) == 0:
            writer.writeheader()
        for d in dict_list:
            writer.writerow(d)

# Main Q-learning loop
last_reward = 0
max_reward = -float('inf')
best_config = None

cpu_cores, cpu_freq, gpu_freq, memory_freq, cl = CPU_CORES_RANGE[-1], CPU_FREQ_RANGE[-1], GPU_FREQ_RANGE[-1], MEMORY_FREQ_RANGE[-1], CL_RANGE[0]
state_index = state_to_index(cpu_cores, cpu_freq, gpu_freq, memory_freq, cl)

for episode in range(num_episodes):
    t1 = time.time()
    actions = choose_action(state_index)

    print("Config before action :", cpu_cores, cpu_freq, gpu_freq, memory_freq, cl)
    print("Action :",actions)

    # Adjust values for each action
    cpu_cores = adjust_value(cpu_cores, actions[0], STEP_SIZES['cpu_cores'], min(CPU_CORES_RANGE), max(CPU_CORES_RANGE))
    cpu_freq = adjust_value(cpu_freq, actions[1], STEP_SIZES['cpu_freq'], min(CPU_FREQ_RANGE), max(CPU_FREQ_RANGE))
    gpu_freq = adjust_value(gpu_freq, actions[2], STEP_SIZES['gpu_freq'], min(GPU_FREQ_RANGE), max(GPU_FREQ_RANGE))
    memory_freq = adjust_value(memory_freq, actions[3], STEP_SIZES['memory_freq'], min(MEMORY_FREQ_RANGE), max(MEMORY_FREQ_RANGE))
    cl = adjust_value(cl, actions[4], STEP_SIZES['cl'], min(CL_RANGE), max(CL_RANGE))

    print("Config after action :", cpu_cores, cpu_freq, gpu_freq, memory_freq, cl)

    state_index = state_to_index(cpu_cores, cpu_freq, gpu_freq, memory_freq, cl)
    close_prohibited = close_freq_prohibited(state_index)
    if state_index in prohibited_configs or close_prohibited:
        print("PROHIBITED CONFIG!")
        epsilon = 0.5
        continue

    t2 = time.time()
    measured_metrics = execute_config(cpu_cores, cpu_freq, gpu_freq, memory_freq, cl)
    elapsed_exec = round(time.time() - t2, 3)
    if not measured_metrics:
        print("EXECUTION PROBLEM!")
        epsilon = 0.5
        continue
    if measured_metrics == "No Device":
        print("No Device")
        break

    reward = calculate_reward(measured_metrics)

    # Prohibited state handling
    if reward == -1:
        print("PROHIBITED CONFIG")
        epsilon = 0.5
        prohibited_configs.add(state_index)
        continue

    action_index = tuple(actions)
    old_q_value = get_q_value(state_index, action_index)
    new_q_value = old_q_value + alpha * (reward + gamma * reward - old_q_value)  # Update Q-value

    update_q_value(state_index, action_index, new_q_value)

    elapsed = round((time.time()-elapsed_exec) - t1, 3)

    configs = {"reward": reward, "xaviernx_time_elapsed": elapsed_exec, "q-learning_time_elapsed": elapsed, "cpu_cores": cpu_cores, "cpu_freq": cpu_freq, "gpu_freq": gpu_freq, "memory_freq": memory_freq, "cl": cl}
    dict_record = [{**configs, **measured_metrics[0]}]
    save_csv(dict_record, f"output_jxavier_{sys.argv[4]}.csv")

    if reward > max_reward:
        max_reward = reward
        best_config = dict_record
    if reward > last_reward - reward_threshold:
        max_saturated_count -= 1
        if max_saturated_count == 0:
            print("Q-Learning is saturated")
            break
    last_reward = reward

    if epsilon > epsilon_min:
        epsilon *= epsilon_decay

    print(f"Episode: {episode}, Reward: {reward}, Max Reward: {max_reward}")

print(f"Best Config: {best_config}")
