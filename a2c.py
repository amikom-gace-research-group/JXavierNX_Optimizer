import numpy as np
import random
import sys
import time
import os
import requests
import csv

print("PID", os.getpid())

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

# Hyperparameters
alpha = 0.1  # Learning rate for policy update
beta = 0.1   # Learning rate for critic (value function) update
gamma = 0.9  # Discount factor
epsilon = 0.6
epsilon_min = 0.01
epsilon_decay = 0.995
reward_threshold = 0.01
num_episodes = 20

# Action constants
ACTIONS = [0, 1, 2, 3, 4, 5, 6]
ACTION_MAPPING = ['cpu_cores', 'cpu_freq', 'gpu_freq', 'memory_freq', 'cl']
action_shape = [len(ACTIONS)] * len(ACTION_MAPPING)

# Step sizes for adjustment
STEP_SIZES = {
    'cpu_cores': (1, 3, 5),
    'cpu_freq': (1, 10, 50),
    'gpu_freq': (1, 10, 50),
    'memory_freq': (1, 10, 50),
    'cl': (1, 2, 0)
}

# Initialize policy (actor) and value (critic) tables
policy_table = {}
value_table = {}

# Prohibited configs
prohibited_configs = set()

def state_to_index(cpu_cores, cpu_freq, gpu_freq, memory_freq, cl):
    return (
        np.searchsorted(CPU_CORES_RANGE, cpu_cores),
        np.searchsorted(CPU_FREQ_RANGE, cpu_freq),
        np.searchsorted(GPU_FREQ_RANGE, gpu_freq),
        np.searchsorted(MEMORY_FREQ_RANGE, memory_freq),
        np.searchsorted(CL_RANGE, cl)
    )

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
    return value

def choose_action(state_index):
    state_key = tuple(state_index)
    if state_key not in policy_table:
        policy_table[state_key] = np.random.rand(np.prod(action_shape))
    
    if random.uniform(0, 1) < epsilon:
        return [random.choice(ACTIONS) for _ in range(len(ACTION_MAPPING))]
    else:
        action_probabilities = softmax(policy_table[state_key])
        return np.unravel_index(np.argmax(action_probabilities), action_shape)

def softmax(x):
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum()

def update_policy(state_index, action_index, advantage):
    state_key = tuple(state_index)
    if state_key not in policy_table:
        policy_table[state_key] = np.random.rand(np.prod(action_shape))
    
    action_probabilities = softmax(policy_table[state_key])
    chosen_action_prob = action_probabilities[np.ravel_multi_index(action_index, action_shape)]
    
    # Update policy using advantage
    policy_table[state_key][np.ravel_multi_index(action_index, action_shape)] += alpha * advantage * (1 - chosen_action_prob)

def update_value(state_index, reward, next_value):
    state_key = tuple(state_index)
    if state_key not in value_table:
        value_table[state_key] = 0.0
    
    # Temporal Difference (TD) error: reward + gamma * next_value - current_value
    td_error = reward + gamma * next_value - value_table[state_key]
    
    # Update value table using TD error
    value_table[state_key] += beta * td_error

def calculate_reward(measured_metrics):
    power = measured_metrics[0]["power_cons"]
    throughput = measured_metrics[0]["throughput"]
    
    if power > POWER_BUDGET or throughput < THROUGHPUT_TARGET:
        return -1
    return (importance_throughput * (throughput / THROUGHPUT_TARGET) +
            importance_power * (POWER_BUDGET / power))

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
                    print("Waiting for Jetson Xavier....")
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
        writer = csv.DictWriter(f, fieldnames=['id', 'reward', 'xaviernx_time_elapsed', 'a2c_time_elapsed', 'cpu_cores', 'cpu_freq', 'gpu_freq', 'memory_freq', 'cl', 'throughput', 'power_cons'])
        if os.path.getsize(filename) == 0:
            writer.writeheader()
        for d in dict_list:
            writer.writerow(d)

# Main A2C loop
cpu_cores, cpu_freq, gpu_freq, memory_freq, cl = 3, 1550, 810, 1700, 2  # Starting values
max_reward = -float('inf')
best_config = None
state_index = state_to_index(cpu_cores, cpu_freq, gpu_freq, memory_freq, cl)
last_reward = 0
max_saturated_count = 5

t_main = time.time()
for episode in range(num_episodes):
    t1 = time.time()
    # Check for prohibited configurations
    if state_index in prohibited_configs:
        print("PROHIBITED CONFIG!")
        break

    t2 = time.time()
    measured_metrics = execute_config(cpu_cores, cpu_freq, gpu_freq, memory_freq, cl)
    elapsed_exec = round(time.time() - t2, 3)
    if not measured_metrics:
        break
    if measured_metrics == "No Device":
        break

    actions = choose_action(state_index)
    
    # Adjust system configuration based on actions
    cpu_cores = adjust_value(cpu_cores, actions[0], STEP_SIZES['cpu_cores'], min(CPU_CORES_RANGE), max(CPU_CORES_RANGE))
    cpu_freq = adjust_value(cpu_freq, actions[1], STEP_SIZES['cpu_freq'], min(CPU_FREQ_RANGE), max(CPU_FREQ_RANGE))
    gpu_freq = adjust_value(gpu_freq, actions[2], STEP_SIZES['gpu_freq'], min(GPU_FREQ_RANGE), max(GPU_FREQ_RANGE))
    memory_freq = adjust_value(memory_freq, actions[3], STEP_SIZES['memory_freq'], min(MEMORY_FREQ_RANGE), max(MEMORY_FREQ_RANGE))
    cl = adjust_value(cl, actions[4], STEP_SIZES['cl'], min(CL_RANGE), max(CL_RANGE))

    next_state_index = state_to_index(cpu_cores, cpu_freq, gpu_freq, memory_freq, cl)

    # Calculate reward and update policy and value tables within inner loop
    reward = calculate_reward(measured_metrics)

    # Prohibited state handling
    if reward == -1:
        print("PROHIBITED CONFIG")
        prohibited_configs.add(state_index)
        state_index = next_state_index
        continue

    # Advantage = TD error (reward + future value estimate - current value)
    next_value = value_table.get(tuple(next_state_index), 0)
    current_value = value_table.get(tuple(state_index), 0)
    advantage = reward + gamma * next_value - current_value
    
    update_policy(state_index, actions, advantage)
    update_value(state_index, reward, next_value)

    elapsed = round((time.time() - elapsed_exec - t1)*1000, 3)

    configs = {
        "reward": reward,
        "xaviernx_time_elapsed": elapsed_exec,
        "a2c_time_elapsed": elapsed,
        "cpu_cores": cpu_cores + 1,
        "cpu_freq": cpu_freq,
        "gpu_freq": gpu_freq,
        "memory_freq": memory_freq,
        "cl": cl
    }
    dict_record = [{**configs, **measured_metrics[0]}]
    save_csv(dict_record, f"a2c_jxavier_{sys.argv[4]}.csv")

    # Update current state
    state_index = next_state_index

    # Track max reward and configurations
    if reward > max_reward:
        max_reward = reward
        best_config = dict_record

    if reward > last_reward - reward_threshold:
        max_saturated_count -= 1
        epsilon = 0.3
        if max_saturated_count == 0:
            print("A2C is saturated")
            break
    
    last_reward = reward
    
    # End of episode, adjust epsilon (for exploration)
    epsilon = max(epsilon_min, epsilon * epsilon_decay)
    print(f"Episode {episode+1} completed with rewards: {reward} in {round(time.time() - tep, 3)}")

print(f"Best configuration after {num_episodes} episodes: {best_config} in {round(time.time() - t_main, 3)}")
