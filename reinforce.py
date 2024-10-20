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

# Hyperparameters
alpha = 0.1
gamma = 0.9
epsilon = 0.6
epsilon_min = 0.01
epsilon_decay = 0.995
reward_threshold = 0.01
num_episodes = 4
max_saturated_count = 5

# Constants and thresholds
POWER_BUDGET = 5000
THROUGHPUT_TARGET = 30
importance_power = 1
importance_throughput = 1

# Action constants
ACTIONS = [0, 1, 2, 3, 4, 5, 6]  # No change, Small/Medium/Large Increase/Decrease
ACTION_MAPPING = ['cpu_cores', 'cpu_freq', 'gpu_freq', 'memory_freq', 'cl']
action_shape = [len(ACTIONS)] * len(ACTION_MAPPING)

STEP_SIZES = {
    'cpu_cores': (1, 3, 5),
    'cpu_freq': (1, 10, 50),
    'gpu_freq': (1, 10, 50),
    'memory_freq': (1, 10, 50),
    'cl': (1, 2, 0)
}

policy_table = {}

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

# Correct softmax action selection
def choose_action(state_index):
    state_key = tuple(state_index)
    if state_key not in policy_table:
        policy_table[state_key] = np.random.rand(np.prod(action_shape))
    
    if random.uniform(0, 1) < epsilon:
        return [random.choice(ACTIONS) for _ in range(len(ACTION_MAPPING))]
    else:
        action_probabilities = policy_table[state_key]
        # Select separate actions for each parameter using softmax
        action_indices = []
        for i in range(len(ACTION_MAPPING)):
            start = i * len(ACTIONS)
            end = start + len(ACTIONS)
            prob_dist = softmax(action_probabilities[start:end])
            action_indices.append(np.random.choice(ACTIONS, p=prob_dist))
        return action_indices

def softmax(x):
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum()

def update_policy(state_index, action_index, reward, cumulative_return):
    state_key = tuple(state_index)
    if state_key not in policy_table:
        policy_table[state_key] = np.random.rand(np.prod(action_shape))
    
    action_probabilities = softmax(policy_table[state_key])
    chosen_action_prob = action_probabilities[np.ravel_multi_index(action_index, action_shape)]
    
    # Update policy with REINFORCE rule
    policy_table[state_key][np.ravel_multi_index(action_index, action_shape)] += alpha * (cumulative_return - reward) * (1 - chosen_action_prob)

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

def calculate_reward(measured_metrics):
    power = measured_metrics[0]["power_cons"]
    throughput = measured_metrics[0]["throughput"]
    
    if power > POWER_BUDGET or throughput < THROUGHPUT_TARGET:
        return -1
    
    return (importance_throughput * (throughput / THROUGHPUT_TARGET) +
            importance_power * (POWER_BUDGET / power))

def save_csv(dict_list, filename):
    with open(filename, 'a', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=['reward', 'xaviernx_time_elapsed', 'reinforce_time_elapsed', 'cpu_cores', 'cpu_freq', 'gpu_freq', 'memory_freq', 'cl', 'throughput', 'power_cons'])
        if os.path.getsize(filename) == 0:
            writer.writeheader()
        for d in dict_list:
            writer.writerow(d)

# Main REINFORCE loop
cpu_cores, cpu_freq, gpu_freq, memory_freq, cl = 3, 1550, 810, 1700, 2
max_reward = -float('inf')
best_config = None
last_reward = 0

state_index = state_to_index(cpu_cores, cpu_freq, gpu_freq, memory_freq, cl)
t_main = time.time()

for episode in range(num_episodes):
    t1 = time.time()
    
    actions_taken = []
    rewards = []
    time_got = []
    state_indexes = []
    configs = []

    for step in range(5):
        actions = choose_action(state_index)
        actions_taken.append(actions)
        
        cpu_cores = adjust_value(cpu_cores, actions[0], STEP_SIZES['cpu_cores'], min(CPU_CORES_RANGE), max(CPU_CORES_RANGE))
        cpu_freq = adjust_value(cpu_freq, actions[1], STEP_SIZES['cpu_freq'], min(CPU_FREQ_RANGE), max(CPU_FREQ_RANGE))
        gpu_freq = adjust_value(gpu_freq, actions[2], STEP_SIZES['gpu_freq'], min(GPU_FREQ_RANGE), max(GPU_FREQ_RANGE))
        memory_freq = adjust_value(memory_freq, actions[3], STEP_SIZES['memory_freq'], min(MEMORY_FREQ_RANGE), max(MEMORY_FREQ_RANGE))
        cl = adjust_value(cl, actions[4], STEP_SIZES['cl'], min(CL_RANGE), max(CL_RANGE))

        state_index = state_to_index(cpu_cores, cpu_freq, gpu_freq, memory_freq, cl)
        state_indexes.append(state_index)
        
        if state_index in prohibited_configs:
            print("PROHIBITED CONFIG!")
            continue
        
        t2 = time.time()
        measured_metrics = execute_config(cpu_cores, cpu_freq, gpu_freq, memory_freq, cl)
        elapsed_exec = round(time.time() - t2, 3)
        time_got.append(elapsed_exec)
        
        if not measured_metrics or measured_metrics == "No Device":
            print("Execution issue or No device found!")
            break
        
        reward = calculate_reward(measured_metrics)

        config = {
            "reward": reward,
            "xaviernx_time_elapsed": elapsed_exec,
            "cpu_cores": cpu_cores+1,
            "cpu_freq": cpu_freq,
            "gpu_freq": gpu_freq,
            "memory_freq": memory_freq,
            "cl": cl,
            "throughput": measured_metrics[0]["throughput"],
            "power_cons": measured_metrics[0]["power_cons"]
        }
        configs.append(config)
        rewards.append(reward)

        if reward == -1:
            print("Prohibited Configuration!")
            prohibited_configs.add(state_index)
            continue
        
        print(f"Action {actions}, Reward: {reward}")

        if reward > max_reward:
            max_reward = reward
            best_config = actions_taken[-1]

        if reward > last_reward - reward_threshold:
            max_saturated_count -= 1
            epsilon = 0.5
            if max_saturated_count == 0:
                print("REINFORCE is saturated")
                break

        last_reward = reward
    
    if rewards:
        cumulative_return = sum([r * (gamma ** i) for i, r in enumerate(rewards)])
        for i, action in enumerate(actions_taken):
            update_policy(state_indexes[i], actions, rewards[i], cumulative_return)

    end_t1 = round(((time.time() - t1) - sum(time_got))*1000, 3)
    for config in configs:
        dict_record = [{'reinforce_time_elapsed': end_t1, **config}]
        save_csv(dict_record, f"reinforce_jxavier_{sys.argv[4]}.csv")

print(f"Best Configuration: {best_config}")
