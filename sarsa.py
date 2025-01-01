import numpy as np
import sys
import time
import os
import csv
import requests
import pandas as pd
from pyDOE import lhs

print("PID", os.getpid())

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

POWER_BUDGET = int(sys.argv[6])

best_throughput = -float('inf')
max_reward = -float('inf')
last_reward = 0

# Hyperparameters
alpha = 0.1
gamma = 0.9
prohibited_addition = 0
epsilon_explore = 0.5
epsilon_exploit = 0.5
epsilon_min = 1e-10  # Minimum epsilon value (always exploit after this threshold)
num_episodes = 100  # Number of episodes to run
reward_threshold = 0.00001

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

# Initialize an empty Q-table as a dictionary
Q_table = {}

def state_to_index(cpu_cores, cpu_freq, gpu_freq, memory_freq, cl):
    return (
        np.searchsorted(sampled_configs['cpu_cores'], cpu_cores),
        np.searchsorted(sampled_configs['cpu_freq'], cpu_freq),
        np.searchsorted(sampled_configs['gpu_freq'], gpu_freq),
        np.searchsorted(sampled_configs['memory_freq'], memory_freq),
        np.searchsorted(sampled_configs['cl'], cl)
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

# Latin Hypercube Sampling to explore new states
def lhs_sampling(num_samples, ranges):
    lhs_samples = lhs(len(ranges), samples=num_samples)
    sampled_values = []
    for i, r in enumerate(ranges):
        sampled_values.append(lhs_samples[:, i] * (r[-1] - r[0]) + r[0])
    return np.array(sampled_values).T

# Generate LHS samples for the exploration phase
def generate_lhs_samples():
    num_samples = 10  # Number of samples per episode
    ranges = [
        (min(ACTIONS), max(ACTIONS) + 1),
        (min(ACTIONS), max(ACTIONS) + 1),
        (min(ACTIONS), max(ACTIONS) + 1),
        (min(ACTIONS), max(ACTIONS) + 1),
        (min(ACTIONS), max(ACTIONS) + 1)
    ]
    samples = lhs_sampling(num_samples, ranges)
    return [tuple(map(int, sample)) for sample in samples]

def get_best_configuration():
    best_q_value = float('-inf')
    best_state = None

    for state, q_values in Q_table.items():
        max_q_index = np.argmax(q_values)  # Find the index of the max Q-value
        max_q_value = q_values[max_q_index]
        
        if max_q_value > best_q_value:
            best_q_value = max_q_value
            best_state = state

    best_state = tuple([conf[int(x)] for x, conf in zip(best_state, [sampled_configs['cpu_cores'], sampled_configs['cpu_freq'], sampled_configs['gpu_freq'], sampled_configs['memory_freq'], sampled_configs['cl']])])
    return best_state

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

def calculate_diversity(lhs_samples, state_key, tau=1.0, max_diversity_score=500):
    """
    Select a configuration for exploration based on diversity scores.

    Args:
        lhs_samples: List of LHS-sampled configurations.
        state_key: Current state (for reference, though not directly used here).
        tau: Temperature parameter for softmax scaling (higher = more uniform).
        max_diversity_score: Maximum value for diversity scores to prevent overflow.

    Returns:
        selected_action: The selected action for exploration.
    """
    # Calculate diversity scores: Higher score for configurations far from each other
    diversity_scores = []
    for sample in lhs_samples:
        diversity_score = sum(abs(np.array(sample) - np.array(state_key))) if state_key else 1
        diversity_scores.append(diversity_score)

    # Clip diversity scores to prevent overflow
    diversity_scores = np.array(diversity_scores)
    diversity_scores = np.clip(diversity_scores, None, max_diversity_score)

    # Convert diversity scores to probabilities using softmax
    exp_scores = np.exp(diversity_scores / tau)

    # Prevent division by zero by adding a small epsilon value
    exp_scores_sum = np.sum(exp_scores)
    if exp_scores_sum == 0:
        exp_scores_sum = 1e-6  # Small value to avoid division by zero

    probabilities = exp_scores / exp_scores_sum

    # Handle any potential NaN values in probabilities
    if np.any(np.isnan(probabilities)):
        probabilities = np.ones_like(probabilities) / len(lhs_samples)  # Default uniform distribution

    # Select an action based on probabilities
    selected_action = lhs_samples[np.random.choice(len(lhs_samples), p=probabilities)]
    
    return selected_action

def choose_action_adaptive(state_index, lhs_samples):
    global epsilon_explore, epsilon_exploit
    state_key = tuple(state_index)
    # Select action based on epsilon
    if (epsilon_explore/epsilon_exploit) > 0.5:
        return calculate_diversity(lhs_samples, state_key), "exploration"
    else:
        # Exploitation: choose best known action
        if state_key not in Q_table:
            return calculate_diversity(lhs_samples, state_key), "exploration" # Use LHS samples for unseen states
        return np.unravel_index(np.argmax(Q_table[state_key]), action_shape), "exploitation"  # Exploit best known action

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

# Calculate reward with shaping
def calculate_reward(measured_metrics):
    power = measured_metrics[0]["power_cons"]
    throughput = measured_metrics[0]["throughput"]
    
    if power > POWER_BUDGET:
        return -(power / POWER_BUDGET)
    
    return throughput / POWER_BUDGET

# CSV saving optimization
def save_csv(dict_list, filename):
    with open(filename, 'a', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=['api_time','id', 'reward', 'phase', 'episode', 'xaviernx_time_elapsed', 'sarsa_time_elapsed', 'cpu_cores', 'cpu_freq', 'gpu_freq', 'memory_freq', 'cl', 'throughput', 'power_cons', 'cpu%', 'gpu%', 'mem%'])
        if os.path.getsize(filename) == 0:
            writer.writeheader()
        for d in dict_list:
            writer.writerow(d)
# Initial configuration (starting in the middle of the range)
cpu_cores, cpu_freq, gpu_freq, memory_freq, cl = max(sampled_configs['cpu_cores']), max(sampled_configs['cpu_freq']), max(sampled_configs['gpu_freq']), max(sampled_configs['memory_freq']), max(sampled_configs['cl'])
state_index = state_to_index(cpu_cores, cpu_freq, gpu_freq, memory_freq, cl)
# Execution loop with adaptive epsilon strategy
for episode in range(num_episodes):
    # Generate LHS samples for this episode
    lhs_samples = generate_lhs_samples()

    # Choose actions based on current state and LHS samples
    actions, phase = choose_action_adaptive(state_index, lhs_samples)

    # Print the chosen configuration for tracking
    print({"cpu_cores": cpu_cores+1, "cpu_freq": cpu_freq, "gpu_freq": gpu_freq, "memory_freq": memory_freq, "cl": cl})

    # Adjust values for the chosen actions
    cpu_cores = adjust_value(cpu_cores, actions[0], STEP_SIZES['cpu_cores'], min(sampled_configs['cpu_cores']), max(sampled_configs['cpu_cores']))
    cpu_freq = adjust_value(cpu_freq, actions[1], STEP_SIZES['cpu_freq'], min(sampled_configs['cpu_freq']), max(sampled_configs['cpu_freq']))
    gpu_freq = adjust_value(gpu_freq, actions[2], STEP_SIZES['gpu_freq'], min(sampled_configs['gpu_freq']), max(sampled_configs['gpu_freq']))
    memory_freq = adjust_value(memory_freq, actions[3], STEP_SIZES['memory_freq'], min(sampled_configs['memory_freq']), max(sampled_configs['memory_freq']))
    cl = adjust_value(cl, actions[4], STEP_SIZES['cl'], min(sampled_configs['cl']), max(sampled_configs['cl']))

    # Convert to new state index
    new_state_index = state_to_index(cpu_cores, cpu_freq, gpu_freq, memory_freq, cl)

    # Check for prohibited configurations
    if new_state_index in prohibited_configs and episode > 0:
        print("PROHIBITED CONFIG, RESET TO BEST CONFIG!")
        cpu_cores, cpu_freq, gpu_freq, memory_freq, cl = get_best_configuration()
        state_index = state_to_index(cpu_cores, cpu_freq, gpu_freq, memory_freq, cl)
        if state_index in prohibited_configs:
            epsilon_explore = 0.5
            epsilon_exploit = 0.5
            continue

    # Execute the chosen configuration and get metrics
    t1 = time.time()
    measured_metrics, api_time = execute_config(cpu_cores, cpu_freq, gpu_freq, memory_freq, cl)
    elapsed_exec = round(time.time() - t1, 3)

    if not measured_metrics or measured_metrics == "No Device":
        print("EXECUTION PROBLEM OR DEVICE UNAVAILABLE!")
        continue

    # Calculate the reward for this configuration
    reward = calculate_reward(measured_metrics)

    if reward < 0:
        print("PROHIBITED CONFIG!")
        prohibited_configs.add(new_state_index)

    new_actions, _ = choose_action_adaptive(new_state_index, lhs_samples)

    # Update Q-values using the old Q-value and the reward
    old_q_value = get_q_value(state_index, actions)
    new_q_value = old_q_value + alpha * (reward + gamma * get_q_value(new_state_index, new_actions) - old_q_value)  # SARSA update
    update_q_value(new_state_index, new_actions, new_q_value)

    # Track the best configuration
    if reward > max_reward and measured_metrics[0]["throughput"] > best_throughput:
        max_reward = reward
        best_config = {
            "api_time": api_time,
            "cpu_cores": cpu_cores+1,
            "cpu_freq": cpu_freq,
            "gpu_freq": gpu_freq,
            "memory_freq": memory_freq,
            "cl": cl
        }
        best_throughput = measured_metrics[0]["throughput"]

    elapsed = round(((time.time() - t1) - elapsed_exec) * 1000, 3)

    # Adaptive strategy: increase epsilon if reward is too low, decrease it if reward is sufficient
    if reward < 0:
        epsilon_explore = min(epsilon_explore * 1.05, 1)  # Increase epsilon if performance is bad
        epsilon_exploit = max(epsilon_exploit * 0.995, epsilon_min)
    else:
        epsilon_explore = max(epsilon_explore * 0.75, epsilon_min)  # Decay epsilon if performance improves
        epsilon_exploit = min(epsilon_exploit * 1.05, 1)
    
    configs = {
        "api_time": api_time,
        "reward": reward,
        "phase":phase,
        "episode": episode,
        "xaviernx_time_elapsed": elapsed_exec,
        "sarsa_time_elapsed": elapsed,
        "cpu_cores": cpu_cores + 1,
        "cpu_freq": cpu_freq,
        "gpu_freq": gpu_freq,
        "memory_freq": memory_freq,
        "cl": cl
    }
    dict_record = [{**configs, **measured_metrics[0]}]
    save_csv(dict_record, f"sarsa_{sys.argv[5]}_{sys.argv[4]}.csv")
    
    last_reward = reward
    state_index = new_state_index

    print(f"Episode: {episode}, Reward: {reward}, Max Reward: {max_reward}")

print(f"Best Config: {best_config} with Reward: {max_reward}")