import numpy as np
import sys
import time
import os
import csv
import requests
import random
from pyDOE import lhs

print("PID", os.getpid())

ACTIONS = [0, 1, 2, 3, 4, 5, 6]
# Configuration ranges for CPU, GPU, and memory
if sys.argv[5] == 'jxavier':
    CPU_CORES_RANGE = range(1, 6)
    CPU_FREQ_RANGE = range(1190, 1909)
    GPU_FREQ_RANGE = range(510, 1111)
    MEMORY_FREQ_RANGE = range(1500, 1867)
    CL_RANGE = range(1, 4)
    ranges = [
        (0, 3),
        (min(ACTIONS), max(ACTIONS) + 1),
        (min(ACTIONS), max(ACTIONS) + 1),
        (min(ACTIONS), max(ACTIONS) + 1),
        (0, 3)
    ]
    action_shape = [len(ACTIONS), len(ACTIONS), len(ACTIONS), len(ACTIONS), len(ACTIONS)]
elif sys.argv[5] == 'jorin-nano':
    CPU_CORES_RANGE = [5]
    CPU_FREQ_RANGE = range(806, 1511)
    GPU_FREQ_RANGE = range(306, 625)
    MEMORY_FREQ_RANGE = [2133]
    CL_RANGE = range(1, 4)
    ranges = [
        (0, 1),
        (min(ACTIONS), max(ACTIONS) + 1),
        (min(ACTIONS), max(ACTIONS) + 1),
        (0, 1),
        (0, 3)
    ]
    action_shape = [1, len(ACTIONS), len(ACTIONS), 1, len(ACTIONS)]

sampled_configs ={
     "cpu_cores": np.array(CPU_CORES_RANGE), 
     "cpu_freq": np.array(CPU_FREQ_RANGE), 
     "gpu_freq": np.array(GPU_FREQ_RANGE), 
     "memory_freq": np.array(MEMORY_FREQ_RANGE), 
     "cl": np.array(CL_RANGE)
}

best_throughput = -float('inf')
max_reward = -float('inf')
last_reward = 0

# Hyperparameters
alpha = 0.1
gamma = 0.9
epsilon_explore = 0.5
epsilon_exploit = 0.5
epsilon_min = 1e-10  # Minimum epsilon value (always exploit after this threshold)
num_episodes = int(sys.argv[6])  # Number of episodes to run

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
def adjust_value(value, action, state, range):
    if action == 1:
        state = min(state + 1, max(range))
    elif action == 2:
        state = max(min(range), state - 1)
    elif action == 3:
        state = min(state + 50, max(range))
    elif action == 4:
        state = max(min(range), state - 50)
    elif action == 5:
        state = min(state + 100, max(range))
    elif action == 6:
        state = max(min(range), state - 100)
    return value[state]

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
    num_samples = 9  # Number of samples per episode
    samples = lhs_sampling(num_samples, ranges)
    return [tuple(map(int, sample)) for sample in samples]

def get_best_configuration():
    best_q_value = float('-inf')
    best_state = None

    for state, q_values in Q_table.items():
        max_q_index = np.argmax(q_values)  # Find the index of the max Q-value
        max_q_value = q_values[max_q_index]
        
        if max_q_value >= best_q_value:
            best_q_value = max_q_value
            best_state = state
        if best_state == None:
            best_state = state

    best_state = tuple([int(conf[int(x)]) for x, conf in zip(best_state, [sampled_configs['cpu_cores'], sampled_configs['cpu_freq'], sampled_configs['gpu_freq'], sampled_configs['memory_freq'], sampled_configs['cl']])])
    return best_state

def get_second_best_configuration(action_index, action_shape, episode):
    best_q_value = float('-inf')
    best_state = None
    second_q_value = float('inf')
    second_best_state = None

    for state, q_values in Q_table.items():
        max_q_index = np.argmax(q_values)  # Find the index of the max Q-value
        max_q_value = q_values[max_q_index]
        
        if max_q_value >= best_q_value:
            best_q_value = max_q_value
            best_state = state
        if best_state == None:
            best_state = state

    best_state = tuple([int(conf[int(x)]) for x, conf in zip(best_state, [sampled_configs['cpu_cores'], sampled_configs['cpu_freq'], sampled_configs['gpu_freq'], sampled_configs['memory_freq'], sampled_configs['cl']])])
    best_state_index = tuple(state_to_index(*best_state))
    if episode < 50:
        for state, q_values in Q_table.items():
            min_q_index = np.argmin(q_values)  # Find the index of the max Q-value
            min_q_value = q_values[min_q_index]
            
            if min_q_value <= second_q_value:
                second_q_value = min_q_value
                second_best_state = state
            if second_best_state == None:
                second_best_state = state

        second_best_state = tuple([int(conf[int(x)]) for x, conf in zip(second_best_state, [sampled_configs['cpu_cores'], sampled_configs['cpu_freq'], sampled_configs['gpu_freq'], sampled_configs['memory_freq'], sampled_configs['cl']])])

        return second_best_state
    else:
        if best_state:
            Q_table[best_state_index][np.ravel_multi_index(tuple(action_index), tuple(action_shape))] = float('-inf')
        
        for state, q_values in Q_table.items():
            max_q_index = np.argmax(q_values)  # Find the index of the max Q-value
            max_q_value = q_values[max_q_index]
            
            if max_q_value >= second_q_value:
                second_q_value = max_q_value
                second_best_state = state
            if second_best_state == None:
                second_best_state = state

        second_best_state = tuple([int(conf[int(x)]) for x, conf in zip(second_best_state, [sampled_configs['cpu_cores'], sampled_configs['cpu_freq'], sampled_configs['gpu_freq'], sampled_configs['memory_freq'], sampled_configs['cl']])])
        Q_table[best_state_index][np.ravel_multi_index(tuple(action_index), tuple(action_shape))] = best_q_value

        return second_best_state

# Retrieve Q-value for a state-action pair
def get_q_value(state_index, action_index):
    state_key = tuple(state_index)
    if state_key not in Q_table:
        Q_table[state_key] = np.zeros(np.prod(action_shape))  # Initialize if not present
    return Q_table[state_key][np.ravel_multi_index(tuple(action_index), tuple(action_shape))]

def update_q_table(state_index, action_index):
    state_key = tuple(state_index)
    if state_key not in Q_table:
        Q_table[state_key] = np.zeros(np.prod(action_shape))  # Initialize if not present
        Q_table[state_key][np.ravel_multi_index(tuple(action_index), tuple(action_shape))] = 0
    else:
        for state, q_values in Q_table.items():
            if state == state_key:
                max_q_index = np.argmax(q_values)  # Find the index of the max Q-value
                max_q_value = q_values[max_q_index]
                Q_table[state_key] = np.zeros(np.prod(action_shape))
                Q_table[state_key][np.ravel_multi_index(tuple(action_index), tuple(action_shape))] = max_q_value

# Update Q-value for a state-action pair
def update_q_value(state_index, action_index, new_value):
    state_key = tuple(state_index)
    if state_key not in Q_table:
        Q_table[state_key] = np.zeros(np.prod(action_shape))  # Initialize if not present
    Q_table[state_key][np.ravel_multi_index(tuple(action_index), tuple(action_shape))] = new_value

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
    global epsilon_explore, epsilon_exploit, action_shape
    state_key = tuple(state_index)
    # Select action based on epsilon
    if (epsilon_explore/epsilon_exploit) > 0.75:
        return calculate_diversity(lhs_samples, state_key), "exploration"
    else:
        # Exploitation: choose best known action
        if state_key not in Q_table:
            return calculate_diversity(lhs_samples, state_key), "exploration" # Use LHS samples for unseen states
        return np.unravel_index(np.argmax(Q_table[state_key]), action_shape), "exploitation"

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

th_target = int(sys.argv[7])
target_pass = False
locked = False

def increment_target(outcome, target):
    global locked
    if not locked:
        if outcome == 'positive':
            target += 5
        else:
            target -= 5
    if target == 0 and not locked:
        target += 3
        locked = True
    return target

# Calculate reward with shaping
def calculate_reward(measured_metrics, target):
    power = measured_metrics[0]["power_cons"]
    throughput = measured_metrics[0]["throughput"]
    
    if throughput <= target:
        return -1e6
    
    return (throughput/power * 1e-6)

# CSV saving optimization
def save_csv(dict_list, filename):
    with open(filename, 'a', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=['api_time','id', 'q-value', 'phase', 'episode', 'xaviernx_time_elapsed', 'ql_time_elapsed', 'cpu_cores', 'cpu_freq', 'gpu_freq', 'memory_freq', 'cl', 'elapsed', 'time_load', 'time_warm', 'time_c', 'th_target', 'throughput', 'power_cons', 'cpu_percent', 'gpu_percent', 'mem_percent'])
        if os.path.getsize(filename) == 0:
            writer.writeheader()
        for d in dict_list:
            writer.writerow(d)

# Initial configuration (starting in the middle of the range)
cpu_cores, cpu_freq, gpu_freq, memory_freq, cl = max(sampled_configs['cpu_cores']), max(sampled_configs['cpu_freq']), max(sampled_configs['gpu_freq']), max(sampled_configs['memory_freq']), max(sampled_configs['cl'])
state_index = state_to_index(cpu_cores, cpu_freq, gpu_freq, memory_freq, cl)
best_action = None
best_q = -float('inf')
episode = 1
max_fail = 5
fail = 0

failed_pt = False
best_config = {
    "th_target": th_target,
    "api_time": 0,
    "cpu_cores": cpu_cores+1,
    "cpu_freq": cpu_freq,
    "gpu_freq": gpu_freq,
    "memory_freq": memory_freq,
    "cl": cl
}
rewards = [0]
set_target = 0

# Execution loop with adaptive epsilon strategy
while episode <= 30:
    t2 = time.time()
    if episode <= (num_episodes) or failed_pt:
        current_thtarget = th_target
        # Generate LHS samples for this episode
        lhs_samples = generate_lhs_samples()

        # Choose actions based on current state and LHS samples
        actions, phase = choose_action_adaptive(state_index, lhs_samples)
        
        # Adjust values for the chosen actions
        cpu_cores = int(adjust_value(sampled_configs['cpu_cores'], actions[0], state_index[0], range(len(CPU_CORES_RANGE))))
        cpu_freq = int(adjust_value(sampled_configs['cpu_freq'], actions[1], state_index[1], range(len(CPU_FREQ_RANGE))))
        gpu_freq = int(adjust_value(sampled_configs['gpu_freq'], actions[2], state_index[2], range(len(GPU_FREQ_RANGE))))
        memory_freq = int(adjust_value(sampled_configs['memory_freq'], actions[3], state_index[3], range(len(MEMORY_FREQ_RANGE))))
        cl = int(adjust_value(sampled_configs['cl'], actions[4], state_index[4], range(len(CL_RANGE))))

        state_key = state_to_index(cpu_cores, cpu_freq, gpu_freq, memory_freq, cl)
        update_q_table(state_key, actions)
        if not failed_pt:
            if state_key in Q_table and bool(get_q_value(state_key, actions)) == True:
                print("STUCK CONFIG, RESET TO DEFAULT CONFIG!")
                epsilon_explore = 0.5
                epsilon_exploit = 0.5
                continue

    else:
        if set_target == 0:
            th_target = best_config['th_target']
        current_thtarget = th_target
        cpu_cores, cpu_freq, gpu_freq, memory_freq, cl = get_best_configuration()
        actions = (0, 0 ,0, 0, 0)
        phase = "testing"

    # # Print the chosen configuration for tracking
    # print({"cpu_cores": cpu_cores+1, "cpu_freq": cpu_freq, "gpu_freq": gpu_freq, "memory_freq": memory_freq, "cl": cl})

    # Convert to new state index
    new_state_index = state_to_index(cpu_cores, cpu_freq, gpu_freq, memory_freq, cl)

    # Check for prohibited configurations
    if (*new_state_index, current_thtarget) in prohibited_configs:
        if episode > (num_episodes) and max_fail <= fail:
            epsilon_explore = 1
            epsilon_exploit = 2
            failed_pt = True
            fail += 1
            continue
        elif episode <= (num_episodes):
            print("PROHIBITED CONFIG, RESET TO DEFAULT CONFIG!")
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
    reward = calculate_reward(measured_metrics, current_thtarget)

    if reward == -1e6:
        print("PROHIBITED CONFIG!")
        prohibited_configs.add((*new_state_index, current_thtarget))
    else:
        failed_pt = False

    if (episode%5) == 0 and episode <= (num_episodes):
        outcome = 'positive' if reward > 0 else 'negative'
        th_target = increment_target(outcome, th_target)
    if episode > (num_episodes):
        if set_target <= 2:
                outcome = 'positive' if reward > 0 else 'negative'
                phase = "exploitation"
                if set_target == 2:
                    if outcome == 'positive':
                        th_target = current_thtarget
                    else:
                        th_target = increment_target(outcome, th_target)
                else:
                    th_target = increment_target(outcome, th_target)
                set_target += 1
        else:
            phase = "testing"
        
    update_q_table(state_index, actions)
    new_actions, _ = choose_action_adaptive(new_state_index, lhs_samples)
    # Update Q-value using the Bellman equation
    old_q_value = get_q_value(state_index, actions)
    max_next_q_value = np.max(Q_table.get(tuple(new_state_index), np.zeros(np.prod(action_shape))))

    new_q_value = old_q_value + alpha * (reward + gamma * max_next_q_value - old_q_value)
    update_q_table(new_state_index, new_actions)
    update_q_value(new_state_index, new_actions, new_q_value)

    # Track the best configuration
    if new_q_value > best_q:
        max_reward = reward
        best_config = {
            "th_target": current_thtarget,
            "api_time": api_time,
            "cpu_cores": cpu_cores+1,
            "cpu_freq": cpu_freq,
            "gpu_freq": gpu_freq,
            "memory_freq": memory_freq,
            "cl": cl
        }
        best_throughput = measured_metrics[0]["throughput"]
        best_q = new_q_value
        best_action = actions

    elapsed = round(((time.time() - t2) - elapsed_exec) * 1000, 3)

    # Adaptive strategy: increase epsilon if reward is too low, decrease it if reward is sufficient
    if reward == -1e6:
        epsilon_explore = min(epsilon_explore * 1.05, 1)  # Increase epsilon if performance is bad
    else:
        epsilon_exploit = min(epsilon_exploit * 1.05, 1)

    configs = {
    "th_target": current_thtarget,
    "api_time": api_time,
    "q-value": new_q_value,
        "phase":phase,
        "episode": episode,
        "xaviernx_time_elapsed": elapsed_exec,
        "ql_time_elapsed": elapsed,
        "cpu_cores": cpu_cores + 1,
        "cpu_freq": cpu_freq,
        "gpu_freq": gpu_freq,
        "memory_freq": memory_freq,
        "cl": cl
    }

    dict_record = [{**configs, **measured_metrics[0]}]

    save_csv(dict_record, f"ql-{num_episodes}_{sys.argv[5]}_{sys.argv[4]}.csv")
    
    # Update state and last reward
    last_reward = reward
    state_index = new_state_index
    rewards.append(reward)

    episode += 1

    print(f"Episode: {episode}, Q-Value: {new_q_value}, Max Q-Value: {best_q}")

print(f"Best Config: {best_config} with Q-Value: {best_q}")
