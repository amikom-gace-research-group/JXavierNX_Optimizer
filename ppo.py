import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import os
import csv
import sys
import time
import requests

# Constants and thresholds
POWER_BUDGET = 5000
THROUGHPUT_TARGET = 30
importance_power = 1
importance_throughput = 1

# Define configuration ranges
CPU_CORES_RANGE = range(1, 6)
CPU_FREQ_RANGE = range(1190, 1909)
GPU_FREQ_RANGE = range(510, 1111)
MEMORY_FREQ_RANGE = range(1500, 1867)
CL_RANGE = range(1, 4)

# Step sizes for adjustment
STEP_SIZES = {
    'cpu_cores': (1, 3, 5),
    'cpu_freq': (1, 10, 50),
    'gpu_freq': (1, 10, 50),
    'memory_freq': (1, 10, 50),
    'cl': (1, 2, 0)
}

# Hyperparameters
alpha = 0.1
gamma = 0.9
epsilon = 0.5
epsilon_min = 0.01
epsilon_decay = 0.995
num_episodes = 5
reward_threshold = 0.01
max_saturated_count = 5

prohibited_configs = set()

# Policy Network
class PolicyNetwork(nn.Module):
    def __init__(self, input_size, output_size):
        super(PolicyNetwork, self).__init__()
        self.fc1 = nn.Linear(input_size, 128)
        self.fc2 = nn.Linear(128, 128)
        self.fc3 = nn.Linear(128, output_size)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        return torch.softmax(self.fc3(x), dim=-1)

# PPO Loss Function
def ppo_loss(old_log_probs, new_log_probs, advantages, epsilon=0.2):
    ratio = torch.exp(new_log_probs - old_log_probs)
    clip = torch.clamp(ratio, 1 - epsilon, 1 + epsilon)
    return -torch.min(ratio * advantages, clip * advantages).mean()

# Calculate Advantages using Generalized Advantage Estimation (GAE)
def calculate_advantages(rewards, values, gamma=0.99, lamda=0.95):
    advantages = []
    gae = 0
    for i in reversed(range(len(rewards))):
        delta = rewards[i] + gamma * values[i+1] - values[i]
        gae = delta + gamma * lamda * gae
        advantages.insert(0, gae)
    return advantages

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
    return value

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

# CSV saving optimization
def save_csv(dict_list, filename):
    with open(filename, 'a', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=['id', 'reward', 'xaviernx_time_elapsed', 'ppo_time_elapsed', 'cpu_cores', 'cpu_freq', 'gpu_freq', 'memory_freq', 'cl', 'throughput', 'power_cons'])
        if os.path.getsize(filename) == 0:
            writer.writeheader()
        for d in dict_list:
            writer.writerow(d)

# Efficient reward calculation
def calculate_reward(measured_metrics):
    power = measured_metrics[0]["power_cons"]
    throughput = measured_metrics[0]["throughput"]
    
    if power > POWER_BUDGET or throughput < THROUGHPUT_TARGET:
        return -1
    
    return (importance_throughput * (throughput / THROUGHPUT_TARGET) +
            importance_power * (POWER_BUDGET / power))

# Main PPO Loop
def ppo_algorithm(policy_network, optimizer):
    t1 = time.time()
    max_reward = -float('inf')
    best_config = None
    time_got = []
    configs = []

    # Initial configuration (starting in the middle of the range)
    cpu_cores = 3
    cpu_freq = 1550
    gpu_freq = 810
    memory_freq = 1700
    cl = 2

    for episode in range(num_episodes):
        # Collect data for each episode
        states, actions, rewards, log_probs = [], [], [], []
        state = np.array([cpu_cores, cpu_freq, gpu_freq, memory_freq, cl])

        for step in range(100):  # Define the number of steps per episode
            state_tensor = torch.tensor(state, dtype=torch.float32)
            action_probs = policy_network(state_tensor)
            action_distribution = torch.distributions.Categorical(action_probs)
            action = action_distribution.sample()

            log_prob = action_distribution.log_prob(action)
            actions.append(action.item())
            log_probs.append(log_prob)

            # Adjust configurations based on actions
            cpu_cores = adjust_value(cpu_cores, actions[0], STEP_SIZES['cpu_cores'], min(CPU_CORES_RANGE), max(CPU_CORES_RANGE))
            cpu_freq = adjust_value(cpu_freq, actions[1], STEP_SIZES['cpu_freq'], min(CPU_FREQ_RANGE), max(CPU_FREQ_RANGE))
            gpu_freq = adjust_value(gpu_freq, actions[2], STEP_SIZES['gpu_freq'], min(GPU_FREQ_RANGE), max(GPU_FREQ_RANGE))
            memory_freq = adjust_value(memory_freq, actions[3], STEP_SIZES['memory_freq'], min(MEMORY_FREQ_RANGE), max(MEMORY_FREQ_RANGE))
            cl = adjust_value(cl, actions[4], STEP_SIZES['cl'], min(CL_RANGE), max(CL_RANGE))

            # Execute configuration and get metrics
            t1 = time.time()
            measured_metrics = execute_config(cpu_cores, cpu_freq, gpu_freq, memory_freq, cl)
            elapsed_exec = round(time.time() - t1, 3)
            time_got.append(elapsed_exec)

            if not measured_metrics:
                print("EXECUTION PROBLEM!")
                continue
            if measured_metrics == "No Device":
                print("No Device/Inference Runtime")
                break

            reward = calculate_reward(measured_metrics)
            rewards.append(reward)

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

            if reward == -1:
                print("Prohibited Configuration!")
                prohibited_configs.add(state)
                continue

            state = np.array([cpu_cores, cpu_freq, gpu_freq, memory_freq, cl])
            
            if reward > max_reward:
                max_reward = reward
                best_config = state

        # Compute the advantages
        values = [0] * len(rewards)  # Replace with value estimates if you have a value network
        advantages = calculate_advantages(rewards, values)

        # Update the policy network using PPO loss
        optimizer.zero_grad()
        for log_prob, adv in zip(log_probs, advantages):
            new_log_probs = policy_network(torch.tensor(state, dtype=torch.float32))
            loss = ppo_loss(log_prob, new_log_probs, adv)
            loss.backward()
        optimizer.step()

        # Epsilon decay for exploration (optional)
        if epsilon > epsilon_min:
            epsilon *= epsilon_decay

        print(f"Episode: {episode}, Max Reward: {max_reward}")

    end_t1 = round(((time.time() - t1) - sum(time_got))*1000, 3)
    end = end_t1 / len(time_got)
    for config in configs:
        dict_record = [{'ppo_time_elapsed': end, **config}]
        save_csv(dict_record, f"ppo_jxavier_{sys.argv[4]}.csv")
    print(f"Best Config: {best_config} in {sum(time_got)+end_t1} sec")

# Initialize the policy network and optimizer
input_size = 5  # State representation: cpu_cores, cpu_freq, gpu_freq, memory_freq, cl
output_size = 7  # Number of actions (e.g., no change, small/medium/large increase/decrease)

policy_network = PolicyNetwork(input_size, output_size)
optimizer = optim.Adam(policy_network.parameters(), lr=0.001)

# Run the PPO algorithm
ppo_algorithm(policy_network, optimizer)
