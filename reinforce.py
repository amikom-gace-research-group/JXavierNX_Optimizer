import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import requests
import random
import time
import sys
import csv
import os

if sys.argv[5] == 'jxavier':
    CPU_CORES_RANGE = range(1, 6)
    CPU_FREQ_RANGE = range(1190, 1909)
    GPU_FREQ_RANGE = range(510, 1111)
    MEMORY_FREQ_RANGE = range(1500, 1867)
    CL_RANGE = range(1, 4)
elif sys.argv[5] == 'jorin-nano':
    CPU_CORES_RANGE = [5]
    CPU_FREQ_RANGE = range(806, 1511)
    GPU_FREQ_RANGE = range(306, 625)
    MEMORY_FREQ_RANGE = [2133]
    CL_RANGE = range(1, 4)

sampled_configs ={
     "cpu_cores": CPU_CORES_RANGE, 
     "cpu_freq": np.linspace(min(CPU_FREQ_RANGE), max(CPU_FREQ_RANGE), 3), 
     "gpu_freq": np.linspace(min(GPU_FREQ_RANGE), max(GPU_FREQ_RANGE), 3), 
     "memory_freq": np.linspace(min(MEMORY_FREQ_RANGE), max(MEMORY_FREQ_RANGE), 3), 
     "cl": CL_RANGE
}

POWER_BUDGET = int(sys.argv[6])

# Hyperparameters
gamma = 0.99
lr = 0.001
num_episodes = 10

prohibited_configs = set()

# Actor Network
class ActorNetwork(nn.Module):
    def __init__(self, input_size, output_sizes):
        super(ActorNetwork, self).__init__()
        self.fc1 = nn.Linear(input_size, 128)
        self.fc2 = nn.Linear(128, 128)
        
        # Separate output layers for each parameter
        self.cpu_cores_out = nn.Linear(128, output_sizes['cpu_cores'])
        self.cpu_freq_out = nn.Linear(128, output_sizes['cpu_freq'])
        self.gpu_freq_out = nn.Linear(128, output_sizes['gpu_freq'])
        self.memory_freq_out = nn.Linear(128, output_sizes['memory_freq'])
        self.cl_out = nn.Linear(128, output_sizes['cl'])

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        
        # Output probabilities for each parameter independently
        cpu_cores_probs = torch.softmax(self.cpu_cores_out(x), dim=-1)
        cpu_freq_probs = torch.softmax(self.cpu_freq_out(x), dim=-1)
        gpu_freq_probs = torch.softmax(self.gpu_freq_out(x), dim=-1)
        memory_freq_probs = torch.softmax(self.memory_freq_out(x), dim=-1)
        cl_probs = torch.softmax(self.cl_out(x), dim=-1)
        
        # Return action probabilities for each parameter
        return {
            'cpu_cores': cpu_cores_probs,
            'cpu_freq': cpu_freq_probs,
            'gpu_freq': gpu_freq_probs,
            'memory_freq': memory_freq_probs,
            'cl': cl_probs
        }

# Adjust configuration values based on action
def adjust_value(value, action):
    unique_values = sorted(value)
    if len(unique_values) != 3:
        return min(unique_values)
    else:
        return unique_values[action]
    
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

def save_csv(dict_list, filename):
    with open(filename, 'a', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=['api_time','episode', 'reward', 'xaviernx_time_elapsed', 'reinforce_time_elapsed', 'cpu_cores', 'cpu_freq', 'gpu_freq', 'memory_freq', 'cl', 'throughput', 'power_cons', 'cpu_percent', 'gpu_percent', 'mem_percent'])
        if os.path.getsize(filename) == 0:
            writer.writeheader()
        for d in dict_list:
            writer.writerow(d)

# Efficient reward calculation
def calculate_reward(measured_metrics):
    power = measured_metrics[0]["power_cons"]
    throughput = measured_metrics[0]["throughput"]
    
    if power > POWER_BUDGET:
        return 1e-6
    
    return (throughput / power) * 1e6

# Main REINFORCE algorithm
def reinforce_algorithm(actor_network, optimizer):
    t1 = time.time()
    max_reward = -float('inf')
    best_throughput = -float('inf')
    best_config = None
    time_got = []
    configs = []
    episode = 0

    # Initial configuration (starting in the middle of the range)
    cpu_cores, cpu_freq, gpu_freq, memory_freq, cl = max(sampled_configs['cpu_cores']), max(sampled_configs['cpu_freq']), max(sampled_configs['gpu_freq']), max(sampled_configs['memory_freq']), max(sampled_configs['cl'])

    for _ in range(num_episodes):
        states, actions, rewards = [], [], []
        state = np.array([cpu_cores, cpu_freq, gpu_freq, memory_freq, cl])

        for _ in range(10):  # Define the number of steps per episode
            episode += 1
            state_tensor = torch.tensor(state, dtype=torch.float32)
            
            # Actor chooses action
            action_probs = actor_network(state_tensor)
            cpu_cores_action = torch.distributions.Categorical(action_probs['cpu_cores']).sample().item()
            cpu_freq_action = torch.distributions.Categorical(action_probs['cpu_freq']).sample().item()
            gpu_freq_action = torch.distributions.Categorical(action_probs['gpu_freq']).sample().item()
            memory_freq_action = torch.distributions.Categorical(action_probs['memory_freq']).sample().item()
            cl_action = torch.distributions.Categorical(action_probs['cl']).sample().item()

            actions_set = (cpu_cores_action, cpu_freq_action, gpu_freq_action, memory_freq_action, cl_action)
            actions.append(actions_set)

            # Adjust values for the chosen actions
            cpu_cores = int(adjust_value(sampled_configs['cpu_cores'], cpu_cores_action))
            cpu_freq = int(adjust_value(sampled_configs['cpu_freq'], cpu_freq_action))
            gpu_freq = int(adjust_value(sampled_configs['gpu_freq'], gpu_freq_action))
            memory_freq = int(adjust_value(sampled_configs['memory_freq'], memory_freq_action))
            cl = int(adjust_value(sampled_configs['cl'], cl_action))

            state = np.array([cpu_cores, cpu_freq, gpu_freq, memory_freq, cl])

            if str(state) in prohibited_configs:
                print("PROHIBITED CONFIG!")
                cpu_cores, cpu_freq, gpu_freq, memory_freq, cl = random.choice(sampled_configs['cpu_cores']), random.choice(sampled_configs['cpu_freq']), random.choice(sampled_configs['gpu_freq']), random.choice(sampled_configs['memory_freq']), random.choice(sampled_configs['cl'])
                continue

            # Execute configuration and get metrics
            t1 = time.time()
            measured_metrics, api_time = execute_config(cpu_cores, cpu_freq, gpu_freq, memory_freq, cl)
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

            if reward == 1e-6:
                print("Prohibited Configuration!")
                prohibited_configs.add(str(state))

            config = {
	            "api_time": api_time,
                "episode": episode,
                "reward": reward,
                "xaviernx_time_elapsed": elapsed_exec,
                "cpu_cores": cpu_cores+1,
                "cpu_freq": cpu_freq,
                "gpu_freq": gpu_freq,
                "memory_freq": memory_freq,
                "cl": cl,
                "throughput": measured_metrics[0]["throughput"],
                "power_cons": measured_metrics[0]["power_cons"],
                "cpu_percent": measured_metrics[0]["cpu_percent"],
                "gpu_percent": measured_metrics[0]["gpu_percent"],
                "mem_percent": measured_metrics[0]["mem_percent"]
            }
            configs.append(config)

            state = np.array([cpu_cores, cpu_freq, gpu_freq, memory_freq, cl])
            states.append(state)

            if reward > max_reward and measured_metrics[0]["throughput"] > best_throughput:
                max_reward = reward
                best_config = state
                best_throughput = measured_metrics[0]["throughput"]

        # Calculate returns
        returns = []
        Gt = 0
        if len(rewards) == 0:
            continue
        for reward in reversed(rewards):
            Gt = reward + gamma * Gt  # Discounted reward
            returns.insert(0, Gt)

        returns = torch.tensor(np.array(returns), dtype=torch.float32)

        # Update the actor network
        optimizer.zero_grad()
        for log_prob, Gt in zip(actions, returns):
            cpu_cores_log_prob = torch.log(action_probs['cpu_cores'][log_prob[0]])
            cpu_freq_log_prob = torch.log(action_probs['cpu_freq'][log_prob[1]])
            gpu_freq_log_prob = torch.log(action_probs['gpu_freq'][log_prob[2]])
            memory_freq_log_prob = torch.log(action_probs['memory_freq'][log_prob[3]])
            cl_log_prob = torch.log(action_probs['cl'][log_prob[4]])
            total_log_prob = (cpu_cores_log_prob + cpu_freq_log_prob + gpu_freq_log_prob +
                            memory_freq_log_prob + cl_log_prob)
            loss = -(total_log_prob * Gt)  # Policy gradient loss
        loss.backward(retain_graph=True)
        optimizer.step()

        print(f"Episode: {episode}, Max Reward: {max_reward}")

    end_t1 = round(((time.time() - t1) - sum(time_got))*1000, 3)
    end = end_t1 / len(time_got)
    for config in configs:
        dict_record = [{'reinforce_time_elapsed': end, **config}]
        save_csv(dict_record, f"reinforce_{sys.argv[5]}_{sys.argv[4]}.csv")
    print(f"Best Config: {best_config} in {sum(time_got)+end_t1} sec")

# Initialize the actor network
input_size = 5  # State representation: cpu_cores, cpu_freq, gpu_freq, memory_freq, cl
output_sizes = {
    'cpu_cores': 3,   # Number of actions for cpu_cores
    'cpu_freq': 3,    # Number of actions for cpu_freq
    'gpu_freq': 3,    # Number of actions for gpu_freq
    'memory_freq': 3, # Number of actions for memory_freq
    'cl': 3           # Number of actions for cl
}  # Number of actions (e.g., no change, small/medium/large increase/decrease)

actor_network = ActorNetwork(input_size, output_sizes)
optimizer = optim.Adam(actor_network.parameters(), lr=lr)

# Run the REINFORCE algorithm
reinforce_algorithm(actor_network, optimizer)
