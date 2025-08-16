import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import requests
import random
import time
import sys
import os
import csv

if sys.argv[5] == 'jxavier':
    CPU_CORES_RANGE = range(1, 6)
    CPU_FREQ_RANGE = range(1190, 1909)
    GPU_FREQ_RANGE = range(510, 1111)
    MEMORY_FREQ_RANGE = range(1500, 1867)
    CL_RANGE = range(1, 4)
    output_sizes = {
    'cpu_cores': 3,   # Number of actions for cpu_cores
    'cpu_freq': 6,    # Number of actions for cpu_freq
    'gpu_freq': 6,    # Number of actions for gpu_freq
    'memory_freq': 6, # Number of actions for memory_freq
    'cl': 3           # Number of actions for cl
    }
elif sys.argv[5] == 'jorin-nano':
    CPU_CORES_RANGE = [5]
    CPU_FREQ_RANGE = range(806, 1511)
    GPU_FREQ_RANGE = range(306, 625)
    MEMORY_FREQ_RANGE = [2133]
    CL_RANGE = range(1, 4)
    output_sizes = {
    'cpu_cores': 1,   # Number of actions for cpu_cores
    'cpu_freq': 6,    # Number of actions for cpu_freq
    'gpu_freq': 6,    # Number of actions for gpu_freq
    'memory_freq': 1, # Number of actions for memory_freq
    'cl': 3           # Number of actions for cl
    }

sampled_configs ={
     "cpu_cores": CPU_CORES_RANGE, 
     "cpu_freq": CPU_FREQ_RANGE, 
     "gpu_freq": GPU_FREQ_RANGE, 
     "memory_freq": MEMORY_FREQ_RANGE, 
     "cl": CL_RANGE
}

# Hyperparameters
gamma = 0.99
lr = 0.001
num_episodes = round((int(sys.argv[6]))/2)

prohibited_configs = set()

# Actor Network with separate output layers for each configuration parameter
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

# Critic Network
class CriticNetwork(nn.Module):
    def __init__(self, input_size):
        super(CriticNetwork, self).__init__()
        self.fc1 = nn.Linear(input_size, 128)
        self.fc2 = nn.Linear(128, 128)
        self.fc3 = nn.Linear(128, 1)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        return self.fc3(x)

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

# CSV saving optimization
def save_csv(dict_list, filename):
    with open(filename, 'a', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=['api_time','id', 'episode', 'reward', 'xaviernx_time_elapsed', 'a2c_time_elapsed', 'cpu_cores', 'cpu_freq', 'gpu_freq', 'mem_freq', 'cl', 'elapsed', 'time_load', 'time_warm', 'time_c', 'th_target', 'throughput', 'power_cons', 'cpu_percent', 'gpu_percent', 'mem_percent'])
        if os.path.getsize(filename) == 0:
            writer.writeheader()
        for d in dict_list:
            writer.writerow(d)

th_target = int(sys.argv[7])
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

# Reward function based on power and throughput metrics
def calculate_reward(measured_metrics, target):
    power = measured_metrics[0]["power_cons"]
    throughput = measured_metrics[0]["throughput"]
    
    if throughput <= target:
        return -1e6
    
    return (throughput/power * 1e-6)

set_target = 0

def exec_trained(best_configs):
    global rewards, configs, set_target, th_target
    outcome = 'positive' if rewards[-1] > 0 else 'negative'
    th_target = increment_target(outcome, th_target)
    for eps in range(int(sys.argv[6])+1, 30):
        current_thtarget = th_target
        t1 = time.time()
        metrics, api_time = execute_config(*best_configs)
        elapsed_exec = round(time.time() - t1, 3)
        if not metrics or metrics == "No Device":
            continue
        if metrics == "No Device":
            break
        reward = calculate_reward(metrics, current_thtarget)
        if set_target <= 2:
            outcome = 'positive' if reward > 0 else 'negative'
            if set_target == 2:
                if outcome == 'positive':
                    th_target = current_thtarget
                else:
                    th_target = increment_target(outcome, th_target)
            else:
                th_target = increment_target(outcome, th_target)
            set_target += 1
        config = {
            "api_time": api_time,
            "episode": eps,
            "reward": reward,
            "xaviernx_time_elapsed": elapsed_exec,
            'a2c_time_elapsed': 0,
            'cpu_cores': int(best_configs[0])+1,
            'cpu_freq': int(best_configs[1]),
            'gpu_freq': int(best_configs[2]),
            'mem_freq': int(best_configs[3]),
            'cl': int(best_configs[4]),
            "elapsed": metrics[0]["elapsed"],
            "time_load": metrics[0]["time_load"],
            "time_warm": metrics[0]["time_warm"],
            "time_c": metrics[0]["time_c"],
            "th_target": current_thtarget,
            "throughput": metrics[0]["throughput"],
            "power_cons": metrics[0]["power_cons"],
            "cpu_percent": metrics[0]["cpu_percent"],
            "gpu_percent": metrics[0]["gpu_percent"],
            "mem_percent": metrics[0]["mem_percent"]
        }
        save_csv([config], f"a2c_{int(sys.argv[6])}_{sys.argv[5]}_{sys.argv[4]}.csv")

# Main A2C algorithm
def a2c_algorithm(actor_network, critic_network, actor_optimizer, critic_optimizer):
    global rewards, configs, th_target
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

        for _ in range(2):  # Define the number of steps per episode
            episode += 1
            current_thtarget = th_target
            state = np.array([cpu_cores, cpu_freq, gpu_freq, memory_freq, cl])
            state_tensor = torch.tensor(state, dtype=torch.float32)

            # Actor chooses an action for each configuration parameter
            action_probs = actor_network(state_tensor)
            cpu_cores_action = torch.distributions.Categorical(action_probs['cpu_cores']).sample().item()
            cpu_freq_action = torch.distributions.Categorical(action_probs['cpu_freq']).sample().item()
            gpu_freq_action = torch.distributions.Categorical(action_probs['gpu_freq']).sample().item()
            memory_freq_action = torch.distributions.Categorical(action_probs['memory_freq']).sample().item()
            cl_action = torch.distributions.Categorical(action_probs['cl']).sample().item()

            actions_set = (cpu_cores_action, cpu_freq_action, gpu_freq_action, memory_freq_action, cl_action)
            actions.append(actions_set)

            state_index = state_to_index(cpu_cores, cpu_freq, gpu_freq, memory_freq, cl)
            
            # Adjust configurations based on actions
            cpu_cores = int(adjust_value(sampled_configs['cpu_cores'], cpu_cores_action, state_index[0], range(len(CPU_CORES_RANGE))))
            cpu_freq = int(adjust_value(sampled_configs['cpu_freq'], cpu_freq_action, state_index[1], range(len(CPU_FREQ_RANGE))))
            gpu_freq = int(adjust_value(sampled_configs['gpu_freq'], gpu_freq_action, state_index[2], range(len(GPU_FREQ_RANGE))))
            memory_freq = int(adjust_value(sampled_configs['memory_freq'], memory_freq_action, state_index[3], range(len(MEMORY_FREQ_RANGE))))
            cl = int(adjust_value(sampled_configs['cl'], cl_action, state_index[4], range(len(CL_RANGE))))

            state = np.array([cpu_cores, cpu_freq, gpu_freq, memory_freq, cl, th_target])

            if str(state) in prohibited_configs:
                print("PROHIBITED CONFIG!")
                cpu_cores, cpu_freq, gpu_freq, memory_freq, cl = random.choice(sampled_configs['cpu_cores']), random.choice(sampled_configs['cpu_freq']), random.choice(sampled_configs['gpu_freq']), random.choice(sampled_configs['memory_freq']), random.choice(sampled_configs['cl'])
                continue

            # Execute configuration and get metrics
            t2 = time.time()
            measured_metrics, api_time = execute_config(cpu_cores, cpu_freq, gpu_freq, memory_freq, cl)
            elapsed_exec = round(time.time() - t2, 3)
            time_got.append(elapsed_exec)

            if not measured_metrics:
                print("EXECUTION PROBLEM!")
                continue
            if measured_metrics == "No Device":
                print("No Device/Inference Runtime")
                break

            reward = calculate_reward(measured_metrics, th_target)
            if (episode % 5) == 0:
                outcome = 'positive' if reward > 0 else 'negative'
                th_target = increment_target(outcome, th_target)
            rewards.append(reward)

            if reward == -1e6:
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
                "mem_freq": memory_freq,
                "cl": cl,
                "elapsed": measured_metrics[0]["elapsed"],
                "time_load": measured_metrics[0]["time_load"],
                "time_warm": measured_metrics[0]["time_warm"],
                "time_c": measured_metrics[0]["time_c"],
                "th_target": th_target,
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

        # Calculate advantages and update networks
        returns, advantages = [], []
        states = np.array(states)
        if states.size == 0 or states.shape[1] != 5:  # replace 5 with the correct feature count
            print("Error: `states` has an unexpected shape:", states.shape)
            continue
        else:
            value_tensor = critic_network(torch.tensor(states, dtype=torch.float32)).detach().numpy()

        for t in range(len(rewards)):
            Gt = sum([gamma ** i * rewards[t + i] for i in range(len(rewards) - t)])  # Return
            advantage = Gt - value_tensor[t]  # Advantage
            returns.append(Gt)
            advantages.append(advantage)

        returns = torch.tensor(np.array(returns), dtype=torch.float32)
        advantages = torch.tensor(np.array(advantages), dtype=torch.float32)

        # Update the actor network
        actor_optimizer.zero_grad()
        actor_loss = 0  # Initialize loss accumulator for the actor

        for action_set, adv in zip(actions, advantages):
            cpu_cores_log_prob = torch.log(action_probs['cpu_cores'][action_set[0]])
            cpu_freq_log_prob = torch.log(action_probs['cpu_freq'][action_set[1]])
            gpu_freq_log_prob = torch.log(action_probs['gpu_freq'][action_set[2]])
            memory_freq_log_prob = torch.log(action_probs['memory_freq'][action_set[3]])
            cl_log_prob = torch.log(action_probs['cl'][action_set[4]])
            
            # Sum log probabilities for each action dimension
            total_log_prob = (cpu_cores_log_prob + cpu_freq_log_prob + gpu_freq_log_prob +
                            memory_freq_log_prob + cl_log_prob)
            
            # Accumulate the loss
            actor_loss += -(total_log_prob * adv)  # Policy gradient loss

        # Now perform the backward pass and update the actor network parameters
        actor_loss.backward()
        actor_optimizer.step()

        # Update the critic network
        critic_optimizer.zero_grad()
        values = critic_network(torch.tensor(np.array(states), dtype=torch.float32)).squeeze()
        critic_loss = nn.MSELoss()(values, returns)
        critic_loss.backward()
        critic_optimizer.step()

        print(f"Episode: {episode}, Max Reward: {max_reward}")

    end_t1 = round(((time.time() - t1) - sum(time_got)) * 1000, 3)
    end = end_t1 / len(time_got)
    for config in configs:
        dict_record = [{'a2c_time_elapsed': end, **config}]
        save_csv(dict_record, f"a2c_{int(sys.argv[6])}_{sys.argv[5]}_{sys.argv[4]}.csv")
    exec_trained(best_config)
    print(f"Best Config: {best_config} in {sum(time_got) + end_t1} sec")


# Initialize the actor and critic networks
input_size = 5  # State representation: cpu_cores, cpu_freq, gpu_freq, memory_freq, cl

actor_network = ActorNetwork(input_size, output_sizes)
critic_network = CriticNetwork(input_size)
actor_optimizer = optim.Adam(actor_network.parameters(), lr=lr)
critic_optimizer = optim.Adam(critic_network.parameters(), lr=lr)

# Run the A2C algorithm
a2c_algorithm(actor_network, critic_network, actor_optimizer, critic_optimizer)
