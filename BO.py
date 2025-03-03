import numpy as np
import sys
import time
import os
import csv
import requests
import matplotlib.animation as animation
import matplotlib.pyplot as plt
from skopt.plots import plot_gaussian_process
from skopt import gp_minimize
from skopt.space import Categorical
from skopt.utils import use_named_args
from skopt.acquisition import gaussian_ei, gaussian_pi, gaussian_lcb

print("PID", os.getpid())

if sys.argv[5] == 'jxavier':
    CPU_CORES_RANGE = np.linspace(1, 5, 3)
    CPU_FREQ_RANGE = range(1190, 1909)
    GPU_FREQ_RANGE = range(510, 1111)
    MEMORY_FREQ_RANGE = range(1500, 1867)
    CL_RANGE = range(1, 4)
elif sys.argv[5] == 'jorin-nano':
    CPU_CORES_RANGE = [5]
    CPU_FREQ_RANGE = range(806, 1510)
    GPU_FREQ_RANGE = range(306, 624)
    MEMORY_FREQ_RANGE = range(1500, 2133)
    CL_RANGE = range(1, 4)

sampled_configs ={
     "cpu_cores": CPU_CORES_RANGE, 
     "cpu_freq": np.linspace(min(CPU_FREQ_RANGE), max(CPU_FREQ_RANGE), 3), 
     "gpu_freq": np.linspace(min(GPU_FREQ_RANGE), max(GPU_FREQ_RANGE), 3), 
     "memory_freq": np.linspace(min(MEMORY_FREQ_RANGE), max(MEMORY_FREQ_RANGE), 3), 
     "cl": CL_RANGE
}

POWER_BUDGET = int(sys.argv[6])

best_throughput = -1e6

# Hyperparameters for Bayesian Optimization
n_calls = 75  # Number of iterations for Bayesian Optimization
n_initial_points = 28

time_got = []

last_rewards = []  # To store recent rewards for saturation check
episode_counter = 0

space = [
    Categorical(sampled_configs['cpu_cores'], name='cpu_cores'),
    Categorical(sampled_configs['cpu_freq'], name='cpu_freq'),
    Categorical(sampled_configs['gpu_freq'], name='gpu_freq'),
    Categorical(sampled_configs['memory_freq'], name='mem_freq'),
    Categorical(sampled_configs['cl'], name='cl')
]

# Function to get the result from the external system
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

# Reward function based on power and throughput metrics
# Efficient reward calculation
def calculate_reward(measured_metrics):
    power = measured_metrics[0]["power_cons"]
    throughput = measured_metrics[0]["throughput"]
    
    if power > POWER_BUDGET:
        return 1e6
    
    return throughput / power

# CSV saving optimization
def save_csv(dict_list, filename):
    with open(filename, 'a', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=['api_time','id', 'reward', 'episode', 'infer_time', 'cpu_cores', 'cpu_freq', 'gpu_freq', 'mem_freq', 'cl', 'power_budget', 'throughput', 'power_cons', 'cpu_percent', 'gpu_percent', 'mem_percent'])
        if os.path.getsize(filename) == 0:
            writer.writeheader()
        for d in dict_list:
            writer.writerow(d)

# The objective function for Bayesian Optimization
@use_named_args(space)
def objective(cpu_cores, cpu_freq, gpu_freq, mem_freq, cl):
    global episode_counter, best_throughput
    print(f"Testing configuration: CPU Cores={cpu_cores+1}, CPU Freq={cpu_freq}, GPU Freq={gpu_freq}, Mem Freq={mem_freq}, CL={cl}")
    
    t1 = time.time()
    measured_metrics, api_time = execute_config(cpu_cores, cpu_freq, gpu_freq, mem_freq, cl)

    elapsed = round(time.time() - t1, 3)
    time_got.append(elapsed)

    if measured_metrics[0]["throughput"]:
        if measured_metrics[0]["throughput"] > best_throughput:
            best_throughput = measured_metrics[0]["throughput"]
    
        if not measured_metrics or measured_metrics == "No Device":
            print("No device detected. Raising an exception to stop optimization.")
            raise RuntimeError("No device detected. Stopping optimization.")  # Raise exception to stop the optimizer

        reward = calculate_reward(measured_metrics)
        print(f"Configuration reward: {reward}")
    
        configs = {
        "reward": reward,
	    "api_time": api_time,
        "episode" : episode_counter,
        "infer_time": elapsed,
        "cpu_cores": int(cpu_cores) + 1,
        "cpu_freq": int(cpu_freq),
        "gpu_freq": int(gpu_freq),
        "mem_freq": int(mem_freq),
        "cl": int(cl),
        "power_budget": POWER_BUDGET,
        }
        result = {**configs, **measured_metrics[0]}
        save_csv([result], f"bo_{sys.argv[5]}_{sys.argv[4]}.csv")

        last_rewards.append(reward)
    
        if reward == 1e6:
            return reward  # Return penalty for invalid config

        episode_counter += 1

        return -reward  # Minimize the negative reward to maximize reward
    return 0


# Custom callback to track the optimization progress
class PhaseTracker:
    def __init__(self, filename="optimization_phases.csv"):
        self.points = []
        self.acquisition_values = []
        self.acquisition_type = []
        self.filename = filename

        with open(self.filename, mode="w", newline="") as file:
            writer = csv.writer(file)
            writer.writerow(["Iteration", "Point", "Acquisition Function", "Phase"])

    def __call__(self, res):
        x = res.x_iters[-1]  # Get the last evaluated point
        self.points.append(x)

        if not res.models:
            phase = "Exploration (Initial Sampling)"
            acquisition_type = "Random"
        else:
            # Transform input using res.space
            x_encoded = res.space.transform([x])

            # Compute acquisition function values
            model = res.models[-1]
            ei = gaussian_ei(x_encoded, model)
            pi = gaussian_pi(x_encoded, model)
            lcb = gaussian_lcb(x_encoded, model)

            self.acquisition_values.append((ei[0], pi[0], lcb[0]))

            if np.argmax([ei[0], pi[0], lcb[0]]) == 0:
                phase = "EI"
                acquisition_type = "EI"
            elif np.argmax([ei[0], pi[0], lcb[0]]) == 1:
                phase = "PI"
                acquisition_type = "PI"
            else:
                phase = "LCB"
                acquisition_type = "LCB"

        iteration = len(self.points)
        with open(self.filename, mode="a", newline="") as file:
            writer = csv.writer(file)
            writer.writerow([iteration, x, acquisition_type, phase])


# Initialize the tracker
tracker = PhaseTracker()

# Main Optimization Loop
try:
    t2 = time.time()
    res = gp_minimize(objective, space, n_calls=n_calls, random_state=42, n_initial_points=n_initial_points, callback=[tracker])
    # Run Bayesian Optimization
    elapsed = round(((time.time() - sum(time_got)) - t2) * 1000, 3)
    elapsed_total = round(time.time() - t2, 3)
    if int(sys.argv[7]):
        from skopt.plots import plot_objective, plot_evaluations

        # Plot the optimization process
        fig, ax = plt.subplots(1, 2, figsize=(12, 6))
        plot_objective(res, ax=ax[0])
        plot_evaluations(res, ax=ax[1])
        plt.savefig("bo-res.png")  # Save as a static image
        plt.close()
    # Output the best found configuration and try the best config on device
    best_params = dict(zip(['cpu_cores', 'cpu_freq', 'gpu_freq', 'mem_freq', 'cl'], res.x))
    print(f"Best configuration found: {best_params} in {elapsed} ms for BO and total time is took {elapsed_total}")
    for _ in range(25):
        objective(tuple(res.x))
except RuntimeError as e:
    print(e)  # Handle exception messages
