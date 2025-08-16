import numpy as np
import sys
import time
import os
import csv
import requests
import matplotlib.pyplot as plt
from skopt import gp_minimize
from skopt.space import Categorical, Integer
from skopt.utils import use_named_args
from skopt.acquisition import gaussian_ei, gaussian_pi, gaussian_lcb

print("PID", os.getpid())

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

best_throughput = -1e6

prohibited_configs = set()

# Hyperparameters for Bayesian Optimization
n_calls = int(sys.argv[6])
n_initial_points = 2

time_got = []

last_rewards = []  # To store recent rewards for saturation check
episode_counter = 0

cores_space = (Categorical(CPU_CORES_RANGE, name='cpu_cores') if len(CPU_CORES_RANGE) == 1 else Integer(min(CPU_CORES_RANGE), max(CPU_CORES_RANGE), name='cpu_cores'))
mem_space = (Categorical(MEMORY_FREQ_RANGE, name='mem_freq') if len(MEMORY_FREQ_RANGE) == 1 else Integer(min(MEMORY_FREQ_RANGE), max(MEMORY_FREQ_RANGE), name='mem_freq'))

# Define the parameter space for Bayesian Optimization
space = [
    cores_space,
    Integer(min(CPU_FREQ_RANGE), max(CPU_FREQ_RANGE), name='cpu_freq'),
    Integer(min(GPU_FREQ_RANGE), max(GPU_FREQ_RANGE), name='gpu_freq'),
    mem_space,
    Integer(min(CL_RANGE), max(CL_RANGE), name='cl')
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

th_target = int(sys.argv[8])

#def avg(val):
#    sum(val)//len(val)

locked = False

def increment_target(outcome, target):
    global locked
    if not locked:
        if outcome == 'negative':
            target += 5
        else:
            target -= 5
    if target == 0 and not locked:
        target += 3
        locked = True
    return target

# Reward function based on power and throughput metrics
# Efficient reward calculation
def calculate_reward(measured_metrics, target):
    power = measured_metrics[0]["power_cons"]
    throughput = measured_metrics[0]["throughput"]
    
    if throughput <= target:
        return 1e6
    
    return -(throughput/power * 1e-6)

# CSV saving optimization
def save_csv(dict_list, filename):
    with open(filename, 'a', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=['api_time','id', 'reward', 'episode', 'infer_time', 'bo_time_elapsed', 'cpu_cores', 'cpu_freq', 'gpu_freq', 'mem_freq', 'cl', 'elapsed', 'time_load', 'time_warm', 'time_c', 'th_target', 'throughput', 'power_cons', 'cpu_percent', 'gpu_percent', 'mem_percent'])
        if os.path.getsize(filename) == 0:
            writer.writeheader()
        for d in dict_list:
            writer.writerow(d)

powers = []
results = []
rewards = []
set_target = 0

# The objective function for Bayesian Optimization
@use_named_args(space)
def objective(cpu_cores, cpu_freq, gpu_freq, mem_freq, cl):
    global episode_counter, best_throughput, powers, results, rewards, th_target, set_target
    episode_counter += 1
    print(f"Testing configuration in eps {episode_counter}: CPU Cores={cpu_cores+1}, CPU Freq={cpu_freq}, GPU Freq={gpu_freq}, Mem Freq={mem_freq}, CL={cl}")

    if (cpu_cores, cpu_freq, gpu_freq, mem_freq, cl, th_target) not in prohibited_configs or episode_counter >= 10:
        current_thtarget = th_target
        t1 = time.time()
        if episode_counter == 1:
            cpu_cores, cpu_freq, gpu_freq, mem_freq, cl = max(CPU_CORES_RANGE), max(CPU_FREQ_RANGE), max(GPU_FREQ_RANGE), max(MEMORY_FREQ_RANGE), max(CL_RANGE)
        measured_metrics, api_time = execute_config(cpu_cores, cpu_freq, gpu_freq, mem_freq, cl)

        elapsed = round(time.time() - t1, 3)
        time_got.append(elapsed)

        if measured_metrics:
            if measured_metrics[0]["throughput"] > best_throughput:
                best_throughput = measured_metrics[0]["throughput"]
        
            if not measured_metrics or measured_metrics == "No Device":
                print("No device detected. Raising an exception to stop optimization.")
                raise RuntimeError("No device detected. Stopping optimization.")  # Raise exception to stop the optimizer

            reward = calculate_reward(measured_metrics, current_thtarget)
            if (episode_counter % 5) == 0 and episode_counter < 10:
                outcome = 'positive' if reward > 0 else 'negative'
                th_target = increment_target(outcome, th_target)
            if episode_counter >= 10:
                if set_target <= 2:
                    outcome = 'positive' if reward > 0 else 'negative'
                    if set_target == 2:
                        if outcome == 'negative':
                            th_target = current_thtarget
                        else:
                            th_target = increment_target(outcome, th_target)
                    else:
                        th_target = increment_target(outcome, th_target)
                    set_target += 1
            rewards.append(reward)
            print(f"Configuration reward: {reward}")

            if reward == 1e6:
                prohibited_configs.add((cpu_cores, cpu_freq, gpu_freq, mem_freq, cl, current_thtarget))

            powers.append(measured_metrics[0]["power_cons"])
        
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
            "elapsed": measured_metrics[0]["elapsed"],
            "time_load": measured_metrics[0]["time_load"],
            "time_warm": measured_metrics[0]["time_warm"],
            "time_c": measured_metrics[0]["time_c"],
            "th_target": current_thtarget,
            'throughput': measured_metrics[0]["throughput"],
            'power_cons': measured_metrics[0]["power_cons"],
            "cpu_percent": measured_metrics[0]["cpu_percent"],
            "gpu_percent": measured_metrics[0]["gpu_percent"],
            "mem_percent": measured_metrics[0]["mem_percent"]
            }
            results.append(configs)

            return reward  # Minimize the negative reward to maximize reward
        else:
            return 0
    else:
        return 1e6

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
    elapsed_tall = round(((time.time() - sum(time_got)) - t2) * 1000, 3)
    elapsed = elapsed_tall / len(time_got)
    for result in results:
        dict_record = [{'bo_time_elapsed': elapsed, **result}]
        save_csv(dict_record, f"bo_{sys.argv[6]}_{sys.argv[5]}_{sys.argv[4]}.csv")
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
    matching_dict = next(
        (d for d in results if d.get('reward') == min(rewards)),
        None
    )
    #if matching_dict:
    #    th_target = matching_dict.get('th_target')
    print(f"Best configuration found (reward {min(rewards)} with target {th_target}): {best_params} in {elapsed} ms for BO and total time is took {elapsed_total}")
    results = []
    for _ in range(20):
        objective(tuple(res.x))
    for result in results:
        dict_record = [{'bo_time_elapsed': 0, **result}]
        save_csv(dict_record, f"bo_{sys.argv[6]}_{sys.argv[5]}_{sys.argv[4]}.csv")
except RuntimeError as e:
    print(e)  # Handle exception messages
