import numpy as np
import sys
import time
import os
import csv
import requests
import random
from pyDOE import lhs
from scipy.stats import norm

print("PID", os.getpid())

if sys.argv[5] == 'jxavier':
    CPU_CORES_RANGE = range(1, 6)
    CPU_FREQ_RANGE = range(1190, 1909)
    GPU_FREQ_RANGE = range(510, 1111)
    MEMORY_FREQ_RANGE = range(1500, 1867)
    CL_RANGE = range(1, 4)
    ranges = [
        (1, 6),
        (1190, 1909),
        (510, 1111),
        (1500, 1867),
        (1, 4)
    ]
    low_pwr = 4000
    high_pwr = 20500
elif sys.argv[5] == 'jorin-nano':
    CPU_CORES_RANGE = [5]
    CPU_FREQ_RANGE = range(806, 1511)
    GPU_FREQ_RANGE = range(306, 625)
    MEMORY_FREQ_RANGE = [2133]
    CL_RANGE = range(1, 4)
    ranges = [
        (5, 6),
        (806, 1511),
        (306, 625),
        (2133, 2134),
        (1, 4)
    ]
    low_pwr = 4700
    high_pwr = 15500

POWER_BUDGET = [power_budget for power_budget in range(low_pwr, high_pwr, 500)]
conf = 0

# Extended Kalman Filter class for throughput prediction (returns mean and variance)
class KalmanFilter:
    def __init__(self):
        self._A = 1  # State transition factor
        self._H = 1  # Measurement matrix
        self._Q = 0.05  # Process noise covariance
        self._R = 0.5  # Measurement noise covariance
        self._P = 0.01  # Estimate error covariance (variance)
        self._S = 0.001
        self._x = 1  # Initial state estimate (throughput)
        self._K = 0.5  # Kalman gain
        self._y = 1  # Measurement residual

    def update(self, measurement, u=0):
        if measurement < 1:
            measurement = 1
            
        self._x = self._A * self._x + u
        self._Q = max(0.3 * self._Q + 0.7 * self._K * self._K * self._y * self._y, 0.1)
        self._P = self._A * self._P * self._A + self._Q
        
        self._y = measurement - (self._H * self._x)
        self._S = self._H * self._P * self._H + self._R
        self._K = (self._P * self._H) / self._S
        
        self._x = self._x + (self._K * self._y)
        self._P = (1 - (self._K * self._H)) * self._P
        
        if self._x < 1:
            self._x = 1
            
        return self._x, self._P  # Return predicted mean and variance (P)


# Kalman Filter class for power prediction
class KalmanFilterPower:
    def __init__(self):
        # Initialize Kalman Filter parameters for power prediction
        self._A = 1
        self._H = 1
        self._Q = 0.0001
        self._R = 0.0001
        self._P = 0.01
        self._S = 0.001
        self._x = 1 # Initial power estimate
        self._K = 0.5
        self._y = 1  # Measurement residual

    def update(self, measurement):
        # State prediction
        self._x = self._A * self._x
        
        # Prediction error covariance update
        self._P = self._A * self._P * self._A + self._Q
        
        # Compute measurement residual
        self._y = measurement - (self._H * self._x)
        
        # Compute Kalman Gain
        self._S = self._H * self._P * self._H + self._R
        self._K = (self._P * self._H) / self._S
        
        # Update state estimate with measurement
        self._x = self._x + (self._K * self._y)
        
        # Update error covariance
        self._P = (1 - (self._K * self._H)) * self._P
        
        return self._x, self._P

# Retrieve the result from the system API
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

# Generate LHS samples for the exploration phase
def generate_lhs_samples():
    num_samples = 9  # Number of samples per episode
    samples = lhs_sampling(num_samples, ranges)
    return [tuple(map(int, sample)) for sample in samples]

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

sampled_configs = []

def sampling(episode):
    global sampled_configs, prohibited_configs, POWER_BUDGET
    if episode < 2:
        for cpu_cores, cpu_freq, gpu_freq, memory_freq, cl, id_pwr_budget in [(min(CPU_CORES_RANGE), min(CPU_FREQ_RANGE), min(GPU_FREQ_RANGE), min(MEMORY_FREQ_RANGE), min(CL_RANGE), 0), (max(CPU_CORES_RANGE), max(CPU_FREQ_RANGE), max(GPU_FREQ_RANGE), max(MEMORY_FREQ_RANGE), max(CL_RANGE), -1)]:
            config = {"cpu_cores": int(cpu_cores), "cpu_freq": int(cpu_freq), "gpu_freq": int(gpu_freq), "memory_freq": int(memory_freq), "cl": cl, "power_budget":POWER_BUDGET[id_pwr_budget]}
            if config in sampled_configs:
                return "stuck"
            sampled_configs.append(config)
    else: # random hypercube
        lhs_samples = generate_lhs_samples()
        power_budget = random.choice(POWER_BUDGET)
        st_state = random.choice(lhs_samples)
        nd_state = calculate_diversity(lhs_samples, st_state)
        for configs in [st_state, nd_state]:
            config = {"cpu_cores": int(configs[0]), "cpu_freq": int(configs[1]), "gpu_freq": int(configs[2]), "memory_freq": int(configs[3]), "cl": int(configs[4]), "power_budget":power_budget}
            if config in sampled_configs:
                return "stuck"
            sampled_configs.append(config)

def select_best_configuration(entries, power_budget, power_variance, episode, num_episodes):
    global conf
    # Step 1: Extract relevant data from entries
    power = np.array([float(entry['power']) for entry in entries])  # Mean power consumption
    throughput = np.array([float(entry['throughput']) for entry in entries])  # Throughput
    configurations = np.array(entries)
    
    # Step 2: Create binary mask for valid configurations
    power_mask = (power <= power_budget).astype(int)  # 1 if within power budget, 0 otherwise

    # Step 3: Calculate power probabilities
    power_probabilities = np.array([calculate_probability(power_budget, p, var) for p, var in zip(power, [power_variance])])

    # Step 4: Create value matrix for scoring
    k_throughput = 1.0  # Weight for throughput (primary objective)
    k_power_probability = 0.5  # Weight for power probability (secondary objective)
    B = 99999999  # Large constant to make valid scores positive
    value_matrix = power_mask * (B + k_throughput * throughput + k_power_probability * power_probabilities)
    
    if power_mask[conf] > 0 and (num_episodes-episode) > 5:
        return None, None
    if power_mask[conf] == 0 or (num_episodes-episode) == 5:
        if power_mask[conf] == 0:
            print("No valid configuration found within the power budget")
        best_index = np.argmax(value_matrix)  # Find the index of the highest score
        best_config = configurations[best_index]
        return best_config, best_index

# -----------------------
# Runtime Execution Loop
# -----------------------

def execute_runtime(num_episodes):
    global conf
    """
    Executes the runtime learning and adjustment process.
    """
    throughput_filter = KalmanFilter()
    power_filter = KalmanFilterPower()
    best_config = None
    best_index = 0
    episode = 0
    power_var = 0.01
    best_config = None
    backup_POWER_BUDGET = POWER_BUDGET

    while episode < num_episodes:
        if not POWER_BUDGET:
            POWER_BUDGET = backup_POWER_BUDGET
        power_budget = POWER_BUDGET[episode % len(POWER_BUDGET)]
        if not best_config:
            out = sampling(episode)
            sampled_config = sampled_configs[episode]
            cpu_cores, cpu_freq, gpu_freq, memory_freq, cl = sampled_config["cpu_cores"], sampled_config["cpu_freq"], sampled_config["gpu_freq"], sampled_config["memory_freq"], sampled_config["cl"]
            if out == 'stuck':
                continue
        t1 = time.time()
        # Simulated runtime execution (replace with actual API call)
        measured_metrics, api_time = execute_config(cpu_cores, cpu_freq, gpu_freq, memory_freq, cl)

        elapsed_exec = round(time.time() - t1, 3)
        if isinstance(measured_metrics, list) or not measured_metrics:
            if not measured_metrics:
                print(f"EXECUTION PROBLEM! {measured_metrics}")
                conf += 1
                config = sampled_configs[conf]
                cpu_cores, cpu_freq, gpu_freq, memory_freq, cl = config["cpu_cores"], config["cpu_freq"], config["gpu_freq"], config["memory_freq"], config["cl"]
                continue
            elif measured_metrics[0]['power_cons'] == 0:
                print("EXECUTION PROBLEM! Power Zero")
                conf += 1
                config = sampled_configs[conf]
                cpu_cores, cpu_freq, gpu_freq, memory_freq, cl = config["cpu_cores"], config["cpu_freq"], config["gpu_freq"], config["memory_freq"], config["cl"]
                continue
        if measured_metrics == "No Device":
            print("No Device/No Inference Runtime")
            break

        throughput = measured_metrics[0]['throughput']
        power = measured_metrics[0]['power_cons']
        cpu =  measured_metrics[0]["cpu_percent"]
        gpu = measured_metrics[0]["gpu_percent"]
        mem = measured_metrics[0]["mem_percent"]

        # Update Kalman Filters
        predicted_throughput, _ = throughput_filter.update(throughput)
        predicted_power, power_var = power_filter.update(power)

        # Update slowdown factor
        slowdown_factor = predicted_throughput / throughput
        estimated_throughput = slowdown_factor * throughput
        power_slowdown_factor = predicted_power / power
        estimated_power = power_slowdown_factor * power

        if episode == 0:
            sampled_configs[0]['power'] = estimated_power
            sampled_configs[0]['throughput'] = estimated_throughput
            sampled_configs[0]['cpu'] = cpu
            sampled_configs[0]['gpu'] = gpu
            sampled_configs[0]['mem'] = mem
            sampled_configs[0]['power_budget'] = power_budget
        else:
            sampled_configs[best_index]['power'] = estimated_power
            sampled_configs[best_index]['throughput'] = estimated_throughput
            sampled_configs[best_index]['cpu'] = cpu
            sampled_configs[best_index]['gpu'] = gpu
            sampled_configs[best_index]['mem'] = mem
            sampled_configs[best_index]['power_budget'] = power_budget

        best = select_best_configuration(sampled_configs, power_budget, power_var, episode, num_episodes)
        best_config, best_index = best

        power_list = [sampled_config['power_cons'] for sampled_config in sampled_configs if sampled_config['power_cons'] != -1]
        POWER_BUDGET = [
            power_budget
            for power_budget in POWER_BUDGET
            if min(power_list) <= power_budget <= max(power_list)
        ]

        elapsed = round(((time.time() - elapsed_exec) - t1) * 1000, 3)
        configs = {
            "api_time": api_time,
            'episode': episode,
            "infer_overhead" : elapsed_exec,
            "alert_overhead" : elapsed,
            "power_budget": power_budget,
            "cpu_cores": int(cpu_cores)+1,
            "cpu_freq": cpu_freq,
            "gpu_freq": gpu_freq,
            "memory_freq": memory_freq,
            "cl": cl,
            "estimated_throughput": estimated_throughput,
            "estimated_power": estimated_power,
            "cpu_percent": measured_metrics[0]["cpu_percent"],
            "gpu_percent": measured_metrics[0]["gpu_percent"],
            "mem_percent": measured_metrics[0]["mem_percent"]
        }
        file = f"alert-online-{sys.argv[6]}_{sys.argv[5]}_{sys.argv[4]}.csv"

        save_csv([configs], file)

        if best_config:
            print(f"[Runtime] Selected configuration {best_config}")

            cpu_cores, cpu_freq, gpu_freq, memory_freq, cl = best_config["cpu_cores"], best_config["cpu_freq"], best_config["gpu_freq"], best_config["memory_freq"], best_config["cl"]
        episode += 1

    if best is None:
        return "No Best"

# -----------------------
# Helper Functions
# -----------------------

def calculate_probability(goal, mean, variance):
    """
    Calculates the probability of meeting a constraint using a Gaussian distribution.
    """
    sigma = np.sqrt(variance)
    z_score = (goal - mean) / sigma if sigma > 0 else float("inf")
    return norm.cdf(z_score)

# CSV saving optimization
def save_csv(dict_list, filename):
    with open(filename, 'a', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=['api_time','episode', 'infer_overhead', 'alert_overhead', 'power_budget', 'cpu_cores', 'cpu_freq', 'gpu_freq', 'memory_freq', 'cl', 'estimated_throughput', 'estimated_power', 'cpu_percent', 'gpu_percent', 'mem_percent'])
        if os.path.getsize(filename) == 0:
            writer.writeheader()
        for d in dict_list:
            writer.writerow(d)

# -----------------------
# Execution
# -----------------------

if __name__ == "__main__":
    execute_runtime(int(sys.argv[6]))

