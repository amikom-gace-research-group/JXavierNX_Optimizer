import numpy as np
import sys
import time
import os
import csv
import requests
import pandas as pd
from scipy.stats import norm

print("PID", os.getpid())

if sys.argv[5] == 'jxavier':
    CPU_CORES_RANGE = np.linspace(1, 6, 3)
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

POWER_BUDGET = int(sys.argv[6])
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

# Stratified sampling: Select a subset of configurations
for cpu_cores in CPU_CORES_RANGE:
    for cpu_freq in np.linspace(min(CPU_FREQ_RANGE), max(CPU_FREQ_RANGE), 3):  # Example: 3 CPU frequency strata
        for gpu_freq in np.linspace(min(GPU_FREQ_RANGE), max(GPU_FREQ_RANGE), 3):
            for memory_freq in np.linspace(min(MEMORY_FREQ_RANGE), max(MEMORY_FREQ_RANGE), 3):
                for cl in CL_RANGE:
                    config = {"cpu_cores": int(cpu_cores), "cpu_freq": int(cpu_freq), "gpu_freq": int(gpu_freq), "memory_freq": int(memory_freq), "cl": cl, 
                              "throughput": 0, "power": float('inf'), "cpu_percent": 0, "gpu_percent": 0, "mem_percent": 0}
                    sampled_configs.append(config)

def apply_configs(config):
    return config["cpu_cores"], config["cpu_freq"], config["gpu_freq"], config["memory_freq"], config["cl"]

def minmax(values, range):
    values = min(values, max(range))
    values = max(min(range), values)
    return int(values)

def generate_neighbor(exist_configs, neighbor_configs):
    new_neighbors = []
    for exist_config, neighbor_config, range in zip(exist_configs, neighbor_configs, (CPU_CORES_RANGE, CPU_FREQ_RANGE, GPU_FREQ_RANGE, MEMORY_FREQ_RANGE, CL_RANGE)):
        if exist_config > neighbor_config:
            new_neighbor = minmax(round(exist_config - abs(exist_config - neighbor_config) / 2), range)
        else:
            new_neighbor = minmax(round(exist_config + abs(exist_config - neighbor_config) / 2), range)
        new_neighbors.append(new_neighbor)
    new_neighbor  = {"cpu_cores": int(new_neighbors[0]), "cpu_freq": int(new_neighbors[1]), "gpu_freq": int(new_neighbors[2]), "memory_freq": int(new_neighbors[3]), "cl": new_neighbors[4]}
    return new_neighbors

def select_best_configuration(entries, power_budget, power_variance, episode, proposed=0):
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
    
    if power_mask[conf] > 0 and episode <= round((int(sys.argv[7])/100)*len(sampled_configs)):
        conf += 1
        next_config = configurations[conf]
        return next_config, conf
    if proposed and round((int(sys.argv[7])/100)*len(sampled_configs)) < episode < 75 and power_mask[conf] > 0:
        best_index = np.argmax(value_matrix)
        best_value = value_matrix[best_index]
        best_config = configurations[best_index]
        if episode < 50:
            second_best_index = np.argmin(value_matrix)
        else:
            best_index = np.argmax(value_matrix)
            value_matrix[best_index] = -np.inf
            second_best_index = np.argmax(value_matrix)
        second_best_config = apply_configs(configurations[second_best_index])
        best_config = apply_configs(best_config)
        new_config = generate_neighbor(best_config, second_best_config)
        value_matrix[best_index] = best_value
        if new_config not in entries:
            entries.append(new_config)
        return new_config, entries.index(new_config)
    if power_mask[conf] == 0 or episode >= 75 or (round((int(sys.argv[7])/100)*len(sampled_configs)) < episode if not proposed else False):
        if power_mask[conf] == 0:
            print("No valid configuration found within the power budget")
        best_index = np.argmax(value_matrix)  # Find the index of the highest score
        best_config = configurations[best_index]
        return best_config, best_index

# -----------------------
# Runtime Execution Loop
# -----------------------

def execute_runtime(num_episodes=100):
    global conf
    """
    Executes the runtime learning and adjustment process.
    """
    throughput_filter = KalmanFilter()
    power_filter = KalmanFilterPower()
    best_config = None
    power_var = 0.01
    cpu_cores, cpu_freq, gpu_freq, memory_freq, cl = sampled_configs[0]["cpu_cores"], sampled_configs[0]["cpu_freq"], sampled_configs[0]["gpu_freq"], sampled_configs[0]["memory_freq"], sampled_configs[0]["cl"]

    for episode in range(num_episodes):
        t1 = time.time()
        # Simulated runtime execution (replace with actual API call)
        measured_metrics, api_time = execute_config(cpu_cores, cpu_freq, gpu_freq, memory_freq, cl)

        elapsed_exec = round(time.time() - t1, 3)
        if isinstance(measured_metrics, list) or not measured_metrics:
            if not measured_metrics:
                print("EXECUTION PROBLEM!")
                conf += 1
                config = sampled_configs[conf]
                cpu_cores, cpu_freq, gpu_freq, memory_freq, cl = config["cpu_cores"], config["cpu_freq"], config["gpu_freq"], config["memory_freq"], config["cl"]
                continue
            elif measured_metrics[0]['power_cons'] == 0:
                print("EXECUTION PROBLEM!")
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
        else:
            sampled_configs[best_index]['power'] = estimated_power
            sampled_configs[best_index]['throughput'] = estimated_throughput
            sampled_configs[best_index]['cpu'] = cpu
            sampled_configs[best_index]['gpu'] = gpu
            sampled_configs[best_index]['mem'] = mem

        best = select_best_configuration(sampled_configs, POWER_BUDGET, power_var, episode, proposed=int(sys.argv[8]))
        best_config, best_index = best

        elapsed = round(((time.time() - elapsed_exec) - t1) * 1000, 3)
        configs = {
            "api_time": api_time,
            'episode': episode,
            "infer_overhead" : elapsed_exec,
            "alert_overhead" : elapsed,
            "power_budget": POWER_BUDGET,
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
        if int(sys.argv[8]):
            file =f"alert-online-proposed-{sys.argv[7]}_{sys.argv[5]}_{sys.argv[4]}.csv"
        else:
            file = f"alert-online-{sys.argv[7]}_{sys.argv[5]}_{sys.argv[4]}.csv"

        save_csv([configs], file)

        print(f"[Runtime] Selected configuration {best_config}")

        cpu_cores, cpu_freq, gpu_freq, memory_freq, cl = best_config["cpu_cores"], best_config["cpu_freq"], best_config["gpu_freq"], best_config["memory_freq"], best_config["cl"]

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
    execute_runtime(num_episodes=100)

