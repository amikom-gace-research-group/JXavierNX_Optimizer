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

sampled_configs = []

# Stratified sampling: Select a subset of configurations
for cpu_cores in CPU_CORES_RANGE:
    for cpu_freq in np.linspace(min(CPU_FREQ_RANGE), max(CPU_FREQ_RANGE), 3):  # Example: 3 CPU frequency strata
        for gpu_freq in np.linspace(min(GPU_FREQ_RANGE), max(GPU_FREQ_RANGE), 3):
            for memory_freq in np.linspace(min(MEMORY_FREQ_RANGE), max(MEMORY_FREQ_RANGE), 3):
                for cl in CL_RANGE:
                    config = {"cpu_cores": int(cpu_cores), "cpu_freq": int(cpu_freq), "gpu_freq": int(gpu_freq), "memory_freq": int(memory_freq), "cl": cl}
                    sampled_configs.append(config)

sampled_configs = pd.DataFrame(sampled_configs)

sampled_configs = sampled_configs.sort_values(
    by=["cpu_cores", "cpu_freq", "gpu_freq", "memory_freq", "cl"]
).reset_index(drop=True)

POWER_BUDGET = int(sys.argv[6])

slowdown_factor = 1.0  # Global slowdown factor (initial)

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

# Function to find the row ID based on configuration values
def get_row_id(cpu_cores, cpu_freq, gpu_freq, memory_freq, cl):
    row = sampled_configs[
        (sampled_configs['cpu_cores'] == cpu_cores) &
        (sampled_configs['cpu_freq'] == cpu_freq) &
        (sampled_configs['gpu_freq'] == gpu_freq) &
        (sampled_configs['memory_freq'] == memory_freq) &
        (sampled_configs['cl'] == cl)
    ]
    return row.index[0] if not row.empty else None

def adjust_configuration(value_matrix, value_matrixes, sampled_configs, configs, best_config):
    global conf
    if value_matrix > max(value_matrixes):
        # increase resources if power probability is high
        configs_id = round(get_row_id(*configs) + 1)
        configs_id = min(configs_id, len(sampled_configs)-1)
        updated_configs = sampled_configs.iloc[configs_id]
        value_matrixes.append(value_matrix)
        conf += 1
        return updated_configs['cpu_cores'], updated_configs['cpu_freq'], updated_configs['gpu_freq'], updated_configs['memory_freq'], updated_configs['cl']
    elif value_matrix == 0:
        configs_id = round(get_row_id(*configs) + 1)
        configs_id = max(get_row_id(*configs) - abs(get_row_id(*configs) - configs_id), 0)
        updated_configs = sampled_configs.iloc[configs_id]
        conf -= 1
        return updated_configs['cpu_cores'], updated_configs['cpu_freq'], updated_configs['gpu_freq'], updated_configs['memory_freq'], updated_configs['cl']
    else:
        return configs if not best_config else best_config

def calculate_probability(goal, m, value_var):
    # Calculate the standard deviation from variance
    sigma = np.sqrt(value_var)
    # Z-score for the normal distribution
    z_score =  (goal - m) / sigma if sigma > 0 else float('inf')
    # Use the CDF to calculate the probability
    return norm.cdf(z_score)

# CSV saving optimization
def save_csv(dict_list, filename):
    with open(filename, 'a', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=['api_time', 'episode', 'infer_overhead', 'alert_overhead', 'power_probability', 'power_budget', 'cpu_cores', 'cpu_freq', 'gpu_freq', 'memory_freq', 'cl', 'estimated_throughput', 'estimated_power'])
        if os.path.getsize(filename) == 0:
            writer.writeheader()
        for d in dict_list:
            writer.writerow(d)

time_got = []
best_config = None
best_throughput = -float('inf')

# Initialize Kalman Filters for throughput and power
throughput_filter = KalmanFilter()
power_filter = KalmanFilterPower()

# Initial configurations (starting in the middle of the range)
cpu_cores, cpu_freq, gpu_freq, memory_freq, cl = min(sampled_configs['cpu_cores']), min(sampled_configs['cpu_freq']), min(sampled_configs['gpu_freq']), min(sampled_configs['memory_freq']), min(sampled_configs['cl'])

value_matrixes = [0]

num_episodes = 100
conf = 0

for episode in range(num_episodes):
    t1 = time.time()
    # Execute configuration and get metrics
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

    # Update Kalman Filters with current metrics
    throughput_measurement = measured_metrics[0]['throughput']
    power_measurement = measured_metrics[0]['power_cons']

    # Update Kalman filters for throughput and power
    predicted_throughput, throughput_var = throughput_filter.update(throughput_measurement)
    predicted_power, power_var = power_filter.update(power_measurement)

    # Update global slowdown factor based on throughput measurement
    slowdown_factor = predicted_throughput / throughput_measurement
    estimated_throughput = slowdown_factor * throughput_measurement
    power_slowdown_factor = predicted_power / power_measurement
    estimated_power = power_slowdown_factor * power_measurement

    # Calculate probability of meeting throughput/power target
    power_probability = calculate_probability(POWER_BUDGET, estimated_power, power_var)

    power_mask = (np.int64(estimated_power) <= np.int64(POWER_BUDGET)).astype(int)

    k_throughput = 1.0  # Weight for throughput (primary objective)
    k_power_probability = 0.5  # Weight for power probability (secondary objective)
    B = 99999999  # Large constant to make valid scores positive
    value_matrix = power_mask * (B + k_throughput * estimated_throughput + k_power_probability * power_probability)
    
    print(f"power_probability {power_probability}")
    elapsed = round(((time.time() - elapsed_exec) - t1) * 1000, 3)
    time_got.append(elapsed+elapsed_exec)
    #Save results to CSV
    configs = {
        "api_time": api_time,
        "episode": episode,
        "infer_overhead" : elapsed_exec,
        "alert_overhead" : elapsed,
        "power_probability": power_probability,
        "power_budget": POWER_BUDGET,
        "cpu_cores": cpu_cores+1,
        "cpu_freq": cpu_freq,
        "gpu_freq": gpu_freq,
        "memory_freq": memory_freq,
        "cl": cl,
        "estimated_throughput": estimated_throughput,
        "estimated_power": estimated_power
    }

    if value_matrix > max(value_matrixes):
        best_config = cpu_cores, cpu_freq, gpu_freq, memory_freq, cl

    # Adjust configurations based on probabilities
    config = cpu_cores, cpu_freq, gpu_freq, memory_freq, cl
    cpu_cores, cpu_freq, gpu_freq, memory_freq, cl = adjust_configuration(value_matrix, value_matrixes, sampled_configs, config, best_config)

    save_csv([configs], f"alert-online_scaled_{sys.argv[5]}_{sys.argv[4]}.csv")


    print(f"Configs: {configs}")

print(f"Best Config: {best_config} in {sum(time_got)} sec")