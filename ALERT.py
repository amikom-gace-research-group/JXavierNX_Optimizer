import numpy as np
import sys
import time
import os
import csv
import requests
from statistics import median
from scipy.stats import norm

print("PID", os.getpid())

# Define configuration ranges
CPU_CORES_RANGE = range(1, 6)  # Number of CPU cores (2 to 6)
CPU_FREQ_RANGE = range(1190, 1909)  # CPU frequency in MHz (1190 to 1908)
GPU_FREQ_RANGE = range(510, 1111)  # GPU frequency in MHz (510 to 1110)
MEMORY_FREQ_RANGE = range(1500, 1867)  # Memory frequency in MHz (1500 to 1866)
CL_RANGE = range(1, 4)  # Concurrency level (1 to 3)

# Constants and thresholds
POWER_BUDGET = 5000  # Power budget in milliWatts
THROUGHPUT_TARGET = 30  # Throughput target in units
slowdown_factor = 1.0  # Global slowdown factor (initial)
scaling_factor = 0.1  # Scaling factor for gradual frequency adjustments
max_saturated_count = 5

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

def adjust_configuration(throughput_probability, power_probability, cpu_freq, gpu_freq, memory_freq, cl):
    if throughput_probability < 0.8:
        # Increase resources if throughput probability is low
        adjustment = int((1 - throughput_probability) * scaling_factor * 100)
        cpu_freq = min(cpu_freq + adjustment, max(CPU_FREQ_RANGE))
        gpu_freq = min(gpu_freq + adjustment, max(GPU_FREQ_RANGE))
        memory_freq = min(memory_freq + adjustment, max(MEMORY_FREQ_RANGE))
        cl = min(cl + 1, max(CL_RANGE))  # Increase concurrency
    elif power_probability < 0.8:
        # Decrease resources if power probability is low
        adjustment = int((1 - power_probability) * scaling_factor * 100)
        cpu_freq = max(cpu_freq - adjustment, min(CPU_FREQ_RANGE))
        gpu_freq = max(gpu_freq - adjustment, min(GPU_FREQ_RANGE))
        memory_freq = max(memory_freq - adjustment, min(MEMORY_FREQ_RANGE))
        cl = max(cl - 1, min(CL_RANGE))  # Decrease concurrency
    return cpu_freq, gpu_freq, memory_freq, cl

def calculate_probability(goal, m, value_var, val):
    # Calculate the standard deviation from variance
    sigma = np.sqrt(value_var)
    
    # Z-score for the normal distribution
    if val:
        z_score = (m - goal) / sigma if sigma > 0 else float('inf')
    else:
        z_score =  (goal - m) / sigma if sigma > 0 else float('inf')
    # Use the CDF to calculate the probability
    return norm.cdf(z_score)

# CSV saving optimization
def save_csv(dict_list, filename):
    with open(filename, 'a', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=['episode', 'infer_overhead', 'alert_overhead', 'throughput_probability', 'power_probability', 'throughput_target', 'power_budget', 'cpu_cores', 'cpu_freq', 'gpu_freq', 'memory_freq', 'cl', 'estimated_throughput', 'estimated_power'])
        if os.path.getsize(filename) == 0:
            writer.writeheader()
        for d in dict_list:
            writer.writerow(d)

time_got = []
best_config = None

# Initialize Kalman Filters for throughput and power
throughput_filter = KalmanFilter()
power_filter = KalmanFilterPower()

# Initial configurations (starting in the middle of the range)
cpu_cores, cpu_freq, gpu_freq, memory_freq, cl = max(CPU_CORES_RANGE), max(CPU_FREQ_RANGE), max(GPU_FREQ_RANGE), max(MEMORY_FREQ_RANGE), max(CL_RANGE)
last_probability = 0, 0

num_episodes = 20

for episode in range(num_episodes):
    t1 = time.time()
    # Execute configuration and get metrics
    measured_metrics = execute_config(cpu_cores, cpu_freq, gpu_freq, memory_freq, cl)

    elapsed_exec = round(time.time() - t1, 3)
    if not measured_metrics:
        print("EXECUTION PROBLEM!")
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
    estimated_power= power_slowdown_factor * power_measurement

    # Calculate probability of meeting throughput/power target
    throughput_probability = calculate_probability(THROUGHPUT_TARGET, estimated_throughput, throughput_var, 1)
    power_probability = calculate_probability(POWER_BUDGET, estimated_power, power_var, 0)

    print(f"throughput probability {throughput_probability}, power_probability {power_probability}")
    elapsed = round(((time.time() - elapsed_exec) - t1) * 1000, 3)
    time_got.append(elapsed+elapsed_exec)
    #Save results to CSV
    configs = {
        "episode": episode,
        "infer_overhead" : elapsed_exec,
        "alert_overhead" : elapsed,
        "throughput_probability" : throughput_probability,
        "power_probability": power_probability,
        "throughput_target": THROUGHPUT_TARGET,
        "power_budget": POWER_BUDGET,
        "cpu_cores": cpu_cores+1,
        "cpu_freq": cpu_freq,
        "gpu_freq": gpu_freq,
        "memory_freq": memory_freq,
        "cl": cl,
        "estimated_throughput": estimated_throughput,
        "estimated_power": estimated_power
    }

    # Adjust configurations based on probabilities
    cpu_freq, gpu_freq, memory_freq, cl = adjust_configuration(
        throughput_probability, power_probability, cpu_freq, gpu_freq, memory_freq, cl
    )

    save_csv([configs], f"alert_jxavier_{sys.argv[4]}.csv")

    if throughput_probability > 0.8 and power_probability > 0.8:
        best_config = configs

    if abs(last_probability[0] - throughput_probability) <= 0.01 and abs(last_probability[1] - power_probability) <= 0.01:
        max_saturated_count -= 1
        if max_saturated_count == 0:
            print("ALERT is saturated")
            break

    last_probability = throughput_probability, power_probability

    print(f"Configs: {configs}")

print(f"Best Config: {best_config} in {sum(time_got)} sec")
