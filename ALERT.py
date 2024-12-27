import numpy as np
import sys
import time
import os
import csv
import requests
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
    CPU_FREQ_RANGE = range(806, 1510)
    GPU_FREQ_RANGE = range(306, 624)
    MEMORY_FREQ_RANGE = range(1500, 2133)
    CL_RANGE = range(1, 3)

POWER_BUDGET = int(sys.argv[6])

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

# -----------------------
# Profiling Configurations
# -----------------------

def profile_configurations():
    """
    Profiles a subset of configurations and returns profiling data.
    """
    if os.path.exists("profiling_alert.csv"):
        print("[Profiling] profiling configurations was profiled.")
        with open("profiling_alert.csv", mode='r', encoding='utf-8') as file:
            csv_reader = csv.DictReader(file)
            # Convert each row into a dictionary and add it to a list
            data = [dict(row) for row in csv_reader]
        return data
    else:
        profiling_data = []
        sampled_configs = []

        # Stratified sampling: Select a subset of configurations
        for cpu_cores in np.linspace(min(CPU_CORES_RANGE), max(CPU_CORES_RANGE), 3):
            for cpu_freq in np.linspace(min(CPU_FREQ_RANGE), max(CPU_FREQ_RANGE), 3):  # Example: 3 CPU frequency strata
                for gpu_freq in np.linspace(min(GPU_FREQ_RANGE), max(GPU_FREQ_RANGE), 3):
                    for memory_freq in np.linspace(min(MEMORY_FREQ_RANGE), max(MEMORY_FREQ_RANGE), 3):
                        for cl in CL_RANGE:
                            config = {"cpu_cores": int(cpu_cores), "cpu_freq": int(cpu_freq), "gpu_freq": int(gpu_freq), "memory_freq": int(memory_freq), "cl": cl}
                            sampled_configs.append(config)

        # Simulated profiling (replace with real measurements on the Jetson Xavier NX)
        for config in sampled_configs:
            measured_metrics, _ = execute_config(config["cpu_cores"], config["cpu_freq"], config["gpu_freq"], config["memory_freq"], config["cl"])
            throughput = measured_metrics[0]['throughput']
            power = measured_metrics[0]['power_cons']
            data = {**config, "throughput": throughput, "power": power}
            profiling_data.append(data)
            with open("profiling_alert.csv", 'a', newline='') as f:
                writer = csv.DictWriter(f, fieldnames=['cpu_cores', 'cpu_freq', 'gpu_freq', 'memory_freq', 'cl', 'throughput', 'power'])
                if os.path.getsize("profiling_alert.csv") == 0:
                    writer.writeheader()
                writer.writerow(data)

        print("[Profiling] Completed profiling configurations.")
        return profiling_data


# -----------------------
# Dynamic Configuration Selection
# -----------------------

def select_best_configuration(profiling_data, power_budget, power_variance):
    """
    Selects the best configuration to maximize throughput under a power budget.

    Args:
        profiling_data: List of profiled configurations with throughput, power metrics, and variance.
        power_budget: Power budget constraint (W).

    Returns:
        dict: The best configuration.
    """
    # Step 1: Extract relevant data from profiling_data
    power = np.array([float(entry['power']) for entry in profiling_data])  # Mean power consumption
    throughput = np.array([float(entry['throughput']) for entry in profiling_data])  # Throughput
    configurations = np.array(profiling_data)

    # Step 2: Create binary mask for valid configurations
    power_mask = (power <= power_budget).astype(int)  # 1 if within power budget, 0 otherwise

    # Step 3: Calculate power probabilities
    power_probabilities = np.array([calculate_probability(power_budget, p, var) for p, var in zip(power, [power_variance])])

    # Step 4: Create value matrix for scoring
    k_throughput = 1.0  # Weight for throughput (primary objective)
    k_power_probability = 0.5  # Weight for power probability (secondary objective)
    B = 99999999  # Large constant to make valid scores positive
    value_matrix = power_mask * (B + k_throughput * throughput + k_power_probability * power_probabilities)

    # Step 5: Select the best configuration
    if np.sum(power_mask) == 0:
        print("No valid configuration found within the power budget.")
        return None

    best_index = np.argmax(value_matrix)  # Find the index of the highest score
    best_config = configurations[best_index]

    return best_config, best_index


# -----------------------
# Runtime Execution Loop
# -----------------------

def execute_runtime(profiling_data, num_episodes=100):
    """
    Executes the runtime learning and adjustment process.
    """
    throughput_filter = KalmanFilter()
    power_filter = KalmanFilterPower()
    best_config = None
    power_var = 0.01

    for episode in range(num_episodes):
        t1 = time.time()

         # Select the best configuration dynamically
        best = select_best_configuration(profiling_data, POWER_BUDGET, power_var)

        if best is None:
            print("[Runtime] No valid configuration found.")
            os.remove("profiling_alert.csv")
            break

        best_config, best_index = best

        # Adjust frequencies to the selected configuration
        cpu_cores, cpu_freq, gpu_freq, memory_freq, cl = best_config["cpu_freq"], best_config["gpu_freq"], best_config["memory_freq"], best_config["cl"]

        t1 = time.time()
        # Simulated runtime execution (replace with actual API call)
        measured_metrics, api_time = execute_config(cpu_cores, cpu_freq, gpu_freq, memory_freq, cl)

        elapsed_exec = round(time.time() - t1, 3)
        if not measured_metrics:
            print("EXECUTION PROBLEM!")
            continue
        if measured_metrics == "No Device":
            print("No Device/No Inference Runtime")
            break

        throughput = measured_metrics[0]['throughput']
        power = measured_metrics[0]['power_cons']

        # Update Kalman Filters
        predicted_throughput, _ = throughput_filter.update(throughput)
        predicted_power, power_var = power_filter.update(power)

        # Update slowdown factor
        slowdown_factor = predicted_throughput / throughput
        estimated_throughput = slowdown_factor * throughput
        power_slowdown_factor = predicted_power / power
        estimated_power= power_slowdown_factor * power

        profiling_data[best_index]['power'] = estimated_power
        profiling_data[best_index]['throughput'] = estimated_throughput

        elapsed = round(((time.time() - elapsed_exec) - t1) * 1000, 3)
        configs = {
            "api_time": api_time,
            "episode": episode+1,
            "infer_overhead" : elapsed_exec,
            "alert_overhead" : elapsed,
            "power_budget": POWER_BUDGET,
            "cpu_cores": cpu_cores+1,
            "cpu_freq": cpu_freq,
            "gpu_freq": gpu_freq,
            "memory_freq": memory_freq,
            "cl": cl,
            "estimated_throughput": estimated_throughput,
            "estimated_power": estimated_power
        }
        save_csv([configs], f"alert_{sys.argv[5]}_{sys.argv[4]}.csv")

        print(f"[Runtime] Episode {episode + 1}: Selected configuration {best_config}")

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
        writer = csv.DictWriter(f, fieldnames=['api_time','episode', 'infer_overhead', 'alert_overhead', 'power_budget', 'cpu_cores', 'cpu_freq', 'gpu_freq', 'memory_freq', 'cl', 'estimated_throughput', 'estimated_power'])
        if os.path.getsize(filename) == 0:
            writer.writeheader()
        for d in dict_list:
            writer.writerow(d)

# -----------------------
# Execution
# -----------------------

if __name__ == "__main__":
    while True:
        # Step 1: Profiling
        profiling_data = profile_configurations()

        # Step 2: Runtime execution
        out = execute_runtime(profiling_data, num_episodes=100)
        if out == "No Best":
            continue
        else:
            break


