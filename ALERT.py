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
    CPU_FREQ_RANGE = range(806, 1511)
    GPU_FREQ_RANGE = range(306, 625)
    MEMORY_FREQ_RANGE = [2133]
    CL_RANGE = range(1, 4)

THROUGHPUT_TARGET = int(sys.argv[6])
target_pass = False

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

# -----------------------
# Profiling Configurations
# -----------------------

def profile_configurations():
    """
    Profiles a subset of configurations and returns profiling data.
    """
    if os.path.exists(f"profiling_{sys.argv[5]}_{sys.argv[4]}.csv"):
        print("[Profiling] profiling configurations was profiled.")
        with open(f"profiling_{sys.argv[5]}_{sys.argv[4]}.csv", mode='r', encoding='utf-8') as file:
            csv_reader = csv.DictReader(file)
            # Convert each row into a dictionary and add it to a list
            data = [dict(row) for row in csv_reader]
        return data
    else:
        profiling_data = []
        sampled_configs = []

        # Stratified sampling: Select a subset of configurations
        for cpu_cores in CPU_CORES_RANGE:
            for cpu_freq in np.linspace(min(CPU_FREQ_RANGE), max(CPU_FREQ_RANGE), 3):  # Example: 3 CPU frequency strata
                for gpu_freq in np.linspace(min(GPU_FREQ_RANGE), max(GPU_FREQ_RANGE), 3):
                    for memory_freq in np.linspace(min(MEMORY_FREQ_RANGE), max(MEMORY_FREQ_RANGE), 3):
                        for cl in CL_RANGE:
                            config = {"cpu_cores": int(cpu_cores), "cpu_freq": int(cpu_freq), "gpu_freq": int(gpu_freq), "memory_freq": int(memory_freq), "cl": cl}
                            sampled_configs.append(config)

        # Simulated profiling (replace with real measurements on the Jetson Xavier NX)
        for config in sampled_configs:
            t1 = time.time()
            measured_metrics, _ = execute_config(config["cpu_cores"], config["cpu_freq"], config["gpu_freq"], config["memory_freq"], config["cl"])
            elapsed_exec = round(time.time() - t1, 3)
            throughput = measured_metrics[0]['throughput']
            power = measured_metrics[0]['power_cons']
            cpu =  measured_metrics[0]["cpu_percent"]
            gpu = measured_metrics[0]["gpu_percent"]
            mem = measured_metrics[0]["mem_percent"]
            elapsed = measured_metrics[0]["elapsed"]
            time_load = measured_metrics[0]["time_load"]
            time_warm = measured_metrics[0]["time_warm"]
            time_c = measured_metrics[0]["time_c"]
            data = {**config, "throughput": throughput, "power": power, "cpu_percent": cpu, "gpu_percent": gpu, "mem_percent": mem, "profiling_time (s)": elapsed_exec, "elapsed": elapsed, "time_load": time_load, "time_warm": time_warm, "time_c": time_c}
            profiling_data.append(data)
            with open(f"profiling_{sys.argv[5]}_{sys.argv[4]}.csv", 'a', newline='') as f:
                writer = csv.DictWriter(f, fieldnames=['profiling_time (s)', 'cpu_cores', 'cpu_freq', 'gpu_freq', 'memory_freq', 'elapsed', 'cl', 'time_load', 'time_warm', 'time_c', 'throughput', 'power', 'cpu_percent', 'gpu_percent', 'mem_percent'])
                if os.path.getsize(f"profiling_{sys.argv[5]}_{sys.argv[4]}.csv") == 0:
                    writer.writeheader()
                writer.writerow(data)

        print("[Profiling] Completed profiling configurations.")
        return profiling_data


# -----------------------
# Dynamic Configuration Selection
# -----------------------

def select_best_configuration(profiling_data, throughput_target, throughput_variance):
    """
    Selects the best configuration to maximize throughput under a power budget.

    Args:
        profiling_data: List of profiled configurations with throughput, power metrics, and variance.
        throughput_target: Power budget constraint (W).

    Returns:
        dict: The best configuration.
    """
    # Step 1: Extract relevant data from profiling_data
    # power = np.array([float(entry['power']) for entry in profiling_data])  # Mean power consumption
    throughput = np.array([float(entry['throughput']) for entry in profiling_data])  # Throughput
    configurations = np.array(profiling_data)

    # Step 2: Create binary mask for valid configurations
    power_mask = (throughput > throughput_target).astype(int)  # 1 if within power budget, 0 otherwise

    # Step 3: Calculate power probabilities
    power_probabilities = np.array([calculate_probability(throughput_target, t, var) for t, var in zip(throughput, [throughput_variance])])

    # Step 4: Create value matrix for scoring
    k_throughput = 1.0  # Weight for throughput (primary objective)
    k_power_probability = 0.5  # Weight for power probability (secondary objective)
    B = 99999999  # Large constant to make valid scores positive
    value_matrix = power_mask * (B + k_throughput * throughput + k_power_probability * power_probabilities)

    # Step 5: Select the best configuration
    if np.sum(power_mask) == 0:
        print("[Runtime] No valid configuration found within the power budget.")
    
    if value_matrix.any():
        best_index = np.argmax(value_matrix)  # Find the index of the highest score
    else:
        best_index = -1
    best_config = configurations[best_index]

    return best_config, best_index

# -----------------------
# Runtime Execution Loop
# -----------------------

def execute_runtime(profiling_data, num_episodes=100):
    global THROUGHPUT_TARGET
    """
    Executes the runtime learning and adjustment process.
    """
    throughput_filter = KalmanFilter()
    power_filter = KalmanFilterPower()
    best_config = None
    throughput_var = 0.01
    target_pass = False

    for episode in range(num_episodes):
        t1 = time.time()

         # Select the best configuration dynamically
        best = select_best_configuration(profiling_data, THROUGHPUT_TARGET, throughput_var)

        best_config, best_index = best

        # Adjust frequencies to the selected configuration
        cpu_cores, cpu_freq, gpu_freq, memory_freq, cl = best_config["cpu_cores"], best_config["cpu_freq"], best_config["gpu_freq"], best_config["memory_freq"], best_config["cl"]

        t1 = time.time()
        # Simulated runtime execution (replace with actual API call)
        measured_metrics, api_time = execute_config(cpu_cores, cpu_freq, gpu_freq, memory_freq, cl)

        elapsed_exec = round(time.time() - t1, 3)
        if isinstance(measured_metrics, list) or not measured_metrics:
            if not measured_metrics:
                print("EXECUTION PROBLEM!")
                continue
            elif measured_metrics[0]['power_cons'] == 0:
                print("EXECUTION PROBLEM!")
                continue
        if measured_metrics == "No Device":
            print("No Device/No Inference Runtime")
            break

        throughput = measured_metrics[0]['throughput']
        power = measured_metrics[0]['power_cons']

        # Update Kalman Filters
        predicted_throughput, throughput_var = throughput_filter.update(throughput)
        predicted_power, _ = power_filter.update(power)

        # Update slowdown factor
        slowdown_factor = predicted_throughput / throughput
        estimated_throughput = slowdown_factor * throughput
        power_slowdown_factor = predicted_power / power
        estimated_power= power_slowdown_factor * power

        if not target_pass:
            if estimated_throughput >= THROUGHPUT_TARGET and (episode % 5) == 0:
                THROUGHPUT_TARGET += 5
            elif (episode % 5) == 0:
                if THROUGHPUT_TARGET > 5:
                    THROUGHPUT_TARGET -= 5
                    if episode != 0:
                        target_pass = True

        profiling_data[best_index]['power'] = estimated_power
        profiling_data[best_index]['throughput'] = estimated_throughput

        elapsed = round(((time.time() - elapsed_exec) - t1) * 1000, 3)
        configs = {
            "api_time": api_time,
            "episode": episode+1,
            "infer_overhead" : elapsed_exec,
            "alert_overhead" : elapsed,
            "throughput_target": THROUGHPUT_TARGET,
            "cpu_cores": int(cpu_cores)+1,
            "cpu_freq": cpu_freq,
            "gpu_freq": gpu_freq,
            "memory_freq": memory_freq,
            "cl": cl,
            "elapsed": measured_metrics[0]["elapsed"],
            "estimated_throughput": estimated_throughput,
            "estimated_power": estimated_power,
            "cpu_percent": measured_metrics[0]["cpu_percent"],
            "gpu_percent": measured_metrics[0]["gpu_percent"],
            "mem_percent": measured_metrics[0]["mem_percent"],
            "time_load": measured_metrics[0]["time_load"],
            "time_warm": measured_metrics[0]["time_warm"],
            "time_c": measured_metrics[0]["time_c"]
        }
        file = f"alert_{sys.argv[5]}_{sys.argv[4]}.csv"
        
        save_csv([configs], file)

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
        writer = csv.DictWriter(f, fieldnames=['api_time','episode', 'infer_overhead', 'alert_overhead', 'throughput_target', 'cpu_cores', 'cpu_freq', 'gpu_freq', 'memory_freq', 'cl', 'elapsed', 'time_load', 'time_warm', 'time_c', 'estimated_throughput', 'estimated_power', 'cpu_percent', 'gpu_percent', 'mem_percent'])
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
        out = execute_runtime(profiling_data, num_episodes=20)
        if out == "No Best":
            continue
        else:
            break


