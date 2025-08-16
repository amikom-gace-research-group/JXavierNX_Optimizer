import requests
import time
import sys
import csv
import numpy as np
import os
from itertools import product

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
        writer = csv.DictWriter(f, fieldnames=['api_time', 'infer_time', 'id', 'cpu_cores', 'cpu_freq', 'gpu_freq', 'memory_freq', 'cl', 'throughput', 'power_cons'])
        if os.path.getsize(filename) == 0:
            writer.writeheader()
        for d in dict_list:
            writer.writerow(d)

def calibrate():
    sampled_configs = []

    # Stratified sampling: Select a subset of configurations
    for cpu_cores in CPU_CORES_RANGE:
        for cpu_freq in np.linspace(min(CPU_FREQ_RANGE), max(CPU_FREQ_RANGE), 3):  # Example: 3 CPU frequency strata
            for gpu_freq in np.linspace(min(GPU_FREQ_RANGE), max(GPU_FREQ_RANGE), 3):
                for memory_freq in np.linspace(min(MEMORY_FREQ_RANGE), max(MEMORY_FREQ_RANGE), 3):
                    for cl in CL_RANGE:
                        config = {"cpu_cores": int(cpu_cores), "cpu_freq": int(cpu_freq), "gpu_freq": int(gpu_freq), "memory_freq": int(memory_freq), "cl": cl}
                        sampled_configs.append(config)
    # Reversed combinations
    for config in reversed(sampled_configs):
        for _ in range(5):
            t1 = time.time()
            measured_metrics, api_time = execute_config(config["cpu_cores"], config["cpu_freq"], config["gpu_freq"], config["memory_freq"], config["cl"])
            elapsed_exec = round(time.time() - t1, 3)
            configs = {
                "api_time": api_time,
                "infer_time": elapsed_exec,
                "cpu_cores": config["cpu_cores"]+1,
                "cpu_freq": config["cpu_freq"],
                "gpu_freq": config["gpu_freq"],
                "memory_freq": config["memory_freq"],
                "cl": config["cl"]
            }
            dict_record = [{**configs, **measured_metrics[0]}]
            save_csv(dict_record, f"calibration_{sys.argv[5]}_{sys.argv[4]}.csv")

    for config in sampled_configs:
        for _ in range(5):
            t1 = time.time()
            measured_metrics, api_time = execute_config(config["cpu_cores"], config["cpu_freq"], config["gpu_freq"], config["memory_freq"], config["cl"])
            elapsed_exec = round(time.time() - t1, 3)
            configs = {
                "api_time": api_time,
                "infer_time": elapsed_exec,
                "cpu_cores": config["cpu_cores"]+1,
                "cpu_freq": config["cpu_freq"],
                "gpu_freq": config["gpu_freq"],
                "memory_freq": config["memory_freq"],
                "cl": config["cl"]
            }
            dict_record = [{**configs, **measured_metrics[0]}]
            save_csv(dict_record, f"calibration_{sys.argv[5]}_{sys.argv[4]}.csv")

if __name__ == "__main__":
    calibrate()
