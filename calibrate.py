import requests
import time
import sys
import csv
import os
from itertools import product

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

# CSV saving optimization
def save_csv(dict_list, filename):
    with open(filename, 'a', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=['api_time', 'infer_time', 'id', 'cpu_cores', 'cpu_freq', 'gpu_freq', 'memory_freq', 'cl', 'throughput', 'power_cons'])
        if os.path.getsize(filename) == 0:
            writer.writeheader()
        for d in dict_list:
            writer.writerow(d)

def calibrate():
    for _ in range(5):  # Repeat the outer loop 5 times
        for cl in [1, 2, 3]:  # Iterate over concurrency levels
            # Reversed combinations
            reversed_combinations = reversed(list(product(
                range(1, 6, 4),  # CPU cores
                range(1190, 1909, 718),  # CPU frequency
                range(510, 1111, 600),  # GPU frequency
                range(1500, 1867, 366)  # Memory frequency
            )))
            for i, x, y, z in reversed_combinations:
                for _ in range(5):
                    t1 = time.time()
                    measured_metrics, api_time = execute_config(i, x, y, z, cl)
                    elapsed_exec = round(time.time() - t1, 3)
                    configs = {
                        "api_time": api_time,
                        "infer_time": elapsed_exec,
                        "cpu_cores": i+1,
                        "cpu_freq": x,
                        "gpu_freq": y,
                        "memory_freq": z,
                        "cl": cl
                    }
                    dict_record = [{**configs, **measured_metrics[0]}]
                    save_csv(dict_record, f"calibration_{sys.argv[5]}_{sys.argv[4]}.csv")

            reversed_combinations = list(product(
                range(1, 6, 4),  # CPU cores
                range(1190, 1909, 718),  # CPU frequency
                range(510, 1111, 600),  # GPU frequency
                range(1500, 1867, 366)  # Memory frequency
            ))
            for i, x, y, z in reversed_combinations:
                for _ in range(5):
                    t1 = time.time()
                    measured_metrics, api_time = execute_config(i, x, y, z, cl)
                    elapsed_exec = round(time.time() - t1, 3)
                    configs = {
                        "api_time": api_time,
                        "infer_time": elapsed_exec,
                        "cpu_cores": i+1,
                        "cpu_freq": x,
                        "gpu_freq": y,
                        "memory_freq": z,
                        "cl": cl
                    }
                    dict_record = [{**configs, **measured_metrics[0]}]
                    save_csv(dict_record, f"calibration_{sys.argv[5]}_{sys.argv[4]}.csv")

                    

if __name__ == "__main__":
    calibrate()
