import requests
import time
import sys
import csv
import os

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

# CSV saving optimization
def save_csv(dict_list, filename):
    with open(filename, 'a', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=['id', 'cpu_cores', 'cpu_freq', 'gpu_freq', 'memory_freq', 'cl', 'throughput', 'power_cons'])
        if os.path.getsize(filename) == 0:
            writer.writeheader()
        for d in dict_list:
            writer.writerow(d)

def calibrate():
    for x in reversed(range(1190, 1909, 360)):
        for y in reversed(range(510, 1111, 300)):
            for z in reversed(range(1500, 1867, 183)):
                for _ in range(5):
                    measured_metrics = execute_config(5, x, y, z, 1)
                    configs = {
                        "cpu_cores": 6,
                        "cpu_freq": x,
                        "gpu_freq": y,
                        "memory_freq": z,
                        "cl": 1
                    }
                    dict_record = [{**configs, **measured_metrics[0]}]
                    save_csv(dict_record, f"calibration_{sys.argv[5]}_{sys.argv[4]}.csv")

    for x in range(1190, 1909, 360):
        for y in range(510, 1111, 300):
            for z in range(1500, 1867, 183):
                for _ in range(5):
                    measured_metrics = execute_config(5, x, y, z, 1)
                    configs = {
                        "cpu_cores": 6,
                        "cpu_freq": x,
                        "gpu_freq": y,
                        "memory_freq": z,
                        "cl": 1
                    }
                    dict_record = [{**configs, **measured_metrics[0]}]
                    save_csv(dict_record, f"calibration_{sys.argv[5]}_{sys.argv[4]}.csv")

                    

if __name__ == "__main__":
    calibrate()
