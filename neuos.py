import numpy as np
import pandas as pd
import os
import time
import requests
import sys
import csv

# Configuration ranges for CPU, GPU, and memory
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

# -----------------------
# Profiling Configurations
# -----------------------

def profile_configurations():
    """
    Profiles a subset of configurations and returns profiling data.
    """
    if os.path.exists("profiling_neuos.csv"):
        print("[Profiling] profiling configurations was profiled.")
        with open("profiling_neuos.csv", mode='r', encoding='utf-8') as file:
            csv_reader = csv.DictReader(file)
            data = []
            for row in csv_reader:
                new_row = {}
                for key, value in row.items():
                    # Attempt to convert the value to float
                    try:
                        float_value = float(value)
                        # Convert to integer by rounding
                        new_row[key] = round(float_value)
                    except ValueError:
                        # Keep the original value if it's not a number
                        new_row[key] = value
                data.append(new_row)
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
            data = {**config, "throughput": throughput, "power": power, "cpu_percent": cpu, "gpu_percent": gpu, "mem_percent": mem, "profiling_time (s)": elapsed_exec}
            profiling_data.append(data)
            with open("profiling_neuos.csv", 'a', newline='') as f:
                writer = csv.DictWriter(f, fieldnames=['profiling_time (s)', 'cpu_cores', 'cpu_freq', 'gpu_freq', 'memory_freq', 'cl', 'throughput', 'power', 'cpu_percent', 'gpu_percent', 'mem_percent'])
                if os.path.getsize("profiling_neuos.csv") == 0:
                    writer.writeheader()
                writer.writerow(data)

        print("[Profiling] Completed profiling configurations.")
        return profiling_data

# SpeedUp and PowerUp table from the NeuOS algorithm
SpeedUp_PowerUp = {'1':[1.0, 1.0], '2':[2.1, 2], '3':[2.8, 1.5]}
chosen_dvfs = [[], [], []]

POWER_BUDGET = int(sys.argv[6])

def select_dvfs(df_prof):
    baseline_dvfs = df_prof["cpu_cores"].min(), df_prof["cpu_freq"].min(), df_prof["gpu_freq"].min(), df_prof["memory_freq"].min(), df_prof["cl"].min()
    baseline = df_prof[(df_prof["cpu_cores"] == baseline_dvfs[0]) & (df_prof["cpu_freq"] == baseline_dvfs[1]) & (df_prof["gpu_freq"] == baseline_dvfs[2]) & (df_prof["memory_freq"] == baseline_dvfs[3]) & (df_prof["cl"] == baseline_dvfs[4])]
    baseline_troughput, baseline_power = round(baseline["throughput"].iloc[0]), round(baseline["power"].iloc[0])
    chosen_dvfs[0] = list(baseline_dvfs)
    moderate = df_prof[(df_prof["throughput"] >= (baseline_troughput * SpeedUp_PowerUp['2'][0])) & (df_prof["power"] <= (baseline_power * SpeedUp_PowerUp['2'][1]))]
    if not moderate.empty:
        moderate_dvfs = moderate["cpu_cores"].min(), moderate["cpu_freq"].min(), moderate["gpu_freq"].min(), moderate["memory_freq"].min(), moderate["cl"].min()
        chosen_dvfs[1] = list(moderate_dvfs)
    high = df_prof[(df_prof["throughput"] >= (baseline_troughput * SpeedUp_PowerUp['3'][0])) & (df_prof["power"] <= (baseline_power * SpeedUp_PowerUp['3'][1]))]
    if not high.empty:
        high_dvfs = high["cpu_cores"].min(), high["cpu_freq"].min(), high["gpu_freq"].min(), high["memory_freq"].min(), high["cl"].min()
        chosen_dvfs[2] = list(high_dvfs)

def update_output(throughput, power, configs, df_prof):
    df_prof["throughput"][(df_prof["cpu_cores"] == configs[0]) & (df_prof["cpu_freq"] == configs[1]) & (df_prof["gpu_freq"] == configs[2]) & (df_prof["memory_freq"] == configs[3]) & (df_prof["cl"] == configs[4])] = throughput
    df_prof["power"][(df_prof["cpu_cores"] == configs[0]) & (df_prof["cpu_freq"] == configs[1]) & (df_prof["gpu_freq"] == configs[2]) & (df_prof["memory_freq"] == configs[3]) & (df_prof["cl"] == configs[4])] = power

def delta_calculator(power_consumed):
    S_p = (POWER_BUDGET + (POWER_BUDGET - power_consumed)) / POWER_BUDGET
    return chosen_dvfs[round(S_p)]

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

# CSV saving optimization
def save_csv(dict_list, filename):
    with open(filename, 'a', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=['api_time','episode', 'infer_overhead (sec)', 'neuos_overhead (ms)', 'lag', 'power_budget', 'cpu_cores', 'cpu_freq', 'gpu_freq', 'memory_freq', 'cl', 'throughput', 'power', 'cpu_percent', 'gpu_percent', 'mem_percent'])
        if os.path.getsize(filename) == 0:
            writer.writeheader()
        for d in dict_list:
            writer.writerow(d)

def execute_runtime(num_episodes=100):
    prof = profile_configurations()

    df_prof = pd.DataFrame(prof)
    select_dvfs(df_prof)
   
    dvfs_config = delta_calculator(float(df_prof['power'][df_prof["power"] <= POWER_BUDGET].max()))
    cpu_cores, cpu_freq, gpu_freq, memory_freq, cl = dvfs_config

    best_power = POWER_BUDGET
    time_got = []
    best_config = None

    for episode in range(num_episodes):  # Example: run for 100 episodes
        # Get current metrics (throughput and power)
        t1 = time.time()
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
        power_consumed = measured_metrics[0]['power_cons']

        confs = cpu_cores, cpu_freq, gpu_freq, memory_freq, cl
        update_output(throughput, power_consumed, confs, df_prof)
        select_dvfs(df_prof)
        
        # Choose DVFS configuration based on LAG and power consumption
        dvfs_config = delta_calculator(power_consumed)

        elapsed = round(((time.time() - elapsed_exec) - t1) * 1000, 3)
        time_got.append(elapsed+elapsed_exec)

        configs = {
            "api_time": api_time,
            "episode": episode,
            "infer_overhead (sec)" : elapsed_exec,
            "neuos_overhead (ms)" : elapsed,
            "power_budget": POWER_BUDGET,
            "cpu_cores": cpu_cores+1,
            "cpu_freq": cpu_freq,
            "gpu_freq": gpu_freq,
            "memory_freq": memory_freq,
            "cl": cl,
            "throughput": throughput,
            "power": power_consumed,
            "cpu_percent": measured_metrics[0]["cpu_percent"],
            "gpu_percent": measured_metrics[0]["gpu_percent"],
            "mem_percent": measured_metrics[0]["mem_percent"]
        }

        # Apply the DVFS configuration
        cpu_cores, cpu_freq, gpu_freq, memory_freq, cl = dvfs_config
        
        save_csv([configs], f"neuos_jxavier_{sys.argv[4]}.csv")
        # Log the results
        print(f"Configs: {configs}")

        # Check if the system meets the target and stop if it stabilizes
        if power_consumed <= best_power:
            best_power = power_consumed
            best_config = configs

    print(f"Best Config: {best_config} in {sum(time_got)} sec")

if __name__ == "__main__":
    execute_runtime()