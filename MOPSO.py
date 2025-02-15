import numpy as np
import requests
import csv
import time
import sys
import os

def set_device_ranges(device_type):
    if device_type == 'jxavier':
        CPU_CORES_RANGE = np.linspace(1, 6, 3)
        CPU_FREQ_RANGE = range(1190, 1909)
        GPU_FREQ_RANGE = range(510, 1111)
        MEMORY_FREQ_RANGE = range(1500, 1867)
        CL_RANGE = range(1, 4)
    elif device_type == 'jorin-nano':
        CPU_CORES_RANGE = [5]
        CPU_FREQ_RANGE = range(806, 1510)
        GPU_FREQ_RANGE = range(306, 624)
        MEMORY_FREQ_RANGE = range(1500, 2133)
        CL_RANGE = range(1, 4)
    else:
        raise ValueError("Invalid device type specified.")
    

    sampled_configs ={
        "CPU_CORES_RANGE": CPU_CORES_RANGE, 
        "CPU_FREQ_RANGE": np.linspace(min(CPU_FREQ_RANGE), max(CPU_FREQ_RANGE), 3), 
        "GPU_FREQ_RANGE": np.linspace(min(GPU_FREQ_RANGE), max(GPU_FREQ_RANGE), 3), 
        "MEMORY_FREQ_RANGE": np.linspace(min(MEMORY_FREQ_RANGE), max(MEMORY_FREQ_RANGE), 3), 
        "CL_RANGE": CL_RANGE
    }

    return sampled_configs


POWER_BUDGET = int(sys.argv[6])

time_got = []
prohibited_configs = set()

# Function to get the result from the external system
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

# Reward function based on power and throughput metrics
def calculate_fitness(measured_metrics):
    power = measured_metrics[0]["power_cons"]
    throughput = measured_metrics[0]["throughput"]
    
    if power > POWER_BUDGET:
        return 1e-6
    
    return (throughput / power) * 1e6

class Particle:
    def __init__(self, problem_size):
        self.position = np.random.uniform(0, 1, problem_size)
        self.velocity = np.zeros(problem_size)
        self.best_position = np.copy(self.position)
        self.best_fitness = -1

    def update_velocity(self, global_best_position, w=0.5, c1=1.5, c2=1.5):
        r1 = np.random.uniform(0, 1, len(self.velocity))
        r2 = np.random.uniform(0, 1, len(self.velocity))
        cognitive = c1 * r1 * (self.best_position - self.position)
        social = c2 * r2 * (global_best_position - self.position)
        self.velocity = w * self.velocity + cognitive + social

    def update_position(self, bounds):
        self.position += self.velocity
        for i in range(len(self.position)):
            if self.position[i] < bounds[i][0]:
                self.position[i] = bounds[i][0]
            elif self.position[i] > bounds[i][1]:
                self.position[i] = bounds[i][1]

# Save results to CSV
def save_csv(results, filename):
    # Write the results to a CSV file
    with open(filename, 'a', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=['api_time', 'episode', 'iteration', 'reward', 'xavier_time_elapsed', 'cpu_cores', 'cpu_freq', 'gpu_freq', 'mem_freq', 'cl', 'throughput', 'power_cons', 'cpu_percent', 'gpu_percent', 'mem_percent'])
        if os.path.getsize(filename) == 0:  # Check if file is empty
            writer.writeheader()  # Write header only once
        writer.writerows(results)

# MOPSO Class
class MOPSO:
    def __init__(self, swarm_size, problem_size, bounds, max_iter, saturation_threshold, config_ranges, api_url, auth_header, power_budget):
        self.swarm_size = swarm_size
        self.problem_size = problem_size
        self.bounds = bounds
        self.max_iter = max_iter
        self.saturation_threshold = saturation_threshold
        self.config_ranges = config_ranges
        self.api_url = api_url
        self.auth_header = auth_header
        self.power_budget = power_budget
        self.swarm = [Particle(problem_size) for _ in range(swarm_size)]
        self.global_best_position = np.zeros(problem_size)
        self.global_best_fitness = -1
        self.best_config = None
        self.best_throughput = -float('inf')

    def optimize(self):
        results = []
        episode = 0
        for iteration in range(self.max_iter):
            best_fitness_this_iter = -1
            for particle in self.swarm:
                episode += 1
                config = [
                    int(particle.position[0] * (self.config_ranges["CPU_CORES_RANGE"][-1] - self.config_ranges["CPU_CORES_RANGE"][0]) + self.config_ranges["CPU_CORES_RANGE"][0]),
                    int(particle.position[1] * (self.config_ranges["CPU_FREQ_RANGE"][-1] - self.config_ranges["CPU_FREQ_RANGE"][0]) + self.config_ranges["CPU_FREQ_RANGE"][0]),
                    int(particle.position[2] * (self.config_ranges["GPU_FREQ_RANGE"][-1] - self.config_ranges["GPU_FREQ_RANGE"][0]) + self.config_ranges["GPU_FREQ_RANGE"][0]),
                    int(particle.position[3] * (self.config_ranges["MEMORY_FREQ_RANGE"][-1] - self.config_ranges["MEMORY_FREQ_RANGE"][0]) + self.config_ranges["MEMORY_FREQ_RANGE"][0]),
                    int(particle.position[4] * (self.config_ranges["CL_RANGE"][-1] - self.config_ranges["CL_RANGE"][0]) + self.config_ranges["CL_RANGE"][0])
                ]
                if tuple(config) in prohibited_configs:
                    print("Prohibited Configuration!")
                    continue
                t2 = time.time()
                metrics, api_time = execute_config(*config)
                elapsed_exec = round(time.time() - t2, 3)
                time_got.append(elapsed_exec)
                if not metrics or metrics == "No Device":
                    continue
                if metrics == "No Device":
                    break

                fitness = calculate_fitness(metrics)

                if fitness == 1e-6:
                    print("Prohibited Configuration!")
                    prohibited_configs.add(tuple(config))

                if fitness > particle.best_fitness and metrics[0]["throughput"] > self.best_throughput:
                    particle.best_fitness = fitness
                    particle.best_position = np.copy(particle.position)
                    self.best_throughput = metrics[0]["throughput"]

                if fitness > self.global_best_fitness:
                    self.best_config = config
                    self.global_best_fitness = fitness
                    self.global_best_position = np.copy(particle.position)

                # Save results to CSV
                result_entry = {
                    "api_time": api_time,
                    "episode": episode,
                    "iteration": iteration,
                    'reward': fitness,
                    'xavier_time_elapsed': elapsed_exec,
                    'cpu_cores': config[0]+1,
                    'cpu_freq': config[1],
                    'gpu_freq': config[2],
                    'mem_freq': config[3],
                    'cl': config[4],
                    'throughput': metrics[0]["throughput"],
                    'power_cons': metrics[0]["power_cons"],
                    "cpu_percent": metrics[0]["cpu_percent"],
                    "gpu_percent": metrics[0]["gpu_percent"],
                    "mem_percent": metrics[0]["mem_percent"]
                }
                results.append(result_entry)

                # Track the best fitness for this iteration
                best_fitness_this_iter = max(best_fitness_this_iter, fitness)

            # Update velocity and position for each particle
            for particle in self.swarm:
                particle.update_velocity(self.global_best_position)
                particle.update_position(self.bounds)

            print(f"Iteration {iteration + 1}/{self.max_iter}, Best Fitness: {self.global_best_fitness}")
            if metrics == "No Device":
                break
        save_csv(results, f"mopso_{sys.argv[5]}_{sys.argv[4]}.csv")  # Save all results to CSV after optimization
        return self.best_config, self.global_best_fitness

# Main Execution
if __name__ == "__main__":
    device_ranges = set_device_ranges(sys.argv[5])
    bounds = np.array([(0, 1) for _ in range(5)])
    mopso = MOPSO(
        swarm_size=10,
        problem_size=5,
        bounds=bounds,
        max_iter=10,
        saturation_threshold=50,
        config_ranges=device_ranges,
        api_url=sys.argv[1],
        auth_header=sys.argv[2],
        power_budget=int(sys.argv[6]),
    )
    # Run the MOPSO algorithm
    t1 = time.time()
    best_config, best_fitness = mopso.optimize()
    elapsed = round(((time.time() - sum(time_got)) - t1) * 1000, 3)
    print(f"Best configuration found: {best_config} in {elapsed} ms")
    print(f"Objective value: {best_fitness}")
