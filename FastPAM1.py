import numpy as np
import pandas as pd
import time
import os
from sklearn.preprocessing import MinMaxScaler

def euclidean_distance(x1, x2):
    return np.linalg.norm(x1 - x2)

def total_cost_data(data, sampled_data, medoids, clusters):
    cost = 0
    for medoid_idx in medoids:
        for j in clusters[medoid_idx]:
            cost += euclidean_distance(data[j], sampled_data[medoid_idx])
    return cost

def total_cost(data, medoids, clusters):
    cost = 0
    for medoid_idx in medoids:
        for j in clusters[medoid_idx]:
            cost += euclidean_distance(data[j], data[medoid_idx])
    return cost

def assign_clusters_data(data, sampled_data, medoids):
    clusters = {medoid_idx: [] for medoid_idx in medoids}
    for i, point in enumerate(data):
        closest_medoid = min(medoids, key=lambda medoid_idx: euclidean_distance(point, sampled_data[medoid_idx]))
        clusters[closest_medoid].append(i)
    return clusters

def assign_clusters(data, medoids):
    clusters = {medoid_idx: [] for medoid_idx in medoids}
    for i, point in enumerate(data):
        closest_medoid = min(medoids, key=lambda medoid_idx: euclidean_distance(point, data[medoid_idx]))
        clusters[closest_medoid].append(i)
    return clusters

# fastpam1
def pam(data, k, max_iterations=1000):
    l = len(data)
    if k == 1:
        medoid = np.median(data, axis=0)
        medoid_idx = np.argmin(np.sum((data - medoid) ** 2, axis=1))
        return [medoid_idx], {medoid_idx: list(range(l))}
    else:
        medoids = np.random.choice(range(l), k, replace=False)

        # Main loop for FASTPAM1 algorithm
        for _ in range(max_iterations):
            clusters = assign_clusters(data, medoids)
            for medoid in medoids:
                medoid_points = [data[j] for j in clusters[medoid]]
                medoid_point = np.median(medoid_points, axis=0)
                medoid = np.argmin(np.sum((medoid_points - medoid_point) ** 2, axis=1))
            old_cost = total_cost(data, medoids, clusters)
            td = old_cost
            nearest_medoids = {}
            second_nearest_medoids = {}
            
            for i in range(l):
                nearest_medoids[i] = min(medoids, key=lambda medoid_idx: euclidean_distance(data[i], data[medoid_idx]))
                second_nearest_medoids[i] = sorted(medoids, key=lambda medoid_idx: euclidean_distance(data[i], data[medoid_idx]))[1]

            DelTDF = 0  # Change in total deviation
            mstar = -1
            xstar = -1
        
            for xj, _ in enumerate(data):
                if xj in medoids:
                    continue
                dj = euclidean_distance(data[nearest_medoids[xj]], data[xj])

                DelTD ={i: 0 for i in range(l)}
                
                for i in medoids:
                    DelTD[i] = -dj
                    
                for xo, point in enumerate(data):
                    if xo == xj:
                        continue
                    doj = euclidean_distance(data[xo], data[xj])
                    n = nearest_medoids[xo]
                    dn = euclidean_distance(data[n], data[xo])
                    ds = euclidean_distance(data[second_nearest_medoids[xo]], data[xo])
                    DelTD[n] += min(doj, ds) - dn
                    if doj < dn:
                        for i in medoids:
                            if i == n:
                                continue
                            DelTD[i] += doj - dn
                i = np.argmin(DelTD)  # Get index of minimum value in DelTD
                if DelTD[i] < DelTDF:
                    DelTDF = DelTD[i]
                    mstar = medoids[i]
                    xstar = xj
            if DelTDF >= 0:
                break
            medoids = np.append(np.delete(medoids, np.where(medoids == mstar)), xstar)
            td = td + DelTDF  

        return td, medoids, clusters
    
def compute_inclusion_probability(deviations, N):
    # Compute the inclusion probability for each data point
    pi = (N * (1 / deviations)) / np.sum(1 / deviations)
    return pi
    
def inclusionProbablities(X, N):
    max_values = np.max(X, axis=0)
    # Compute the deviation for each data point
    deviations = np.sum(max_values - X, axis=1)
    # print(deviations[:10])
    return compute_inclusion_probability(deviations, N)

    
    
# Function to normalize data and sample it
def sample(X):
    if int(len(X) / 2) == 0:
        return X
    
    scaler = MinMaxScaler()
    X_Scaled = scaler.fit_transform(X)
    N = int(len(X_Scaled) / 2)
    pi = inclusionProbablities(X_Scaled, N)  # Assuming this function is defined elsewhere
    
    sampled_indices = set()  # Set to keep track of sampled indices
    indices_to_delete = set()  # Set to keep track of indices to delete
    
    temp = X.copy()
    
    while len(sampled_indices) < N and len(temp) > 1:
        i, j = np.random.choice(range(len(temp)), 2, replace=False)
        
        if pi[i] + pi[j] >= 1:
            # Select one of the units
            if pi[i] >= pi[j]:
                sampled_indices.add(i)
                indices_to_delete.add(i)
            else:
                sampled_indices.add(j)
                indices_to_delete.add(j)
        else:
            # Remove the smaller probability element
            if pi[i] >= pi[j]:
                indices_to_delete.add(j)
            else:
                indices_to_delete.add(i)
        
        temp = np.delete(temp, list(indices_to_delete), axis=0)
        pi = np.delete(pi, list(indices_to_delete))
        indices_to_delete.clear()
        
    # Create the sampled dataset
    sampled_X = X[list(sampled_indices)]
    
    return sampled_X



# Function to run PAM on a file
def run_pam_on_file(file_path):
    df = pd.read_csv(file_path, header=None)
    X = df.to_numpy()
    Sampled_X = sample(X)
    # Sampled_X = X
    if len(Sampled_X) == 0:
        print(f"Sampled dataset is empty for file: {file_path}")
        return
    k = 2  
    
    # Run PAM algorithm
    start_time = time.time()
    td, medoids, clusters = pam(Sampled_X, k)  # Ignore returned medoids and clusters
    end_time = time.time()
    
    # Calculate duration
    duration = end_time - start_time
    min_td, min_medoids, min_clusters = td, medoids, clusters
    for _ in range(10):  # Run the algorithm multiple times to find the configuration with the minimum total deviation
        td, medoids, clusters = pam(Sampled_X, k)
        if td < min_td:
            min_td, min_medoids, min_clusters = td, medoids, clusters
            
    clusters = assign_clusters_data(X, Sampled_X, min_medoids)
    
    total_deviation = total_cost_data(X, Sampled_X, min_medoids, clusters)
    
    # Print min_td and time taken
    print("File:", file_path)
    print("Min_td:", np.sqrt(total_deviation))
    print("Time taken:", duration, "seconds\n")
    print("")

# Main execution
if __name__ == "__main__":
    # Directory containing CSV files
    directory = "featurevector"

    # List all CSV files in the directory
    csv_files = [os.path.join(directory, f) for f in os.listdir(directory) if f.endswith('.csv')]

    # Iterate over each CSV file and run PAM algorithm
    for file_path in csv_files:
        run_pam_on_file(file_path)
