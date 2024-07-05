import numpy as np
import time
import pandas as pd
import os

def euclidean_distance(x1, x2):
    """Compute the Euclidean distance between two points."""
    return np.linalg.norm(x1 - x2)

def total_cost(data, medoids, clusters):
    """Calculate the total cost (sum of distances) of the clustering."""
    cost = 0
    for medoid_idx in medoids:
        for j in clusters[medoid_idx]:
            cost += euclidean_distance(data[j], data[medoid_idx])
    return cost

def assign_clusters(data, medoids):
    """Assign data points to the nearest medoids."""
    clusters = {medoid_idx: [] for medoid_idx in medoids}
    for i, point in enumerate(data):
        closest_medoid = min(medoids, key=lambda medoid_idx: euclidean_distance(point, data[medoid_idx]))
        clusters[closest_medoid].append(i)
    return clusters

def pam(data, k, max_iterations=1000):
    """Partitioning Around Medoids (PAM) algorithm."""
    l = len(data)
    # Initialize medoids: select k random data points as initial medoids
    medoids = np.random.choice(range(l), k, replace=False)
    clusters = assign_clusters(data, medoids)
    old_cost = total_cost(data, medoids, clusters)
    td = old_cost

    for _ in range(max_iterations):
        # Iterate over each non-medoid point and try swapping with each medoid
        for x in range(l):
            if x not in medoids:
                for m in range(k):
                    # Swap the current non-medoid point with the m-th medoid
                    new_medoids = medoids.copy()
                    new_medoids[m] = x
                    new_clusters = assign_clusters(data, new_medoids)
                    new_cost = total_cost(data, new_medoids, new_clusters)

                    # If the new clustering has lower cost, update medoids and clusters
                    if new_cost < old_cost:
                        medoids = new_medoids
                        clusters = new_clusters
                        old_cost = new_cost
                        td = old_cost
        # If no improvement, break the loop
        else:
            break

    return td, medoids, clusters

# Function to run PAM on a file
def run_pam_on_file(file_path):
    df = pd.read_csv(file_path, header=None)
    X = df.to_numpy()
    k = 2  # Number of clusters
    
    # Run PAM algorithm
    start_time = time.time()
    td, medoids, clusters = pam(X, k)  # Ignore returned medoids and clusters
    end_time = time.time()
    duration = end_time - start_time
    # Print min_td and time taken
    print("File:", file_path)
    print("Min_td:", np.sqrt(td))
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
    
