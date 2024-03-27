import numpy as np
import numpy as np


def euclidean_distance(x1, x2):
    return np.linalg.norm(x1 - x2)


def total_cost(data, medoids, clusters):
    cost = 0
    for medoid_idx in medoids:
        for j in clusters[medoid_idx]:
            cost += euclidean_distance(data[j], data[medoid_idx])
    return cost


def assign_clusters(data, medoids):
    clusters = {medoid_idx: [] for medoid_idx in medoids}
    for i, point in enumerate(data):
        closest_medoid = min(
            medoids, key=lambda medoid_idx: euclidean_distance(point, data[medoid_idx]))
        clusters[closest_medoid].append(i)
    return clusters


def pam(data, k, max_iterations=1000):
    l = len(data)
    if k == 1:
        medoid = np.median(data, axis=0)
        medoid_idx = np.argmin(np.sum((data - medoid) ** 2, axis=1))
        return [medoid_idx], {medoid_idx: list(range(l))}
    else:
        medoids = np.random.choice(range(l), k, replace=False)
        # medoids = np.arange(k)
        # print(medoids)
        clusters = assign_clusters(data, medoids)
        old_cost = total_cost(data, medoids, clusters)
        td = old_cost

        # Main loop for FASTPAM1 algorithm
        for _ in range(max_iterations):
            clusters = assign_clusters(data, medoids)
            # Cache indices to nearest and second nearest medoids
            nearest_medoids = {}
            second_nearest_medoids = {}
            for i in range(l):
                nearest_medoids[i] = min(
                    medoids, key=lambda medoid_idx: euclidean_distance(data[i], data[medoid_idx]))
                second_nearest_medoids[i] = sorted(
                    medoids, key=lambda medoid_idx: euclidean_distance(data[i], data[medoid_idx]))[1]
            # print(second_nearest_medoids[2])
            DelTDF = 0  # Change in total deviation
            mstar = -1
            xstar = -1
            # for xj, _ in enumerate(data):
            #     if xj in medoids:
            #         continue
            #     print(data[xj])
            for xj, _ in enumerate(data):
                if xj in medoids:
                    continue
                dj = euclidean_distance(data[nearest_medoids[xj]], data[xj])
                # DelTD = {i: 0 for i in medoids}  # Initialize DelTD with all medoid indices
                DelTD = {i: 0 for i in range(l)}
                for i in medoids:
                    DelTD[i] = -dj
                for xo, point in enumerate(data):
                    if xo == xj:
                        continue
                    doj = euclidean_distance(data[xo], data[xj])
                    n = nearest_medoids[xo]
                    dn = euclidean_distance(data[n], data[xo])
                    ds = euclidean_distance(
                        data[second_nearest_medoids[xo]], data[xo])
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
            medoids.remove(mstar)
            medoids.append(xstar)
            td = td + DelTDF

        return medoids, clusters


if __name__ == '__main__':
    # Example usage
    X = np.array([[3, 4], [9, 10], [5, 6], [7, 8], [1, 2]])
    k = 2

    # Run PAM algorithm
    medoids, clusters = pam(X, k)

    # Print results
    print("Medoids:", X[medoids])
    for medoid_idx, cluster_points in clusters.items():
        print(f"Cluster {X[medoid_idx]}: {X[cluster_points]}")
