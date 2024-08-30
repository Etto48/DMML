import numpy as np

def k_means(dataset: np.ndarray, k: int, max_iter: int = 100) -> list[set[int]]:
    N = len(dataset)
    def dist(a: np.ndarray, b: np.ndarray) -> float:
        return np.linalg.norm(a - b)
    def nearest_centroid(centroids: np.ndarray, point: np.ndarray) -> int:
        min_dist = float("inf")
        nearest = -1
        for i in range(len(centroids)):
            d = dist(centroids[i], point)
            if d < min_dist:
                min_dist = d
                nearest = i
        return nearest
    centroids = np.array([dataset[np.random.randint(0, N)] for _ in range(k)])
    for _ in range(max_iter):
        clusters = [set() for _ in range(k)]
        for i in range(N):
            clusters[nearest_centroid(centroids, dataset[i])].add(i)
        new_centroids = np.zeros((k, dataset.shape[1]))
        for i in range(k):
            if len(clusters[i]) == 0:
                new_centroids[i] = dataset[np.random.randint(0, N)]
            else:
                new_centroids[i] = np.mean([dataset[j] for j in clusters[i]], axis=0)
        if np.array_equal(centroids, new_centroids):
            break
        centroids = new_centroids
    return clusters