import numpy as np

def cosine_similarity(x, a, b, normalized: bool = True) -> float:
    v1 = x - a
    v2 = x - b
    distance = np.linalg.norm(v1) * np.linalg.norm(v2) + 1e-10
    if normalized:
        return np.dot(v1, v2) / distance
    else:
        return np.dot(v1, v2), distance

def angle_based_outlier_detection(dataset: np.ndarray, threshold: float, iterations: int, distance_weighted: bool = False) -> list[int]:
    outliers = []
    for i, x in enumerate(dataset):
        avg_similarity = 0
        total_distance = 0
        for _ in range(iterations):
            a, b = dataset[np.random.choice(len(dataset), 2, replace=False)]
            if distance_weighted:
                similarity, distance = cosine_similarity(x, a, b, normalized=False)
                total_distance += distance
                avg_similarity += similarity
            else:
                total_distance += 1
                avg_similarity += cosine_similarity(x, a, b, normalized=True)
        avg_similarity /= total_distance
        if avg_similarity > threshold:
            outliers.append(i)
    return outliers

if __name__ == "__main__":
    NDIM = 100

    dataset = np.random.randn(500, NDIM)
    outlier_index = np.random.randint(0, 500)
    print(f"Injecting outlier at index {outlier_index}")
    dataset[outlier_index] += 3
    
    outliers = angle_based_outlier_detection(dataset, 0.9, 10, True)
    print(f"Detected outliers: {outliers}")