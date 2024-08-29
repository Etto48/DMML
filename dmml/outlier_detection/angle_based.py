import numpy as np

def cosine_similarity(x, a, b) -> float:
    v1 = x - a
    v2 = x - b
    distance = np.dot(v1, v1) * np.dot(v2, v2) + 1e-10
    return np.dot(v1, v2) / distance

def random_index_except(len_dataset: int, indices: list[int]) -> list[float]:
    indices = sorted(list(set(indices)))
    assert len(indices) < len_dataset
    assert all(0 <= i < len_dataset for i in indices)
    i = np.random.randint(0, len_dataset - len(indices))
    for j in indices:
        if i >= j:
            i += 1
    return i

def var_cosine_similarity(dataset: np.ndarray, x: int | np.ndarray, iterations: int) -> float:
    if isinstance(x, int):
        i = x
        x = dataset[x]
    else:
        i = None
    values = []
    for _ in range(iterations):
        if i is None:
            a_index = random_index_except(len(dataset), [])
            b_index = random_index_except(len(dataset), [a_index])
        else:
            a_index = random_index_except(len(dataset), [i])
            b_index = random_index_except(len(dataset), [i, a_index])
        a = dataset[a_index]
        b = dataset[b_index]
        similarity = cosine_similarity(x, a, b)
        values.append(similarity)
    return np.var(values)

def angle_based_outlier_detection(dataset: np.ndarray, threshold: float, iterations: int) -> list[int]:
    outliers = []
    for i in range(len(dataset)):
        std_similarity = var_cosine_similarity(dataset, i, iterations)
        if std_similarity < threshold:
            outliers.append(i)
    return outliers

if __name__ == "__main__":
    NDIM = 100

    dataset = np.random.randn(500, NDIM)
    outlier_index = np.random.randint(0, 500)
    print(f"Injecting outlier at index {outlier_index}")
    dataset[outlier_index] += 3
    
    outliers = angle_based_outlier_detection(dataset, 0.000000005, 10)
    print(f"Detected outliers: {outliers}")