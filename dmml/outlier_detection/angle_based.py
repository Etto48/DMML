import sys
sys.path.append(f'{__file__}/../../../')
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

def var_cosine_similarity(dataset: np.ndarray | list[list[float]], x: int | np.ndarray, iterations: int) -> float:
    if isinstance(x, int):
        i = x
        if isinstance(dataset, np.ndarray):
            x = dataset[i]
        elif isinstance(dataset, list):
            x = dataset[i]
        else:
            raise ValueError("Invalid type for dataset")
    else:
        i = None
    values = []
    if isinstance(dataset, np.ndarray):
        dataset = dataset.tolist()
    elif isinstance(dataset, list):
        pass
    else:
        raise ValueError("Invalid type for dataset")
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


def angle_based_outlier_factor(dataset: np.ndarray, iterations: int) -> list[tuple[int, float]]:
    factors = []
    for i in range(len(dataset)):
        std_similarity = var_cosine_similarity(dataset, i, iterations)
        factors.append((i, std_similarity))
    return factors

def elbow_method(values: list[float], limit_index: None) -> int:
    max_index = len(values) if limit_index is None else min(limit_index, len(values))
    dvalues = [values[i] - values[i - 1] for i in range(1, max_index)]
    mean = np.mean(dvalues)
    std = np.std(dvalues)
    last_index = None
    for i in range(1, max_index):
        if dvalues[i - 1] > mean + 3 * std:
            last_index = i
    return last_index

def angle_based_outlier_detection(dataset: np.ndarray, iterations: int, **kwargs) -> list[int]:
    """
    Detect outliers in a dataset using the Angle-Based Outlier Detection method.
    
    Parameters:
    - dataset `np.ndarray`: the dataset to analyze
    - iterations `int`: the number of iterations to compute the variance of the cosine similarity
    - method `str`: the method to use to detect outliers. either "elbow" or "threshold", if None, "elbow" method will be used
    - threshold `float`: needed if method is "threshold". the threshold to use to detect outliers
    - no_assumptions `bool`: if True, the method will not assume that at least half of the dataset is not outliers, this may lead to a lot of false positives
    """
    factors = angle_based_outlier_factor(dataset, iterations)
    method = kwargs.get("method") or "elbow"
    if method == "elbow":
        no_assumptions = kwargs.get("no_assumptions") or False
        factors = sorted(factors, key=lambda x: x[1])
        # we assume that at least half of the dataset is not outliers
        elbow_index = elbow_method([np.log(factor) for i, factor in factors], len(factors) // 2 if not no_assumptions else None)
        if elbow_index is None:
            return []
        elbow_outliers = []
        for i, (index, _) in enumerate(factors):
            if i < elbow_index:
                elbow_outliers.append(index)
        return elbow_outliers
    elif method == "threshold":
        threshold = kwargs.get("threshold")
        outliers = [i for i, factor in factors if factor < threshold]
        return outliers
    else:
        raise ValueError(f"Invalid method {method}")

if __name__ == "__main__":
    from matplotlib import pyplot as plt
    from dmml.clustering.datasets import demo_dataset
    
    NUM_OUTLIERS = 10
    DATASET_SIZE = 500
    
    dataset = np.array(demo_dataset(DATASET_SIZE, "three_lines"))
    outliers = np.random.uniform(-0.5, 0.5, (NUM_OUTLIERS, 2))
    dataset = np.concatenate([dataset, outliers])
    
    outliers = angle_based_outlier_detection(dataset, 100)
    
    factors = angle_based_outlier_factor(dataset, 100)
    factors = sorted(factors, key=lambda x: x[1])
    elbow_index = elbow_method([np.log(factor) for i, factor in factors], len(factors) // 2)
    dvalues = [np.log(factors[i][1]) - np.log(factors[i - 1][1]) for i in range(1, len(factors))]
    mean = np.mean(dvalues)
    std = np.std(dvalues)
    
    plt.subplot(2, 1, 1)
    plt.plot(range(len(factors)), [np.log(factor) for i, factor in factors])
    if elbow_index is not None:
        plt.axvline(x=elbow_index, color="red")
    plt.plot(range(1, len(factors)), dvalues)
    plt.axhline(y=mean + 3 * std, color="red")
    plt.axhline(y=mean, color="red")
    plt.subplot(2, 1, 2)
    plt.scatter(dataset[:, 0], dataset[:, 1])
    for i in range(len(factors)):
        if i in outliers:
            plt.scatter(dataset[i, 0], dataset[i, 1], color="red", marker="o")
    
    plt.show()
    
    