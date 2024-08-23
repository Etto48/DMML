import numpy as np

def x_squared(entry: np.ndarray, average: np.ndarray) -> float:
    return np.sum((entry - average)**2 / average)

def x_squared_outlied_detection(dataset: np.ndarray) -> list[int]:
    average = np.mean(dataset, axis=-1)
    outliers = []
    
    average_x_squared = np.mean([x_squared(dataset[:,i], average) for i in range(dataset.shape[-1])])
    std_x_squared = np.std([x_squared(dataset[:,i], average) for i in range(dataset.shape[-1])])

    for i in range(dataset.shape[-1]):
        x_squared_i = x_squared(dataset[:,i], average)
        if x_squared_i > average_x_squared + 3 * std_x_squared:
            outliers.append(i)
    return outliers

if __name__ == "__main__":
    NDIM = 5
    dataset = np.random.randn(NDIM, 500)
    index = np.random.randint(0, dataset.shape[-1])
    dataset[:,index] += 10
    print(f"Injected outlier at index {index}")
    
    outliers = x_squared_outlied_detection(dataset)
    print(f"Detected outliers: {outliers}")