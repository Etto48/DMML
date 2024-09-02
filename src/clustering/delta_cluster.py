import numpy as np

def residual(IJ: np.ndarray) -> float:
    row_avg = np.mean(IJ, axis=1)
    col_avg = np.mean(IJ, axis=0)
    total_avg = np.mean(IJ)
    residual = 0
    for i in range(IJ.shape[0]):
        for j in range(IJ.shape[1]):
            residual += (IJ[i, j] - row_avg[i] - col_avg[j] + total_avg)**2
    residual /= IJ.shape[0] * IJ.shape[1]
    return residual

def residual_row(IJ: np.ndarray, i: int, row_avg: np.ndarray, col_avg: np.ndarray, total_avg: np.ndarray) -> float:
    residual = 0
    for j in range(IJ.shape[1]):
        residual += (IJ[i, j] - row_avg[i] - col_avg[j] + total_avg)**2
    residual /= IJ.shape[1]
    return residual

def residual_col(IJ: np.ndarray, j: int, row_avg: np.ndarray, col_avg: np.ndarray, total_avg: np.ndarray) -> float:
    residual = 0
    for i in range(IJ.shape[0]):
        residual += (IJ[i, j] - row_avg[i] - col_avg[j] + total_avg)**2
    residual /= IJ.shape[0]
    return residual

def extract_submatrix(A: np.ndarray, rows: set[int], cols: set[int]) -> np.ndarray:
    return A[np.ix_(list(rows), list(cols))]

if __name__ == "__main__":
    const = np.array([
        [60, 60, 60, 60, 60],
        [60, 60, 60, 60, 60],
        [60, 60, 60, 60, 60],
        [60, 60, 60, 60, 60],
        [60, 60, 60, 60, 60],
    ])
    
    const_on_rows = np.array([
        [10, 10, 10, 10, 10],
        [20, 20, 20, 20, 20],
        [50, 50, 50, 50, 50],
        [0, 0, 0, 0, 0],
    ])
    
    coherent = np.array([
        [10, 50, 30, 70, 20],
        [20, 60, 40, 80, 30],
        [50, 90, 70, 110, 60],
        [0, 40, 20, 60, 10],
    ])
    
    coherent_on_rows = np.array([
        [10, 50, 30, 70, 20],
        [20, 100, 50, 1000, 30],
        [50, 100, 90, 120, 80],
        [0, 80, 20, 100, 10]
    ])
    
    print(residual(const))
    print(residual(const_on_rows))
    print(residual(coherent))
    print(residual(coherent_on_rows))