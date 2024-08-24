import sys
sys.path.append(f'{__file__}/../../../')
from dmml.graph.random_graph import random_graph, plot_graph
import numpy as np

def neighbours(A: np.ndarray, i: int) -> list[int]:
    return [j for j in range(A.shape[0]) if A[i][j] == 1]

def simrank(A: np.ndarray, c: float, iterations: int) -> np.ndarray:
    nodes = A.shape[0]
    S = np.identity(nodes)
    for _ in range(iterations):
        S_new = np.zeros((nodes, nodes))
        for i in range(nodes):
            neighbours_i = neighbours(A, i)
            for j in range(nodes):
                neighbours_j = neighbours(A, j)
                if i == j:
                    S_new[i][j] = 1
                else:
                    S_new[i][j] = c/(len(neighbours_i) * len(neighbours_j)) * sum([S[x][y] for x in neighbours_i for y in neighbours_j])
        S = S_new
    return S

if __name__ == '__main__':
    A = random_graph(10, 2)
    S = simrank(A, 0.5, 100)
    for i in range(10):
        for j in range(10):
            print(f"{i} {j}: {S[i][j]:.2f}")
    plot_graph(A)
    