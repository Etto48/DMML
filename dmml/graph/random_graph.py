import numpy as np
import matplotlib.pyplot as plt
import networkx as nx

def random_graph(nodes: int, average_degree: int) -> np.ndarray:

    A = np.ndarray((nodes, nodes))

    for i in range(nodes):
        for j in range(nodes):
            if i != j and np.random.random() < average_degree/nodes:
                A[i][j] = 1
            else:
                A[i][j] = 0
        if np.sum(A[i]) == 0:
            j = np.random.randint(0, nodes)
            j = j if j != i else (j + 1) % nodes
            A[i][j] = 1
    return A


def plot_graph(A: np.ndarray):
    G = nx.from_numpy_array(A)
    nx.draw(G, with_labels=True)
    plt.show()
    
if __name__ == '__main__':
    A = random_graph(20, 2)
    plot_graph(A)