import sys
sys.path.append(f'{__file__}/../../../')
import matplotlib.pyplot as plt
import numpy as np

def random_dataset(size: int, cluster_spread: float, min_coords: list[float], max_coords: list[float]):
    dataset = []
    dims = len(min_coords)
    assert dims == len(max_coords)
    num_seeds = np.random.randint(2, 5)
    for _ in range(size):
        while True:
            if len(dataset) < 2 * num_seeds:
                other = None
            elif len(dataset) >= 2 * num_seeds:
                other_index = np.random.choice(len(dataset))
                other = dataset[other_index]
            delta = [np.random.uniform(-(max_coords[j]-min_coords[j]),(max_coords[j]-min_coords[j]))*cluster_spread for j in range(dims)]
            if other is None:
                point = [np.random.uniform(min_coords[j], max_coords[j]) for j in range(dims)]
                dataset.append(point)
                break
            else:
                new_position = [other[j] + delta[j] for j in range(dims)]
                dataset.append(new_position)
                break
    return dataset
    
def dist(p: list[float], q: list[float]) -> float:
    return sum([(p[i] - q[i])**2 for i in range(len(p))])**0.5
    
def demo_dataset(size: int, kind: str, with_labels=False) -> list[list[float]] | tuple[list[list[float]], list[int]]:
    dataset = []
    labels = []
    match kind:
        case "uniform":
            dataset = np.random.uniform(-0.5, 0.5, (size, 2)).tolist()
            labels = np.zeros(size, dtype=int)
        case "normal":
            dataset = np.random.normal(0, 0.1, (size, 2)).tolist()
            labels = np.zeros(size, dtype=int)
        case "concentric":
            r_1 = 0.2
            r_2 = 0.5
            std_r = 0.03
            dataset = []
            labels = []
            for _ in range(size):
                angle = np.random.rand() * 2 * np.pi
                is_smaller = np.random.rand() < 0.3
                labels.append(0 if is_smaller else 1)
                radius = np.random.normal(r_1 if is_smaller else r_2, std_r)
                dataset.append([radius * np.cos(angle), radius * np.sin(angle)])
        case "two_moons":
            r = 0.5
            std_r = 0.03
            center_1 = [-0.25, 0.125]
            center_2 = [0.25, -0.125]
            dataset = []
            labels = []
            for _ in range(size):
                angle = np.random.rand() * np.pi
                is_upper = np.random.rand() < 0.5
                labels.append(0 if is_upper else 1)
                center = center_1 if is_upper else center_2
                angle += np.pi if is_upper else 0
                radius = np.random.normal(r, std_r)
                dataset.append([center[0] + radius * np.cos(angle), center[1] + radius * np.sin(angle)])
        case "two_circles":
            r = 0.2
            center_1 = [-0.25, 0]
            center_2 = [0.25, 0]
            dataset = []
            labels = []
            for _ in range(size):
                radius = np.random.rand() * r
                angle = np.random.rand() * 2 * np.pi
                is_left = np.random.rand() < 0.5
                labels.append(0 if is_left else 1)
                center = center_1 if is_left else center_2
                dataset.append([center[0] + radius * np.cos(angle), center[1] + radius * np.sin(angle)])
        case "three_lines":
            length = 0.5
            std_width = 0.02
            center_1 = [0.3, -0.35]
            center_2 = [0.1, -0.4]
            center_3 = [0.15, 0.1]
            angle = np.pi * 3 / 4
            dataset = []
            labels = []
            for _ in range(size):
                is_left = np.random.rand() < 1/3
                is_middle = 1/3 <= np.random.rand() < 2/3
                center = center_1 if is_left else center_2 if is_middle else center_3
                labels.append(0 if is_left else 1 if is_middle else 2)
                l = np.random.rand() * length
                width = np.random.normal(0, std_width)
                x = center[0] + l * np.cos(angle) + width * np.cos(angle + np.pi / 2)
                y = center[1] + l * np.sin(angle) + width * np.sin(angle + np.pi / 2)
                dataset.append([x, y])
        case "two_spirals":
            std_r = 0.01
            r = 0.5
            skip = 0.1
            dataset = []
            labels = []
            for _ in range(size):
                t = np.random.rand() * (1 - skip) + skip
                first = np.random.rand() < 0.5
                labels.append(0 if first else 1)
                radius = t * r + np.random.normal(0, std_r)
                angle = 4 * np.pi * t + (np.pi if first else 0)
                x = radius * np.cos(angle)
                y = radius * np.sin(angle)
                dataset.append([x, y])
        case "smiley":
            face_r = 0.5
            std_face = 0.02
            eye_r = 0.1
            eye_1 = [-0.2, 0.2]
            eye_2 = [0.2, 0.2]
            mouth_r = 0.3
            std_mouth = 0.02
            dataset = []
            labels = []
            for _ in range(size):
                part = np.random.choice([0, 1, 2, 3], p=[0.4, 0.15, 0.15, 0.3])
                if part == 0: # face
                    angle = np.random.rand() * 2 * np.pi
                    radius = np.random.normal(face_r, std_face)
                    dataset.append([radius * np.cos(angle), radius * np.sin(angle)])
                elif part == 1 or part == 2: # eyes
                    angle = np.random.rand() * 2 * np.pi
                    radius = np.random.rand() * eye_r
                    if part == 1:
                        x = eye_1[0] + radius * np.cos(angle)
                        y = eye_1[1] + radius * np.sin(angle)
                    else:
                        x = eye_2[0] + radius * np.cos(angle)
                        y = eye_2[1] + radius * np.sin(angle)
                    dataset.append([x, y])
                elif part == 3: # mouth
                    angle = np.random.rand() * np.pi + np.pi
                    radius = np.random.normal(mouth_r, std_mouth)
                    x = radius * np.cos(angle)
                    y = radius * np.sin(angle)
                    dataset.append([x, y])
                labels.append(part)
        case "clusters":
            cluster_count = 5
            std = 0.1
            dataset = []
            labels = []
            centers = [[0.3, -0.4], [-0.3, 0.3], [0.1, -0.1], [-0.4, -0.3], [0.4, 0.3]]
            for _ in range(size):    
                cluster = np.random.choice(cluster_count, p=[0.5, 0.2, 0.05, 0.1, 0.15])
                center = centers[cluster]
                dx = np.random.normal(0, std)
                dy = np.random.normal(0, std)
                x = center[0] + dx
                y = center[1] + dy
                dataset.append([x, y])
                labels.append(cluster)
        case _:
            raise ValueError(f"Unknown kind {kind}")
    return (dataset, labels) if with_labels else dataset

if __name__ == "__main__":
    #dataset = random_dataset(50, 0.2, 0.1, [0, 0], [10, 10])
    from dmml.clustering.plotting import plot_clusters
    
    plt.subplot(3, 3, 1)
    dataset, labels = demo_dataset(500, "uniform", with_labels=True)
    clusters = [set([i for i, l in enumerate(labels) if l == label]) for label in set(labels)]
    plot_clusters(dataset, clusters)
    plt.subplot(3, 3, 2)
    dataset, labels = demo_dataset(500, "normal", with_labels=True)
    clusters = [set([i for i, l in enumerate(labels) if l == label]) for label in set(labels)]
    plot_clusters(dataset, clusters)
    plt.subplot(3, 3, 3)
    dataset, labels = demo_dataset(500, "concentric", with_labels=True)
    clusters = [set([i for i, l in enumerate(labels) if l == label]) for label in set(labels)]
    plot_clusters(dataset, clusters)
    plt.subplot(3, 3, 4)
    dataset, labels = demo_dataset(500, "two_moons", with_labels=True)
    clusters = [set([i for i, l in enumerate(labels) if l == label]) for label in set(labels)]
    plot_clusters(dataset, clusters)
    plt.subplot(3, 3, 5)
    dataset, labels = demo_dataset(500, "two_circles", with_labels=True)
    clusters = [set([i for i, l in enumerate(labels) if l == label]) for label in set(labels)]
    plot_clusters(dataset, clusters)
    plt.subplot(3, 3, 6)
    dataset, labels = demo_dataset(500, "three_lines", with_labels=True)
    clusters = [set([i for i, l in enumerate(labels) if l == label]) for label in set(labels)]
    plot_clusters(dataset, clusters)
    plt.subplot(3, 3, 7)
    dataset, labels = demo_dataset(500, "two_spirals", with_labels=True)
    clusters = [set([i for i, l in enumerate(labels) if l == label]) for label in set(labels)]
    plot_clusters(dataset, clusters)
    plt.subplot(3, 3, 8)
    dataset, labels = demo_dataset(500, "smiley", with_labels=True)
    clusters = [set([i for i, l in enumerate(labels) if l == label]) for label in set(labels)]
    plot_clusters(dataset, clusters)
    plt.subplot(3, 3, 9)
    dataset, labels = demo_dataset(500, "clusters", with_labels=True)
    clusters = [set([i for i, l in enumerate(labels) if l == label]) for label in set(labels)]
    plot_clusters(dataset, clusters)
    
    plt.show()
    