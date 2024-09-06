import matplotlib.pyplot as plt
import numpy as np

class Point:
    id = None
    position = []
    reachability_distance = np.inf
    core_distance = np.inf
    processed = False
    
    def __init__(self, position: list[float], id: str):
        self.position = position
        self.id = id
    
    def __str__(self):
        return f'Point {self.id} at {self.position}'

class PriorityQueue:
    def __init__(self):
        self.queue: list[str] = []
        
    def insert(self, id: str, key):
        insert_index = 0
        for i, p in enumerate(self.queue):
            if key(p) > key(id):
                insert_index = i
                break
            
        self.queue.insert(insert_index, id)
        
    def move_up(self, id: str, key):
        self.queue.remove(id)
        self.insert(id, key)
        
    def pop(self) -> str:
        return self.queue.pop(0)
    
    def empty(self) -> bool:
        return len(self.queue) == 0

def dist(p: Point, q: Point) -> float:
    return sum([(p.position[i] - q.position[i])**2 for i in range(len(p.position))])**0.5

def get_neighbors(DB: dict[str,Point], id: str, eps: float) -> list[str]:
    neighbors = []
    for q in DB.values():
        if id != q.id and dist(DB[id], q) < eps:
            neighbors.append(q.id)
    return neighbors

def core_distance(DB: dict[str,Point], id: str, eps: float, MinPts: int) -> float:
    neighbors = get_neighbors(DB, id, eps)
    if len(neighbors) < MinPts:
        return np.inf
    distances = [dist(DB[id], DB[q]) for q in neighbors]
    distances.sort()
    return distances[MinPts - 1]

def update(DB: dict[str,Point], neighbors: list[str], p: str, seeds: PriorityQueue, eps: float, MinPts: int):
    coredist = core_distance(DB, p, eps, MinPts)
    for o in neighbors:
        if not DB[o].processed:
            new_reach_dist = max(coredist, dist(DB[p],DB[o]))
            if DB[o].reachability_distance == np.inf: # o is not in Seeds
                DB[o].reachability_distance = new_reach_dist
                seeds.insert(o, lambda x: DB[x].reachability_distance)
            else: # o in Seeds, check for improvement
                if new_reach_dist < DB[o].reachability_distance:
                    DB[o].reachability_distance = new_reach_dist
                    seeds.move_up(o, lambda x: DB[x].reachability_distance)

def optics(DB: list[list[int]], eps, MinPts):
    DB = {str(i): Point(p, str(i)) for i, p in enumerate(DB)}
    ret = []
    
    for (id, p) in DB.items():
        p.reachability_distance = np.inf
    for (id,p) in DB.items():
        if p.processed:
            continue
        neighbors = get_neighbors(DB, id, eps)
        p.processed = True
        ret.append(id)
        if core_distance(DB, id, eps, MinPts) != np.inf:
            seeds = PriorityQueue()
            update(DB, neighbors, id, seeds, eps, MinPts)
            while not seeds.empty():
                q_id = seeds.pop()
                neighbors_prime = get_neighbors(DB, q_id, eps)
                DB[q_id].processed = True
                ret.append(q_id)
                if core_distance(DB, q_id, eps, MinPts) != np.inf:
                    update(DB, neighbors_prime, q_id, seeds, eps, MinPts)
    
    return DB, ret

if __name__ == "__main__":
    from dmml.clustering.datasets import random_dataset
    DB = random_dataset(100, 0.2, 0.1, [0, 0], [10, 10])
    DB, order = optics(DB, 40, 2)
    plt.subplot(2, 1, 1)
    plt.plot([min(DB[id].reachability_distance, 10) for id in order])
    for i, id in enumerate(order):
        plt.text(i, min(DB[id].reachability_distance, 10), id)
    plt.subplot(2, 1, 2)
    plt.scatter([p.position[0] for p in DB.values()], [p.position[1] for p in DB.values()])
    for i, id in enumerate(order):
        plt.text(DB[id].position[0], DB[id].position[1], id)
    plt.show()
