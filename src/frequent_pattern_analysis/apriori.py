import itertools as it
from src.frequent_pattern_analysis.preprocess import preprocess_csv
from tqdm import tqdm

class FrequentPattern:
    pattern: tuple[int] = ()
    support: int = 0
    def __init__(self, pattern: tuple[int], support: int):
        self.pattern = pattern
        self.support = support
    
    def __str__(self):
        return f"{self.pattern}: {self.support}"

    def __repr__(self):
        return str(self)
    
    def __eq__(self, other: 'FrequentPattern'):
        return set(self.pattern) == set(other.pattern)

class AssociationRule:
    antededent: tuple[int] = ()
    consequent: tuple[int] = ()
    confidence: float = 0.0
    def __init__(self, antededent: tuple[int], consequent: tuple[int], confidence: float):
        self.antededent = antededent
        self.consequent = consequent
        self.confidence = confidence
    
    def print_with_labels(self, labels: dict[int,str]):
        ant = [labels[x] for x in self.antededent]
        con = [labels[x] for x in self.consequent]
        print(f"{ant} -> {con} {self.confidence*100:.2f}%")
    
    def __str__(self):
        return f"{self.antededent} -> {self.consequent} {self.confidence*100:.2f}%"

    def __repr__(self):
        return str(self)
    
    def __eq__(self, other: 'AssociationRule'):
        return set(self.antededent) == set(other.antededent) and set(self.consequent) == set(other.consequent)        
        
def find_rules(dataset: list[list[int]], min_support: float, min_confidence: float) -> list[AssociationRule]:
    max_id = max([max(x) for x in dataset])
    
    items = [0 for _ in range(0, max_id + 1)]

    for d in tqdm(dataset, "Checking 1-Patterns"):
        for d_i in d:
            items[d_i] += 1

    frequent_1 = []
    for i in range(len(items)):
        if items[i] >= len(dataset) * min_support:
            frequent_1.append(FrequentPattern((i,), items[i]))

    patterns = [x for x in it.combinations([p.pattern[0] for p in frequent_1], 2)]
    patterns_counts = [0 for _ in range(len(patterns))]

    for d in tqdm(dataset, "Checking 2-Patterns"):
        for i, itemset in enumerate(patterns):
            if all(x in d for x in itemset):
                patterns_counts[i] += 1

    frequent_2 = []
    for i in range(len(patterns)):
        if patterns_counts[i] >= len(dataset) * min_support:
            frequent_2.append(FrequentPattern(patterns[i], patterns_counts[i]))

    frequent = [frequent_1, frequent_2]

    for k in it.count(2):
        patterns = []
        
        frequent_k_minus_1: list[FrequentPattern] = frequent[k - 1]
        for f1 in frequent_k_minus_1:
            for f2 in frequent_k_minus_1:
                if f1 != f2:
                    differences = 0
                    for item in f1.pattern:
                        if item not in f2.pattern:
                            differences += 1
                            if differences > 1:
                                break
                    if differences == 1:
                        new_pattern = f1.pattern + tuple([x for x in f2.pattern if x not in f1.pattern])
                        new_pattern = tuple(sorted(new_pattern))
                        if new_pattern not in patterns:
                            patterns.append(new_pattern)
        
        patterns_counts = [0 for _ in range(len(patterns))]
        for d in tqdm(dataset, f"Checking {k + 1}-Patterns"):
            for i, itemset in enumerate(patterns):
                if all(x in d for x in itemset):
                    patterns_counts[i] += 1
        
        frequent_k = []
        for i in range(len(patterns)):
            if patterns_counts[i] >= len(dataset) * min_support:
                frequent_k.append(FrequentPattern(patterns[i], patterns_counts[i]))
        
        if len(frequent_k) == 0:
            break
        
        frequent.append(frequent_k)
        
    rules = []

    frequent_lookup = {}
    for f in frequent:
        for fp in f:
            frequent_lookup[fp.pattern] = fp

    frequent_list = [f for f in frequent_lookup.values()]

    for fp in tqdm(frequent_list, "Generating rules"):
        for i in range(1, len(fp.pattern)):
            for ant in it.combinations(fp.pattern, i):
                con = tuple([x for x in fp.pattern if x not in ant])
                ant_count = 0
                con_count = 0
                ant_count = frequent_lookup[ant].support
                con_count = frequent_lookup[fp.pattern].support
                
                confidence = con_count / ant_count
                if confidence >= min_confidence:
                    rules.append(AssociationRule(ant, con, confidence))
                        
    unique_rules = []
    for r in rules:
        if r not in unique_rules:
            unique_rules.append(r)
    rules: list[AssociationRule] = unique_rules

    rules.sort(key=lambda x: x.confidence, reverse=True)
    return rules

if __name__ == "__main__":
    dataset = []

    with open(f'{__file__}/../../../datasets/retail.dat', 'r') as f:
        for line in f:
            dataset.append(list(map(int, line.split())))
        
    MIN_SUPPORT = 0.01
    MIN_CONFIDENCE = 0.5

    rules = find_rules(dataset, MIN_SUPPORT, MIN_CONFIDENCE)

    for r in rules:
        print(r)