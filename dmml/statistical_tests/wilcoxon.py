import scipy
import numpy as np
import scipy.stats

def wilcoxon_test(distribution_a: list[float], distribution_b: list[float]) -> float:
    """
    Perform the Wilcoxon signed-rank test on two samples of scores.
    Returns the p-value of the test (the probability that the two samples
    were drawn from the same distribution).
    """
    assert len(distribution_a) == len(distribution_b), "Both distributions must have the same length"
    
    differences = [b - a for a, b in zip(distribution_a, distribution_b)]
    sorted_diff = sorted(differences, key=abs)
    ranks = range(1, len(differences) + 1)
    W_plus = sum([r for r, d in zip(ranks, sorted_diff) if d >= 0])
    W_minus = sum([r for r, d in zip(ranks, sorted_diff) if d < 0])
    W_min = min(W_plus, W_minus)
    n = len(differences)
    mu = n * (n + 1) / 4
    sigma = (n * (n + 1) * (2 * n + 1) / 24)**0.5
        
    z = (W_min - mu) / sigma
    p = 1-scipy.stats.norm.cdf(abs(z))
    return p*2

if __name__ == "__main__":
    distribution_1 = np.random.normal(0, 1, 100).tolist()
    distribution_2 = np.random.normal(0, 1, 100).tolist()
    p = wilcoxon_test(distribution_1, distribution_2)
    print(f"p-value: {p}")