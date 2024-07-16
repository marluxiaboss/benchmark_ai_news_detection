import numpy as np


def bootstrap_score(scores_list: list[int], n_bootstraps: int=1000, seed: int=42):
    np.random.seed(seed)
    
    means = []
    for _ in range(n_bootstraps):
        bootstrap_sample = np.random.choice(scores_list, len(scores_list), replace=True)
        winrate_A = np.mean(bootstrap_sample)
        means.append(winrate_A)
        
    lower_bound = np.percentile(means, 2.5)
    upper_bound = np.percentile(means, 97.5)
    
    return np.mean(means), lower_bound, upper_bound

