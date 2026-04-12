
"""
Each clustbench.dataset object has number of features most important from our standpoint include
- .data - raw data
- .labels list of arrays of labels,  we wil always utilize only the first of them (labels0)
- .n_clusters number of clusters 

"""
import tqdm
import clustbench

def cluster_dataset_(X, nclusters, labels, algorithm, alg_name):
    if alg_name == "GaussianMixture":
        alg = algorithm(n_components=nclusters)
    else:
        alg = algorithm(n_clusters=nclusters)
    fit = alg.fit_predict(X)
    return {'Y_pred': fit + 1}

def cluster_datasets_(datasets, algorithm, alg_name):
    results = {}
    for (battery, dataset) in  tqdm.tqdm(datasets.items(), desc=f"Clustering datasets with {alg_name}"):
        results[battery] = cluster_dataset_(X=dataset['X'], nclusters=dataset['n_clusters'], labels=dataset['labels'], algorithm=algorithm, alg_name=alg_name)
    return results