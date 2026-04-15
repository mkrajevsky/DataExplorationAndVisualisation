import numpy as np
from sklearn.cluster import DBSCAN
from sklearn.neighbors import NearestNeighbors


def estimate_eps(data, k=4):
    """
    Automatyczna estymacja parametru epsilon
    """
    neigh = NearestNeighbors(n_neighbors=k)
    nbrs = neigh.fit(data)
    distances, _ = nbrs.kneighbors(data)
    distances = np.sort(distances[:, k-1], axis=0)
    return np.percentile(distances, 90)

def run_dbscan_evaluation(ds_name, dataset):
    X = dataset.data
    y_true = dataset.labels[0]

    mask_not_natural_noise = (y_true != 0)
    X_clean = X[mask_not_natural_noise]
    y_true_clean = y_true[mask_not_natural_noise]
    eps_val = estimate_eps(X_clean, k=4)
    dbscan = DBSCAN(eps=eps_val, min_samples=4)
    y_dbscan = dbscan.fit_predict(X_clean)

    mask_not_db_noise = (y_dbscan != -1)
    if not np.any(mask_not_db_noise):
        return None

    X_final = X_clean[mask_not_db_noise]
    y_true_final = y_true_clean[mask_not_db_noise]
    y_dbscan_final = y_dbscan[mask_not_db_noise]
    _clusters = len(np.unique(y_dbscan_final))
    if _clusters < 2:
        print(f"Dataset {ds_name}: skipped DBSCAN found only {_clusters} cluster")
        return None

   
    result = {
        'X': X_final,
        'y_true': y_true_final,
        'y_dbscan': y_dbscan_final,
    }
    return result