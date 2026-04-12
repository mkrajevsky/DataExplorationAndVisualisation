from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score

from genieclust.cluster_validity import calinski_harabasz_index, generalised_dunn_index, silhouette_index
from genieclust.compare_partitions import compare_partitions
def evaluate_clustering(X, y_true, y_pred):
    metrics = {
        'ARI': adjusted_rand_score(y_true, y_pred),
        'NMI': normalized_mutual_info_score(y_true, y_pred),
        'NCA' : compare_partitions(X, y_true, y_pred)['nca'],
        'Silhouette': silhouette_index(X, y_pred),
        'Calinski-Harabasz': calinski_harabasz_index(X, y_pred),
        'Generalised Dunn': generalised_dunn_index(X, y_pred)
    }
    
    return metrics
