import clustbench
"""
Preprocess datasets with the same function to ensure consistency
genieclust.preprocess_data is used to preprocess the data,
but it does not handle labels or n_clusters,
so we create a wrapper function to extract those as well
"""

def preprocess_dataset(dataset):
    nclusters= dataset.n_clusters[0]
    data =  clustbench.preprocess_data(dataset.data)
    mask = dataset.labels[0]!=0
    X = data[mask]
    labels = dataset.labels[0]
    return {'X': X, 'labels': labels, 'n_clusters': nclusters}

def preprocess_datasets(datasets):
    preprocessed = {}
    for (battery, dataset) in datasets.items():
        preprocessed[(battery)] = preprocess_dataset(dataset)
    return preprocessed
