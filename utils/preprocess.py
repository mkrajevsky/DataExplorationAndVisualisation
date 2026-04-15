import clustbench
"""
Preprocess datasets with the same function to ensure consistency
genieclust.preprocess_data is used to preprocess the data,
but it does not handle labels or n_clusters,
so  a wrapper function is created to extract those as well
"""

def preprocess_dataset(dataset, run_preprocessing=True):
    nclusters= dataset.n_clusters[0] #label0 are the main labels, we will ignore the rest
    data =  clustbench.preprocess_data(dataset.data) if run_preprocessing else dataset.data
    mask = dataset.labels[0]!=0 #find noise points
    X = data[mask] # remove noise points
    labels = dataset.labels[0]
    labels = labels[mask] # remove noise points from labels as well
    return {'X': X, 'labels': labels, 'n_clusters': nclusters}

def preprocess_datasets(datasets):
    preprocessed = {}
    for (battery, dataset) in datasets.items():
        preprocessed[battery] = preprocess_dataset(dataset)
    return preprocessed
