import clustbench
import tqdm
def load_datasets(path = 'data/datasets/clustering-data-v1-1.1.0'):
    clustbench.get_battery_names(path=path)
    batteries = ["sipu", "uci", "fcps", "graves"]
    exclude_list = ["birch1", "birch2", "worms_2", "worms_64"] # too large datasets to load
    datasets = {}
    for battery in batteries:
        for dataset in tqdm.tqdm(clustbench.get_dataset_names(battery, path=path), desc=f"Loading datasets from {battery}"):
            if dataset not in exclude_list:
                clustbench.load_dataset(battery, dataset, path=path)
                datasets[(battery, dataset)] = clustbench.load_dataset(battery, dataset, path=path)
            
    return datasets

if __name__ == "__main__":
    datasets = load_datasets()  
    print(datasets.keys())