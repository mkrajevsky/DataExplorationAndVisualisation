# DataExplorationAndVisualisation
Project Overview: Clustering Algorithms Evaluation
The goal of my experiment is to evaluate the overall quality of selected clustering algorithms on a wide range of benchmark datasets.

### Algorithms 
following clustering algorithms will be evaluated: 
- **K-means**

        sklearn.cluster.KMeans(n_clusters)

- **Gaussian mixture model**

        sklearn.mixture.GaussianMixture(n_components)

- **Genie** with parameter $g\in \{0.1,0.3,0.5,0.7,0.9\}$

        genieclust.Genie(n_clusters, gini_threshold = g)

- **Agglomerative Hierarchical Clustering** (with single, average, complete, and Ward linkage)

        sklearn.cluster.AgglomerativeClustering(n_clusters, linkage = linkage)

- **Spectral clustering**

        sklearn.cluster.SpectralClustering(n_clusters)

- **DBSCAN is considered separetely

        sklearn.cluster.DBSCAN(eps)


## Datasets

To evaluate the quality of the clustering algorithms, I will utilize the Benchmark Suite for Clustering Algorithms, available at: clustering-benchmarks.gagolewski.com. This suite consists of benchmark batteries frequently used in the literature.

In this project, I will use all datasets from the following collections:

sipu (16 data sets)

uci (8 data sets)

fcps (9 data sets)

graves (10 data sets)


### Methodology
For evaluation of clustering algorithms following  external cluster measures will be used:

- Adjusted Rank Index (ARI)

    $$ARI = (RI - Expected_{RI}) / (\max(RI) - Expected_{RI})$$

    The Rand Index (RI) computes a similarity measure between two clusterings by considering all pairs of samples and counting pairs that are assigned in the same or different clusters in the predicted and true clusterings.https://scikit-learn.org/stable/modules/generated/sklearn.metrics.adjusted_rand_score.html

            sklearn.metrics.adjusted_rand_score



- Normalised Clustering Accuracy (NCA)


    NCA is the averaged percentage of correctly classified points in each cluster above the perfectly uniform label distribution. Note that the matching between the cluster labels is performed automatically by finding the best permutation $\sigma$ 

            clustbench.get_score


- Normalised Mutual Information (NMI)

    NMI is a normalization of the Mutual Information (MI) score to scale the results between 0 (no mutual information) and 1 (perfect correlation). In this function, mutual information is normalized by some generalized mean of H(labels_true) and H(labels_pred), defined by the

            sklearn.metrics.normalized_mutual_info_score


As well as  internal cluster validity measures :
- Silhouette

    The silhouette value is a measure of how similar an object is to its own cluster (cohesion) compared to other clusters (separation). The silhouette value ranges from −1 to +1, where a high value indicates that the object is well matched to its own cluster and poorly matched to neighboring clusters

        genieclust.cluster_validity.silhuette_index

- Caliński-Harabasz

    The score is defined as ratio of the sum of between-cluster dispersion and of within-cluster dispersion.

        genieclust.cluster_validity.calinski_harabasz_index

- 

