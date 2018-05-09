import numpy
from sklearn.cluster import KMeans
from sklearn.cluster import AffinityPropagation
from sklearn.cluster import MeanShift, estimate_bandwidth
from sklearn.cluster import DBSCAN
from sklearn import metrics
from sklearn.metrics import confusion_matrix
from sklearn.neighbors import NearestNeighbors
from sklearn import mixture
import includes.dataset_helper as dsf
import includes.plots as plots
import os
from pathlib import Path
import math
import skimage.measure
import hdbscan


def find_nn_distance(input_array):
    print("calculating nearest neighbors distance for {} vectors...".format(len(input_array)))
    neigh = NearestNeighbors(n_neighbors=5, metric='euclidean')
    neigh.fit(input_array)
    dist, _ = neigh.kneighbors(input_array)
    return dist[:,1]


def make_nn_distance(input_array, output_dir, dataset_path):
    distance_array = find_nn_distance(input_array)
    plots.visualize_nnd(distance_array, dataset_path, output_dir)
    return numpy.percentile(distance_array, 95)


def kmeans_run(input_array, num_clusters):
    kmeans = KMeans(n_clusters = num_clusters).fit(input_array)
    return kmeans.labels_, num_clusters


def aff_prop_run(input_array):
    af = AffinityPropagation().fit(input_array)
    cluster_centers_indices = af.cluster_centers_indices_
    labels = af.labels_
    n_clusters_ = len(cluster_centers_indices)
    return labels, n_clusters_


def mean_shift_run(input_array):
    #bandwidth = estimate_bandwidth(input_array, n_samples=1000, n_jobs=-1)
    ms = MeanShift(n_jobs=-1)
    ms.fit(input_array)
    labels = ms.labels_
    #cluster_centers = ms.cluster_centers_
    labels_unique = numpy.unique(labels)
    n_clusters_ = len(labels_unique)
    return labels, n_clusters_


def db_scan_run(input_array, params):
    db = DBSCAN(eps = params['eps'] if 'eps' in params else 0.5).fit(input_array)
    labels = db.labels_
    n_clusters_ = len(set(labels)) - (1 if -1 in labels else 0)
    return labels, n_clusters_


def hdbscan_run(input_array, params):

    db = hdbscan.HDBSCAN(min_cluster_size=10)
    labels = db.fit_predict(input_array)
    n_clusters_ = len(set(labels)) - (1 if -1 in labels else 0)
    return labels, n_clusters_


def hierarch_clustering(input_array, params={'threshold':1.5}):
    import scipy.cluster.hierarchy as hcluster
    labels = hcluster.fclusterdata(input_array, params['threshold'], criterion="distance")
    return labels, len(set(labels))


def bayesian_gaussian_mixture(input_array, params):
    try:
        dpgmm = mixture.BayesianGaussianMixture(n_components=params['components'],
                                                covariance_type='full', init_params='random').fit(input_array)
        clusters = dpgmm.predict(input_array)
    except numpy.linalg.linalg.LinAlgError as err:
        print("Bayesian clustering failed due to LineArgError: {}".format(err))
        return None, None
    except ValueError as err:
        print("Bayesian ValueError: {}".format(err))
        return None, None
    unique_elements, labels = numpy.unique(clusters, return_inverse=True)
    return labels, len(unique_elements)


def convert_to_bool(clust_fit, num_clusters, dataset_path):
    if dataset_path is None:
        return clust_fit, num_clusters
    _, img_params = dsf.extract_images(dataset_path)
    if img_params['srcNumRows'] < 1 or img_params['srcNumCols'] < 1:
        return clust_fit, num_clusters
    # finding the cluster with nerves
    clustered_image = clust_fit.copy()
    # zero labels are ignored by function
    clustered_image += 1
    clustered_image = clustered_image.reshape(img_params['srcNumRows'], img_params['srcNumCols'])
    props = skimage.measure.regionprops(clustered_image)
    min_solidity = math.inf
    for prop in props:
        if min_solidity > prop.solidity:
            min_label = prop.label
            min_solidity = prop.solidity
    clustered_image = clustered_image.flatten()
    clustered_image[clustered_image != min_label] = 0
    clustered_image[clustered_image == min_label] = 1
    clusters_list = numpy.unique(clustered_image)
    return clustered_image, len(clusters_list)


def cluster_with_bench(input_array, type, labels, output_dir, dataset_path, num_clusters = None, params = None, dat_file=None):
    print("clustering {}, num clusters = {} ...".format(type, num_clusters))
    if type == 'kmeans':
        clust_fit, num_clusters = kmeans_run(input_array, num_clusters)
    elif type == 'affprop':
        clust_fit, num_clusters = aff_prop_run(input_array)
    elif type == 'hierarch':
        clust_fit, num_clusters = hierarch_clustering(input_array)
    elif type == 'dbscan':
        clust_fit, num_clusters = db_scan_run(input_array, params)
    elif type == 'hdbscan':
        clust_fit, num_clusters = hdbscan_run(input_array, params)
    elif type == 'bayesian':
        clust_fit, num_clusters = bayesian_gaussian_mixture(input_array, params)

    if num_clusters is not None and num_clusters > 0:
        clust_fit, bool_num_clusters = convert_to_bool(clust_fit, num_clusters, dat_file)
        clust_fit_bench, input_array_bench, labels_bench = clust_fit, input_array, labels

        if len(input_array) > 3000 and labels is not None:
            confusion_result = {'matrix':confusion_matrix(labels_bench, clust_fit_bench).astype(numpy.str).tolist(),
            'labelsStructure': list(numpy.unique(labels_bench,return_counts=True)[1].astype(numpy.str)),
            'clustersStructure':list(numpy.unique(clust_fit_bench, return_counts=True)[1].astype(numpy.str))}
            output_name = "{}confusion-{}-{}_{}.json".format(output_dir, type, num_clusters, dsf.file_name_hash(dataset_path))
            dsf.write_cfg(output_name, confusion_result)
            print("too many pictures ({}), reducing to 3000 for bench!".format(len(input_array_bench)))
        if len(input_array_bench) < 3000 and labels is not None:
            plots.visualize_tsne_2d(dataset_path, input_array_bench, clust_fit_bench, num_clusters=num_clusters,
                          output_dir=output_dir, cluster_type=type)
            bench_params = bench_clustering(clust_fit_bench, input_array_bench, labels_bench)
            bench_params['clustersNum'] = num_clusters
            bench_params['classesNum'] = len(numpy.unique(labels_bench))
            output_name = output_dir + 'benchmark-' + type + '-' + str(bench_params['classesNum']) + '-' + str(
                num_clusters) + '_' + dsf.file_name_hash(dataset_path) + '.json'
            dsf.write_cfg(output_name, bench_params)
            print("{} clustered with {} clusers".format( type, num_clusters))
        return clust_fit, num_clusters
    else:
        print(type + " end up with zero clusters or failed!")
        return None, None


def bench_clustering(clusteredData, input_data, labels):
    benchmarks = {}
    benchmarks['homogeneity'] = metrics.homogeneity_score(labels, clusteredData)
    benchmarks['completeness'] = metrics.completeness_score(labels, clusteredData)
    benchmarks['v-measure'] = metrics.v_measure_score(labels, clusteredData)
    benchmarks['ARI'] = metrics.adjusted_rand_score(labels, clusteredData)
    benchmarks['AMI'] = metrics.adjusted_mutual_info_score(labels,  clusteredData)
    benchmarks['silhouette'] = metrics.silhouette_score(input_data, clusteredData, metric='euclidean').item()
    return benchmarks


def clean_old_data(run_dir):
    print("deleting old data")
    for p in Path(run_dir).glob("benchmark-*"):
        p.unlink()
    for p in Path(run_dir).glob("plot-*"):
        p.unlink()


def clustering_main(out_filename, run_config, dat_file = None):
    run_tag, run_dir = run_config

    print("loading vectors...")

    all_vectors = out_filename.replace(".dat","_all.vct")
    if os.path.isfile(all_vectors):
        vectors_all, vectors_params_all = dsf.load_vectors(all_vectors, run_config=[run_tag, run_dir])
        labels_all, labels_params_all = dsf.extract_labels(all_vectors)
        for num_clusters in range(max(3, labels_params_all['numClasses'] if labels_all is not None else 3),
                                 (labels_params_all['numClasses'] + 4) if labels_all is not None else 6):
            kmeans_clusters, kmeans_num_clusters = cluster_with_bench(vectors_all, 'kmeans', labels_all, output_dir=run_dir,
                                                                      dataset_path=out_filename, num_clusters=num_clusters, dat_file=dat_file)
            dsf.add_clusters_layer(dat_file, kmeans_clusters, num_clusters, run_config=[run_tag, run_dir], cluster_type="kmeans", clust_original=kmeans_num_clusters)
        if labels_all is not None:
            dsf.add_clusters_layer(dat_file, labels_all, labels_params_all['numClasses'], run_config=[run_tag, run_dir], cluster_type="ground")

    vectors, vectors_params = dsf.load_vectors(out_filename, run_config=[run_tag, run_dir])
    labels, labels_params = dsf.extract_labels(out_filename)

    plots.visualize_tsne_2d(out_filename, vectors, labels, num_clusters=labels_params['numClasses'], output_dir=run_dir,
                          cluster_type="ground")
    eps = make_nn_distance(vectors, dataset_path=out_filename, output_dir=run_dir)
    cluster_with_bench(vectors, 'dbscan', labels, output_dir=run_dir, dataset_path=out_filename, params={'eps': eps})
    cluster_with_bench(vectors, 'hdbscan', labels, output_dir=run_dir, dataset_path=out_filename)
    #cluster_with_bench(vectors, 'bayesian', labels, output_dir=run_dir, dataset_path=out_filename, params={'components': 8})
    for num_clusters in range(max(2, labels_params['numClasses'] - 2), labels_params['numClasses'] + 4):
        cluster_with_bench(vectors, 'kmeans', labels, output_dir=run_dir, dataset_path=out_filename,
                           num_clusters=num_clusters)
