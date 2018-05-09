import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt, numpy as np
import includes.dataset_helper as dsf
from sklearn import (manifold, datasets, decomposition, ensemble,
                     discriminant_analysis, random_projection)


def visualise2d(data, labels, output_filename, cluster_type=None):
    '''Scatter-plot to file'''
    num_classes = len(np.unique(labels))
    if num_classes < 15:
        return visualise2d_markers(data, labels, output_filename, cluster_type=cluster_type)
    plt.ioff()
    plt.clf()
    for curClass in range(num_classes):
        sc = plt.scatter(data[:,0], data[:,1], c=labels,
                         cmap= plt.cm.Set1 if num_classes<10 else plt.cm.nipy_spectral, marker=".")
    cb = plt.colorbar(sc)
    plt.savefig(output_filename)


def visualise2d_markers(data, labels, output_filename, cluster_type=None):
    '''Scatter-plot with different markers'''
    data = np.asarray(data)
    markers = [".",",","o","v","^","<",">","1","2","3","4","8","s","p","P","*","h","H","+","x","X","D","d","|","_"]
    records = []
    classes = np.unique(labels)

    color = plt.cm.gist_ncar(np.linspace(0,1,len(classes)))

    for n in range(len(data)):
        records.append([data[n][0], data[n][1],
                        markers[labels[n]], color[labels[n]]])
    fig, ax = plt.subplots()
    s_arr = []
    for m in classes:
        points = np.take(data, [i for i, x in enumerate(labels) if x == m], axis=0)
        s = ax.scatter(points[:,0], points[:,1], color=color[m],
                       marker=markers[m])
        s_arr.append(s)
    plt.legend(tuple(s_arr), tuple(["{} #{}"
                                   .format("Class" if cluster_type=="ground" else "Cluster", x+1) for x in classes]),
                                     scatterpoints=1,loc='best', ncol=1, fontsize=8, framealpha=1.0,fancybox=False)
    plt.xlabel('x')
    plt.ylabel('y')
    plt.savefig(output_filename)


def visualize_tsne_2d(dataset_path, data, labels, cluster_type = None, num_clusters = None, output_dir = None):
    X_tsne = _reduce_tsne(data, 2)
    output_filename = output_dir + 'plot-' + cluster_type + '-' + str(num_clusters) + '_' + dsf.file_name_hash(dataset_path) + '.png'
    visualise2d(X_tsne, labels, output_filename, cluster_type=cluster_type)


def _reduce_tsne(X, dimensions):
    print("Computing t-SNE embedding")
    tsne = manifold.TSNE(n_components=dimensions, init='pca', random_state=0)
    return tsne.fit_transform(X)


def visualize_nnd(input_array, dataset_path, output_dir):
    '''visualize nearest neighbors distance'''
    output_filename = output_dir + 'NND-' + '_' + dsf.file_name_hash(dataset_path) + '.png'
    plt.clf()
    plt.hist(input_array, bins=20)
    plt.grid(True)
    plt.savefig(output_filename)
