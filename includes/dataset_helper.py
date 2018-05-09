import numpy
import os
from includes.dataset_class import DataSet
from includes.dataset_lazy_class import DataSetLazy
import json
import collections
import colorsys

Datasets = collections.namedtuple('Datasets', ['train', 'test', 'all', 'params'])
ConvLayer = collections.namedtuple('ConvLayer', ['wSize', 'ch', 'strd', 'mxpl'])

def _read32(bytestream):
  dt = numpy.dtype(numpy.uint32).newbyteorder('>')
  return numpy.frombuffer(bytestream.read(4), dtype=dt)[0]


def _write32(bytestream, value):
  dt = numpy.dtype(numpy.uint32).newbyteorder('>')
  bytestream.write(numpy.array([value], dtype=dt))


def write_meta(fout, data):
    _write32(fout, 2051)
    for value in data:
        _write32(fout, value)


def checksymbol(bytestream):
    checksymbol = _read32(bytestream)
    if checksymbol != 2051:
        raise ValueError("Invalid file format")


def _extract_meta(bytestream, valueNames = []):
    checksymbol(bytestream)
    output = {}
    for val in valueNames:
        output[val] = _read32(bytestream)
    return output


def file_cut_extension(path):
    return os.path.splitext(path)[0]

def extract_file_name(path):
    _, file_name = os.path.split(path)
    return file_name

def extract_file_name_cut_extension(path):
    return os.path.splitext(extract_file_name(path))[0]


def replace_dir(input_path, newDir):
    _, file_name = os.path.split(input_path)
    newDir += '/' if newDir[-1]=='/' else ''
    return newDir + file_name


def extract_images(path):
    file_name = path
    with open(file_name, 'rb') as bytestream:
        params = _extract_meta(bytestream, ['srcNumRows', 'srcNumCols','numImages','imageHeight','imageWidth','imageChannels'])
        read_size = params['imageChannels'] * params['imageHeight'] * params['imageWidth'] *  params['numImages']
        data = numpy.fromfile(bytestream, dtype=numpy.uint8)
        assert read_size == data.shape[0]
        return data.reshape(params['numImages'], params['imageHeight'], params['imageWidth'], params['imageChannels']), params


#Same as extract_images, but for lazy loading. Need params and open handler rewinded to start position
def _load_images_handler(path):
    file_name = path
    bytestream = open(file_name, 'rb')
    params = _extract_meta(bytestream, ['srcNumRows', 'srcNumCols','numImages','imageHeight','imageWidth','imageChannels'])
    params['itemBytesSize'] = params['imageChannels'] * params['imageHeight'] * params['imageWidth']
    params['startPosition'] = bytestream.tell()
    return bytestream, params


def load_vectors(path, run_config):
    file_name = file_cut_extension(path) + '.vct'
    if run_config is not None:
        file_name = replace_dir(file_name, run_config[1])
    with open(file_name, 'rb') as bytestream:
        params = _extract_meta(bytestream, ['srcNumRows', 'srcNumCols','numVectors','sizeVector'])
        data = numpy.fromfile(bytestream, dtype = numpy.float32, count = params['numVectors'] * params['sizeVector'])
    return data.reshape(params['numVectors'], params['sizeVector']), params


def extract_labels(path, one_hot=False):
    file_name = file_cut_extension(path) + ".labels"
    if not os.path.isfile(file_name):
        return None, {}
    with open(file_name, 'rb') as bytestream:
        params = _extract_meta(bytestream, ['numLabels', 'numClasses'])
        buf = bytestream.read(params['numLabels'])
        labels = numpy.frombuffer(buf, dtype=numpy.uint8)
        _, labels_stat = numpy.unique(labels, return_counts=True)
        print("Labels count by each class:", labels_stat)
        if one_hot:
            labels = dense_to_one_hot(labels, params['numClasses'])
        return labels, params


def extract_img_names(path):
    file_name = file_cut_extension(path) + ".meta"
    if not os.path.isfile(file_name):
        return None
    return numpy.loadtxt(file_name, dtype=numpy.str, delimiter=os.linesep)


def _get_n_colors(N=5):
    HSV_tuples = [(x * 1.0 / N, 0.5, 0.5) for x in range(N)]
    out = []
    for rgb in HSV_tuples:
        rgb = map(lambda x: int(x * 255), colorsys.hsv_to_rgb(*rgb))
        out.append(list(rgb))
    return out


def add_clusters_layer(dataset_path, clustered, num_clusters, run_config = None, cluster_type = "none", stride=2, clust_original = None):
    import imageio
    if num_clusters == 0:
        print("{} end up with {} classes. Skipping image construction".format(cluster_type, num_clusters))
        return
    layer_colors = {2: [[255,0,0], [0,255,0]],
                   3: [[255, 0, 0], [0, 255, 0], [0, 0, 255]],
                   4: [[255, 0, 0], [0, 255, 0], [0, 0, 255], [255, 255, 0]],
                   5: [[255, 0, 0], [0, 255, 0], [0, 0, 255], [255, 255, 0], [255, 0, 255]],
                   6: [[255, 0, 0], [0, 255, 0], [0, 0, 255], [255, 255, 0], [255, 0, 255], [0, 255, 255]]}

    images, imgParams = extract_images(dataset_path)
    i = 0
    if num_clusters in layer_colors:
        color_set = layer_colors[num_clusters]
    else:
        color_set = _get_n_colors(num_clusters)

    new_image = numpy.empty_like(images)
    print("applying mask to patch...")
    for i in range(imgParams['numImages']):
        img_mask = numpy.full((imgParams['imageHeight'], imgParams['imageWidth'], imgParams['imageChannels']), color_set[clustered[i]])
        #img_mask = img_mask.reshape(imgParams['imageHeight'], imgParams['imageWidth'], 3)
        new_image[i] = 0.65 * images[i] + 0.35 * img_mask
    #stride <= patchtsize!!!
    output_image = numpy.empty([stride*imgParams['srcNumRows'] + (imgParams['imageHeight']-stride), stride*imgParams['srcNumCols'] + (imgParams['imageWidth'] - stride),imgParams['imageChannels'] ])
    i = 0
    print("rebuilding big image from patches...")
    for y_pos in range(imgParams['srcNumRows']):
        for x_pos in range(imgParams['srcNumCols']):
         output_image[y_pos*stride:y_pos*stride + imgParams['imageHeight'], x_pos*stride:x_pos*stride + imgParams['imageWidth']] = new_image[i]
         i += 1
    output_name_with_path = file_cut_extension(dataset_path) + '.png'
    if run_config is not None:
        output_name_with_path = "{}{}-restored-o{}-c{}_{}.png".format(run_config[1], cluster_type, clust_original, num_clusters, file_name_hash(dataset_path))
    print("saving image")
    imageio.imwrite(output_name_with_path, output_image)
    print("{} saved".format(output_name_with_path))


def merge_arrays(array):
    output = {}
    for item in array:
        for param in item:
            if param not in output:
                output[param] = []
            output[param].append(item[param])
    return output


def file_name_hash(name):
    return extract_file_name_cut_extension(name).replace('-','_')


def read_data_sets(path, one_hot=True, reshape=False, validation_size=1000, shuffle=False, k_fold=0):

    input_images, input_images_params = extract_images(path)
    input_labels, input_labels_params = extract_labels(path, one_hot=one_hot)
    input_names = extract_img_names(path)
    params = {**input_images_params, **input_labels_params}
    params["sourceFilePath"] = path
    all = DataSet(input_images, input_labels, input_names)

    if shuffle:
        if os.path.isfile(file_cut_extension(path) + '.npy'):
            print("fixed permutation is loaded for {}".format(path))
            perm0 = numpy.load(file_cut_extension(path) + '.npy')
        else:
            perm0 = numpy.arange(params['numImages'])
            numpy.random.shuffle(perm0)
        input_images = input_images[perm0]
        if input_labels is not None:
            input_labels = input_labels[perm0]
        if input_names is not None:
            input_names = input_names[perm0]

    if (validation_size*(k_fold+1) > params['numImages']):
        k_fold = 0
        print("k_fold param is set to 0 for {}, since test_size = {}, images = {}".format(path, validation_size, params['numImages']))
    testIndices = numpy.arange(k_fold*validation_size,(k_fold+1)*validation_size)
    trainIndices = numpy.delete(numpy.arange(params['numImages']), testIndices)

    train_images = input_images[trainIndices]
    train_labels = input_labels[trainIndices] if input_labels is not None else None
    train_names = input_names[trainIndices] if input_names is not None else None
    test_images = input_images[testIndices]
    test_labels = input_labels[testIndices] if input_labels is not None else None
    test_names = input_names[testIndices] if input_names is not None else None


    #options = dict(dtype=dtype, reshape=reshape, seed=seed)

    train = DataSet(train_images, train_labels, train_names, reshape=reshape)
    #validation = DataSet(test_images, test_labels, test_names, reshape=reshape)
    test = DataSet(test_images, test_labels, test_names, reshape=reshape)
    return  Datasets(train=train, test=test, all=all, params=params)

#TODO: update
def read_data_sets_lazy(path, one_hot=True, reshape=False, validation_size=1000, shuffle=False):
    images_handler, input_images_params = _load_images_handler(path)
    input_labels, input_labels_params = extract_labels(path, one_hot=one_hot)
    params = {**input_images_params, **input_labels_params}
    params["sourceFilePath"] = path
    training_size = params['numImages'] - validation_size
    indices = numpy.arange(params['numImages'])
    all = DataSetLazy(images_handler, indices, input_labels, dataSetParams= params)
    indices = indices[:] #just in case
    if shuffle:
        numpy.random.shuffle(indices)

    train = DataSetLazy(images_handler, indices[:training_size], input_labels, dataSetParams= params, reshape=reshape)
    test = DataSetLazy(images_handler, indices[training_size:], input_labels, dataSetParams= params, reshape=reshape)
    return Datasets(train=train, validation=None, test=test, all=all, params=params)


def dense_to_one_hot(labels_dense, num_classes):
    """Convert class labels from scalars to one-hot vectors."""
    num_labels = labels_dense.shape[0]
    index_offset = numpy.arange(num_labels) * num_classes
    labels_one_hot = numpy.zeros((num_labels, num_classes), dtype=numpy.uint8)
    labels_one_hot.flat[index_offset + labels_dense.ravel()] = 1
    return labels_one_hot


def one_hot_to_dense(array):
    return numpy.asarray([numpy.where(r==1)[0][0] for r in array], dtype=numpy.uint8)


def subdirs(path):
    """Yield directory names not starting with '.' under given path."""
    for entry in os.scandir(path):
        if not entry.name.startswith('.') and entry.is_dir():
            yield entry.name


def write_cfg(path, dict, append=False):
    with open(path, 'a' if append else 'w') as outfile:
        outfile.write(json.dumps(dict, sort_keys=True, indent=4) + "\n")