import numpy
from tensorflow.python.framework import random_seed
import os
from pathlib import Path
from tensorflow.python.platform import gfile
from PIL import Image
import re
import array
from tensorflow.contrib.learn.python.learn.datasets import base
from sklearn.feature_extraction import image as featureExtraction
import includes.dataset_helper as dsf
import ntpath
import math as m
from tensorflow.contrib.learn.python.learn.datasets.mnist import extract_images as extract_images_mnist
from tensorflow.contrib.learn.python.learn.datasets.mnist import extract_labels as extract_labels_mnist
import tensorflow as tf
import json
import math
import pickle


def path_leaf(path):
    head, tail = ntpath.split(path)
    return tail or ntpath.basename(head)


def patches2dat(image_path, patch_size, stride):
    image_name = path_leaf(image_path)
    patches_name = dsf.file_cut_extension(image_name) + "_" + str(patch_size) + ".dat"
    mask_name = dsf.file_cut_extension(image_path) + "_mask.bmp"
    labels_name = dsf.file_cut_extension(image_name) + "_" + str(patch_size) + ".labels"
    patches, img_shape = tf_slice_pic_to_patches(image_path, patch_size, stride)
    with open("data/sparc/" + patches_name, 'wb') as foutPatches:
        # numRowsPatches, numColsPatches, totalPatches, patchSizeX, patchsizeY, channels
        dsf.write_meta(foutPatches,
                       [patches.shape[1], patches.shape[2], patches.shape[1] * patches.shape[2], patch_size,
                        patch_size, img_shape[2]])
        patches.flatten().tofile(foutPatches)
    if os.path.isfile(mask_name):
        mask_patches, mask_shape = tf_slice_pic_to_patches(mask_name, patch_size, stride)
        mask_patches = mask_patches.astype(numpy.uint8).reshape(-1, patch_size ** 2)
        mask_patches = numpy.sum(mask_patches, axis=-1)
        cut_limit = numpy.vectorize(lambda e: (1 if e > 35 * 255 else 0))
        mask_patches = cut_limit(mask_patches)
        mask_patches = mask_patches.astype(numpy.uint8)
        with open("data/sparc/" + labels_name, 'wb') as foutPatches:
            # numRowsPatches, numColsPatches, totalPatches, patchSizeX, patchsizeY, channels
            classes, per_class = numpy.unique(mask_patches, return_counts=True)
            dsf.write_meta(foutPatches, [mask_patches.shape[0], len(classes)])
            mask_patches.tofile(foutPatches)
        print(per_class)
    else:
        mask_patches = None
    print(
        "image '{}', patch size - {}x{}, patches - {}x{}, labels - {}. Done.".format(image_path,
                                                                                     patch_size,
                                                                                     patch_size,
                                                                                     patches.shape[1],
                                                                                     patches.shape[2],
                                                                                     mask_patches.shape[0]
                                                                                     if mask_patches is not None
                                                                                            else "None"))


def tf_slice_pic_to_patches(image_path, patch_size, stride):
    with Image.open(image_path) as img:
        img_numpy = numpy.asarray(img)
    img_place_holder = tf.placeholder(numpy.uint8, shape=(1, img_numpy.shape[0],
                                                        img_numpy.shape[1],
                                                        img_numpy.shape[2] if len(img_numpy.shape) == 3 else 1),
                                                        name="BigImagePH")
    clust_quarters = tf.extract_image_patches(images=img_place_holder,
                                             ksizes=[1, patch_size, patch_size, 1],
                                             strides=[1, stride, stride, 1],
                                             rates=[1, 1, 1, 1],
                                             padding='VALID', name="SplitIntoPatches")
    session = tf.Session()
    size = (1, img_numpy.shape[0], img_numpy.shape[1], img_numpy.shape[2] if len(img_numpy.shape) == 3 else 1)
    return session.run(clust_quarters, {img_place_holder: img_numpy.reshape(size)}), img_numpy.shape


def convert_mnist(image_path, labels_path):
    with gfile.Open(image_path, 'rb') as f:
        images = extract_images_mnist(f)
    with open("data/mnist/mnist.dat", 'wb') as foutPatches:
        dsf.write_meta(foutPatches, [0, 0, images.shape[0], images.shape[1], images.shape[2], images.shape[3]])
        images.flatten().tofile(foutPatches)
    print("image '{}', patch size - {}x{}, patches - {}. Done.".format(image_path, images.shape[1], images.shape[2],
                                                                       images.shape[0]))
    with gfile.Open(labels_path, 'rb') as f:
        labels = extract_labels_mnist(f)
    with open("data/mnist/mnist.labels", 'wb') as foutLabels:
        counts = numpy.unique(labels)
        dsf.write_meta(foutLabels, [labels.shape[0], len(counts)])
        labels.tofile(foutLabels)


def convert_cifar(image_path):
    def rgb2gray(rgb):
        return numpy.dot(rgb[..., :3], [0.299, 0.587, 0.114]).astype(numpy.uint8)

    with open("data/cifar/cifar.dat", 'wb') as foutPatches:
        dsf.write_meta(foutPatches, [0, 0, 50000, 32, 32, 1])

    with open("data/cifar/cifar.labels", 'wb') as foutLabels:
        dsf.write_meta(foutLabels, [50000, 10])
    for i in range(1, 6):
        with open(image_path + str(i), 'rb') as f:
            dict = pickle.load(f, encoding='bytes')
            shape = int(math.sqrt(dict[b'data'].shape[1] / 3))
        with open("data/cifar/cifar.dat", 'ab') as foutPatches:
            images = rgb2gray(dict[b'data'].reshape((dict[b'data'].shape[0], shape, shape, 3), order='F'))
            images.flatten().tofile(foutPatches)
        with open("data/cifar/cifar.labels", 'ab') as foutLabels:
            labels = numpy.asarray(dict[b'labels']).astype(numpy.uint8)
            labels.tofile(foutLabels)
    print("image '{}', patch size - {}x{}, patches - {}. Done.".format(image_path, 32, 32, 50000))


def export_random_patches(path, x=64, y=64, num=64):
    images = []  # categories (dir names)
    with open("data/sparc/images.dat", 'wb') as foutImages:
        with open("data/sparc/labels.dat", 'wb') as foutLabels:
            for dir in subdirs(path):
                for entry in os.scandir(path + dir):
                    cat = re.match(r'^[0-9]+ ([A-Z]{1}[a-zA-Z0-9 ]+?) (Slide|[0-9\.\%]+).+\.tif$', entry.name)
                    if not cat:
                        print(entry.name + " -- bad category")
                        continue
                    print("processing " + entry.name)
                    cat = cat.group(1);
                    cat not in images and images.append(cat)
                    value = images.index(cat)

                    with Image.open(path + dir + "/" + entry.name) as img:
                        img_numpy = numpy.asarray(img)
                    patches = featureExtraction.extract_patches_2d(img_numpy, (y, x), num)
                    # numpy.savetxt(foutImages, patches.flatten())
                    patches.flatten().tofile(foutImages)
                    foutLabels.write(bytes([value] * num))

    with open("data/sparc/classes.csv", 'w') as fp:
        fp.write(",".join(images))
    print("extraction is completed")
    input_images = numpy.fromfile("data/sparc/images.dat", dtype=numpy.uint8).reshape((-1, x * y * 3))
    input_labels = numpy.fromfile("data/sparc/labels.dat", dtype=numpy.uint8)
    perm = numpy.arange(input_labels.size)
    numpy.random.shuffle(perm)
    input_images = input_images[perm]
    input_labels = input_labels[perm]
    with open("data/sparc/images.dat", 'wb') as foutImages:
        dsf._write32(foutImages, 2051)
        dsf._write32(foutImages, input_labels.size)  # numimages
        dsf._write32(foutImages, y)  # rows
        dsf._write32(foutImages, x)  # cols
        dsf._write32(foutImages, 3)  # channels
        input_images.tofile(foutImages)

    with open("data/sparc/labels.dat", 'wb') as foutLabels:
        dsf._write32(foutLabels, 2051)
        dsf._write32(foutLabels, input_labels.size)  # numimages
        dsf._write32(foutLabels, len(images))  # rows
        input_labels.tofile(foutLabels)


def convert_bardot(img_path, classes_limit=None, min_pics_in_class=None, max_pics_in_class=None,
                   exclude_file_path=None, resize=None, exclude_array=None):
    excluded_classes = None
    if exclude_file_path is not None:
        exclude_file_path = dsf.file_cut_extension(exclude_file_path)
        excluded_classes = json.load(open(exclude_file_path + '.json'))['classes']
        print("converting classes with filter of {} classes".format(len(excluded_classes)))
    images = []
    img_size = None
    if img_path[:-1] != '/':
        img_path += '/'
    with open("data/bardot/bardot.dat.temp", 'wb') as foutPatches:
        with open("data/bardot/bardot.labels.temp", 'wb') as foutLabels:
            with open("data/bardot/bardot.meta", 'w') as foutMeta:
                image_count = 0
                for dir in subdirs(img_path):
                    cat = re.match(r'^[A-Z]+ [0-9]+', dir)
                    if not cat:
                        print(dir + " -- bad category")
                        continue
                    cat = cat.group();
                    if exclude_file_path is not None and cat in excluded_classes \
                            or (exclude_array is not None and cat in exclude_array):
                        print(dir + " -- excluded category")
                        continue
                    cat_dir = os.listdir(img_path + dir)
                    # class has enough images?
                    if min_pics_in_class is not None and len(cat_dir) < min_pics_in_class:
                        continue;
                    # exceed limit of classes?
                    if cat not in images:
                        if classes_limit is None or classes_limit > len(images):
                            images.append(cat)
                        else:
                            continue
                    # exceeds number of pics?
                    if max_pics_in_class is not None:
                        cat_dir = cat_dir[:max_pics_in_class]

                    for entry in cat_dir:
                        if (not re.match(r'.+\.png', entry)):
                            continue
                        value = images.index(cat)
                        foutLabels.write(value.to_bytes(1, byteorder='little'))
                        print("{}\t{}".format(entry, cat), file=foutMeta)
                        img = Image.open(img_path + dir + "/" + entry)
                        if resize is not None:
                            img = img.resize((resize, resize))
                        img_numpy = numpy.asarray(img)
                        img_size = img_numpy.shape
                        img_numpy.flatten().tofile(foutPatches)
                        image_count += 1
    if img_size is None:
        raise Exception("No eligible pictures found!")
    if exclude_file_path is None:
        out_file_base = "data/bardot/bardot"
    else:
        out_file_base = exclude_file_path + "-aug"

    if resize is not None:
        out_file_base += "-{}x{}".format(resize, resize)
    dsf.write_cfg("{}-{}.json".format(out_file_base, len(images)), {"classes": images})
    with open("{}-{}.dat".format(out_file_base, len(images)), 'wb') as foutPatches:
        with open("data/bardot/bardot.dat.temp", 'rb') as finPatches:
            dsf.write_meta(foutPatches, [0, 0, image_count, img_size[0], img_size[1], img_size[2] if 2 in img_size else 1])
            bytes = finPatches.read(100000)
            while bytes:
                foutPatches.write(bytes)
                bytes = finPatches.read(100000)
    os.remove("data/bardot/bardot.dat.temp")

    with open("{}-{}.labels".format(out_file_base, len(images)), 'wb') as foutLabels:
        with open("data/bardot/bardot.labels.temp", 'rb') as finLabels:
            dsf.write_meta(foutLabels, [image_count, len(images)])
            bytes = finLabels.read(100000)
            while bytes:
                foutLabels.write(bytes)
                bytes = finLabels.read(100000)
    os.remove("data/bardot/bardot.labels.temp")
    os.rename("data/bardot/bardot.meta", "{}-{}.meta".format(out_file_base, len(images)))
    print("{} images of {}x{}x{} from {} classes converted".format(image_count, img_size[0], img_size[1],
                                                                   img_size[2] if 2 in img_size else 1, len(images)))


def subdirs(path):
    """Yield directory names not starting with '.' under given path."""
    for entry in os.scandir(path):
        if not entry.name.startswith('.') and entry.is_dir():
            yield entry.name


# TFRECORD
def _int64_feature(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))


def _bytes_feature(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))


def convert_to_tf(data_set, name):
    """Converts a dataset to tfrecords."""
    images = data_set.images
    labels = data_set.labels
    # num_examples = data_set.num_examples
    num_examples = images.shape[0]

    if images.shape[0] != labels.shape[0]:
        raise ValueError('Images size %d does not match label size %d.' %
                         (images.shape[0], labels.shape[0]))
    rows = images.shape[1]
    cols = images.shape[2]
    depth = images.shape[3]

    file_name = name + '.tfrecords'
    print('Writing', file_name)
    with tf.python_io.TFRecordWriter(file_name) as writer:
        for index in range(num_examples):
            image_raw = images[index].tostring()
            example = tf.train.Example(
                features=tf.train.Features(
                    feature={
                        'height': _int64_feature(rows),
                        'width': _int64_feature(cols),
                        'depth': _int64_feature(depth),
                        'label': _int64_feature(int(labels[index])),
                        'image_raw': _bytes_feature(image_raw)
                    }))
            writer.write(example.SerializeToString())


def create_permutation(path):
    input_images, input_images_params = dsf.extract_images(path)
    input_labels, input_labels_params = dsf.extract_labels(path, one_hot=False)
    input_names = dsf.extract_img_names(path)
    params = {**input_images_params, **input_labels_params}

    perm_name = dsf.file_cut_extension(path) + '.npy'
    perm0 = numpy.arange(params['numImages'])
    numpy.random.shuffle(perm0)
    numpy.save(perm_name, perm0)
    print("{} permutation saved".format(len(perm0)))





if __name__ == '__main__':
    # create_permutation("data/bardot/bardot-5-aug-5.dat")
    patches2dat("data/sparc/10352_cut.tif", patch_size=8, stride=2)
    # convert_mnist("data/mnist/train-images-idx3-ubyte.gz", "data/mnist/train-labels-idx1-ubyte.gz")
    # convert_bardot("data/bardot/training_set_new", classesLimit=5, minPicsInClass=310, maxPicsInClass=310)
    # convert_bardot("data/bardot/training_set_new", classesLimit=5, minPicsInClass=150, maxPicsInClass=150,
    #                excludeFilePath="data/bardot/bardot-5")
    # convert_cifar("data/cifar/data_batch_")