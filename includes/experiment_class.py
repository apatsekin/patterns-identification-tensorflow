from abc import abstractmethod

import tensorflow as tf
from tensorflow.contrib.tensorboard.plugins import projector
import includes.dataset_helper as dsf
import numpy
import includes.debug as dbg
import os
import math
from operator import mul
from functools import reduce


class Experiment:
    """
    Core class for all feature learning models
    """
    def __init__(self, **params):
        # set user params
        self.params = params
        # define default user params for core class
        self._set_default_params()
        # setting missing params from default params to user params
        for param in self._default_params:
            if param not in self.params:
                self.params[param] = self._default_params[param]

        # set properties for core class
        self._define_properties()

        self._save_config(self.params['run_config'][1], self.params)
        self._load_dataset(self.params['dataset_path'])

        # set default placeholders
        with tf.variable_scope("Placeholders"):
            self._define_placeholders()

        # tensorboard
        self._log_dir = "logs/train/{}/{}".format(self.params['tag_name'], self.params['run_config'][0])
        self.projectorConfig = projector.ProjectorConfig()

        print("init completed. tag: {}, classes: {}, elements: {}, size: {}x{}x{}"
              .format(self.params['tag_name'],
                      'none' if 'numLabels' not in self.dataset.params else self.dataset.params['numClasses'],
                      self.dataset.params['numImages'],
                      self.dataset.params['imageHeight'],
                      self.dataset.params['imageWidth'],
                      self.dataset.params['imageChannels']))

    def finish_up(self):
        if 'train_writer' in self.tboard:
            # if self.dataset.test.fileNames is not None:
            print("writing projector config to log dir")
            projector.visualize_embeddings(self.tboard['train_writer'], self.projectorConfig)
            self.tboard['train_writer'].close()
        print("destroing tensorflow graph...")

        if hasattr(self, 'session'):
            self.session.close()

    def _init_tf_vars(self):
        '''Initialize only uninitialized vars'''
        global_vars = tf.global_variables()
        is_not_initialized = self.session.run([tf.is_variable_initialized(var) for var in global_vars])
        not_initialized_vars = [v for (v, f) in zip(global_vars, is_not_initialized) if not f]
        # print[str(i.name) for i in not_initialized_vars]  # only for testing
        if len(not_initialized_vars):
            self.session.run(tf.variables_initializer(not_initialized_vars))

    def _set_default_params(self):
        self._default_params = {'test_size': 310, 'training_batch_size': 200, 'k_fold': 0}

    def _define_placeholders(self):
        with tf.name_scope('io_tensors'):
            self.input_tensor = tf.placeholder(tf.float32, [None, self.dataset.params['imageHeight'],
                                                            self.dataset.params['imageWidth'],
                                                            self.dataset.params['imageChannels']], name="input_tensor")
            self.tboard['input_images'] = tf.summary.image('input', self.input_tensor, 6)

    def _define_properties(self):
        self.tboard = {}
        # graph output objects
        self.graph = {}
        self.session = tf.Session()

    def _save_config(self, path, dict, append=False):
        dsf.write_cfg(path + 'config.json', dict, append)

    def _load_dataset(self, path):
        self.dataset = dsf.read_data_sets(path, one_hot=True, reshape=False, shuffle=True,
                                          validation_size=self.params['test_size'],
                                          k_fold=self.params['k_fold'])

    def _setup_embedding_vector(self, index, test_vectors):
        with tf.name_scope('EmbeddingVectors'):
            embeded_shape = [len(self.dataset.test.images), self._get_tensor_flat_size(self.graph['features_layer'])]
            self.embeddingInput = tf.placeholder(tf.float32, embeded_shape, name="EmbeddingVector" + index)
            embeded_vector = tf.get_variable("embeded_vector_" + index, shape=embeded_shape)
            if 'embeded_vectors' not in self.graph:
                self.graph['embeded_vectors'] = []
            self.graph['embeded_vectors'].append(embeded_vector)
            self.graph['embeded_vector_assign'] = tf.assign(embeded_vector, self.embeddingInput)
            self._init_tf_vars()
            self._create_meta_for_embedding(index, embeded_vector)
            self.session.run([self.graph['embeded_vector_assign']], {self.embeddingInput: test_vectors})

    def _generate_sprites(self):
        if (self.dataset.params['imageWidth'] < 64):
            down_sample_rate = 1
        else:
            down_sample_rate = int(self.dataset.params['imageWidth'] / 64)
        sprites = numpy.copy(self.dataset.test.images)
        if down_sample_rate > 1:
            sprites = - (sprites[:, ::down_sample_rate, ::down_sample_rate, :] * 255) + 255
        else:
            sprites = sprites * 255
        sprites = sprites.astype(numpy.uint8)
        output_size = 64 if down_sample_rate > 1 else self.dataset.params['imageWidth']
        return self._reconstruct_from_patches(sprites, output_size), output_size


    def _reconstruct_from_patches(self, patches, patch_size):
        map_size = int(math.ceil(math.sqrt(len(self.dataset.test.images))))
        out_img = numpy.full((map_size * patch_size, map_size * patch_size, patches.shape[-1]), 255, dtype=numpy.uint8)
        i = 0
        for row in range(map_size):
            for col in range(map_size):
                out_img[row * patches.shape[1]:(row + 1) * patches.shape[1],
                        col * patches.shape[2]:(col + 1) * patches.shape[2]] = patches[i]
                i += 1
                if i >= patches.shape[0]:
                    return out_img
        return out_img

    def _get_tensor_flat_size(self, input_tensor):
        if not isinstance(input_tensor, list):
            shape = input_tensor.get_shape().as_list()
        else:
            shape = input_tensor.copy()
        shape.pop(0)
        return reduce(mul, shape, 1)

    def _create_meta_for_embedding(self, index, embeded_vector):
        metadata_filename = os.path.join(self._log_dir, "metadata_{}.tsv".format(index))
        if self.dataset.test.fileNames is not None or self.dataset.test.labels is not None:
            if self.dataset.test.fileNames is not None:
                metadata = numpy.copy(self.dataset.test.fileNames)
                metadata = numpy.insert(metadata, 0, "{}\t{}".format("file_name", "Class"))
            elif self.dataset.test.labels is not None:
                #copy, convert onehot to dense and convert int to string
                metadata = numpy.char.mod('%d',dsf.one_hot_to_dense(numpy.copy(self.dataset.test.labels)))
                #metadata = numpy.insert(metadata, 0, "Class")
            numpy.savetxt(metadata_filename, metadata, fmt='%s',
                          delimiter=os.linesep)
        embedding = self.projectorConfig.embeddings.add()
        embedding.tensor_name = embeded_vector.name
        if os.path.isfile(metadata_filename):
            embedding.metadata_path = dsf.extract_file_name(metadata_filename)
        # sprites are generated from dataset.test images anyway, since
        sprite, sprite_size = self._generate_sprites()
        sprite_input = tf.placeholder(tf.uint8, sprite.shape, name="Sprite" + index)
        sprite_to_png = tf.image.encode_png(sprite_input)
        sprite_imag_path = os.path.join(self._log_dir, "sprite{}.png".format(index))
        sprite_write = tf.write_file(tf.constant(sprite_imag_path), sprite_to_png)
        embedding.sprite.image_path = "sprite{}.png".format(index)
        embedding.sprite.single_image_dim.extend([sprite_size, sprite_size])
        self.session.run([sprite_write], {sprite_input: sprite})

    def _save_model_builder(self):
        builder = tf.saved_model.builder.SavedModelBuilder(os.path.join(self.params['run_config'][1], "modelBuilder"))
        builder.add_meta_graph_and_variables(self.session, [tf.saved_model.tag_constants.SERVING])
        builder.save()

    def _rotate_image(self, input_image):
        if (self.params['rotate_images']):
            # rotate image on random degree
            twopi = tf.constant(2 * math.pi)
            rotate_degree = tf.random_uniform([1], minval=0, maxval=twopi, dtype=tf.float32)[0]
            input_image = tf.contrib.image.rotate(self.input_tensor, rotate_degree)
        return input_image

    def extract_vectors_from_dataset(self, out_filename, type="test"):
        print("creating vectors ({})...".format(type))
        filename_suffix = ""
        if type == "test":
            dts = self.dataset.test
        elif type == "all":
            dts = self.dataset.all
            filename_suffix += "_all"
        else:
            raise ValueError("Dataset type is wrong ('type' param)")

        vct_file_name = dsf.file_cut_extension(out_filename) + filename_suffix + '.vct'

        labels_file_name = dsf.file_cut_extension(out_filename) + filename_suffix + '.labels'
        if 'features_layer' not in self.graph:
            get_size_vector = self._extract_features(dts.images[:1])
            size_vector = numpy.prod(get_size_vector.shape)
        else:
            size_vector = numpy.prod(self.graph['features_layer'].get_shape().as_list()[1:])

        with open(vct_file_name, 'wb') as foutVectors:
            dsf.write_meta(foutVectors,
                           [0, 0, len(dts.images), size_vector])

            if dts.labels is not None:
                fout_labels = open(labels_file_name, 'wb')
                dsf.write_meta(fout_labels, [len(dts.labels), self.dataset.params['numClasses']])
            test_vectors = []
            # data for embedings need to be stored in saved TF model in a Variable
            # variable will be named with dataset file_name suffix

            for test_input_batch, test_labels_batch in dts.split_to_batches(self.params['training_batch_size']):
                if dts.labels is not None:
                    if len(test_labels_batch.shape) != 1:
                        test_labels_batch = dsf.one_hot_to_dense(test_labels_batch)
                converted_batch = self._extract_features(test_input_batch)
                test_vectors.append(converted_batch)
                converted_batch.flatten().tofile(foutVectors)
                if dts.labels is not None:
                    test_labels_batch.tofile(fout_labels)
            if dts.labels is not None:
                fout_labels.close()

            test_vectors = numpy.concatenate(test_vectors, axis=0)
            if type == "test":
                self._setup_embedding_vector(
                    index=dsf.extract_file_name_cut_extension(dsf.file_cut_extension(out_filename)).replace("-", ""),
                    vectors=test_vectors)
                #self.session.run([self.graph['embeded_vector_assign']], {self.embeddingInput: test_vectors})
            # self.saver = tf.train.Saver(max_to_keep=2)
            if 'embeded_vectors' in self.graph:
                tf.train.Saver(self.graph['embeded_vectors'], max_to_keep=2).save(self.session,
                                                                              os.path.join(self._log_dir,
                                                                                           "modelEmbeded.ckpt"),
                                                                              100000)

        print("vectors saved to files!")

    @abstractmethod
    def _hook_post_tf_init(self):
        pass

    @abstractmethod
    def _train_batch(self, iteration):
        pass

    @abstractmethod
    def _build_network(self):
        pass

    @abstractmethod
    def _setup_embedding_vector(self, index, vectors):
        pass

    @abstractmethod
    def _extract_features(self, test_input_batch):
        pass

    def train(self):
        self._build_network()

        # create object and dir for tensorboard
        self.tboard['merged_summary'] = tf.summary.merge_all()
        self.tboard['train_writer'] = tf.summary.FileWriter(self._log_dir, self.session.graph)
        self.saver = tf.train.Saver(max_to_keep=1)
        # if self.dataset.test.fileNames is not None:
        #    self.createMeta()
        self._init_tf_vars()
        self._hook_post_tf_init()
        # if self.dataset.test.fileNames is not None:
        #   self.session.run([spriteWrite], {spriteInput: sprite})

        for i in range(self.params['training_steps']):
            self._train_batch(i)


