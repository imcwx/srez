import srez_demo
import srez_input
import srez_model
import srez_train
import srez_test

import sys
import os.path
import random
import numpy as np
# import imageio
# from PIL import Image

import tensorflow as tf

FLAGS = tf.app.flags.FLAGS

# Configuration (alphabetically)
tf.app.flags.DEFINE_integer('batch_size', 32,
                            "Number of samples per batch.")
# 16

tf.app.flags.DEFINE_string('checkpoint_dir', 'checkpoint',
                           "Output folder where checkpoints are dumped.")

tf.app.flags.DEFINE_integer('checkpoint_period', 20000,
                            "Number of batches in between checkpoints")
# 10000

tf.app.flags.DEFINE_string('dataset', 'data',
                           "Path to the dataset directory.")
# dataset

tf.app.flags.DEFINE_float('epsilon', 1e-8,
                          "Fuzz term to avoid numerical instability")

tf.app.flags.DEFINE_string('run', 'demo',
                           "Which operation to run [demo|train|test].")

tf.app.flags.DEFINE_float('gene_l1_factor', .90,
                          "Multiplier for generator L1 loss term")
# 0.90

tf.app.flags.DEFINE_float('learning_beta1', 0.9,
                          "Beta1 parameter used for AdamOptimizer")
# 0.5
# 0.9

tf.app.flags.DEFINE_float('learning_rate_start', 0.00020,
                          "Starting learning rate used for AdamOptimizer")

tf.app.flags.DEFINE_integer('learning_rate_half_life', 2500,
                            "Number of batches until learning rate is halved")
# 5000
# 1000

tf.app.flags.DEFINE_bool('log_device_placement', False,
                         "Log the device where variables are placed.")

tf.app.flags.DEFINE_integer('sample_size', 128,
                            "Image sample size in pixels. Range [64,128]")
# 64

tf.app.flags.DEFINE_integer('summary_period', 500,
                            "Number of batches between summary data dumps")
# 200

tf.app.flags.DEFINE_integer('random_seed', 0,
                            "Seed used to initialize rng.")

tf.app.flags.DEFINE_integer('test_vectors', 25000,
                            """Number of features to use for testing""")
# 16

tf.app.flags.DEFINE_string('train_dir', 'train',
                           "Output folder where training logs are dumped.")

tf.app.flags.DEFINE_integer('train_time', 240,
                            "Time in minutes to train the model")
# 20

# I Added, Not Alphabetically.
tf.app.flags.DEFINE_string('test_dir', 'test',
                           "Test folder where images are unseen test set.")

tf.app.flags.DEFINE_string('predict_dir', 'predict',
                           "Output folder for unseen test set.")

tf.app.flags.DEFINE_boolean('allow_gpu_growth', True,
                            "Set whether to allow GPU growth.")

tf.app.flags.DEFINE_integer('test_size', 32,
                            "Test pixel size in pixels")

tf.app.flags.DEFINE_integer('crop_size', 32,
                            "Image crop size in pixels")

tf.app.flags.DEFINE_integer("learning_rate_reduction", 0.90,
                            "The fraction of reduction in learning rate.")
# 0.5
# 0.91


def prepare_dirs(delete_train_dir=False):
    # Create checkpoint dir (do not delete anything)
    if not tf.gfile.Exists(FLAGS.checkpoint_dir):
        tf.gfile.MakeDirs(FLAGS.checkpoint_dir)

    # Cleanup train dir
    if delete_train_dir:
        if tf.gfile.Exists(FLAGS.train_dir):
            tf.gfile.DeleteRecursively(FLAGS.train_dir)
        tf.gfile.MakeDirs(FLAGS.train_dir)

    # Return names of training files
    if not tf.gfile.Exists(FLAGS.dataset) or \
            not tf.gfile.IsDirectory(FLAGS.dataset):
        raise FileNotFoundError("Could not find folder `%s'" % (FLAGS.dataset,))

    filenames = tf.gfile.ListDirectory(FLAGS.dataset)
    filenames = sorted(filenames)
    random.shuffle(filenames)
    filenames = [os.path.join(FLAGS.dataset, f) for f in filenames]

    return filenames


def prepare_test_dir():
    # Check test dir Exist
    if not tf.gfile.Exists(FLAGS.test_dir) or \
            not tf.gfile.IsDirectory(FLAGS.test_dir):
        raise FileNotFoundError("Could not find folder `%s'" % (FLAGS.test_dir,))

    # Check predict dir Exist
    if not tf.gfile.Exists(FLAGS.predict_dir) or \
            not tf.gfile.IsDirectory(FLAGS.predict_dir):
        raise FileNotFoundError("Could not find folder `%s'" % (FLAGS.predict_dir,))

    filenames = tf.gfile.ListDirectory(FLAGS.test_dir)
    filenames = sorted(filenames)
    random.shuffle(filenames)
    filenames = [os.path.join(FLAGS.test_dir, f) for f in filenames]

    return filenames


def prepare_test16_dir():
    # Check test dir Exist
    if not tf.gfile.Exists(FLAGS.test_dir16) or \
            not tf.gfile.IsDirectory(FLAGS.test_dir16):
        raise FileNotFoundError("Could not find folder `%s'" % (FLAGS.test_dir16,))

    # Check predict dir Exist
    if not tf.gfile.Exists(FLAGS.predict_dir16) or \
            not tf.gfile.IsDirectory(FLAGS.predict_dir16):
        raise FileNotFoundError("Could not find folder `%s'" % (FLAGS.predict_dir16,))

    filenames = tf.gfile.ListDirectory(FLAGS.test_dir16)
    filenames = sorted(filenames)
    random.shuffle(filenames)
    filenames = [os.path.join(FLAGS.test_dir16, f) for f in filenames]

    return filenames


def setup_tensorflow():
    # Create session
    # config = tf.ConfigProto(log_device_placement=FLAGS.log_device_placement, device_count={'GPU': 2})
    config = tf.ConfigProto(log_device_placement=FLAGS.log_device_placement)
    config.gpu_options.allow_growth = FLAGS.allow_gpu_growth
    config.gpu_options.per_process_gpu_memory_fraction = 0.80
    config.gpu_options.allocator_type = 'BFC'
    sess = tf.Session(config=config)
    # tf.device('/gpu:1')

    # Initialize rng with a deterministic seed
    with sess.graph.as_default():
        tf.set_random_seed(FLAGS.random_seed)

    random.seed(FLAGS.random_seed)
    np.random.seed(FLAGS.random_seed)

    summary_writer = tf.summary.FileWriter(FLAGS.train_dir, sess.graph)

    return sess, summary_writer


def _test(onefilename=False):
    # Load checkpoint
    if not tf.gfile.IsDirectory(FLAGS.checkpoint_dir):
        raise FileNotFoundError("Could not find folder `%s'" % (FLAGS.checkpoint_dir,))

    # Prepare directories
    if onefilename:
        filenames = [onefilename]
    else:
        # Load test set
        if not tf.gfile.IsDirectory(FLAGS.test_dir):
            raise FileNotFoundError("Could not find folder `%s'" % (FLAGS.test_dir,))
        filenames = prepare_test_dir()

    # Setup global tensorflow state
    sess, summary_writer = setup_tensorflow()

    # Setup async input queues
    test_features, test_labels = srez_input.test_inputs(sess, filenames)

    # Create and initialize model
    [gene_minput, gene_moutput,
     gene_output, gene_var_list,
     disc_real_output, disc_fake_output, disc_var_list] = \
        srez_model.create_model(sess, test_features, test_labels)

    # Restore variables from checkpoint
    saver = tf.train.Saver()
    filename = 'checkpoint_new.txt'
    filename = os.path.join(FLAGS.checkpoint_dir, filename)

    saver.restore(sess, filename)

    test_data = TrainData(locals())
    td = test_data
    test_feature, test_label = sess.run([test_features, test_labels])
    feed_dict = {gene_minput: test_feature}
    gene_output = sess.run(gene_moutput, feed_dict=feed_dict)

    if onefilename:
        srez_test.predict_one(test_data, gene_output)
    else:
        srez_test.predict(test_data, test_feature, test_label, gene_output)


def _test16(onefilename=False):
    # Load checkpoint
    if not tf.gfile.IsDirectory(FLAGS.checkpoint_dir):
        raise FileNotFoundError("Could not find folder `%s'" % (FLAGS.checkpoint_dir,))

    # Load test set
    if not tf.gfile.IsDirectory(FLAGS.test_dir):
        raise FileNotFoundError("Could not find folder `%s'" % (FLAGS.test_dir,))

    # Setup global tensorflow state
    sess, summary_writer = setup_tensorflow()

    # Prepare directories
    if os.path.isfile(onefilename):
        filenames = [onefilename]
    elif os.path.isdir(onefilename):
        filenames = [os.path.join(onefilename, f) for f in os.listdir(onefilename) if
                     os.path.isfile(os.path.join(onefilename, f))]

    # im = Image.open(onefilename)
    # size = im.size

    # Setup async input queues
    test_features, test_labels = srez_input.test_inputs(sess, filenames)

    # Create and initialize model
    [gene_minput, gene_moutput,
     gene_output, gene_var_list,
     disc_real_output, disc_fake_output, disc_var_list] = \
        srez_model.create_model(sess, test_features, test_labels)

    # Restore variables from checkpoint
    saver = tf.train.Saver()
    filename = 'checkpoint_new.txt'
    filename = os.path.join(FLAGS.checkpoint_dir, filename)

    saver.restore(sess, filename)

    for file in filenames:
        test_features, test_labels = srez_input.test_inputs(sess, [file])

        test_feature, test_label = sess.run([test_features, test_labels])
        feed_dict = {gene_minput: test_label}
        gene_output = sess.run(gene_moutput, feed_dict=feed_dict)

        srez_test.predict_one(sess, test_feature, test_label, gene_output, file)


def _demo():
    # Load checkpoint
    if not tf.gfile.IsDirectory(FLAGS.checkpoint_dir):
        raise FileNotFoundError("Could not find folder `%s'" % (FLAGS.checkpoint_dir,))

    # Setup global tensorflow state
    sess, summary_writer = setup_tensorflow()

    # Prepare directories
    filenames = prepare_dirs(delete_train_dir=False)

    # Setup async input queues
    features, labels = srez_input.setup_inputs(sess, filenames)

    # Create and initialize model
    [gene_minput, gene_moutput,
     gene_output, gene_var_list,
     disc_real_output, disc_fake_output, disc_var_list] = \
        srez_model.create_model(sess, features, labels)

    # Restore variables from checkpoint
    saver = tf.train.Saver()
    filename = 'checkpoint_new.txt'
    filename = os.path.join(FLAGS.checkpoint_dir, filename)

    saver.restore(sess, filename)

    # Execute demo
    srez_demo.demo1(sess)


class TrainData(object):
    def __init__(self, dictionary):
        self.__dict__.update(dictionary)


def _train():
    # Restore variables from checkpoint if EXISTS
    # if tf.gfile.IsDirectory(FLAGS.checkpoint_dir):
    #     filename = 'checkpoint_new.txt'
    #     filename = os.path.join(FLAGS.checkpoint_dir, filename)
    #     saver = tf.train.Saver()
    #     if tf.gfile.Exists(filename):
    #         saver.restore(tf.Session(), filename)
    #         print("Restored previous checkpoint. "
    #               "Warning, Batch number restarted.")

    # Setup global tensorflow state
    sess, summary_writer = setup_tensorflow()

    # Prepare directories
    # all_filenames = prepare_dirs(delete_train_dir=True)
    all_filenames = prepare_dirs(delete_train_dir=False)

    # Separate training and test sets
    train_filenames = all_filenames[:-FLAGS.test_vectors]
    test_filenames = all_filenames[-FLAGS.test_vectors:]

    # TBD: Maybe download dataset here

    # Setup async input queues
    train_features, train_labels = srez_input.setup_inputs(sess, train_filenames)
    test_features, test_labels = srez_input.setup_inputs(sess, test_filenames)

    # Add some noise during training (think denoising autoencoders)
    noise_level = .03
    noisy_train_features = train_features + \
                           tf.random_normal(train_features.get_shape(), stddev=noise_level)

    # Create and initialize model
    [gene_minput, gene_moutput,
     gene_output, gene_var_list,
     disc_real_output, disc_fake_output, disc_var_list] = \
        srez_model.create_model(sess, noisy_train_features, train_labels)

    gene_loss = srez_model.create_generator_loss(disc_fake_output, gene_output, train_features)
    disc_real_loss, disc_fake_loss = \
        srez_model.create_discriminator_loss(disc_real_output, disc_fake_output)
    disc_loss = tf.add(disc_real_loss, disc_fake_loss, name='disc_loss')

    (global_step, learning_rate, gene_minimize, disc_minimize) = \
        srez_model.create_optimizers(gene_loss, gene_var_list,
                                     disc_loss, disc_var_list)

    # Train model
    train_data = TrainData(locals())
    srez_train.train_model(train_data)


def main(argv=None):
    if len(sys.argv) == 2:
        filename = sys.argv[1]
        _test16(filename)
    elif len(sys.argv) > 2:
        if FLAGS.run == 'demo':
            _demo()
        elif FLAGS.run == 'train':
            _train()
        elif FLAGS.run == 'test':
            _test()


if __name__ == '__main__':
    tf.app.run()
