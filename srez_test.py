import numpy as np
import os.path
import scipy.misc
import tensorflow as tf
import time

FLAGS = tf.app.flags.FLAGS


def predict(test_data, feature, label, gene_output, suffix="predicted", max_samples=8):
    td = test_data

    size = [label.shape[1], label.shape[2]]

    nearest = tf.image.resize_nearest_neighbor(feature, size)
    nearest = tf.maximum(tf.minimum(nearest, 1.0), 0.0)

    bicubic = tf.image.resize_bicubic(feature, size)
    bicubic = tf.maximum(tf.minimum(bicubic, 1.0), 0.0)

    clipped = tf.maximum(tf.minimum(gene_output, 1.0), 0.0)

    image = tf.concat([nearest, bicubic, clipped, label], 2)

    image = image[0:max_samples, :, :, :]
    image = tf.concat([image[i, :, :, :] for i in range(max_samples)], 0)
    image = td.sess.run(image)

    filename = '%s.png' % suffix
    filename = os.path.join(FLAGS.predict_dir, filename)
    scipy.misc.toimage(image, cmin=0., cmax=1.).save(filename)
    print("    Saved %s" % (filename,))


def predict_one(test_data, feature, label, gene_output):
    td = test_data
    name = td.filenames[0]
    name = os.path.basename(name)
    name = os.path.splitext(name)[0]

    size = [label.shape[1], label.shape[2]]

    nearest = tf.image.resize_nearest_neighbor(feature, size)
    nearest = tf.maximum(tf.minimum(nearest, 1.0), 0.0)

    image = tf.concat([nearest], 2)
    image = image[0, :, :, :]
    image = td.sess.run(image)

    filename = '%s_resize.png' % name
    filename = os.path.join(FLAGS.predict_dir, filename)
    scipy.misc.toimage(image, cmin=0., cmax=1.).save(filename)
    print("    Saved %s" % (filename,))

    clipped = tf.maximum(tf.minimum(gene_output, 1.0), 0.0)

    image = tf.concat([clipped], 2)
    image = image[0, :, :, :]

    image = td.sess.run(image)

    filename = '%s_predicted.png' % name
    filename = os.path.join(FLAGS.predict_dir, filename)
    scipy.misc.toimage(image, cmin=0., cmax=1.).save(filename)
    print("    Saved %s" % (filename,))