from random import randint
import pyaudio
import wave
import tensorflow as tf
import numpy as np
import librosa
import glob
import os
import load_file as loadfile

CHUNK = 1024
FORMAT = pyaudio.paInt16
CHANNELS = 1
RATE = 16000
RECORD_SECONDS = 1
n_dim = 80
n_hidden_units_one = 280
n_hidden_units_two = 350
sd = 1 / np.sqrt(n_dim)
learning_rate = 0.000008
batch_size = 290
n_classes = 3


def train_model_w_new_data():
    X = tf.placeholder(tf.float32, [None, n_dim])
    Y = tf.placeholder(tf.float32, [None, n_classes])

    W_1 = tf.Variable(tf.random_normal([n_dim, n_hidden_units_one], mean=0, stddev=sd))
    b_1 = tf.Variable(tf.random_normal([n_hidden_units_one], mean=0, stddev=sd))
    h_1 = tf.nn.tanh(tf.matmul(X, W_1) + b_1)

    W_2 = tf.Variable(tf.random_normal([n_hidden_units_one, n_hidden_units_two], mean=0, stddev=sd))
    b_2 = tf.Variable(tf.random_normal([n_hidden_units_two], mean=0, stddev=sd))
    h_2 = tf.nn.sigmoid(tf.matmul(h_1, W_2) + b_2)

    W = tf.Variable(tf.random_normal([n_hidden_units_two, n_classes], mean=0, stddev=sd))
    b = tf.Variable(tf.random_normal([n_classes], mean=0, stddev=sd))
    y_ = tf.nn.softmax(tf.matmul(h_2, W) + b)
    sess = tf.InteractiveSession()
    sess.run(tf.global_variables_initializer())

    #
    correct_prediction = tf.equal(tf.argmax(y_, 1), tf.argmax(Y, 1))
    max_prediction = tf.argmax(y_, 1)
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    cost_function = tf.reduce_mean(-tf.reduce_sum(Y * tf.log(y_)))
    optimizer = tf.train.GradientDescentOptimizer(learning_rate).minimize(cost_function)
    features, labels = loadfile.get_audio_features("./", "datatrain")
    labels = loadfile.one_hot_encode_with_test_file(labels, n_classes)
    #features_test, labels_test = loadfile.get_audio_features("./", "c")
    #labels_test = loadfile.one_hot_encode_with_test_file(labels_test, n_classes)

    print labels
    """if(num>1):
        tf.train.Saver().restore(sess, "./model/data")"""
    nSteps = 3000
    for i in range(nSteps):
        # batch_size=randint(100, 500)

        # run the training step with feed of data
        offset = (nSteps * batch_size) % (features.shape[0] - batch_size)
        # Generate a minibatch.
        batch_data = features[offset:(offset + batch_size), :]
        batch_labels = labels[offset:(offset + batch_size), :]
        optimizer.run(feed_dict={X: features, Y: labels})
        #if (i + 1) % 10 == 0:
            #y_pred = sess.run(accuracy, feed_dict={X: features_test, Y: labels_test})
            #print y_pred
    save_path = tf.train.Saver().save(sess, "./model/data")
    print("Model saved")

print '************** Train *************'
train_model_w_new_data()
