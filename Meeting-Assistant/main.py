from threading import Thread
import record_audio as ra
import load_file as lf
import speech2text as sp2t
import tensorflow as tf
import numpy as np
import time
file_name = [""]
batch_size = 290
training_epochs = 59
n_dim = 60
n_hidden_units_one = 280
n_hidden_units_two = 300
sd = 1 / np.sqrt(n_dim)
learning_rate = 0.0001
n_classes = 8


def classifier(features_test):
    features_test = features_test.reshape(1, 60)
    # X is the input array, contaning mfccs data
    X = tf.placeholder(tf.float32, [None, n_dim])
    # Y contains true labels output
    Y = tf.placeholder(tf.float32, [None, n_classes])

    # Multi-layer neural network
    W_1 = tf.Variable(tf.random_normal([n_dim, n_hidden_units_one], mean=0, stddev=sd))
    b_1 = tf.Variable(tf.random_normal([n_hidden_units_one], mean=0, stddev=sd))
    h_1 = tf.nn.tanh(tf.matmul(X, W_1) + b_1)

    W_2 = tf.Variable(tf.random_normal([n_hidden_units_one, n_hidden_units_two], mean=0, stddev=sd))
    b_2 = tf.Variable(tf.random_normal([n_hidden_units_two], mean=0, stddev=sd))
    h_2 = tf.nn.sigmoid(tf.matmul(h_1, W_2) + b_2)

    W = tf.Variable(tf.random_normal([n_hidden_units_two, n_classes], mean=0, stddev=sd))
    b = tf.Variable(tf.random_normal([n_classes], mean=0, stddev=sd))

    # Output calc(Result)
    y_ = tf.nn.softmax(tf.matmul(h_2, W) + b)
    sess = tf.InteractiveSession()
    sess.run(tf.global_variables_initializer())
    # Restore the model
    tf.train.Saver().restore(sess, "./model/data")
    y_pred = sess.run(tf.argmax(y_, 1), feed_dict={X: features_test})
    users = open("USER.txt", "r")
    list_users = []
    for user in users:
        user = user.split("\n")[0]
        list_users.append(user)
    return list_users[y_pred[0]]


def listen_file_name(fn):
    tmp = fn[0]
    while True:
        if tmp != fn[0]:
            print "PATH:  ", fn[0]
            # classify and speech to text
            result_text = ThreadWithReturnValue(target=collect_text, args=(fn[0]))
            result_text.start()
            tmp = fn[0]


def collect_text(fn):
    features_test = lf.extract_feature(fn)
    audio_file = fn
    label = ThreadWithReturnValue(target=classifier, args=features_test)
    return_text = ThreadWithReturnValue(target=sp2t.speech_2_text, args=audio_file)
    label.start()
    return_text.start()
    label = label.join()
    return_text = return_text.join()
    text = fn + "  " + label + ": " + return_text
    conversations = open("CONVERSATIONS.txt", "a")
    conversations.write(text)
    conversations.write('\n\n')


class ThreadWithReturnValue(Thread):
    def __init__(self, group=None, target=None, name=None,
                 args=(), kwargs={}, Verbose=None):
        Thread.__init__(self, group, target, name, args, kwargs, Verbose)
        self._return = None

    def run(self):
        if self._Thread__target is not None:
            self._return = self._Thread__target(self._Thread__args,
                                                **self._Thread__kwargs)

    def join(self):
        Thread.join(self)
        return self._return


record = ThreadWithReturnValue(target=ra.record, args=file_name)
listen_record = ThreadWithReturnValue(target=listen_file_name, args=file_name)
record.start()
listen_record.start()
