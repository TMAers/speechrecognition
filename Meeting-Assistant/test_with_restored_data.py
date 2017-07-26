import glob
import os
import librosa
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import load_file

from matplotlib.pyplot import specgram
batch_size=290
training_epochs = 59
n_dim =  60
n_hidden_units_one = 280 
n_hidden_units_two = 300
sd = 1 / np.sqrt(n_dim)
learning_rate = 0.001
n_classes=6

#X is the input array, contaning mfccs data
X = tf.placeholder(tf.float32,[None,n_dim])
#Y contains true labels output
Y = tf.placeholder(tf.float32,[None,n_classes])

#Convolutional neural network
W_1 = tf.Variable(tf.random_normal([n_dim,n_hidden_units_one], mean = 0, stddev=sd))
b_1 = tf.Variable(tf.random_normal([n_hidden_units_one], mean = 0, stddev=sd))
h_1 = tf.nn.tanh(tf.matmul(X,W_1) + b_1)

W_2 = tf.Variable(tf.random_normal([n_hidden_units_one,n_hidden_units_two], mean = 0, stddev=sd))
b_2 = tf.Variable(tf.random_normal([n_hidden_units_two], mean = 0, stddev=sd))
h_2 = tf.nn.sigmoid(tf.matmul(h_1,W_2) + b_2)

W = tf.Variable(tf.random_normal([n_hidden_units_two,n_classes], mean = 0, stddev=sd))
b = tf.Variable(tf.random_normal([n_classes], mean = 0, stddev=sd))
#Output calc(Result)
y_ = tf.nn.softmax(tf.matmul(h_2,W) + b)

#
cost_function = tf.reduce_mean(-tf.reduce_sum(Y * tf.log(y_)))
optimizer = tf.train.GradientDescentOptimizer(learning_rate).minimize(cost_function)

correct_prediction = tf.equal(tf.argmax(y_,1), tf.argmax(Y,1))
max_prediction=tf.argmax(y_,1)
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

sess = tf.InteractiveSession()
sess.run(tf.global_variables_initializer())
#Get features and labels from 'testdata' folder
features_test, labels_test = load_file.get_audio_features("./","testdata")
labels_test=load_file.one_hot_encode_with_test_file(labels_test,n_classes)

#Restore the model
tf.train.Saver().restore(sess, "./model/data")

y_true = sess.run(tf.argmax(Y,1),feed_dict={Y: labels_test })
y_pred = sess.run(tf.argmax(y_,1),feed_dict={X: features_test })
y_true=y_true+1
y_pred=y_pred+1

print "True labels:  ", y_true
print "Prediction:   ", y_pred
y_pred = sess.run(y_,feed_dict={X: features_test ,Y:labels_test}
print "Accuracy:     ", y_pred



