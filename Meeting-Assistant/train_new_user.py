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
n_classes = 4


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
    features, labels = loadfile.get_audio_features("./", "record218train106")
    labels = loadfile.one_hot_encode_with_test_file(labels, n_classes)
    features_test, labels_test = loadfile.get_audio_features("./", "c")
    labels_test = loadfile.one_hot_encode_with_test_file(labels_test, n_classes)

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


"""
def get_ten_input(number):
	sentences=open("TEXT.txt","r")
        list_ =sentences.readlines()
	print "Please speak some sentenses below:"
	for i in range(1,11):
		print i,". ", list_[randint(0,103)]
		print "Enter to start record!"
		p = pyaudio.PyAudio()
		frames = []
		enter = raw_input()
		if enter=="":
			print "recording !!!"
			stream = p.open(format=FORMAT,
                	channels=CHANNELS,
                	rate=RATE,
                	input=True,
                	frames_per_buffer=CHUNK)
			#enter1=raw_input()
			for n in range(0, int(RATE / CHUNK * RECORD_SECONDS)):
    				data = stream.read(CHUNK)
    				frames.append(data)

			print("done recording !!!")

			stream.stop_stream()
			stream.close()
			p.terminate()
			file_name="./audiodata/"+str(number)+"_"+str(i)+".wav"
			wf = wave.open(file_name, 'wb')
			wf.setnchannels(CHANNELS)
			wf.setsampwidth(p.get_sample_size(FORMAT))
			wf.setframerate(RATE)
			wf.writeframes(b''.join(frames))
			wf.close()
			print "Saved to ",file_name"""


def add_new_user(new_user, users):
    check = False
    num_users = 0
    for user in users:
        num_users = num_users + 1
        user = user.split("\n")[0]
        if new_user == user:
            print "Your voice data is exist in database"
            return 0
    if check == False:
        users.write(new_user)
        users.write('\n')
        print "Added your name to USER.txt"
    # get_ten_input(num_users+1)
    return num_users + 1


print '************** Train *************'
"""new_user = raw_input("Enter your name: ")
users=open("USER.txt","r+")
num=add_new_user(new_user,users)"""
# if(num!=0):
train_model_w_new_data()
