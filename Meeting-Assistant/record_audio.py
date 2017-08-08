# coding: utf8
import pyaudio
import wave
from threading import Thread
import logging
import time
import load_file as lf
import tensorflow as tf
import numpy as np
import speech2text as sp2t
import threading

CHUNK = 1024
FORMAT = pyaudio.paInt16
CHANNELS = 2
RATE = 44100
RECORD_SECONDS = 3
RECORD_OUTPUT_NAME = "record-"
RECORD_OUTPUT_FOLDER = "./records/"
SPEECH_TO_TEXT_RECORD_FOLDER = "./speech/"
SPEECH_TO_TEXT_RECORD_NAME = "speech-"

list_ = []

logging.basicConfig(level=logging.DEBUG,
                    format='(%(threadName)-10s) %(message)s',
                    )
file_name = [""]
batch_size = 290
training_epochs = 59
n_dim = 60
n_hidden_units_one = 280
n_hidden_units_two = 300
sd = 1 / np.sqrt(n_dim)
learning_rate = 0.0001
n_classes = 8

lock = threading.Lock()


def join_file(list_file, count):
    data = []
    for infile, _, _ in list_:
        w = wave.open(infile, 'rb')
        data.append([w.getparams(), w.readframes(w.getnframes())])
        w.close()
    name = SPEECH_TO_TEXT_RECORD_FOLDER + SPEECH_TO_TEXT_RECORD_NAME + "-" + list_file[0][1] + str(count) + ".wav"
    output = wave.open(name, 'wb')
    output.setparams(data[0][0])
    num = len(data)
    for i in range(0, num):
        output.writeframes(data[i][1])
    output.close()
    return name


def classifier(features_test):
    logging.debug("start")
    features_test = features_test.reshape(1, 60)
    # X is the input array, containing mfccs data
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
    logging.debug("stop")
    return list_users[y_pred[0]]


def num(x):
    switcher = {
        0: "/home/ptphu/Documents/pyaudio/records/1_70968-0036.wav",
        1: "/home/ptphu/Documents/pyaudio/records/1_70968-0036.wav",
        2: "/home/ptphu/Documents/pyaudio/records/3_123286-0017.wav",
        3: "/home/ptphu/Documents/pyaudio/records/3_123286-0017.wav",
        4: "/home/ptphu/Documents/pyaudio/records/4_133604-0025.wav",
        5: "/home/ptphu/Documents/pyaudio/records/4_133604-0025.wav",
        6: "/home/ptphu/Documents/pyaudio/records/4_133604-0025.wav",
        7: "/home/ptphu/Documents/pyaudio/records/5_134686-0020.wav",
        8: "/home/ptphu/Documents/pyaudio/records/5_134686-0020.wav",
        9: "/home/ptphu/Documents/pyaudio/records/2_122617-0029.wav",
        10: "/home/ptphu/Documents/pyaudio/records/2_122617-0029.wav",
        11: "/home/ptphu/Documents/pyaudio/records/2_122617-0029.wav",
        12: "/home/ptphu/Documents/pyaudio/records/1_70968-0036.wav",
        13: "/home/ptphu/Documents/pyaudio/records/1_70968-0036.wav",
        14: "/home/ptphu/Documents/pyaudio/records/1_70968-0036.wav",

    }
    return switcher.get(x, "nothing")


def save_file(frame, p, count, fn):
    if count == -1:
        print("Start Record!!!")
        return 0
    logging.debug("start")
    name = RECORD_OUTPUT_FOLDER + RECORD_OUTPUT_NAME + str(count) + ".wav"
    wf = wave.open(name, 'wb')
    wf.setnchannels(CHANNELS)
    wf.setsampwidth(p.get_sample_size(FORMAT))
    wf.setframerate(RATE)
    wf.writeframes(b''.join(frame))
    wf.close()
    # print "File saved in: " + name
    fn[0] = name
    logging.debug("exit")
    # check list: if list null, add current file to list (file_name, class, count)
    # else if: sub(current_count,first_element_count) < 2, append current file to list
    # else if: sub(current_count,first_element_count) >= 2, check equal(current_class, first_element_class)
    # if True: append current file to list, join all files in list and convert to text, list = []
    # if False: join all files in list and convert to text, list = [], add current file to list
    if count < 15:
        name = num(count)
    features_test = lf.extract_feature(name)
    lock.acquire()

    if len(list_) == 0:
        label = classifier(features_test)
        list_.append([name, label, count])
    else:
        first_element_count = list_[0][2]
        if (count - first_element_count) < 2:
            label = classifier(features_test)
            list_.append([name, label, count])
        else:
            label = classifier(features_test)
            first_element_class = list_[0][1]
            if label == first_element_class:
                list_.append([name, label, count])
                conversations = open("log.txt", "a")
                conversations.write(str(list_))
                conversations.write('\n\n')
                conversations.write("####################")
                conversations.write('\n\n')
                # join 3 file in list
                speech = join_file(list_, count)
                text = sp2t.speech_2_text(speech)
                # clear list
                list_[:] = []

            else:
                a = 1
                # join 2 file in list
                speech = join_file(list_, count)
                text = sp2t.speech_2_text(speech)
                # clear list
                list_[:] = []
                list_.append([name, label, count])

            result = speech + ": " + text
            print result
            # conversations = open("CONVERSATIONS.txt", "a")
            # conversations.write(result)
            # conversations.write('\n\n')
    conversations = open("log.txt", "a")
    conversations.write(str(list_))
    conversations.write('\n\n')
    conversations.write("####################")
    conversations.write('\n\n')
    lock.release()


class ThreadWithReturnValue(Thread):
    def __init__(self, group=None, target=None, name=None,
                 args=(), lock=None):
        Thread.__init__(self, group, target, name, args, lock)
        self._return = None
        self.lock = lock

    def run(self):
        if self._Thread__target is not None:
            self._return = self._Thread__target(self._Thread__args,
                                                **self._Thread__kwargs)

    def join(self):
        Thread.join(self)
        return self._return


def record(file_name):
    # range_ = int(RATE / CHUNK * RECORD_SECONDS)
    logging.debug("start")
    p = pyaudio.PyAudio()
    # width = p.get_sample_size(FORMAT)
    count = -1

    stream = p.open(format=FORMAT,
                    channels=CHANNELS,
                    rate=RATE,
                    input=True,
                    frames_per_buffer=CHUNK)
    while True:
        frames = []
        for i in range(0, int(RATE / CHUNK * RECORD_SECONDS)):
            data = stream.read(CHUNK)
            frames.append(data)
        # file_name[0] = save_file(frames, p, count)

        savefile = Thread(name="savefile", target=save_file, args=(frames, p, count, file_name,))
        savefile.setDaemon(True)
        savefile.start()
        # save_file(frames, p, count, file_name)
        count = count + 1
    stream.stop_stream()
    stream.close()
    p.terminate()


a = [""]
record(a)
