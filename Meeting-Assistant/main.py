# coding: utf8
import pyaudio
import wave
from threading import Thread
import load_file as lf
import tensorflow as tf
import numpy as np
import speech2text as sp2t
import threading
import requests
from pydub import AudioSegment

CHUNK = 1024
FORMAT = pyaudio.paInt16
CHANNELS = 1
RATE = 16000
RECORD_SECONDS = 1
RECORD_OUTPUT_NAME = "record-"
RECORD_OUTPUT_FOLDER = "./record/"
SPEECH_TO_TEXT_RECORD_FOLDER = "./speech/"
SPEECH_TO_TEXT_RECORD_NAME = "speech"

list_ = []
file_name = [""]
n_dim = 80
n_classes = 3
sd = 1 / np.sqrt(n_dim)

# GradientDescentOptimizer
n_hidden_units_one = 280
n_hidden_units_two = 350

# AdamOptimizer
n_hidden_1 = 256
n_hidden_2 = 256
n_hidden_3 = 256

lock = threading.Lock()


def join_file(list_file, count):
    """
    Join files in list, save new file in folder "Speech"
    :param list_file:
    :param count:
    :return: Path of new file
    """
    data = []
    for infile, _ in list_file:
        w = wave.open(infile, 'rb')
        data.append([w.getparams(), w.readframes(w.getnframes())])
        w.close()
    name = SPEECH_TO_TEXT_RECORD_FOLDER + SPEECH_TO_TEXT_RECORD_NAME + "-" + str(list_file[0][1]) + str(count) + ".wav"
    output = wave.open(name, 'wb')
    output.setparams(data[0][0])
    num = len(data)
    for i in range(0, num):
        output.writeframes(data[i][1])
    output.close()
    return name


def check_silence(dir):
    audio = AudioSegment.from_wav(dir)
    vol = audio.rms
    print dir + ":   " + str(vol)
    if vol < 102:
        return True, vol
    else:
        return False, vol


# def classifier(features_test):
#         """
#         :param features_test: mfcc feature of file
#         Feed data to restored model
#         :return: Predict label of data
#         """
#         features_test = features_test.reshape(1, 80)
#         weights = {
#             'h1': tf.Variable(tf.random_normal([n_dim, n_hidden_1], mean=0, stddev=sd)),
#             'h2': tf.Variable(tf.random_normal([n_hidden_1, n_hidden_2], mean=0, stddev=sd)),
#             'h3': tf.Variable(tf.random_normal([n_hidden_2, n_hidden_3], mean=0, stddev=sd)),
#             'out': tf.Variable(tf.random_normal([n_hidden_3, n_classes], mean=0, stddev=sd))
#         }
#         biases = {
#             'b1': tf.Variable(tf.random_normal([n_hidden_1], mean=0, stddev=sd)),
#             'b2': tf.Variable(tf.random_normal([n_hidden_2], mean=0, stddev=sd)),
#             'b3': tf.Variable(tf.random_normal([n_hidden_3], mean=0, stddev=sd)),
#             'out': tf.Variable(tf.random_normal([n_classes], mean=0, stddev=sd))
#         }
#         X = tf.placeholder(tf.float32, [None, n_dim])
#         Y = tf.placeholder(tf.float32, [None, n_classes])
#
#         layer_1 = tf.add(tf.matmul(X, weights['h1']), biases['b1'])
#         layer_1 = tf.nn.tanh(layer_1)
#         # Hidden layer with RELU activation
#         layer_2 = tf.add(tf.matmul(layer_1, weights['h2']), biases['b2'])
#         layer_2 = tf.nn.sigmoid(layer_2)
#         # Hidden layer with RELU activation
#         layer_3 = tf.add(tf.matmul(layer_2, weights['h3']), biases['b3'])
#         layer_3 = tf.nn.sigmoid(layer_3)
#         # Output layer with linear activation
#         out_layer = tf.matmul(layer_3, weights['out']) + biases['out']
#         y_ = tf.nn.softmax(out_layer)
#         sess = tf.InteractiveSession()
#         sess.run(tf.global_variables_initializer())
#         # Restore the model
#         tf.train.Saver().restore(sess, "./adammodel/data")
#         y_pred = sess.run(tf.argmax(y_, 1), feed_dict={X: features_test})
#         y_tt = sess.run(y_, feed_dict={X: features_test})
#         users = open("USER.txt", "r")
#         list_users = []
#         for user in users:
#             user = user.split("\n")[0]
#             list_users.append(user)
#         return list_users[y_pred[0]], y_tt
def classifier(features_test):
    """
    :param features_test: mfcc feature of file
    Feed data to restored model
    :return: Predict label of data
    """
    features_test = features_test.reshape(1, 80)
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
    y_tt = sess.run(y_, feed_dict={X: features_test})
    users = open("USER.txt", "r")
    list_users = []
    for user in users:
        user = user.split("\n")[0]
        list_users.append(user)
    return list_users[y_pred[0]], y_tt


def post_text(name, text):
    """
    POST json file to web service
    :param name: Name of speaker
    :param text:
    :return: None
    """
    item = {"name": name, "text": text}
    print item
    respond = requests.post('https://example.com', json=item)
    if respond.status_code != 201:
        print respond.status_code


# def save_file(frame, p, count):
#     """Save recorded file and check list"""
#     if count == -1:
#         print("Start Record!!!")
#         return 0
#     lock.acquire()
#     # Save file
#     name = RECORD_OUTPUT_FOLDER + RECORD_OUTPUT_NAME + str(count) + ".wav"
#     wf = wave.open(name, 'wb')
#     wf.setnchannels(CHANNELS)
#     wf.setsampwidth(p)
#     wf.setframerate(RATE)
#     wf.writeframes(b''.join(frame))
#     wf.close()
#     # print "File saved in: " + name
#     """
#     check list: if list null, add current file to list (file_name, class, count)
#     else if: sub(current_count,first_element_count) < 2, append current file to list
#     else if: sub(current_count,first_element_count) >= 2, check equal(current_class, first_element_class)
#     if True: append current file to list, join all files in list and convert to text, list = []
#     if False: join all files in list and convert to text, list = [], add current file to list
#     """
#     # Check silence
#     vol = check_silence(name)
#
#     # Get mfcc
#     features_test = lf.extract_feature(name)
#
#     if len(list_) == 0:
#         label, o = classifier(features_test)
#         list_.append([name, label, count])
#     else:
#         first_element_count = list_[0][2]
#         if (count - first_element_count) < 2:
#             label, o = classifier(features_test)
#             list_.append([name, label, count])
#         else:
#             label, o = classifier(features_test)
#             first_element_class = list_[0][1]
#             if label == first_element_class:
#                 list_.append([name, label, count])
#                 conversations = open("log.txt", "a")
#                 conversations.write(str(list_))
#                 conversations.write('\n\n')
#                 conversations.write(str(o))
#                 conversations.write('\n\n')
#                 conversations.write(str(count)+":   "+str(vol))
#                 conversations.write("####################")
#                 conversations.write('\n\n')
#                 # join 3 file in list
#                 speech = join_file(list_, count)
#                 text = sp2t.speech_2_text(speech)
#                 # clear list
#                 list_[:] = []
#
#             else:
#                 second_element_class = list_[1][1]
#                 if first_element_class == second_element_class:
#                     # join 2 file in list
#                     speech = join_file(list_, count)
#                     text = sp2t.speech_2_text(speech)
#                     # clear list
#                     list_[:] = []
#                     list_.append([name, label, count])
#                 else:
#                     list_.append([name, label, count])
#                     conversations = open("log.txt", "a")
#                     conversations.write(str(list_))
#                     conversations.write('\n\n')
#                     conversations.write(str(o))
#                     conversations.write('\n\n')
#                     conversations.write(str(count)+":   "+str(vol))
#                     conversations.write("####################")
#                     conversations.write('\n\n')
#                     speech = join_file(list_, count)
#                     text = sp2t.speech_2_text(speech)
#                     first_element_class = second_element_class
#                     list_[:] = []
#
#             result = first_element_class + ": " + text.encode("utf8")
#             # result = unicode(result, errors='ignore')
#             post_text(first_element_class, text)
#             # Write speech-text to file
#             conversations = open("CONVERSATIONS.txt", "a")
#             conversations.write(result)
#             conversations.write('\n\n')
#     # Write list_ log to file
#     conversations = open("log.txt", "a")
#     conversations.write(str(list_))
#     conversations.write('\n\n')
#     conversations.write(str(o))
#     conversations.write('\n\n')
#     conversations.write(str(count)+":   "+str(vol))
#     conversations.write("####################")
#     conversations.write('\n\n')
#     lock.release()

def save_file(frame, sample_size, count):
    """Save recorded file and check list"""
    if count == -1:
        print("Start Record!!!")
        return 0
    lock.acquire()
    # Save file
    name = RECORD_OUTPUT_FOLDER + RECORD_OUTPUT_NAME + str(count) + ".wav"
    wf = wave.open(name, 'wb')
    wf.setnchannels(CHANNELS)
    wf.setsampwidth(sample_size)
    wf.setframerate(RATE)
    wf.writeframes(b''.join(frame))
    wf.close()
    # Check silence
    vol, val = check_silence(name)
    # If true, classify and speech-to-text if len > 1s
    # Else, if len < 10s, append current file (1s) to list, else
    if vol:
        if len(list_) > 1:
            speech = join_file(list_, count)
            text = sp2t.speech_2_text(speech)
            list_[:] = []
            features = lf.extract_feature(speech)
            label, o = classifier(features)
            result_text = str(label) + " :" + str(text)
            # Write speech-text to file
            conversations = open("CONVERSATIONS.txt", "a")
            conversations.write(result_text)
            conversations.write('\n\n')
            conversations.write(str(o))
            conversations.write('\n\n')
        else:
            list_[:] = []
    else:
        if len(list_) < 10:
            list_.append([name, count])
            conversations = open("log.txt", "a")
            conversations.write(str(list_))
            conversations.write('\n\n')
            conversations.write(str(count) + ":   " + str(val))
            conversations.write("####################")
            conversations.write('\n\n')
        else:
            speech = join_file(list_, count)
            text = sp2t.speech_2_text(speech)
            list_[:] = []
            features = lf.extract_feature(speech)
            label, o = classifier(features)
            result_text = str(label) + " :" + str(text)
            # Write speech-text to file
            conversations = open("CONVERSATIONS.txt", "a")
            conversations.write(result_text)
            conversations.write('\n\n')
            conversations.write(str(o))
            conversations.write('\n\n')
            list_.append([name, count])
            conversations = open("log.txt", "a")
            conversations.write(str(list_))
            conversations.write('\n\n')
            conversations.write(str(count) + ":   " + str(val))
            conversations.write("####################")
            conversations.write('\n\n')
    lock.release()


def record():
    """Start recording, save to file every X seconds"""
    range_ = int(RATE / CHUNK * RECORD_SECONDS)
    p = pyaudio.PyAudio()
    sample_size = p.get_sample_size(FORMAT)
    count = -1
    #wf = wave.open("1_speech-4245.wav",'rb')

    stream = p.open(format=FORMAT,
                    channels=CHANNELS,
                    rate=RATE,
                    input=True,
                    frames_per_buffer=CHUNK)
    while True:
        frames = []
        for i in range(0, range_):
            data = stream.read(CHUNK)
            frames.append(data)
        new_frame = frames
        savefile = Thread(name="savefile", target=save_file, args=(new_frame, sample_size, count,))
        savefile.setDaemon(True)
        savefile.start()
        count = count + 1
    stream.stop_stream()
    stream.close()
    p.terminate()

if __name__ == '__main__':
    record()
