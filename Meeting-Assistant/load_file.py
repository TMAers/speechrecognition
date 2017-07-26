import glob
import os
import librosa
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from matplotlib.pyplot import specgram

#Return mfccs of sound
def extract_feature(file_name):
	X, sample_rate = librosa.load(file_name)
    	#stft = np.abs(librosa.stft(X))
    	mfccs = np.mean(librosa.feature.mfcc(y=X, sr=sample_rate, n_mfcc=60).T,axis=0)
	print mfccs
    	return mfccs
#Read data
#Return two values, first value is array of data's features, second value is array of labels
def get_audio_features(parent_dir,sub_dir,file_ext="*.wav"):
	features, labels = np.empty((0,60)), np.empty(0)
        for fn in glob.glob(os.path.join(parent_dir, sub_dir, file_ext)):
            	try:
              		mfccs= extract_feature(fn)
            	except Exception as e:
              		print "Error encountered while parsing file: ", e
             		continue
            	ext_features = np.hstack([mfccs])
            	features = np.vstack([features,ext_features])
            	labels = np.append(labels, (fn.split('_')[0]).split("/")[2])
	print labels
    	return np.array(features), np.array(labels, dtype = np.int)
	
#Convert array of labels to one-hot vector
def one_hot_encode(labels):
    n_labels = len(labels)
    n_unique_labels = len(np.unique(labels))
    one_hot_encode = np.zeros((n_labels,n_unique_labels))
    one_hot_encode[np.arange(n_labels), labels-1] = 1
    return one_hot_encode,n_unique_labels
def one_hot_encode_with_test_file(labels,n_classes):
    n_labels = len(labels)
    n_unique_labels = n_classes
    one_hot_encode = np.zeros((n_labels,n_unique_labels))
    one_hot_encode[np.arange(n_labels), labels-1] = 1
    return one_hot_encode

