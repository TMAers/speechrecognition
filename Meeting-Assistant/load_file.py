import glob
import os
import librosa
import numpy as np


def extract_feature(file_name):
    """
    :param file_name: Path of audio file
    :return: mfcc of sound
    """
    X, sample_rate = librosa.load(file_name)
    mfccs = np.mean(librosa.feature.mfcc(y=X, sr=sample_rate, n_mfcc=80).T, axis=0)
    return mfccs


def get_audio_features(parent_dir, sub_dir, file_ext="*.wav"):
    """
    Read data
    :param parent_dir:
    :param sub_dir:
    :param file_ext:
    :return: Two values, first value is array of mfccs, second value is array of labels
    """
    features, labels = np.empty((0, 80)), np.empty(0)
    for fn in glob.glob(os.path.join(parent_dir, sub_dir, file_ext)):
        try:
            mfccs = extract_feature(fn)
        except Exception as e:
            print "Error encountered while parsing file: ", e
            continue
        ext_features = np.hstack([mfccs])
        features = np.vstack([features, ext_features])
        labels = np.append(labels, (fn.split('_')[0]).split("/")[2])
    return np.array(features), np.array(labels, dtype=np.int)


def one_hot_encode(labels):
    """Convert array of labels to one-hot vector"""
    n_labels = len(labels)
    n_unique_labels = len(np.unique(labels))
    one_hot_encode = np.zeros((n_labels, n_unique_labels))
    one_hot_encode[np.arange(n_labels), labels - 1] = 1
    return one_hot_encode, n_unique_labels


def one_hot_encode_with_test_file(labels, n_classes):
    n_labels = len(labels)
    n_unique_labels = n_classes
    one_hot_encode = np.zeros((n_labels, n_unique_labels))
    one_hot_encode[np.arange(n_labels), labels - 1] = 1
    return one_hot_encode
