# Copyright Nicholas Larus-Stone 2018.
import numpy as np
import os
import pickle
from asimov import app

def file_allowed(filename, exts):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in exts

def get_train_test_data(froot):
    try:
        with open('./data/train_test_{0}.npz'.format(froot), 'rb') as f:
            data = np.load(f)
            x_train, y_train, x_test, y_test = data['x_train'], data['y_train'], data['x_test'], data['y_test']
    except FileNotFoundError:
        return None
    return x_train, x_test, y_train, y_test

def write_train_test_data(froot, x_train, x_test, y_train, y_test):
    with open('./data/train_test_{0}.npz'.format(froot), 'wb') as f:
        np.savez(f, x_train=x_train, x_test=x_test, y_train=y_train, y_test=y_test)
    classes = np.unique(y_train).tolist()
    with open('./models/{0}.classes'.format(froot), 'wb') as f:
        pickle.dump(classes, f)

def topn(arr, n=5):
    return arr.argsort(axis=1)[0][(-1 * n):]
