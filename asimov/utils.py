from asimov import app
import os
import pickle
import numpy as np

def file_allowed(filename, exts):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in exts

def valid_file_formatting(f):
    i = 0
    for line in f:
        l = line.decode("utf-8")
        line_list = l.split('\t', 1)
        # TODO: More error checking
        if len(line_list) != 2:
            if app.config.DEBUG:
                print(line_list)
            return False
        if len(line_list[1].split('\n')) != 2:
            if app.config.DEBUG:
                print(line_list)
            return False
    return True

# TODO ADD VALIDATION
def valid_test_data(text):
    return True

def get_train_test_data(froot):
    try:
        with open('./data/train_test_{0}.npz'.format(froot), 'rb') as f:
            data = np.load(f)
            x_train, y_train, x_test, y_test = data['x_train'], data['y_train'], data['x_test'], data['y_test']
    except FileNotFoundError:
        return None
    """if os.path.exists('./data/x_train_{0}.txt'.format(froot)):
        with open('./data/x_train_{0}.txt'.format(froot), 'rb') as f:
            x_train = pickle.load(f)
    else:
        return None
    if os.path.exists('./data/x_test_{0}.txt'.format(froot)):
        with open('./data/x_test_{0}.txt'.format(froot), 'rb') as f:
            x_test = pickle.load(f)
    else:
        return None
    if os.path.exists('./data/y_train_{0}.txt'.format(froot)):
        with open('./data/y_train_{0}.txt'.format(froot), 'rb') as f:
            y_train = pickle.load(f)
    else:
        return None
    if os.path.exists('./data/y_test_{0}.txt'.format(froot)):
        with open('./data/y_test_{0}.txt'.format(froot), 'rb') as f:
            y_test = pickle.load(f)
    else:
        return None"""
    return x_train, x_test, y_train, y_test

def write_train_test_data(froot, x_train, x_test, y_train, y_test):
    with open('./data/train_test_{0}.npz'.format(froot), 'wb') as f:
        np.savez(f, x_train=x_train, x_test=x_test, y_train=y_train, y_test=y_test)
    """with open('./data/x_train_{0}.txt'.format(froot), 'wb') as f:
        x_train.savetxt('./data/x_train_{0}.txt'.format(froot)), f)
    with open('./data/x_test_{0}.txt'.format(froot), 'wb') as f:
        pickle.dump(x_test, f)
    with open('./data/y_train_{0}.txt'.format(froot), 'wb') as f:
        pickle.dump(y_train, f)
    with open('./data/y_test_{0}.txt'.format(froot), 'wb') as f:
        pickle.dump(y_test, f)"""
