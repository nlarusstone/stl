# Copyright Nicholas Larus-Stone 2018.
import pickle
import ast
import re
import keras.backend as K
from keras.models import Model, save_model, load_model
from keras.layers import Dense, Activation, Dropout, Input, merge
from keras.wrappers.scikit_learn import KerasClassifier
from keras.callbacks import EarlyStopping
from lime.lime_text import LimeTextExplainer
# Having issues with tkAgg backedn
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, precision_recall_fscore_support
from sklearn.pipeline import make_pipeline
from .utils import get_train_test_data, write_train_test_data, topn

max_words = 1000
np.random.seed(42)

def train_model(file_path, froot, model_type, new=False):
    # Try to read in split up data if possible
    if not new:
        data = get_train_test_data(froot)
        if data:
            x_train, x_test, y_train, y_test = data
    elif new or not data:
        x, y = [], []
        invalid_lines = 0
        with open(file_path, 'rb') as file_obj:
            for line in file_obj:
                l = line.decode("utf-8")
                # Ensure valid file formatting
                try:
                    line_list = l.split('\t', 1)
                    label, text = line_list
                    x.append(text)
                    y.append(label)
                except:
                    # If error, just skip that line (and keep track of number of errors)
                    invalid_lines += 1
            x = np.array(x)
            y = np.array(y)
            # Only return error if >50% of lines are incorrectly formatted
            if invalid_lines > x.shape[0]:
                return 'More than half the lines are incorrectly formatted'

        x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2)
        write_train_test_data(froot, x_train, x_test, y_train, y_test)
    classes = np.unique(y_train)

    model_path = './models/{0}_{1}.pkl'.format(froot, model_type)
    # For NNs, need to load the pipeline and model separately due to bug in pickle library
    try:
        with open(model_path, 'rb') as model_f:
            pipeline = pickle.load(model_f)
        if model_type != 'mnb':
            model = load_model('./models/{0}_{1}_model.h5'.format(froot, model_type))
            pipeline.steps.append(('nn', model))
    except (FileNotFoundError, EOFError):
        if model_type == 'mnb':
            model = naive_bayes()
        elif model_type == 'ffnn':
            early_stopping = EarlyStopping(monitor='val_loss', patience=2, verbose=0, mode='auto')
            model = KerasClassifier(build_fn=ffnn, nclasses=classes.shape[0], epochs=10, batch_size=128,
                                            verbose=0, validation_split=0.2, callbacks=[early_stopping])
        elif model_type == 'rnn':
            model == KerasClassifier(rnn)
        else:
            return 'Invalid model type'

        # Create and fit pipeline
        vectorizer = vectorize()
        pipeline = make_pipeline(vectorizer, model)
        pipeline.fit(x_train, y_train)

        # Save model for test usage
        with open(model_path, 'wb') as model_f:
            # Need to save the Keras model separately from the pipeline due to a bug in pickle
            if model_type != 'mnb':
                model_step = pipeline.steps.pop(-1)[1]
                save_model(model_step.model, './models/{0}_{1}_model.h5'.format(froot, model_type))
            pickle.dump(pipeline, model_f)
            if model_type != 'mnb':
                pipeline.steps.append(('nn', model_step))

    # Get training metrics
    preds = pipeline.predict(x_test)
    acc = accuracy_score(y_test, preds)
    precision, recall, fscore, supp = map(lambda x: np.round(x, decimals=3), precision_recall_fscore_support(y_test, preds))
    cnf_mat = confusion_matrix(y_test, preds)
    return {'cnf_mat': cnf_mat, 'classes': classes, 'accuracy': acc,
                'precision': precision, 'recall': recall, 'fscore': fscore}

def vectorize():
    return TfidfVectorizer(lowercase=False, max_features=max_words)
    #tokenizer = Tokenizer(num_words=max_words)
#tokenizer.fit_on_texts(x)
#return tokenizer.texts_to_matrix(x, mode='count')

def naive_bayes():
    mnb = MultinomialNB()
    return mnb

def attn_dense(inputs, idx=1):
    # Keeps track of the attention
    attention_probs = Dense(int(inputs.shape[1]), activation='relu', name='attention_vec_{0}'.format(idx))(inputs)
    # Actual provides the connections in the NN
    attention_lay = merge([inputs, attention_probs], output_shape=1000, name='attention_layer_{0}'.format(idx), mode='mul')
    return attention_lay

# Creates a simple NN for multiclass classification
def ffnn(nclasses=20):
    inputs = Input(shape=(max_words,))
    attn_1 = Dense(512)(attn_dense(inputs, idx=1))
    drop_1 = Dropout(0.5)(attn_1)
    attn_2 = Dense(512)(attn_dense(drop_1, idx=2))
    drop_2 = Dropout(0.5)(attn_2)
    output = Dense(nclasses, activation='softmax')(drop_2)
    model = Model(input=[inputs], output=output)
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model

# TODO: If had more time, would implement a RNN model
def rnn():
    pass

def get_activations(model, layer, X):
    # From https://github.com/keras-team/keras/issues/41
    get_acts = K.function([model.get_layer(layer).input, K.learning_phase()], [model.get_layer(layer).output,])
    activations = get_acts([X,0])
    return activations

def show_attention(pipeline, x_val):
    vec = pipeline.named_steps['tfidfvectorizer']
    model = pipeline.named_steps['nn']
    x_val_trans = vec.transform([x_val]).toarray()
    activations = get_activations(model, 'attention_vec_1', x_val_trans)[0]
    top_acts = topn(activations, 5)
    attn_words = []
    for k,v in vec.vocabulary_.items():
        if v in top_acts:
            attn_words.append(k)
    return attn_words

def explain(x_val, pred_probs, classes):
    explainer = LimeTextExplainer(class_names=classes)
    exp = explainer.explain_instance(x_val, pred_probs, num_features=5, top_labels=2)
    poss_labels = exp.available_labels()
    exp.save_to_file('./asimov/templates/exp.html')

    # Small hack to remove a phrase that ruins jinja templating
    with open('./asimov/templates/exp.html', 'r') as f:
        filedata = f.read()
    # This string is always there because it comes from 3rd party code
    filedata = filedata.replace('_.templateSettings.interpolate = /{{([\s\S]+?)}}/g;', '')
    # Write the file out again
    with open('./asimov/templates/exp.html', 'w') as f:
        f.write(filedata)

def predict(model_path, x_val):
    froot, model_type = model_path.rsplit('_', 1)
    with open('./models/{0}'.format(model_path), 'rb') as model_f:
        pipeline = pickle.load(model_f)
    model_type = model_type.split('.')[0]
    # Need to load Keras model separately due to pickle bug
    if model_type == 'ffnn':
        model = load_model('./models/{0}_{1}_model.h5'.format(froot, model_type))
        pipeline.steps.append(('nn', model))
    with open('./models/{0}.classes'.format(froot), 'rb') as f:
        classes = pickle.load(f)
    attn = None
    pred = pipeline.predict([x_val])
    if model_type == 'ffnn':
        explain(x_val, pipeline.predict, classes)
        attn = show_attention(pipeline, x_val)
        pred = classes[np.argmax(pred)]
    else:
        explain(x_val, pipeline.predict_proba, classes)
        pred = pred[0]
    return pred, attn

def visualize_cnf_mat(cnf_mat, classes):
    fig = plt.figure(figsize=(10, 10))
    classes = ast.literal_eval(classes.replace(' ', ','))
    nclasses = len(classes)
    cnf_mat = np.matrix(cnf_mat, dtype=float).reshape((nclasses, nclasses))
    # Normalize confusion matrix to scale out of 1
    cm = cnf_mat / cnf_mat.sum(axis=1)
    plt.imshow(cm)
    plt.colorbar()
    tick_marks = np.arange(nclasses)
    plt.xticks(tick_marks, classes, rotation=75)
    plt.yticks(tick_marks, classes)
    plt.xlabel('Predicted label')
    plt.ylabel('Actual label')
    plt.tight_layout()
    return fig
