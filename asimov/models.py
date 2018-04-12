import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import matplotlib.pyplot as plt
import pickle
import os
from io import BytesIO
import ast
max_words = 1000

def train_model(file_path, froot):
    x, y = [], []
    if os.path.exists('./data/x_train_{0}.txt'.format(froot)):
    with open('./data/x_train_{0}.txt'.format(froot), 'rb') as f:
    x_train_tok = pickle.load(f)
    with open('./data/x_val_{0}.txt'.format(froot), 'rb') as f:
    x_val_tok = pickle.load(f)
    with open('./data/y_train_{0}.txt'.format(froot), 'rb') as f:
    y_train = pickle.load(f)
    with open('./data/y_val_{0}.txt'.format(froot), 'rb') as f:
    y_val = pickle.load(f)
    else:
    with open(file_path, 'rb') as file_obj:
    for line in file_obj:
    l = line.decode("utf-8")
    line_list = l.split('\t', 1)
    label, text = line_list
    x.append(text)
    y.append(label)
    x = np.array(x)

 63     #    pickle.dump(vectorizer, f)
 64 
 65     # Save model for test usage
 66     #file_name = file_path.rsplit('.', maxsplit=1)[0].rsplit('/', maxsplit=1)[-1]
 67     model_path = './models/{0}_{1}.pkl'.format(froot, model_type)
 68     #if os.path.exists(model_path):
 69     #    model_f = open(model_path, 'rb')
 70     #    nb = pickle.load(model_f)
 71     #else:
 72     #    nb = naive_bayes(x_train_tok, y_train)
 73     #model_f = open(model_path, 'wb')
 74     with open(model_path, 'wb') as model_f:
 75         pickle.dump(pipeline, model_f)
 76     classes = np.unique(y_test)
 77     with open('./models/{0}_classes.pkl'.format(froot), 'wb') as f:
 78         pickle.dump(classes, f)
 79     #model_f.close()
 80 
 81     # Get training metrics
 82     preds = pipeline.predict(x_test)
 83     acc = accuracy_score(y_test, preds)
 84     precision, recall, fscore, supp = map(lambda x: np.round(x, decimals=3), precision_recall_fscore_support(y_test, preds))
 85     cnf_mat = confusion_matrix(y_test, preds)
 86     return (None, {'cnf_mat': cnf_mat, 'classes': classes, 'accuracy': acc,
 87             'precision': precision, 'recall': recall, 'fscore': fscore})
 88 
 89 def vectorize():
 90     return TfidfVectorizer(lowercase=False)
 91     #tokenizer = Tokenizer(num_words=max_words)
 92     #tokenizer.fit_on_texts(x)
 93     #return tokenizer.texts_to_matrix(x, mode='count')
 94 
 95 def naive_bayes():
 96     mnb = MultinomialNB()
 97     return mnb
 98 
 99 def ffnn():
100     model = Sequential()
101     model.add(Dense(512, input_shape=(max_words,)))
102     model.add(Activation('relu'))
103     model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
104     return model
105 
106 def rnn():
107     pass
108 
109 def rnn_attn():
110     pass
111 
112 def explain(x_val, pred_probs, classes):
113     from lime.lime_text import LimeTextExplainer
114     explainer = LimeTextExplainer(class_names=classes.tolist())
115     exp = explainer.explain_instance(x_val, pred_probs, num_features=5, top_labels=2)
116     poss_labels = exp.available_labels()

    def predict()

def visualize_cnf_mat(cnf_mat, classes):
    norm = True
    fig = plt.figure(figsize=(10, 10))
    classes = ast.literal_eval(classes.replace(' ', ','))
    nclasses = len(classes)
    cnf_mat = np.matrix(cnf_mat, dtype=float).reshape((nclasses, nclasses))
    if norm:
        cm = cnf_mat / cnf_mat.sum(axis=1)
    else:
        cm = cnf_mat
    plt.imshow(cm)
    plt.colorbar()
    tick_marks = np.arange(nclasses)
    plt.xticks(tick_marks, classes, rotation=75)
    plt.yticks(tick_marks, classes)
    plt.tight_layout()
    return fig#mpld3.fig_to_html(fig)


