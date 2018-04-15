# Copyright Nicholas Larus-Stone 2018.
from io import BytesIO
import os
from flask import render_template, request, send_file, abort, make_response
from werkzeug.utils import secure_filename
from asimov import app
from .constants import model_types
from .models import train_model, visualize_cnf_mat, predict
from .utils import file_allowed

@app.route('/', methods=['GET'])
def home():
    return render_template('index.html')

@app.route('/fig/<cnfmat>/<classes>', methods=['GET'])
def fig(cnfmat, classes):
    fig = visualize_cnf_mat(cnfmat, classes)
    img = BytesIO()
    fig.savefig(img)
    img.seek(0)
    return send_file(img, mimetype='image/png')

@app.route('/train', methods=['GET'])
def train():
    # List valid text files that have been uploaded
    uploaded_files = filter(lambda x: file_allowed(x, app.config['TRAINING_EXTENSIONS']), 
            os.listdir(app.config['UPLOAD_FOLDER']))    
    return render_template('train.html', files=uploaded_files, model_types=model_types)

@app.route('/train', methods=['POST'])
def upload_train_file():
    # User selected already uploaded file
    if 'Selected' in request.form:
        new = False
        filename = request.form["uploadedFilesSelect"]
        if filename == '':
            return make_response(render_template('error.html', 
                error_message='No file selected'), 400)
    elif 'Upload' in request.form:
        new = True
        # Ensure user actually uploaded a file
        if 'trainFile' not in request.files:
            return make_response(render_template('error.html', 
                error_message='No file uploaded'), 400)
        f = request.files['trainFile']
        if not f or f.filename == '':
            return make_response(render_template('error.html', 
                error_message='File needs a name'), 400)
        # Ensure the file is of the right type for text
        if file_allowed(f.filename, app.config['TRAINING_EXTENSIONS']):
            filename = secure_filename(f.filename)
        else:
            return make_response(render_template('error.html', 
                error_message='File needs to be of the type: {0}'.format(app.config['TRAINING_EXTENSIONS'])), 
                400)
    else:
        return make_response(render_template('error.html', 
            error_message='Navigation error'), 404)

    model_type = request.form["modelTypeSelect"]
    froot = filename.rsplit('.', maxsplit=1)[0].split('/')[-1]
    # Save to our server for easier usage later
    fpath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    if not os.path.exists(fpath):
        f.save(fpath)

    # While training on file, also check for valid formatting
    train = train_model(fpath, froot, model_type, new=new)
    if type(train) == str:
            return make_response(render_template('error.html', 
                error_message=train), 400)
    else:
        return render_template('train_results.html', cnf_mat=train['cnf_mat'],
            classes=train['classes'], acc=train['accuracy'],
            precision=train['precision'], recall=train['recall'],
            fscore=train['fscore'])

@app.route('/test', methods=['GET', 'POST'])
def test():
    trained_models = filter(lambda x: file_allowed(x, app.config['MODEL_EXTENSIONS']), 
            os.listdir(app.config['MODEL_FOLDER']))
    # If it's a get request, don't need to render any prediction
    if request.method == 'POST':
        text = request.form['testText']
        model_path = request.form['trainedModelsSelect']
        # attn is only used if the model is a neural network
        # The predict method writes the prediction out to a file
        attn = predict(model_path, text)
        return render_template('test.html', prediction=True, models=trained_models, attention=attn)
    else:
        return render_template('test.html', prediction=False, models=trained_models, attention=None)
