# Supervised Text Learning Platform
Can be accessed at https://asimov-cc.herokuapp.com/
This web application allows anyone to train a machine learning model on a collection of text documents of the following form:
label \t text \n
label \t text \n ...

Once the model is trained, it is stored on the server and can be used to predict the label of a new piece of text.

The free Heroku instance has only a small amount of memory and cannot handle large quantities of text.
Additionally, there is a hard timeout limit that can sometimes interfere with predictions.
Thus, if training on large text corpuses, it is recommend to run the app locally.
To run this app locally, simply set your FLASK_APP environment variable to point to run.py in the root directory and then call flask run.

## Current models
Currently, the platform only supports very simple machine learning models for document classification.
Users can select either a Multinomial Naive Bayes or a 2-layer Neural Net as their model.

## Interpretability
A key focus of this platform is understanding why our text classification models make predictions as they due.
Despite using typically uninterpretable methods, we are able to provide explanations for our predictions.
We use LIME (https://arxiv.org/pdf/1602.04938.pdf) to explain the predictions using local features.
Additionally, for the neural models, we use attention to explain which features are the most important.

## Future work
Allowing users to choose hyperparameters
Adding new models (e.g. RNNs)
User accounts
Improved user interaction and design
