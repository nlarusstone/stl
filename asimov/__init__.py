from flask import Flask

app = Flask(__name__, instance_relative_config=True)
app.config.from_object('config.DevelopmentConfig')
# Load instance specific config vars
app.config.from_pyfile('config.py', silent=True)

from asimov import views

# Initialize application
def create_app():
    return app
