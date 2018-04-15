# Copyright Nicholas Larus-Stone 2018.
from flask import Flask

app = Flask(__name__, instance_relative_config=True)
app.config.from_object('config.ProductionConfig')

from asimov import views

# Initialize application
def create_app():
    return app
