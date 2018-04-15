# Copyright Nicholas Larus-Stone 2018.

class BaseConfig(object):
    DEBUG = False
    TESTING = False
    UPLOAD_FOLDER = './uploads'
    MODEL_FOLDER = './models'
    TRAINING_EXTENSIONS = set(['txt', 'csv'])
    MODEL_EXTENSIONS = set(['pkl'])

class DevelopmentConfig(BaseConfig):
    DEBUG = True
    TESTING = True
    TEMPLATES_AUTO_RELOAD = True

class ProductionConfig(BaseConfig):
    DEBUG = False
    TESTING = False

class TestingConfig(BaseConfig):
    DEBUG = False
    TESTING = True
