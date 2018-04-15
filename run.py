# Copyright Nicholas Larus-Stone 2018.
from asimov import create_app

app = create_app()
#app.config["SEND_FILE_MAX_AGE_DEFAULT"] = 30
app.secret_key = 'ASIMOV-CODING-CHALLENGE'
app.config['SESSION_TYPE'] = 'filesystem'

if __name__ == "__main__":
    app.run()
