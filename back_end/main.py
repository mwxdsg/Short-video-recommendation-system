# -*- coding: utf-8 -*-

from flask import Flask
from isort import ImportKey
from play import video_bp
from user import user_bp
from search import search_bp
from forecast import admin_bp
from data_load import load_video_database

app = Flask(__name__)



app.register_blueprint(video_bp)
app.register_blueprint(user_bp)
app.register_blueprint(search_bp)
app.register_blueprint(admin_bp)

if __name__ == "__main__":
    app.run(debug=True, port=5000)
