from flask import Flask,render_template,request

import FeatureExtraction
import pickle

app = Flask(__name__)

@app.route('/')