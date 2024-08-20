from flask import Flask, render_template, request
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
from io import BytesIO
import base64
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from model import train_and_predict_linear, train_and_predict_logistic, plot_results
from sklearn.metrics import accuracy_score

app = Flask(__name__)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/about')
def about():
    return render_template('about.html')

@app.route('/gallery')
def gallery():
    return render_template('gallery.html')

@app.route('/model')
def index():
    # Load the iris dataset
    iris = load_iris()
    iris_pd = pd.DataFrame(iris.data, columns=iris.feature_names)
    iris_pd['target'] = iris.target

    # Prepare data for training and prediction using Linear Regression
    _, _, _, y_test_lin, pred_lin = train_and_predict_linear(iris_pd)
    plot_url_lin = plot_results(y_test_lin, pred_lin, 'linear')

    # Prepare data for training and prediction using Logistic Regression
    _, _, _, y_test_log, pred_log = train_and_predict_logistic(iris_pd)
    plot_url_log = plot_results(y_test_log, pred_log, 'logistic')

    # Calculate accuracy for Logistic Regression
    accuracy_log = accuracy_score(y_test_log, pred_log)

    return render_template('regression.html', plot_url_lin=plot_url_lin, plot_url_log=plot_url_log, accuracy_log=accuracy_log)

if __name__ == '__main__':
    port = int(os.environ.get("PORT", 5000))
    app.run(host='0.0.0.0', port=port)