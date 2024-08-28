from flask import Flask, render_template, jsonify, request
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot as plt
import plotly.express as px
import seaborn as sns

import os
from io import BytesIO
import base64
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from model import train_and_predict_linear, train_and_predict_logistic, plot_results
from sklearn.metrics import accuracy_score
import json

app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/about')
def about():
    return render_template('about.html')

@app.route('/gallery')
def gallery():
    return render_template('gallery.html')

@app.route('/regression')
def regression():
    # Load the iris dataset
    iris = load_iris()
    iris_pd = pd.DataFrame(iris.data, columns=iris.feature_names)
    iris_pd['target'] = iris.target
    data = {
        'features' : iris_pd.columns[:-1].tolist(),
        'species' : iris_pd['target'].tolist(),
        'data' : iris_pd.iloc[:,:-1].values.tolist(),
        'species_names': iris.target_names.tolist()
    }

    mean_values = iris_pd.groupby('target').mean()
    radar_data = {
        'labels': iris.feature_names,  # Feature names
        'datasets': [
            {
                'label': 'Setosa',
                'data':  mean_values.loc[0].tolist(),
                'borderColor': 'rgba(255, 99, 132, 1)',
                'backgroundColor': 'rgba(255, 99, 132, 0.2)'
            },
            {
                'label': 'Versicolor',
                'data': mean_values.loc[1].tolist(),
                'borderColor': 'rgba(54, 162, 235, 1)',
                'backgroundColor': 'rgba(54, 162, 235, 0.2)'
            },
            {
                'label': 'Virginica',
                'data': mean_values.loc[2].tolist(),
                'borderColor': 'rgba(75, 192, 192, 1)',
                'backgroundColor': 'rgba(75, 192, 192, 0.2)'
            }
        ]
    }

    # Prepare data for training and prediction using Linear Regression
    _, _, _, y_test_lin, pred_lin = train_and_predict_linear(iris_pd)
    plot_url_lin = plot_results(y_test_lin, pred_lin, 'linear')

    # Prepare data for training and prediction using Logistic Regression
    _, _, _, y_test_log, pred_log = train_and_predict_logistic(iris_pd)
    plot_url_log = plot_results(y_test_log, pred_log, 'logistic')

    # Calculate accuracy for Logistic Regression
    accuracy_log = accuracy_score(y_test_log, pred_log)

    return render_template('regression.html', data=data, radar_data=radar_data, plot_url_lin=plot_url_lin, plot_url_log=plot_url_log, accuracy_log=accuracy_log)

if __name__ == '__main__':
    port = int(os.environ.get("PORT", 5000))
    app.run(host='0.0.0.0', port=port)