import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.ensemble import RandomForestClassifier
import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot as plt
from io import BytesIO
import base64

def train_and_predict_linear(iris_pd):
    x = iris_pd.drop(labels='petal width (cm)', axis=1)
    y = iris_pd['petal width (cm)']
    X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=42)  # Use 30% test size for more samples
    Lin = LinearRegression()
    Lin.fit(X_train, y_train)
    pred = Lin.predict(X_test)
    return X_train, X_test, y_train, y_test, pred

def train_and_predict_logistic(iris_pd):
    # Use the original target for classification
    x = iris_pd.drop(labels='target', axis=1)
    y = iris_pd['target']
    X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=42)  # Use 30% test size for more samples
    Log = LogisticRegression(max_iter=200, multi_class='ovr')
    Log.fit(X_train, y_train)
    pred = Log.predict(X_test)
    return X_train, X_test, y_train, y_test, pred



def plot_results(y_test, pred, plot_type):
    plt.figure(figsize=(6, 6))
    if plot_type == 'linear':
        plt.scatter(y_test, pred)
        plt.plot([min(y_test), max(y_test)], [min(y_test), max(y_test)], '--k', label="Correct prediction")
        plt.xlabel('True petal width (cm)')
        plt.ylabel('Predicted petal width (cm)')
        plt.title("Real vs predicted petal widths (cm)")
    elif plot_type == 'logistic':
        plt.scatter(y_test, pred)
        plt.plot([min(y_test), max(y_test)], [min(y_test), max(y_test)], '--k', label="Correct prediction")
        plt.xlabel('True class')
        plt.ylabel('Predicted class')
        plt.title("Actual vs predicted classifications")
    plt.legend()
    plt.tight_layout()

    # Save the plot to a BytesIO object
    buf = BytesIO()
    plt.savefig(buf, format='png')
    buf.seek(0)
    plot_url = base64.b64encode(buf.getvalue()).decode('utf8')
    plt.close()
    return plot_url