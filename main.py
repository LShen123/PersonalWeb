import sklearn
from sklearn import datasets
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import LogisticRegression
from sklearn import metrics
from sklearn.metrics import accuracy_score, recall_score, precision_score

iris = datasets.load_iris()
iris_pd = pd.DataFrame(iris.data, columns = iris.feature_names)
iris_pd.head()
tar = pd.get_dummies(iris.target)
print(tar)

x = iris_pd.drop(labels = 'petal width (cm)', axis = 1)
y = iris_pd['petal width (cm)']
X_train, X_test, y_train, y_test = train_test_split(x,y,test_size = 0.2, random_state = 42)
Lin = LinearRegression()
Lin.fit(X_train, y_train)
pred = Lin.predict(X_test)

plt.figure(figsize = (4,4))
plt.scatter(y_test,pred)
plt.plot([0,2.5],[0,2.5],'--k',label="Correct prediction")
plt.axis('tight')
plt.xlabel('True petal width (cm)')
plt.ylabel('Predicted petal width (cm)')
plt.title("Real vs predicted petal widths (cm)")
plt.legend()
plt.tight_layout()
plt.show()