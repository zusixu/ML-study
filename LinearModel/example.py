"""
使用sklearn的线性回归模型
"""
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.datasets import make_classification

x = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
y = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]

model = LinearRegression()
model.fit(np.array(x).reshape(-1, 1), np.array(y))

print(model.predict(np.array(11).reshape(-1, 1)))

lda = LinearDiscriminantAnalysis()
x, y = make_classification(n_samples=2000, n_features=2, n_informative=2, n_redundant=0,
                               n_classes=2, n_clusters_per_class=1, random_state=42)

lda.fit(x, y)

print(lda.predict(np.array([1, 2]).reshape(1, -1)))