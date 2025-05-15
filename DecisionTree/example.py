"""使用sklearn实现CART决策树分类器"""
from sklearn.tree import DecisionTreeClassifier
import numpy as np

def train_cart_classifier(X, y):
    """
    使用CART算法训练决策树分类器
    参数:
        X: 特征数据，二维数组
        y: 标签数据，一维数组
    返回:
        clf: 训练好的决策树分类器
    """
    clf = DecisionTreeClassifier(criterion='gini')  # CART默认使用gini
    clf.fit(X, y)
    return clf

def predict_cart_classifier(clf, X_test):
    """
    使用训练好的CART决策树分类器进行预测
    参数:
        clf: 已训练的决策树分类器
        X_test: 测试特征数据
    返回:
        y_pred: 预测标签
    """
    y_pred = clf.predict(X_test)
    return y_pred

# 示例数据
X = np.array([[1, 2], [3, 4], [5, 6], [7, 8]])
y = np.array([0, 1, 0, 1])

clf = train_cart_classifier(X, y)
y_pred = predict_cart_classifier(clf, X)
print("预测结果：", y_pred)