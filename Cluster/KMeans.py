"""实现加权的kmeans算法"""
import random
import numpy as np
from sklearn.datasets import make_classification


class KMeans:
    def __init__(self, k, max_iter=100):
        self.k = k
        self.max_iter = max_iter
        self.centroids = None

    def fit(self, X, y=None):
        # 随机选择k个初始质心
        n_samples, n_features = X.shape
        self.centroids = X[random.sample(range(n_samples), self.k)]

        for i in range(self.max_iter):
            # 计算每个样本到质心的距离
            distances = np.sqrt(((X - self.centroids[:, np.newaxis]) ** 2).sum(axis=2))

            # 分配样本到最近的质心
            labels = np.argmin(distances, axis=0)

            # 更新质心
            new_centroids = np.array([X[labels == i].mean(axis=0) for i in range(self.k)])
            # 计算当代准确率
            acc = self.accuracy(y, labels)
            print(f"第{i}代训练准确率：{acc}")
            # 如果质心没有变化，则停止迭代
            if np.all(self.centroids == new_centroids):
                print("质心没有变化，停止迭代")
                break

            self.centroids = new_centroids

    def accuracy(self, y_true, y_pred):
        # 计算准确率
        correct = np.sum(y_true == y_pred)
        total = len(y_true)
        acc = correct / total
        return acc

    def predict(self, X):
        # 计算每个样本到质心的距离
        distances = np.sqrt(((X - self.centroids[:, np.newaxis]) ** 2).sum(axis=2))

        # 分配样本到最近的质心
        labels = np.argmin(distances, axis=0)

        return labels


if __name__ == '__main__':
    # 生成随机数据
    X, y = make_classification(n_samples=100, n_features=2, n_informative=2, n_redundant=0, n_clusters_per_class=1,
                               random_state=42)

    # 训练kmeans模型
    kmeans = KMeans(k=2)
    kmeans.fit(X, y=y)

    # 预测  
    y_pred = kmeans.predict(X)

    # 准确率
    accuracy = np.mean(y_pred == y)
    print("准确率:", accuracy)
