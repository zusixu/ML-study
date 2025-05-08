import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.datasets import make_classification

"""
LDA模型实现
"""

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
plt.rcParams['axes.unicode_minus'] = False    # 用来正常显示负号

class LDA:
    def __init__(self, n_components=None):
        self.n_components = n_components
        self.mean_vectors = []
        self.cov_matrices = []
        self.class_labels = []

    def covariance(self, X):
        mean = np.mean(X, axis=0)
        n_samples = X.shape[0]
        covariance = (X - mean).T.dot(X - mean) / (n_samples - 1)
        return covariance

    def fit(self, X, y):
        self.class_labels = np.unique(y)
        row_indices = [np.where(y == label)[0] for label in self.class_labels]
        for i, label in enumerate(self.class_labels):
            X_class = X[row_indices[i]]
            self.mean_vectors.append(np.mean(X_class, axis=0))
            self.cov_matrices.append(self.covariance(X_class))
        try:
            self.w = np.linalg.inv(self.cov_matrices[0] + self.cov_matrices[1]).dot(self.mean_vectors[0] - self.mean_vectors[1])
        except np.linalg.LinAlgError:
            print("矩阵不可逆")
            self.w = np.linalg.pinv(self.cov_matrices[0] + self.cov_matrices[1]).dot(self.mean_vectors[0] - self.mean_vectors[1])

    def predict(self, X):
        proj_mean_0 = self.mean_vectors[0].dot(self.w)
        proj_mean_1 = self.mean_vectors[1].dot(self.w)
        # 计算决策阈值（两个投影均值的中点）
        threshold = (proj_mean_0 + proj_mean_1) / 2
        y_pred = np.dot(X, self.w)
        if proj_mean_0 > proj_mean_1:
            return np.where(y_pred >= threshold, self.class_labels[0], self.class_labels[1])
        else:
            return np.where(y_pred < threshold, self.class_labels[0], self.class_labels[1])

    def plot_decision_boundary(self, X, y, title):
        plt.figure()
        plt.scatter(X[:, 0], X[:, 1], c=y, cmap='viridis')
        plt.title(title)
        plt.xlabel('特征1')
        plt.ylabel('特征2')
        plt.show()

    def plot_projection(self, X, y, title):
        """
        绘制数据在LDA投影方向上的分布。

        Args:
            X: 数据点。
            y: 数据标签。
            title: 图表标题。
        """
        if self.w is None:
            raise RuntimeError("模型尚未训练，请先调用 fit 方法。")

        X_proj = X.dot(self.w)
        plt.figure(figsize=(8, 4))

        for label in self.class_labels:
            # 绘制每个类别投影后的直方图
            plt.hist(X_proj[y == label], bins=20, alpha=0.6, label=f'类别 {label}')

        plt.title(title)
        plt.xlabel('投影到 LDA 方向的值')
        plt.ylabel('频数')
        plt.legend()
        plt.grid(True)
        plt.show()


if __name__ == '__main__':
    # 生成用于分类任务的示例数据
    X, y = make_classification(n_samples=2000, n_features=2, n_informative=2, n_redundant=0,
                               n_classes=2, n_clusters_per_class=1, random_state=42)

    # 划分训练集和测试集        
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # 打印数据形状
    print("训练集形状:", X_train.shape, y_train.shape)
    print("测试集形状:", X_test.shape, y_test.shape)

    # 初始化LDA模型
    lda_model = LDA()

    # 训练模型
    lda_model.fit(X_train, y_train)

    # 预测测试集
    y_pred = lda_model.predict(X_test)

    # 分类准确率
    accuracy = np.mean(y_pred == y_test)
    print("分类准确率:", accuracy)

    # 预测可视化
    lda_model.plot_decision_boundary(X_test, y_test, title="LDA决策边界")
    # 投影可视化
    lda_model.plot_projection(X_test, y_test, title="LDA投影")