import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.datasets import make_classification
"""
线性回归模型学习使用，实现了线性回归和逻辑回归的预测和可视化
"""
# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
plt.rcParams['axes.unicode_minus'] = False    # 用来正常显示负号

def generate_regression_data(n_samples=100, noise=0.3):
        """
        生成用于回归任务的示例数据
        
        参数:
        n_samples: int, 样本数量
        noise: float, 噪声水平
        
        返回:
        X: ndarray, 特征数据
        y: ndarray, 目标变量
        """
        # 生成特征数据
        X = np.linspace(0, 10, n_samples).reshape(-1, 1)
        
        # 生成目标变量（带有一些非线性特征和噪声）
        y = 2 * X + 1 + np.sin(X) + noise * np.random.randn(n_samples, 1)
        
        return X, y
def generate_classification_data(n_samples=100, n_features=2, random_state=42):
    """
    生成用于二分类任务的示例数据

    Args:
        n_samples (int): 样本数量.
        n_features (int): 特征数量.
        random_state (int): 随机种子，保证数据可复现.

    Returns:
        tuple: 包含特征数据 X (ndarray) 和目标标签 y (ndarray).
               y 的标签为 0 或 1.
    """
    X, y = make_classification(n_samples=n_samples, n_features=n_features,
                               n_informative=2, n_redundant=0, n_clusters_per_class=1,
                               flip_y=0.1, class_sep=1.5, random_state=random_state)
    # 将 y 转换为 (n_samples, 1) 的形状
    return X, y.reshape(-1, 1)


class LinearRegression:
    def __init__(self, learning_rate=0.01, num_iterations=1000):
        self.learning_rate = learning_rate
        self.num_iterations = num_iterations
        self.weights = None


    
    def sigmoid(self,x):
        return 1/(1+np.exp(-x))

    def linear_regression(self, X_train, y_train):
        """
        使用线性回归拟合训练数据

        参数:
        X_train: ndarray, 训练特征数据
        y_train: ndarray, 训练目标变量
        """
        b = np.ones((X_train.shape[0],1))
        b.reshape(-1,1)
        X_train = np.hstack((X_train,b))
        self.weights = np.linalg.inv(X_train.T.dot(X_train)).dot(X_train.T).dot(y_train)

    def linear_classification(self, X_train, y_train, circle, alpha):
        """
        使用线性分类拟合训练数据

        参数:
        X_train: ndarray, 训练特征数据
        y_train: ndarray, 训练目标变量
        """
        b = np.ones((X_train.shape[0],1))
        b.reshape(-1,1)
        # X_train = np.hstack((X_train,b))
        X_train = np.concatenate((X_train,b),axis=1)
        self.weights = np.zeros((X_train.shape[1], 1))

        dw = np.zeros((X_train.shape[1],1))
        for i in range(circle):
            z = X_train.dot(self.weights)
            h = self.sigmoid(z)
            dw = (1 / X_train.shape[0])*X_train.T.dot(h - y_train)
            self.weights -= alpha * dw 
            # dw = X_train.T.dot(self.sigmoid(X_train.dot(self.weights)) - y_train) 
            # self.weights -= alpha * dw / X_train.shape[0]
        

    def _add_intercept(self, X):
        """
        在特征矩阵 X 的第一列添加截距项（全为 1）

        Args:
            X (ndarray): 特征矩阵 (n_samples, n_features).

        Returns:
            ndarray: 添加了截距项的特征矩阵 (n_samples, n_features + 1).
        """
        intercept = np.ones((X.shape[0], 1))
        intercept.reshape(-1,1)
        return np.hstack((X,intercept))
        # return np.concatenate((X,intercept),axis=1)

    def predict_proba(self, X):
        """
        预测样本属于类别 1 的概率

        Args:
            X (ndarray): 特征数据 (n_samples, n_features).

        Returns:
            ndarray: 每个样本属于类别 1 的概率 (n_samples, 1).
        """
        if self.weights is None:
            raise ValueError("模型尚未训练！请先调用 fit 方法。")
        X_b = self._add_intercept(X)
        z = X_b.dot(self.weights)
        return self.sigmoid(z)

    def plot_regression_line(X, y, w):
        """
        绘制回归直线

        参数:
        X: ndarray, 特征数据
        y: ndarray, 目标变量
        w: ndarray, 回归系数    
        """
        b = np.ones((X.shape[0],1))
        b.reshape(-1,1)
        X_train = np.hstack((X,b))
        plt.figure(figsize=(10, 6))
        plt.scatter(X, y, color='blue', alpha=0.5, label='数据点')
        plt.plot(X, X_train.dot(w), color='red', linewidth=2, label='回归直线')
        plt.xlabel('X')
        plt.ylabel('y')
        plt.title('线性回归拟合')
        plt.legend()
        plt.grid(True)
        plt.show()  

    def plot_decision_boundary(self, X, y, title="逻辑回归决策边界"):
        """
        绘制二维数据的决策边界

        Args:
            X (ndarray): 特征数据 (n_samples, 2).
            y (ndarray): 真实标签 (n_samples, 1).
            model (LogisticRegression): 训练好的逻辑回归模型.
            title (str): 图表标题.
        """
        if X.shape[1] != 2:
            print("只能为二维特征数据绘制决策边界。")
            return

        plt.figure(figsize=(10, 6))

        # 绘制数据点
        plt.scatter(X[y.flatten() == 0][:, 0], X[y.flatten() == 0][:, 1], color='blue', alpha=0.7, label='类别 0')
        plt.scatter(X[y.flatten() == 1][:, 0], X[y.flatten() == 1][:, 1], color='red', alpha=0.7, label='类别 1')

        # 创建网格来绘制决策边界
        x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
        y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
        xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.02),
                        np.arange(y_min, y_max, 0.02))

        # 对网格中的每个点进行预测
        grid_points = np.c_[xx.ravel(), yy.ravel()]
        Z = self.predict(grid_points)
        Z = Z.reshape(xx.shape)

        # 绘制等高线图（决策边界）
        plt.contourf(xx, yy, Z, alpha=0.3, cmap=plt.cm.coolwarm)

        plt.xlabel('特征 1')
        plt.ylabel('特征 2')
        plt.title(title)
        plt.legend()
        plt.grid(True)
        plt.show()

    def predict(self, X, threshold=0.5):
        """
        预测样本的类别标签

        Args:
            X (ndarray): 特征数据 (n_samples, n_features).
            threshold (float): 概率阈值，用于区分类别 0 和 1.

        Returns:
            ndarray: 预测的类别标签 (n_samples, 1), 值为 0 或 1.
        """
        probabilities = self.predict_proba(X)
        return (probabilities >= threshold).astype(int)


if __name__ == "__main__":
    # 生成数据
    X, y = generate_classification_data(n_samples=200, n_features=2, random_state=42)

    # 划分训练集和测试集
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    # 打印数据形状
    print("训练集形状:", X_train.shape, y_train.shape)
    print("测试集形状:", X_test.shape, y_test.shape)
    model = LinearRegression()
    model.linear_classification(X_train, y_train,20000,0.1)
    # 预测测试集
    y_pred = model.predict(X_test)

    
    # 分类准确率
    accuracy = np.mean(y_pred == y_test)
    print("分类准确率:", accuracy)
    # 预测可视化
    model.plot_decision_boundary(X, y, title="逻辑回归决策边界")
    
    
    # 绘制回归直线
    # plot_regression_line(X, y, w)