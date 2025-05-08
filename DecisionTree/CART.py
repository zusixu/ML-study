"""
Cart树实现
"""
import numpy as np
from collections import defaultdict

# 缺失值统一设置为Nan
NAN = 'Nan'

def type_check(func):
    def wrapper(*args, **kwargs):
        # 获取函数的类型注解
        annotations = func.__annotations__
        # 检查参数类型
        for arg_name, arg_type in annotations.items():
            if arg_name != 'return':
                arg_value = kwargs.get(arg_name) if arg_name in kwargs else args[list(func.__code__.co_varnames).index(arg_name)]
                if not isinstance(arg_value, arg_type):
                    raise TypeError(f"参数 {arg_name} 必须是 {arg_type.__name__} 类型")
        return func(*args, **kwargs)
    return wrapper

class node:
    def __init__(self):
        self.feature = None # 特征
        self.threshold = None # 分割阈值
        self.right = None # 右子树
        self.left = None # 左子树
        self.label = None # 叶节点label
        self.isleaf = False  # 是否为叶节点


class CART:
    def __init__(self):
        self.root = node()

    

    def calGini(self, data: np.ndarray)-> float:
        """
        计算基尼指数
        param:
            data: 训练数据
        return:
            gini: 基尼指数
        """
        numEnts = np.sum(data[:, -2])
        labelCounts = defaultdict(int)
        labellist = set(data[:, -1])
        for label in labellist:
            labelCounts[label] = np.sum(data[data[:, -1] == label, -2])
        gini = 1.0
        for key in labelCounts:
            prob = float(labelCounts[key])/numEnts
            gini -= prob*prob
        return gini

    def calMse(self, data: np.ndarray, y_pred: float)-> float:
        """
        计算均方误差
        param:
            data: 训练数据
            y_pred: 预测值
        return:
            mse: 均方误差
        """
        numEnts = np.sum(data[:, -2])
        mse = np.sum((data[:, -1] - y_pred)**2*data[:,-2]) / numEnts
        return mse

    def splitDataSetWithNull(self, data: np.ndarray, attrIndex: int, threshold, attrs_type: str)-> tuple[np.ndarray, np.ndarray]:
        """
        根据属性阈值划分含有缺失值的数据集
        param:
            data: 训练数据
            attrIndex: 特征索引
            threshold: 分割阈值
            attrs_type: 特征类型
        return: 
            data_left: 左子树数据
            data_right: 右子树数据
        """
        
        data_nan = data[data[:, attrIndex] == NAN]
        # 连续特征
        if attrs_type == 'num':
            left = data[data[:, attrIndex] <= threshold]
            right = data[data[:, attrIndex] > threshold]
        # 离散特征
        elif attrs_type == 'cat':
            left = data[data[:, attrIndex] == threshold]
            right = data[data[:, attrIndex] != threshold]
        else:
            raise ValueError("attrs_type must be 'num' or 'cat'")
        # 输出结果
        data_left = np.concatenate((left, data_nan), axis=0)
        data_right = np.concatenate((right, data_nan), axis=0)
        data_left = np.delete(data_left, attrIndex, axis=1)
        data_right = np.delete(data_right, attrIndex, axis=1)
        return data_left, data_right
    def isSame(self, data: np.ndarray)-> bool:
        """
        判断数据集属性取值是否一致
        param:
            data: 训练数据
        return:
            True or False
        """
        # 获取特征列数（排除最后两列：标签列和权重列）
        n_features = data.shape[1] - 2
        
        # 如果没有特征列，返回True
        if n_features <= 0:
            return True
            
        # 取第一行作为参考
        first_row = data[0, :n_features]
        
        # 比较每一行与第一行是否相同
        for i in range(1, data.shape[0]):
            current_row = data[i, :n_features]
            # 如果任何一行与第一行不同，返回False
            if not np.array_equal(first_row, current_row):
                return False
                
        # 所有行都相同，返回True
        return True

    @type_check
    def buildTree(self, task: str, data: np.ndarray, attrs: list, attrs_type: list)-> node:
        """
        递归构建CART树
        param:
            task: 任务类型，分类(classfication)或回归(regression)
            data: 训练数据
            attrs: 特征列表
            attrs_type: 特征类型
        return:
            node: 树节点
        """
        classlist = data[:, -1]
        # 类别完全相同，停止划分
        if len(set(classlist)) == 1:
            node = Node()
            node.label = classlist[0]
            node.isleaf = True
            return node
        # 所有特征均已使用，返回出现次数最多的类别
        if len(attrs) == 0:
            node = Node()
            node.label = max(set(classlist), key=classlist.tolist().count)
            node.isleaf = True
            return node
        # 所有样本相同，返回出现次数最多的类别
        if self.isSame(data):
            node = Node()
            node.label = max(set(classlist), key=classlist.tolist().count)
            node.isleaf = True
            return node

    def fit(self, task: str, data: np.ndarray, attrs: list, attrs_type: list)-> node:
        """
        构建CART树
        param:
            task: 任务类型，分类(classfication)或回归(regression)
            data: 训练数据
            attrs: 特征列表
            attrs_type: 特征类型
        return: 
            self.root: 根节点
        """
        # 添加计数权重
        data = np.concatenate((data, np.ones((len(data), 1))), axis=1)
        # 构建CART树
        self.root = self.buildTree(task, data, attrs, attrs_type)
        return self.root


    def predict(self, X):
        pass