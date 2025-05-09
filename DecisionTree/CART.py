"""
Cart树实现
"""
from scipy.stats import f



import pandas as pd
import numpy as np
from math import inf
from collections import defaultdict
import copy

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

class Node:
    def __init__(self):
        self.feature = None # 特征
        self.threshold = None # 分割阈值
        self.right = None # 右子树
        self.left = None # 左子树
        self.label = None # 叶节点label
        self.isleaf = False  # 是否为叶节点


class CART:
    def __init__(self):
        self.root = Node()

    

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

    def calMse(self, data: np.ndarray)-> tuple[float,float]:
        """
        计算均方误差
        param:
            data: 训练数据
        return:
            mse: 均方误差
            y_pred: 预测值
        """
        numEnts = np.sum(data[:, -2])
        y_pred = np.sum(data[:, -1]*data[:, -2]) / numEnts
        mse = np.sum((data[:, -1] - y_pred)**2*data[:,-2]) / numEnts
        return mse, y_pred


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
        data_left = data_left[np.where(data_left[:, attrIndex] == NAN), -2] * np.sum(left[:, -2]) / (np.sum(left[:, -2]) + np.sum(right[:, -2]))
        data_right = np.concatenate((right, data_nan), axis=0)
        data_right = data_right[np.where(data_right[:, attrIndex] == NAN), -2] * np.sum(right[:, -2]) / (np.sum(left[:, -2]) + np.sum(right[:, -2]))
        # 删除特征列
        if attrs_type == 'cat':
            data_left = np.delete(data_right, attrIndex, axis=1)
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

    def chooseBestValueandThreshold(self, task: str, data: np.ndarray, index: int, attrs_type: list)-> tuple[float, str or float]:
        """
        选择最优划分属性
        param:
            task: 任务类型，分类(classfication)或回归(regression)
            data: 训练数据
            index: 特征索引
            attrs_type: 特征类型
        return:
            bestthreshold: 最优划分阈值
            bestcalmetric: 最优划分指标
        """
        uniqueVals = list(set(data[:, index]))
        bestthreshold = None
        bestcalmetric = inf
        if NAN in uniqueVals:
            uniqueVals.remove(NAN)
            subdata = data[np.where(data[:, index] != NAN)]
        else:
            subdata = copy.copy(data)
        # 连续特征
        if attrs_type[index] == 1:
            # 只有一个取值，只有左子数没有右子树
            if len(uniqueVals) == 1:
                subdata_left = subdata[np.where(subdata[:, index] <= uniqueVals[0])]
                if task == 'classification':
                    bestthreshold = uniqueVals[0]
                    bestcalmetric = self.calGini(subdata_left)
                elif task == 'regression':
                    bestcalmetric, bestthreshold = self.calMse(subdata_left)
            else:
                for j in range(len(uniqueVals)-2):
                        threshold = (uniqueVals[j] + uniqueVals[j+1]) / 2
                        subdata_left = subdata[np.where(subdata[:, index] <= threshold)]
                        subdata_right = subdata[np.where(subdata[:, index] > threshold)]
                        if task == 'classification':
                            calmetric = self.calGini(subdata_left) + self.calGini(subdata_right)
                        elif task == 'regression':
                            calmetric, _ = self.calMse(subdata_left) + self.calMse(subdata_right)
                        if calmetric < bestcalmetric:
                            bestcalmetric = calmetric
                            bestthreshold = threshold
        # 离散特征
        elif attrs_type[index] == 0:
            for j in range(len(uniqueVals)):
                threshold = uniqueVals[j]
                subdata_left = subdata[np.where(subdata[:, index] == uniqueVals[j])]
                subdata_right = subdata[np.where(subdata[:, index] != uniqueVals[j])]
                if task == 'classification':
                    calmetric = self.calGini(subdata_left) + self.calGini(subdata_right)
                elif task =='regression':
                    calmetric, _ = self.calMse(subdata_left) + self.calMse(subdata_right)
                if calmetric < bestcalmetric:
                    bestcalmetric = calmetric
                    bestthreshold = threshold
        return bestcalmetric, bestthreshold

    def chooseBestFeature(self, task: str, data: np.ndarray, attrs_type: list)-> tuple[float, int]:
        """
        选择最优划分属性
        param:
            task: 任务类型，分类(classfication)或回归(regression)
            data: 训练数据
            attrs_type: 特征类型
        return:
            bestthreshold: 最优划分阈值
            bestfeatureIndex: 最优划分属性索引
        """
        featureNum = data.shape[1] - 2
        threshold = 0
        bestfeatureIndex = 0
        bestthreshold = None
        metric = 0
        bestmetric = inf

        for i in range(featureNum):
            metric, threshold = self.chooseBestValueandThreshold(task, data, i, attrs_type)
            if metric < bestmetric:
                bestmetric = metric
                bestfeatureIndex = i
                bestthreshold = threshold
        return bestthreshold, bestfeatureIndex
           
                    
    @type_check
    def buildTree(self, task: str, data: np.ndarray, attrs: list, attrs_type: list)-> Node:
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

        # 选择最优划分属性
        bestthreshold, bestfeatureIndex = self.chooseBestFeature(task, data, attrs_type)
        # 创建分支节点
        node = Node()
        node.feature = attrs[bestfeatureIndex]
        node.threshold = bestthreshold

        # 离散特征
        if attrs_type[bestfeatureIndex] == 0:
            # 划分数据集
            data_left, data_right = self.splitDataSetWithNull(data, bestfeatureIndex, bestthreshold, "cat")
            # 递归构建子树
            attrs_left = attrs[:bestfeatureIndex] + attrs[bestfeatureIndex+1:]
            attrs_type_left = attrs_type[:bestfeatureIndex] + attrs_type[bestfeatureIndex+1:]
            node.left = self.buildTree(task, data_left, attrs_left, attrs_type_left)
            node.right = self.buildTree(task, data_right, attrs, attrs_type)

        # 连续特征
        elif attrs_type[bestfeatureIndex] == 1:
            # 划分数据集
            data_left, data_right = self.splitDataSetWithNull(data, bestfeatureIndex, bestthreshold, "num")
            # 递归构建子树
            node.left = self.buildTree(task, data_left, attrs, attrs_type)
            node.right = self.buildTree(task, data_right, attrs, attrs_type)
        return node
        

    def fit(self, task: str, data: np.ndarray, attrs: list, attrs_type: list)-> Node:
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

if __name__ == '__main__':
    import os
    script_dir = os.path.dirname(os.path.abspath(__file__))
    data_path = os.path.join(script_dir, "data.csv")
    data = pd.read_csv(data_path, encoding='gbk')
    attributes = data.columns[:-1]
    attributes = list(attributes)
    attributeProps = [0,1,0]
    data = data.values
    cart = CART()
    cart.fit('classification', data, attributes, attributeProps)
    print("finish")