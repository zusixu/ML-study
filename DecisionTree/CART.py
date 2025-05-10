"""
Cart树实现
"""
import pandas as pd
import numpy as np
from math import inf
from collections import defaultdict
import os
import copy
from ucimlrepo import fetch_ucirepo 
  


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
        self.classlabel = None # 类别
        self.threshold = None # 分割阈值
        self.right = None # 右子树
        self.left = None # 左子树
        self.label = None # 叶节点label
        self.isleaf = False  # 是否为叶节点
    
    def to_dot(self, dot_lines, node_id=0):
        """
        递归生成 graphviz dot 格式内容
        :param dot_lines: 存储 dot 语句的列表
        :param node_id: 当前节点的唯一编号
        :return: 当前节点编号，下一可用编号
        """
        this_id = node_id
        if self.isleaf:
            dot_lines.append(f'    node{this_id} [label="Leaf\\nlabel={self.label}", shape=box, style=filled, color=lightgrey];')
        else:
            dot_lines.append(f'    node{this_id} [label="{self.feature}\\n{self.classlabel}\\n<= {self.threshold}"];')
        next_id = this_id + 1
        # 递归左子树
        if self.left:
            left_id, next_id = self.left.to_dot(dot_lines, next_id)
            dot_lines.append(f'    node{this_id} -> node{left_id} [label="True"];')
        # 递归右子树
        if self.right:
            right_id, next_id = self.right.to_dot(dot_lines, next_id)
            dot_lines.append(f'    node{this_id} -> node{right_id} [label="False"];')
        return this_id, next_id

class CART:
    def __init__(self, min_samples_split=100, min_impurity_decrease=1e-2, max_depth=15):
        """
        初始化CART树
        :param min_samples_split: 节点再分裂所需的最小样本数
        :param min_impurity_decrease: 分裂后最小纯度提升
        :param max_depth: 树的最大深度
        """
        self.root = Node()
        self.min_samples_split = min_samples_split
        self.min_impurity_decrease = min_impurity_decrease
        self.max_depth = max_depth
    

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
        if numEnts == 0 or len(data) == 0:
            return float('inf'), 0.0
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
        data_not_nan = data[data[:, attrIndex] != NAN]
        
        # 连续特征
        if attrs_type == 'num':
            left = data_not_nan[data_not_nan[:, attrIndex] <= threshold]
            right = data_not_nan[data_not_nan[:, attrIndex] > threshold]
        # 离散特征
        elif attrs_type == 'cat':
            left = data_not_nan[data_not_nan[:, attrIndex] == threshold]
            right = data_not_nan[data_not_nan[:, attrIndex] != threshold]
        else:
            raise ValueError("attrs_type must be 'num' or 'cat'")
        
        # 输出结果
        data_left = np.concatenate((left, data_nan), axis=0) if len(data_nan) > 0 and len(left) > 0 else left if len(left) > 0 else data_nan if len(data_nan) > 0 else np.empty((0, data.shape[1]))
        data_right = np.concatenate((right, data_nan), axis=0) if len(data_nan) > 0 and len(right) > 0 else right if len(right) > 0 else data_nan if len(data_nan) > 0 else np.empty((0, data.shape[1]))
        
        if len(data_nan) != 0 and (len(left) > 0 or len(right) > 0):
            total_weight = np.sum(left[:, -2]) + np.sum(right[:, -2]) if len(left) > 0 and len(right) > 0 else np.sum(left[:, -2]) if len(left) > 0 else np.sum(right[:, -2])
            if len(left) > 0:
                left_weight = np.sum(left[:, -2]) / total_weight
                data_left[np.where(data_left[:, attrIndex] == NAN), -2] = data_left[np.where(data_left[:, attrIndex] == NAN), -2] * left_weight
            if len(right) > 0:
                right_weight = np.sum(right[:, -2]) / total_weight
                data_right[np.where(data_right[:, attrIndex] == NAN), -2] = data_right[np.where(data_right[:, attrIndex] == NAN), -2] * right_weight
        
        # 删除特征列
        if attrs_type == 'cat':
            data_left = np.delete(data_left, attrIndex, axis=1)
        
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
                for j in range(len(uniqueVals)-1):
                        threshold = (uniqueVals[j] + uniqueVals[j+1]) / 2
                        subdata_left = subdata[np.where(subdata[:, index] <= threshold)]
                        subdata_right = subdata[np.where(subdata[:, index] > threshold)]
                        if task == 'classification':
                            calmetric = self.calGini(subdata_left) + self.calGini(subdata_right)
                        elif task == 'regression':
                            calmetric_L, _ = self.calMse(subdata_left) 
                            calmetric_R, _ = self.calMse(subdata_right)
                            calmetric = calmetric_L + calmetric_R
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
                    calmetric_L, _ = self.calMse(subdata_left)
                    calmetric_R, _ = self.calMse(subdata_right)
                    calmetric = calmetric_L + calmetric_R
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
           
                    
    
    def buildTree(self, task: str, data: np.ndarray, attrs: list, attrs_type: list, depth = 1)-> Node:
        """
        递归构建CART树
        param:
            task: 任务类型，分类(classfication)或回归(regression)
            data: 训练数据
            attrs: 特征列表
            attrs_type: 特征类型
            depth: 树的深度
        return:
            node: 树节点
        """
        classlist = data[:, -1]
        if task == 'classification':
            # 类别完全相同，停止划分
            if len(set(classlist)) == 1:
                node = Node()
                node.label = classlist[0]
                node.isleaf = True
                return node
            
        elif task == 'regression':
            # 样本数量收敛，停止划分
            if len(data) <= self.min_samples_split:
                node = Node()
                node.label = np.mean(classlist)
                node.isleaf = True
                return node
        # 所有特征均已使用，返回出现次数最多的类别
        if len(attrs) == 0:
            node = Node()
            if task == 'classification':
                node.label = max(set(classlist), key=classlist.tolist().count)
            elif task == 'regression':
                node.label = np.mean(classlist)
            node.isleaf = True
            return node
        # 所有样本相同，返回出现次数最多的类别
        if self.isSame(data):
            node = Node()
            if task == 'classification':
                node.label = max(set(classlist), key=classlist.tolist().count)
            elif task =='regression':
                node.label = np.mean(classlist)
            node.isleaf = True
            return node

        # 预剪枝条件：最大深度
        if self.max_depth is not None and depth > self.max_depth:
            node = Node()
            if task == 'classification':
                node.label = max(set(classlist), key=classlist.tolist().count)
            elif task == 'regression':
                node.label = np.mean(classlist)
            node.isleaf = True
            return node

        # 选择最优划分属性
        bestthreshold, bestfeatureIndex = self.chooseBestFeature(task, data, attrs_type)

        # 预剪枝条件：分裂提升不足
        if task == 'classification':
            impurity_parent = self.calGini(data)
            data_left, data_right = self.splitDataSetWithNull(data, bestfeatureIndex, bestthreshold, "cat" if attrs_type[bestfeatureIndex]==0 else "num")
            impurity_left = self.calGini(data_left)
            impurity_right = self.calGini(data_right)
            impurity_decrease = impurity_parent - (len(data_left)/len(data))*impurity_left - (len(data_right)/len(data))*impurity_right
            if impurity_decrease < self.min_impurity_decrease :
                node = Node()
                node.label = max(set(classlist), key=classlist.tolist().count)
                node.isleaf = True
                return node
        elif task == 'regression':
            impurity_parent, _ = self.calMse(data)
            data_left, data_right = self.splitDataSetWithNull(data, bestfeatureIndex, bestthreshold, "cat" if attrs_type[bestfeatureIndex]==0 else "num")
            impurity_left, _ = self.calMse(data_left)
            impurity_right, _ = self.calMse(data_right)
            impurity_decrease = impurity_parent - (len(data_left)/len(data))*impurity_left - (len(data_right)/len(data))*impurity_right
            if impurity_decrease < self.min_impurity_decrease :
                node = Node()
                node.label = np.mean(classlist)
                node.isleaf = True
                return node

        # 创建分支节点
        node = Node()
        node.feature = attrs[bestfeatureIndex]
        node.threshold = bestthreshold

        # 离散特征
        if attrs_type[bestfeatureIndex] == 0:
            node.classlabel = 'cat'
            # 划分数据集
            data_left, data_right = self.splitDataSetWithNull(data, bestfeatureIndex, bestthreshold, "cat")
            # 递归构建子树
            attrs_left = attrs[:bestfeatureIndex] + attrs[bestfeatureIndex+1:]
            attrs_type_left = attrs_type[:bestfeatureIndex] + attrs_type[bestfeatureIndex+1:]
            node.left = self.buildTree(task, data_left, attrs_left, attrs_type_left,depth+1)
            node.right = self.buildTree(task, data_right, attrs, attrs_type, depth+1)

        # 连续特征
        elif attrs_type[bestfeatureIndex] == 1:
            node.classlabel = 'num'
            # 划分数据集
            data_left, data_right = self.splitDataSetWithNull(data, bestfeatureIndex, bestthreshold, "num")
            # 递归构建子树
            node.left = self.buildTree(task, data_left, attrs, attrs_type, depth+1)
            node.right = self.buildTree(task, data_right, attrs, attrs_type, depth+1)
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
        # 添加计数权重到最后一列的左侧
        label_column = data[:, -1].reshape(-1, 1)  # 提取最后一列（标签列）
        data_without_label = data[:, :-1]  # 获取除最后一列外的所有数据
        data = np.concatenate((data_without_label, np.ones((len(data), 1)), label_column), axis=1)
        # 构建CART树
        self.root = self.buildTree(task, data, attrs, attrs_type,depth=1)
        return self.root


    def predict(self, data: np.ndarray, attribute: list, )-> np.ndarray:
        """
        预测
        param:
            data: 测试数据
            attribute: 特征列表
        return:
            y_pred: 预测值
        """
        if len(data.shape) == 1:
            data = np.array([data])
        y_pred = np.empty(len(data), dtype=object)
        for i in range(len(data)):
            node = self.root
            while not node.isleaf:
                if node.classlabel == 'cat':
                    if data[i][attribute.index(node.feature)] == node.threshold:
                        node = node.left
                    else:
                        node = node.right
                elif node.classlabel == 'num':
                    if data[i][attribute.index(node.feature)] <= node.threshold:
                        node = node.left
                    else:
                        node = node.right
            y_pred[i] = node.label
        return y_pred
        

if __name__ == '__main__':
    # import os
    # script_dir = os.path.dirname(os.path.abspath(__file__))
    # data_path = os.path.join(script_dir, "data.csv")
    # data = pd.read_csv(data_path, encoding='gbk')
    # attributes = data.columns[:-1]
    # attributes = list(attributes)
    # attributeProps = [0,1,0]
    # data = data.values
    # cart = CART()
    # cart.fit('classification', data, attributes, attributeProps)
    # print("classification train finish")
    # # 准确率
    # y_true = data[:,-1]
    # y_pred = cart.predict(data[:,:-1], attributes)
    # accuracy = sum(y_pred == y_true) / len(data)
    # print(f"classification accuracy: {accuracy}")

    # 回归任务测试
    # fetch dataset 
    wine_quality = fetch_ucirepo(id=186) 
    
    # data (as pandas dataframes) 
    X = wine_quality.data.features 
    y = wine_quality.data.targets 
    
    # metadata 
    print(wine_quality.metadata) 
    
    # variable information 
    print(wine_quality.variables)
    reg_data_df = pd.concat([X, y], axis=1)
    reg_attributes = list(X.columns)
    reg_attributeProps = [1,1,1,1,1,1,1,1,1,1,1]
    reg_data = reg_data_df.values
    reg_cart = CART(min_samples_split=1, max_depth=20,min_impurity_decrease=0.00001)
    reg_cart.fit('regression', reg_data, reg_attributes, reg_attributeProps)
    print("regression train finish")
    
    # 计算均方误差
    reg_y_true = reg_data[:,-1]
    reg_y_pred = reg_cart.predict(reg_data[:,:-1], reg_attributes)
    mse = np.mean((reg_y_pred.astype(float) - reg_y_true) ** 2)
    print(f"regression MSE: {mse}")
    # 生成 graphviz dot 文件
    dot_lines = ["digraph CART {", "    node [fontname=\"FangSong\"];"]
    reg_cart.root.to_dot(dot_lines)
    dot_lines.append("}")
    with open("cart_tree.dot", "w", encoding="utf-8") as f:
        f.write("\n".join(dot_lines))
    print("已生成 dot 文件：cart_tree.dot，请用 graphviz 渲染。")