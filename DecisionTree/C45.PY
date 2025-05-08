"""
这是一个决策树算法的实现，仍然存在问题，比如使用了numpy库偷懒
而不是使用List来实现数据集的分割，可视化是直接AI生成的，没有添加剪枝操作
"""
import numpy as np
import pandas as pd
import copy
import math
import operator
from collections import Counter,defaultdict
from numpy import inf
import matplotlib.pyplot as plt
import matplotlib.patches as patches
# 缺失值
NAN = 'Nan'
# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'SimSun', 'Arial Unicode MS']  # 尝试多种中文字体
plt.rcParams['axes.unicode_minus'] = False    # 用来正常显示负号

class node:
    def __init__(self, attribute=None, attributeValue=None, PivotValue=None, label=None, isleaf=False):
        """
        初始化节点
        param:
            attribute: 节点的属性
            attributeValue: 节点若为离散值时的属性值
            PivotValue: 节点若为连续值时的划分点
            child: 节点的子节点
            label: 节点为叶子节点时的标签
            isleaf: 节点是否为叶子节点
        """
        self.child = []
        self.label = None
        self.attribute = None
        self.attributeValue = []
        self.PivotValue = None
        self.isleaf = False

    def get_label(self):
        try:
            return self.label
        except:
            print("get label error, node is not leaf node")
            return None
    
    def get_depth(self):
        """
        计算树的深度
        return:
            depth: 树的深度
        """
        if self.isleaf:
            return 1
        else:
            max_depth = 0
            for child in self.child:
                depth = child.get_depth()
                if depth > max_depth:
                    max_depth = depth
            return max_depth + 1

    def get_width(self):
        """
        计算树的宽度（叶子节点数量）
        return:
            width: 树的宽度
        """
        if self.isleaf:
            return 1
        else:
            width = 0
            for child in self.child:
                width += child.get_width()
            return width
    def _draw_node(self, ax, x, y, node_width, node_height, level_height, total_width, x_offset=0):
        """
        递归绘制节点及其子节点
        param:
            ax: matplotlib轴对象
            x: 节点中心x坐标
            y: 节点中心y坐标
            node_width: 节点宽度
            node_height: 节点高度
            level_height: 层级高度
            total_width: 总宽度
            x_offset: x坐标偏移量
        return:
            leaf_positions: 叶子节点位置列表
        """
        # 绘制当前节点
        if self.isleaf:
            # 叶子节点用圆角矩形表示
            rect = patches.FancyBboxPatch(
                (x - node_width/2, y - node_height/2),
                node_width, node_height,
                boxstyle=patches.BoxStyle("Round", pad=0.2),
                facecolor='lightgreen', edgecolor='black', alpha=0.7
            )
            ax.add_patch(rect)
            # 添加标签文本
            ax.text(x, y, f"类别: {self.label}", ha='center', va='center', fontsize=10)
            return [(x, y)]
        else:
            # 非叶子节点用矩形表示
            rect = patches.Rectangle(
                (x - node_width/2, y - node_height/2),
                node_width, node_height,
                facecolor='lightblue', edgecolor='black', alpha=0.7
            )
            ax.add_patch(rect)
            
            # 添加属性文本
            if self.PivotValue is not None:
                # 连续属性
                node_text = f"{self.attribute}\n阈值: {self.PivotValue:.2f}"
            else:
                # 离散属性
                node_text = f"{self.attribute}"
            ax.text(x, y, node_text, ha='center', va='center', fontsize=10)
            
            # 计算子节点位置
            child_count = len(self.child)
            if child_count == 0:
                return [(x, y)]
            
            # 计算子节点的总宽度
            child_total_width = sum(child.get_width() for child in self.child)
            
            # 绘制子节点
            leaf_positions = []
            child_x_offset = x - (total_width / 2)
            
            for i, child in enumerate(self.child):
                # 计算子节点宽度比例
                child_width_ratio = child.get_width() / child_total_width if child_total_width > 0 else 1/child_count
                
                # 计算子节点中心位置
                child_width = total_width * child_width_ratio
                child_x = child_x_offset + (child_width / 2)
                child_y = y - level_height
                
                # 绘制连接线
                ax.plot([x, child_x], [y - node_height/2, child_y + node_height/2], 'k-')
                
                # 在连接线上添加属性值标签
                label_x = (x + child_x) / 2
                label_y = (y - node_height/2 + child_y + node_height/2) / 2
                
                # 获取边的标签（属性值）
                edge_label = str(child.attributeValue)
                ax.text(label_x, label_y, edge_label, ha='center', va='center', 
                       bbox=dict(facecolor='white', alpha=0.7, edgecolor='none'), fontsize=8)
                
                # 递归绘制子节点
                child_leaves = child._draw_node(
                    ax, child_x, child_y, node_width, node_height, 
                    level_height, child_width, child_x_offset
                )
                leaf_positions.extend(child_leaves)
                
                # 更新x偏移量
                child_x_offset += child_width
            
            return leaf_positions
    
    def visualize(self, figsize=(12, 8), title="决策树可视化"):
        """
        可视化决策树
        param:
            figsize: 图形大小
            title: 图形标题
        """
        # 创建图形和轴
        fig, ax = plt.subplots(figsize=figsize)
        
        # 计算树的深度和宽度
        depth = self.get_depth()
        width = self.get_width()
        
        # 设置节点大小和层级高度
        node_width = 1.5
        node_height = 0.8
        level_height = 2.0
        
        # 计算图形总宽度和高度
        total_width = max(width * 2, 10)  # 确保最小宽度
        total_height = depth * level_height
        
        # 设置轴的范围
        ax.set_xlim(-total_width/2, total_width/2)
        ax.set_ylim(-total_height, 1)
        
        # 绘制根节点及其子节点
        self._draw_node(ax, 0, 0, node_width, node_height, level_height, total_width)
        
        # 设置标题和隐藏坐标轴
        ax.set_title(title)
        ax.axis('off')
        
        # 调整布局并显示
        plt.tight_layout()
        plt.show()

class C45:
    def __init__(self):
        self.root = node()
       
    
    def calcShannonEnt(self, data: np.ndarray, index: int):
        """
        计算数据集的信息熵
        param:
            data: 数据集
            index: 数据集的属性索引
        return:
            shannonEnt: 数据集的信息熵
        """
        numEntries = data.shape[0]
        labelCounts = defaultdict(int)
        for featVec in data:
            if featVec[index] != NAN:
                currentLabel = featVec[-1]
                # if currentLabel not in labelCounts.keys():
                #     labelCounts[currentLabel] = 0
                labelCounts[currentLabel] += 1
        shannonEnt = 0.0
        for key in labelCounts:
            prob = float(labelCounts[key]) / numEntries
            shannonEnt -= prob * math.log(prob, 2)
        return shannonEnt
        
    def calcInfoGainRatio(self, data: np.ndarray, index: int, attributeProps: list):
        """
        计算信息增益率
        param:
            data: 数据集
            index: 数据集的属性
            attributeProps: 数据集的标签
        return:
            infoGainRatio: 信息增益率
            PivotValue: 划分点
        """
        # if type(data) != np.ndarray:
        #     data = np.array(data)
        baseEnt = self.calcShannonEnt(data, index)
        # 获得属性的所有值
        attributeValues = [example[index] for example in data]
        attributeValues = list(set(attributeValues))
        # 新信息熵
        newEnt = 0.0
        # 划分点
        PivotValue = None
        IV = 0.0
        # 总样本数量
        totalNum = data.shape[0]
        # 非空样本数量
        nonEmptyNum = sum(1 for x in data[:, index] if x != NAN)

        # 离散特征
        if attributeProps[index] == 0:
            for value in attributeValues:
                subData = data[np.where(data[:, index] == value)]
                prob1 = len(subData) / totalNum
                if value != NAN:
                    prob2 = len(subData) / nonEmptyNum
                    newEnt += prob2 * self.calcShannonEnt(subData, index)
                IV -= prob1 * math.log(prob1, 2)

        # 连续特征
        else:
            # 过滤掉缺失值
            if NAN in attributeValues:
                attributeValues.remove(NAN)
                # 计算空值样本数
                emptyNum = totalNum - nonEmptyNum
                probNull = emptyNum / totalNum
                if probNull > 0:
                    IV -= probNull * math.log(probNull, 2)
            # 对属性值进行排序
            attributeValues = sorted(attributeValues)
            minEntropy = inf # 定义最小熵
            # 如果只有一个值，则只有左子树没有右子树
            if len(attributeValues) == 1:
                subDataL = data[np.where(data[:, index] <= attributeValues[0])]
                probL = len(subDataL) / totalNum
                minEntropy = probL * self.calcShannonEnt(subDataL, index)
                IV -= probL * math.log(probL, 2)
                PivotValue = attributeValues[0]
            else:
                # 初始化最佳概率值
                bestprobL = 0
                bestprobR = 0
                # 计算划分点
                for i in range(len(attributeValues)-1):
                    curPivotValue = (attributeValues[i] + attributeValues[i+1]) / 2
                    # 过滤掉缺失值
                    filteredData = data[np.where(data[:, index] != NAN)]
                    subDataL = filteredData[np.where(filteredData[:, index] <= curPivotValue)]
                    subDataR = filteredData[np.where(filteredData[:, index] > curPivotValue)]
                    probL = len(subDataL) / totalNum
                    probR = len(subDataR) / totalNum
                    newEnt = probL * self.calcShannonEnt(subDataL, index) + probR * self.calcShannonEnt(subDataR, index)
                    if newEnt < minEntropy:
                        minEntropy = newEnt
                        PivotValue = curPivotValue
                        bestprobL = probL
                        bestprobR = probR
            newEnt = minEntropy
            IV -= bestprobL * math.log(bestprobL, 2) + bestprobR * math.log(bestprobR, 2)

        if IV == 0:
            IV = 0.0000000001
        gain = nonEmptyNum / totalNum * (baseEnt - newEnt)
        gainRatio = gain / IV
        return gainRatio, PivotValue 

    def chooseBestFeatureToSplit(self, data: np.ndarray, attributes: list, attributeProps: list):
        """
        选择最优的划分属性
        param:
            data: 数据集
            attributes: 数据集的属性
            attributeProps: 数据集的标签    
        return:
            bestFeature: 最优的划分属性
            bestPivotValue: 最优的划分属性的划分点
        """
        numAttributes = len(attributes)
        bestGainRatio = -inf
        bestFeature = -1
        bestPivotValue = None
        for i in range(numAttributes):
            gainRatio, pivotValue = self.calcInfoGainRatio(data, i, attributeProps)
            if gainRatio > bestGainRatio:
                bestGainRatio = gainRatio
                bestFeature = i
                bestPivotValue = pivotValue
        return bestFeature, bestPivotValue

    def majorityCnt(self, classList):
        """
        计算数据集的标签
        param:
            classList: 数据集的标签
        return:
            label: 数据集的标签
        """
        classCount = defaultdict(int)
        for vote in classList:
            if vote not in classCount.keys():
                classCount[vote] = 0
            classCount[vote] += 1
        sortedClassCount = sorted(classCount.items(), key=operator.itemgetter(1), reverse=True)
        return sortedClassCount[0][0]

    def isSame(self, data: np.ndarray):
        """
        判断数据集的属性是否都相同
        param:
            data: 数据集
        return:
            isSame: 数据集的属性是否都相同
        """
        for i in range(len(data[0])-1):
            for j in range(len(data)):
                if data[j][i] != data[0][i]:
                    return False    
        return True 
       
    def splitDataSetWithNull(self, data: np.ndarray, index: int, value, AttrType='N')->np.ndarray:
        """
        根据属性值划分含有缺失值的数据集
        param:
            data: 数据集
            index: 数据集的属性
            value: 数据集的属性值
            AttrType: 数据集的属性类型
        return:
            datanew: 数据集
        """
        datanew = []
        nullData = []
        if AttrType == 'N':
            for featVec in data:
                if featVec[index] == value:
                    reducedFeatVec = featVec[:index]
                    reducedFeatVec = np.append(reducedFeatVec, featVec[index+1:])
                    datanew.append(reducedFeatVec)
                elif featVec[index] == NAN:
                    reducedFeatVec = featVec[:index]
                    reducedFeatVec = np.append(reducedFeatVec, featVec[index+1:])
                    nullData.append(reducedFeatVec)
        elif AttrType == 'L':
            for featVec in data:
                if featVec[index] != NAN:
                    if featVec[index] <= value:
                        datanew.append(featVec)
                elif featVec[index] == NAN:
                    nullData.append(featVec)
        elif AttrType == 'R':
            for featVec in data:
                if featVec[index]!= NAN:
                    if featVec[index] > value:
                        datanew.append(featVec)
                elif featVec[index] == NAN:
                    nullData.append(featVec)
                    
        return np.array(datanew)
                
    def createTree(self, data: np.ndarray,attributes: list, attributeProps: list):
        """
        递归创建决策树
        param:
            data: 数据集
            attributes: 数据集的属性
            attributeProps: 数据集的标签
        """
        curnode = node()

        classList = [example[-1] for example in data]
        # 如果数据集的标签都是同一个标签，则返回该标签
        if classList.count(classList[0]) == len(classList):
            curnode.label = classList[0]
            curnode.isleaf = True
            return curnode
        # 如果数据集的属性为空，则返回数据集的标签
        if len(attributes) == 0:
            curnode.label = self.majorityCnt(classList)
            curnode.isleaf = True
            return curnode
        # 如果剩余样本的所有属性取值相同，则返回次数最多的类标签
        if self.isSame(data):
            curnode.label = self.majorityCnt(classList)
            curnode.isleaf = True
            return curnode
        # 计算最优分类特征的索引，若为连续属性，则还返回连续属性的最优划分点
        bestFeature, bestPivotValue = self.chooseBestFeatureToSplit(data, attributes, attributeProps)
        # 离散属性
        if attributeProps[bestFeature] == 0:
            bestFeatureLabel = attributes[bestFeature]
            curnode.attribute = bestFeatureLabel
            attributesnew = copy.copy(attributes)
            attributePropsnew = copy.copy(attributeProps)
            # 删除最优分类特征
            del(attributesnew[bestFeature])
            del(attributePropsnew[bestFeature])
            # 获得最优分类特征的所有取值
            featureValues = [example[bestFeature] for example in data]
            featureValues = list(set(featureValues))
            if NAN in featureValues:
                featureValues.remove(NAN)
            # 遍历所有取值
            for value in featureValues:
                # 获得包含空值的子数据集
                subData = self.splitDataSetWithNull(data, bestFeature, value, AttrType='N')
                # 递归调用createTree函数
                subattributes = copy.copy(attributesnew)
                subattributeProps = copy.copy(attributePropsnew)
                subnode = self.createTree(subData, subattributes, subattributeProps)
                subnode.attributeValue = value
                curnode.child.append(subnode)
        # 连续属性
        else:
            bestFeatureLabel = attributes[bestFeature]
            curnode.attribute = bestFeatureLabel
            curnode.PivotValue = bestPivotValue
            attributesnew = copy.copy(attributes)
            attributePropsnew = copy.copy(attributeProps)
            # 构建左子树
            subDataL = self.splitDataSetWithNull(data, bestFeature, bestPivotValue, AttrType='L')
            subattributesL = copy.copy(attributesnew)
            subattributePropsL = copy.copy(attributePropsnew)
            subnodeL = self.createTree(subDataL, subattributesL, subattributePropsL)
            subnodeL.attributeValue = "<=" + str(bestPivotValue)
            curnode.child.append(subnodeL)
            # 构建右子树
            subDataR = self.splitDataSetWithNull(data, bestFeature, bestPivotValue, AttrType='R')
            subattributesR = copy.copy(attributesnew)
            subattributePropsR = copy.copy(attributePropsnew)   
            subnodeR = self.createTree(subDataR, subattributesR, subattributePropsR)
            subnodeR.attributeValue = ">" + str(bestPivotValue)
            curnode.child.append(subnodeR)
        return curnode
    
    def predict(self, data: np.ndarray, attributes: list)->np.ndarray:
        """
        预测数据集的标签
        param:
            data: 数据集
        return:
            label: 数据集的标签
        """
        # 如果是单个样本（一维数组），则转换为二维数组
        if len(data.shape) == 1:
            data = np.array([data])
        
        # 存储预测结果
        predictions = []
        
        # 对每个样本进行预测
        for sample in data:
            # 从根节点开始遍历
            node = self.root
            
            # 当前节点不是叶子节点时，继续遍历
            while not node.isleaf:
                # 获取当前节点的属性
                attribute = node.attribute
                # 获取属性在数据集中的索引
                attr_index = None
                for i, attr in enumerate(attributes):
                    if attr == attribute:
                        attr_index = i
                        break
                
                # 如果是连续属性
                if node.PivotValue is not None:
                    # 如果样本的属性值小于等于划分点，则走左子树
                    if sample[attr_index] <= node.PivotValue:
                        node = node.child[0]
                    # 否则走右子树
                    else:
                        node = node.child[1]
                # 如果是离散属性
                else:
                    # 获取样本的属性值
                    attr_value = sample[attr_index]
                    # 查找对应的子节点
                    found = False
                    for child in node.child:
                        if child.attributeValue == attr_value:
                            node = child
                            found = True
                            break
                    # 如果没有找到对应的子节点，则选择第一个子节点（默认策略）
                    if not found and len(node.child) > 0:
                        node = node.child[0]
                    # 如果没有子节点，则当前节点为叶子节点
                    elif not found:
                        break
            
            # 将叶子节点的标签作为预测结果
            predictions.append(node.label)
        
        # 如果只有一个样本，则返回单个结果，否则返回结果列表
        if len(predictions) == 1:
            return predictions[0]
        else:
            return np.array(predictions)


if __name__ == '__main__':
    import os
    script_dir = os.path.dirname(os.path.abspath(__file__))
    data_path = os.path.join(script_dir, "data.csv")
    data = pd.read_csv(data_path, encoding='gbk')
    attributes = data.columns[:-1]
    attributes = list(attributes)
    attributeProps = [0,1,0]
    data = data.values
    c45 = C45()
    c45.root = c45.createTree(data, attributes, attributeProps)
    print("finish")
    # 分类准确率
    y_true = data[:, -1]
    y_pred = c45.predict(data[:, :-1], attributes)
    accuracy = sum(y_pred == y_true) / len(y_true)
    print(f"分类准确率: {accuracy:.4f}")
    # 可视化决策树
    c45.root.visualize(title="C4.5 决策树可视化")
