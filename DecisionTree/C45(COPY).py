import copy
import operator
from math import log
from numpy import inf
import pandas as pd

NAN = 'Nan'  # 缺失值定义


def calcShannonEnt(dataSet: list, labelIndex: int):
    """
    计算对应属性索引下样本的香农熵
    :param dataSet: 样本
    :param labelIndex: 属性索引
    :return: shannonEnt 香农熵
    """
    numEntries = 0  # 样本数(按权重计算）
    labelCounts = {}
    # 遍历样本，计算每类的权重
    for featVec in dataSet:
        # 样本的属性不为空
        if featVec[labelIndex] != NAN:
            weight = featVec[-2]
            numEntries += weight
            currentLabel = featVec[-1]  # 当前样本的类别
            # 如果样本类别不在labelCounts
            if currentLabel not in labelCounts.keys():
                # 添加该类别，令该类别权重为0
                labelCounts[currentLabel] = .0
            # 添加该类别的权重
            labelCounts[currentLabel] += weight
    shannonEnt = .0
    for key in labelCounts:  # 计算信息熵
        prob = labelCounts[key] / numEntries
        shannonEnt -= prob * log(prob, 2)
    return shannonEnt


def splitDataSet(dataSet: list, axis: int, value, AttrType='N'):
    """
    划分数据集
    :param dataSet: 数据集
    :param axis: 按第几个特征划分
    :param value: 划分特征的值
    :param AttrType: N-离散属性; L-小于等于value值; R-大于value值
    :return: 对应axis为value（连续情况下则为大于或小于value）的数据集dataSet的子集
    """
    subDataSet = []
    # N-离散属性
    if AttrType == 'N':
        for featVec in dataSet:
            if featVec[axis] == value:
                reducedFeatVec = featVec[:axis]
                reducedFeatVec.extend(featVec[axis + 1:])
                subDataSet.append(reducedFeatVec)
    # L-小于等于value值
    elif AttrType == 'L':
        for featVec in dataSet:
            # 样本axis对应属性非空
            if featVec[axis] != NAN:
                if featVec[axis] <= value:
                    # 无需减少该特征
                    subDataSet.append(featVec)
    # R-大于value值
    elif AttrType == 'R':
        for featVec in dataSet:
            if featVec[axis] != NAN:
                if featVec[axis] > value:
                    # 无需减少该特征
                    subDataSet.append(featVec)
    else:
        exit(0)
    return subDataSet


def calcTotalWeight(dataSet: list, labelIndex: int, isContainNull: bool):
    """
    计算样本对某个特征值的总样本数(按权重计算)
    :param dataSet: 数据集
    :param labelIndex: 属性索引
    :param isContainNull: 是否包含空值
    :return: 样本的总权重
    """
    totalWeight = .0
    # 遍历样本
    for featVec in dataSet:
        # 样本权重
        weight = featVec[-2]
        # 不包含空值并且该属性非空
        if isContainNull is False and featVec[labelIndex] != NAN:
            # 非空样本树，按权重计算
            totalWeight += weight
        # 包含空值
        if isContainNull is True:
            # 总样本数
            totalWeight += weight
    return totalWeight


def splitDataSetWithNull(dataSet: list, axis: int, value, AttrType='N'):
    """
    划分含有缺失值的数据集
    :param dataSet: 数据集
    :param axis: 按第几个特征划分
    :param value: 划分特征的值
    :param AttrType: N-离散属性; L-小于等于value值; R-大于value值
    :return: 按value划分的数据集dataSet的子集
    """
    # 属性值未缺失样本子集
    subDataSet = []
    # 属性值缺失样本子集
    nullDataSet = []
    # 计算非空样本总权重
    totalWeightV = calcTotalWeight(dataSet, axis, False)
    # N-离散属性
    if AttrType == 'N':
        for featVec in dataSet:
            if featVec[axis] == value:
                reducedFeatVec = featVec[:axis]
                reducedFeatVec.extend(featVec[axis + 1:])
                subDataSet.append(reducedFeatVec)
            # 样本该属性值缺失
            elif featVec[axis] == NAN:
                reducedNullVec = featVec[:axis]
                reducedNullVec.extend(featVec[axis + 1:])
                nullDataSet.append(reducedNullVec)
    # L-小于等于value值
    elif AttrType == 'L':
        for featVec in dataSet:
            # 样本该属性值未缺失
            if featVec[axis] != NAN:
                if value is None or featVec[axis] < value:
                    subDataSet.append(featVec)
            # 样本该属性值缺失
            elif featVec[axis] == NAN:
                nullDataSet.append(featVec)
    # R-大于value值
    elif AttrType == 'R':
        for featVec in dataSet:
            # 样本该属性值未缺失
            if featVec[axis] != NAN:
                if featVec[axis] > value:
                    subDataSet.append(featVec)
            # 样本该属性值缺失
            elif featVec[axis] == NAN:
                nullDataSet.append(featVec)
    # 计算此分支中非空样本的总权重
    totalWeightSub = calcTotalWeight(subDataSet, -1, True)
    # 缺失值样本按权值比例划分到分支中
    for nullVec in nullDataSet:
        nullVec[-2] = nullVec[-2] * totalWeightSub / totalWeightV
        subDataSet.append(nullVec)
    return subDataSet


def calcGainRatio(dataSet: list, labelIndex: int, labelType: bool):
    """
    计算信息增益率，返回信息增益率和连续属性的划分点
    :param dataSet: 数据集
    :param labelIndex: 属性索引
    :param labelType: 属性类型，0为离散，1为连续
    :return: 信息增益率和连续属性的划分点
    """
    # 计算根节点的信息熵
    baseE = calcShannonEnt(dataSet, labelIndex)
    # 对应labelIndex的特征值向量
    featVec = [row[labelIndex] for row in dataSet]
    # featVec值的种类
    uniqueVals = set(featVec)
    newE = .0  # 新信息熵
    bestPivotValue = None  # 最佳划分属性
    IV = .0  # 该变量取自西瓜书
    # 总样本权重
    totalWeight = calcTotalWeight(dataSet, labelIndex, True)
    # 非空样本权重
    totalWeightV = calcTotalWeight(dataSet, labelIndex, False)
    # 对离散的特征
    if labelType == 0:
        # 按属性值划分数据集，计算各子集的信息熵
        for value in uniqueVals:
            # 划分数据集
            subDataSet = splitDataSet(dataSet, labelIndex, value)
            # 计算子集总权重
            totalWeightSub = calcTotalWeight(subDataSet, labelIndex, True)
            # 过滤空属性
            if value != NAN:
                prob = totalWeightSub / totalWeightV
                newE += prob * calcShannonEnt(subDataSet, labelIndex)
            prob1 = totalWeightSub / totalWeight
            IV -= prob1 * log(prob1, 2)
    # 对连续的特征
    else:
        uniqueValsList = list(uniqueVals)
        # 过滤空属性
        if NAN in uniqueValsList:
            uniqueValsList.remove(NAN)
            # 计算空值样本的总权重，用于计算IV
            dataSetNull = splitDataSet(dataSet, labelIndex, NAN)
            totalWeightN = calcTotalWeight(dataSetNull, labelIndex, True)
            probNull = totalWeightN / totalWeight
            if probNull > 0:
                IV += -1 * probNull * log(probNull, 2)
        # 属性值排序
        sortedUniqueVals = sorted(uniqueValsList)
        minEntropy = inf # 定义最小熵
        # 如果UniqueVals只有一个值，则说明只有左子集，没有右子集
        if len(sortedUniqueVals) == 1:
            totalWeightL = calcTotalWeight(dataSet, labelIndex, True)
            probL = totalWeightL / totalWeightV
            minEntropy = probL * calcShannonEnt(dataSet, labelIndex)
            IV = -1 * probL * log(probL, 2)
        # 如果UniqueVals只有多个值，则计算划分点
        else:
            for j in range(len(sortedUniqueVals) - 1):
                pivotValue = (sortedUniqueVals[j] + sortedUniqueVals[j + 1]) / 2
                # 对每个划分点，划分得左右两子集
                dataSetL = splitDataSet(dataSet, labelIndex, pivotValue, 'L')
                dataSetR = splitDataSet(dataSet, labelIndex, pivotValue, 'R')
                # 对每个划分点，计算左右两侧总权重
                totalWeightL = calcTotalWeight(dataSetL, labelIndex, True)
                totalWeightR = calcTotalWeight(dataSetR, labelIndex, True)
                probL = totalWeightL / totalWeightV
                probR = totalWeightR / totalWeightV
                Ent = probL * calcShannonEnt(dataSetL, labelIndex) + probR * calcShannonEnt(dataSetR, labelIndex)
                # 取最小的信息熵
                if Ent < minEntropy:
                    minEntropy = Ent
                    bestPivotValue = pivotValue
                    probL1 = totalWeightL / totalWeight
                    probR1 = totalWeightR / totalWeight
                    IV += -1 * (probL1 * log(probL1, 2) + probR1 * log(probR1, 2))
        newE = minEntropy
    gain = totalWeightV / totalWeight * (baseE - newE)
    # 避免IV为0（属性只有一个值的情况下）
    if IV == 0.0:
        IV = 0.0000000001
    gainRatio = gain / IV
    return gainRatio, bestPivotValue


def chooseBestFeatureToSplit(dataSet: list, labelProps: list):
    """
    选择最佳数据集划分方式
    :param dataSet: 数据集
    :param labelProps: 属性类型，0离散，1连续
    :return: 最佳划分属性的索引和连续属性的最佳划分值
    """
    numFeatures = len(labelProps)  # 属性数
    bestGainRatio = -inf  # 最大信息增益
    bestFeature = -1  # 最优划分属性索引
    bestPivotValue = None  # 连续属性的最佳划分值
    for featureI in range(numFeatures):  # 对每个特征循环
        gainRatio, bestPivotValuei = calcGainRatio(dataSet, featureI, labelProps[featureI])
        # 取信息益率最大的特征
        if gainRatio > bestGainRatio:
            bestGainRatio = gainRatio
            bestFeature = featureI
            bestPivotValue = bestPivotValuei
    return bestFeature, bestPivotValue


def majorityCnt(classList: list, weightList: list):
    """
    返回出现次数最多的类别(按权重计)
    :param classList: 类别
    :param weightList: 权重
    :return: 出现次数最多的类别
    """
    classCount = {}
    # 计算classCount
    for cls, wei in zip(classList, weightList):
        if cls not in classCount.keys():
            classCount[cls] = .0
        classCount[cls] += wei
    # 排序
    sortedClassCount = sorted(classCount.items(), key=operator.itemgetter(1), reverse=True)
    # 仅剩一个类别
    if len(sortedClassCount) == 1:
        return sortedClassCount[0][0], sortedClassCount[0][1]
    # 剩余多个类别，返回出现次数最多的类别
    return sortedClassCount[0][0], sortedClassCount[0][1]


def isSame(dataSet: list):
    """
    比较样本特征是否相同
    :param dataSet: 数据集
    :return: 相同True，否则False
    """
    for j in range(len(dataSet[0])-2):
        for i in range(1, len(dataSet)):
            if not dataSet[i][j] == dataSet[0][j]:
                return False
    return True


def createTree(dataSet: list, labels: list, labelProps: list):
    """
    创建决策树（Decision Tree）
    :param dataSet: 数据集
    :param labels: 属性集
    :param labelProps: 属性类型，0离散，1连续
    :return: 决策树
    """
    classList = [sample[-1] for sample in dataSet]  # 类别向量
    weightList = [sample[-2] for sample in dataSet]  # 权重向量
    # 如果只剩一个类别，返回并退出
    if classList.count(classList[0]) == len(classList):
        totalWeight = calcTotalWeight(dataSet, 0, True)
        return classList[0], totalWeight
    # 如果所有特征都遍历完了，返回出现次数最多的类别，并退出
    if len(dataSet[0]) == 1:
        return majorityCnt(classList, weightList)
    # 如果剩余样本特征相同，返回出现次数最多的类别，并退出
    if isSame(copy.copy(dataSet)):
        return majorityCnt(classList, weightList)
    # 计算最优分类特征的索引，若为连续属性，则还返回连续属性的最优划分点
    bestFeat, bestPivotValue = chooseBestFeatureToSplit(dataSet, labelProps)
    # 对离散的特征
    if labelProps[bestFeat] == 0:
        bestFeatLabel = labels[bestFeat]
        myTree = {bestFeatLabel: {}}
        labelsNew = copy.copy(labels)
        labelPropertyNew = copy.copy(labelProps)
        # 已经选择的离散特征不再参与分类
        del (labelsNew[bestFeat])
        del (labelPropertyNew[bestFeat])
        featValues = [sample[bestFeat] for sample in dataSet]
        # 最佳花划分属性包含的所有值
        uniqueValue = set(featValues)
        # 删去缺失值
        uniqueValue.discard(NAN)
        # 遍历每个属性值，递归构建树
        for value in uniqueValue:
            subLabels = labelsNew[:]
            subLabelProperty = labelPropertyNew[:]
            myTree[bestFeatLabel][value] = createTree(splitDataSetWithNull(dataSet, bestFeat, value),
                                                      subLabels, subLabelProperty)
    # 对连续特征，不删除该特征，分别构建左子树和右子树
    else:
        bestFeatLabel = labels[bestFeat] + '<' + str(bestPivotValue)
        myTree = {bestFeatLabel: {}}
        subLabels = labels[:]
        subLabelProperty = labelProps[:]
        # 构建左子树
        valueLeft = 'Y'
        myTree[bestFeatLabel][valueLeft] = createTree(splitDataSetWithNull(dataSet, bestFeat, bestPivotValue, 'L'),
                                                      subLabels, subLabelProperty)
        # 构建右子树
        valueRight = 'N'
        myTree[bestFeatLabel][valueRight] = createTree(splitDataSetWithNull(dataSet, bestFeat, bestPivotValue, 'R'),
                                                       subLabels, subLabelProperty)
    return myTree


if __name__ == '__main__':
    # 读取数据文件
    fr = open(r'D:\\project\DataScience\DecisionTree\data.csv')
    data = [row.strip().split(',') for row in fr.readlines()]
    labels = data[0][0:-1]  # labels：属性
    dataset = data[1:]  # dataset：数据集(初始样本)
    labelProperties = [0, 1, 0]  # labelProperties：属性标识，0为离散，1为连续
    # 样本权重初始化
    for row in dataset:
        row.insert(-1, 1.0)
    # 按labelProperties连续化离散属性
    for row in dataset:
        for i, lp in enumerate(labelProperties):
            # 若标识为连续属性，则转化为float型
            if lp:
                row[i] = float(row[i])
    # C4.5算法生成决策树
    trees = createTree(copy.copy(dataset), copy.copy(labels), copy.copy(labelProperties))
    print(trees)
