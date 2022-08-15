# -*- coding: utf-8 -*-
"""
Created on Thu Aug 11 17:15:46 2022
@author: xiao.chen
"""
import numpy as np
import collections
import dataSetFunc as dsf

"""  Func Type: kNN
input: 
    inX: 测试集
    dataSet: 训练集
    labels: 分类标签
    k: kNN算法参数，选择距离最小的k个点
output: 
    label: 分类结果
"""
def classify0(inX, dataSet, labels, k):
    # 计算测试集与训练集的距离,二范数,无需对向量进行扩展
    # # distances = np.sum((np.tile(inX,dataSet.shape[0]) - dataSet)**2)**0.5
    # axis=0 按列求和; axis=1 按行求和
    distances = np.sum((inX - dataSet)**2, axis=1)**0.5
    # argsort()获取正向排序下的值对应的原数组的Index
    klabels = [labels[x] for x in distances.argsort()[0:k]]
    # Counter方法返回klabels中值出现的次数，返回 值:次数 键-值对
    # most_common(k) 方法返回前k个出现次数最多的键-值对组成的元组列表
    # [0][0]获取对应元组列表的第一个元组的第一个元素(键)
    label = collections.Counter(klabels).most_common(1)[0][0]
    return label


"""  Func Type: classify test
input: 
    fileName: 数据集文件名
    ratio: 测试集比例
    colSize: datingSet的列数，即数据集特征量，默认包含所有特征量
output: 
    errorRatio: 错误率
"""
def datingClassTest(fileName, ratio, colSize=3):
    # 从文件读取数据集
    matGroup, labelsVector = dsf.file2matrix(fileName)
    # 将数据集进行归一化
    normMatSet, maxVals, minVals = dsf.autoNorm(matGroup[:, :colSize])

    # 将数据集分为测试集和训练集
    m = normMatSet.shape[0]  # 获取数据集总行数
    testSize = int(m * ratio)  # 取数据集中的 ratio行 数据作为测试集，其余为训练集
    trainingSets = normMatSet[testSize:m, :]
    trainingLabels = labelsVector[testSize:m]
    testSets = normMatSet[:testSize, :]
    testLabels = labelsVector[:testSize]
    errorCount = 0  # 错误量初始化
    for i in range(testSize):
        label = classify0(testSets[i, :], trainingSets, trainingLabels, 4)  # 有3组分类，故k选4
        if label != testLabels[i]:
            errorCount += 1
    return round(errorCount/testSize*100, 2)


def main():
    # classify0
    group, labels = dsf.createDataSet()  # 创建数据集
    test = [101, 20]  # 测试集
    test_class = classify0(test, group, labels, 3)  # kNN分类
    print(test_class)  # 打印分类结果

    # helen date test
    fileName = 'datingTestSet.txt'
    bestR = 0
    bestColSize = 0
    bestErrorRatio = 100
    for ratio in np.arange(0.2, 0.5, 0.05):
        for colSize in range(2, 4):
            errorRatio = datingClassTest(fileName, ratio, colSize)
            print("For", round(ratio*100, 1), "% test data, using", colSize,
                  "column dataSet, there is", errorRatio, "% errors")
            if bestErrorRatio > errorRatio:
                bestErrorRatio = errorRatio
                bestR = ratio
                bestColSize = colSize
    print("For best training,", round(bestR*100, 1), "% test data, with", bestColSize,
          "column dataSet, have", bestErrorRatio, "% errors")


if __name__ == '__main__':
    main()
