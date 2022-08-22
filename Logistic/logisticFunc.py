# -*- coding: utf-8 -*-
"""
Created on Tue Aug  9 10:28:13 2022
@author: xiao.chen

"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def sigmoid(x):
    return 1. / (1 + np.exp(-x))

# # sigmoid函数绘制
def plotSigmoid():
    x = np.arange(-10, 10, 0.1)
    y = sigmoid(x)
    plt.plot(x, y)

# # Logistic 代价函数
def cost(theta, x, y):
    theta = np.mat(theta)       # n * 1
    x = np.mat(x)       # m * n
    y = np.mat(y)       # m * 1

    r1 = np.multiply(-y, np.log(sigmoid(x * theta)))
    r2 = np.multiply(-(1 - y), np.log(1 - sigmoid(x * theta)))

    return np.sum(r1 + r2) / len(x)


def classifyVector(X, theta):
    prob = sigmoid(sum(X*theta))
    if prob >= 0.5:
        return 1
    else:
        return 0

# # 梯度上升算法， 求解 theta 最大值
def gradAscent(trainingSet, trainingLabels, maxCycles=500):
    trainingSet = np.mat(trainingSet)       # 为方便矩阵运算，将数组转换为矩阵
    trainingLabels = np.mat(trainingLabels).transpose()     # m * 1 列向量
    m, n = trainingSet.shape        # 获取参数矩阵大小 m * n
    theta = np.ones((n, 1))     # 待计算参数向量, n * 1
    alpha = 0.001       # 学习率，梯度下降速率
    costList = []       # 代价函数变化趋势
    for k in range(maxCycles):
        error = trainingLabels - sigmoid(trainingSet*theta)
        theta = theta + alpha * trainingSet.transpose() * error
        costList.append(cost(theta, trainingSet, trainingLabels))

    # # 绘制 代价函数 随 迭代次数的变化情况
    fig = plt.figure()
    axis = fig.add_subplot()
    axis.plot(range(maxCycles), costList)
    plt.title("cost function")
    plt.xlabel('cycles')
    plt.ylabel('cost')

    # 将矩阵转换为数组
    return np.array(theta)


# # 从文件获取训练数据，为方便线性拟合，添加1列全1训练集
def file2mat(fileName):
    with open(fileName) as f:
        # # pandas 读取txt, 间隔为一个或多个空格, 无列名
        txt = pd.read_csv(f, sep='\\s+', header=None)
        # # iloc获取特定行列元素
        numSet = txt.iloc[:, :txt.shape[1] - 1]  # 获取前n-1列数据作为训练集
        numLabels = txt.iloc[:, txt.shape[1] - 1]  # 获取最后1列数据作为训练集标签
        # # 将数据转换为numpy数组,并在首列插入一列 1 用于 theta0 运算
        numSet = numSet.to_numpy()
        numSet = np.insert(numSet, 0, np.ones(len(numSet)), axis=1)
    return numSet, numLabels.to_numpy()


# # 绘制训练集 和 回归方程
def plotSet(trainingSet, trainingLabels, theta):
    posx1 = []
    posx2 = []
    negx1 = []
    negx2 = []
    for i in range(len(trainingLabels)):
        if trainingLabels[i] == 1:
            posx1.append(trainingSet[i, 1])
            posx2.append(trainingSet[i, 2])
        else:
            negx1.append(trainingSet[i, 1])
            negx2.append(trainingSet[i, 2])
    fig = plt.figure()
    axis = fig.add_subplot()
    axis.scatter(posx1, posx2, color='red')
    axis.scatter(negx1, negx2, color='green')

    x = np.arange(-3.0, 3.0, 0.1)
    y = - (theta[0] + theta[1] * x)/theta[2]
    axis.plot(x, y, color='black')
    plt.title("Best Fit")
    plt.xlabel('X1')
    plt.ylabel('X2')


def main():
    plotSigmoid()
    fileName = 'testSet.txt'
    trainingSet, trainingLabels = file2mat(fileName)
    theta = gradAscent(trainingSet, trainingLabels)
    plotSet(trainingSet, trainingLabels, theta)
    plt.show()


if __name__ == '__main__':
    main()
