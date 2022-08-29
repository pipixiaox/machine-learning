# -*- coding: utf-8 -*-
"""
Created on Thu Aug 11 14:38:15 2022
@author: xiao.chen
"""
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.lines as mlines

"""  Func Type: 创建数据集
input: 
    None
output: 
    group, 数据集(训练集)
    labels, 分类标签
"""
def createDataSet():
    # 二维特征矩阵X， m = 4
    group = np.array([[1, 101],
                      [5, 89],
                      [108, 5],
                      [115, 8]])
    
    # 1代表爱情片，2代表动作片
    labels = [1, 1, 2, 2]
    return group, labels


"""  Func Type: 从txt文件导入数据，并对数据进行分类
input: 
    filename, 文件名
output: 
    matGroup, 数据集 特征矩阵
    labelsVector, 分类标签
"""
def file2matrix(filename):
    with open(filename) as f:
        # 按行读取txt文档
        # 第1列:飞行常客里程 第2列:玩游戏时间 第3列:冰淇淋量 第4列:喜欢程度
        arrayFromLines = f.readlines()
        lines = len(arrayFromLines)
        # 初始化输出结果
        matGroup = np.zeros((lines, 3))  # 返回数据集，lines行3列，不包含 喜欢程度
        labelsVector = []
        
        index = 0  # 数据集初始行
        for line in arrayFromLines:
            line = line.strip()  # 删除字符串头尾指定字符，默认为空格或换行符
            datas = line.split()
            matGroup[index, :] = datas[0:3]  # 逐行保存前三项数据到矩阵中
            
            # 保存分类标签
            if datas[-1] == 'didntLike':
                labelsVector.append(1)
            elif datas[-1] == 'smallDoses':
                labelsVector.append(2)
            elif datas[-1] == 'largeDoses':
                labelsVector.append(3)
            index += 1
    # matGroup = autoNorm(matGroup)
    return matGroup, labelsVector


"""  Func Type: 对数据进行归一化
input: 
    dataSet: 原始数据集
output:
    normDataSet: 归一化后的数据集
    maxVals: 数据集最大值
    minVals: 数据集最小值
"""
def autoNorm(dataSet):
    normDataSet = np.zeros(dataSet.shape)
    # 对列向量分别进行归一化  ~[0,1]
    for i in range(dataSet.shape[1]):
        maxVals = np.max(dataSet[:, i])
        minVals = np.min(dataSet[:, i])
        normDataSet[:, i] = (dataSet[:, i]-minVals)/(maxVals-minVals)
    return normDataSet, maxVals, minVals


"""  Func Type: 对数据进行展示
input: 
    x,y: 数据x,y坐标轴
    labeltext: title,x,y 标签列表
    labelsVector: 喜欢程度
output: None parameter
    scatter plot
"""
def showdatas(x, y, labeltext, labelsVector):
    # 对不同喜欢程度采用不同的颜色进行展示
    labelsColors = []
    for i in labelsVector:
        if i == 1:
            labelsColors.append("red")  # didntLike
        if i == 2:
            labelsColors.append("green")  # smallDoses
        if i == 3:
            labelsColors.append("blue")  # largeDoses
    # 显示中文
    plt.rcParams['font.sans-serif'] = ['SimHei']
    # 初始化图
    fig = plt.figure()
    axis = fig.add_subplot()
    # 绘制散点图
    axis.scatter(x, y, color=labelsColors)
    # 设置散点图标题，xy坐标
    axis.set_title(labeltext[0], fontsize=15)
    axis.set_xlabel(labeltext[1], fontsize=13)
    axis.set_ylabel(labeltext[2], fontsize=13)
    # 设置散点图图例legend
    red = mlines.Line2D([], [], color='red', marker='.', markersize=12, label='didntLike')
    green = mlines.Line2D([], [], color='green', marker='.', markersize=12, label='smallDoses')
    blue = mlines.Line2D([], [], color='blue', marker='.', markersize=12, label='largeDoses')
    axis.legend(handles=[red, green, blue], loc='upper right')
    plt.show()



def main():
    ## kNN-1  classify film type
    group, labels = createDataSet()
    print(group, "\n", labels)
    # 将矩阵按列拆分
    x, y = np.hsplit(group, group.shape[1])
    # 训练集可视化
    plt.ion()  # 触发交互模式
    plt.scatter(x, y)
    plt.show()  # 默认显示模式为阻塞模式(程序将会暂停，并不会继续执行)；交互模式下图将一闪而过
    plt.pause(3)  # 避免图一闪而过，可以设置等待时间

    ## kNN-2  helen date
    filename = 'datingTestSet.txt'
    matGroup, labelsVector = file2matrix(filename)
    normMatGroup, maxVals, minVals = autoNorm(matGroup)
    # 将矩阵按列拆分
    x1, x2, x3 = np.hsplit(normMatGroup, normMatGroup.shape[1])
    ## 训练集可视化
    # 飞行里程数与玩游戏时间
    labeltext1 = ["飞行里程数与玩游戏时间", "飞行里程数", "玩游戏时间"]
    plt.ion()  # 触发交互模式
    showdatas(x1, x2, labeltext1, labelsVector)
    plt.pause(3)  # 避免图一闪而过，可以设置等待时间
    # 飞行里程数与冰淇淋量
    labeltext2 = ["飞行里程数与冰淇淋摄入量", "飞行里程数", "冰淇淋摄入量"]
    plt.ion()  # 触发交互模式
    showdatas(x1, x3, labeltext2, labelsVector)
    plt.pause(3)  # 避免图一闪而过，可以设置等待时间
    # 玩游戏时间与冰淇淋摄入量
    labeltext3 = ["玩游戏时间与冰淇淋摄入量", "玩游戏时间", "冰淇淋摄入量"]
    plt.ion()  # 触发交互模式
    showdatas(x2, x3, labeltext3, labelsVector)
    plt.pause(3)  # 避免图一闪而过，可以设置等待时间


    # 重建训练集，仅包含分类清晰的列，如飞行里程数与玩游戏时间
    newSet1 = np.hstack((x1, x2))  # hstack 添加新的列; vstack 添加新的行


if __name__ == "__main__":
    main()
