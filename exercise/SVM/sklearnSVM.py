# -*- coding: utf-8 -*-
"""
@Time    : 2022/8/23 15:53
@Author  : xiao.chen
"""

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn import svm

def file2mat(fileName):
	fileDirs = os.listdir(fileName)
	# m = len(fileDirs)       # 样本数
	m = 500
	numSets = np.zeros((m, 1024), dtype=int)     # 初始化数据集
	labels = np.zeros((m, 1), dtype=int)     # 初始化数据标签
	for i in range(m):
		labels[i] = fileDirs[i].split(sep='_')[0]       # 从文件名获取数据标签信息
		with open(fileName+fileDirs[i]) as f:
			# 通过限制converters, 将单行数据类型限制为str字符串
			imgDf = pd.read_csv(f, encoding='utf-8', sep='\\n',
			                    converters={0: str}, header=None, engine='python')
			imgSr = imgDf.iloc[:, 0]        # 将读取的dataframe转换为series类型
			extrSr = imgSr.str.extractall(r"(\d)")      # 将每行字符串提取为单个数字字符
			numSets[i, :] = extrSr.T.to_numpy()     # 将 1024*1 dataframe转置, 并转为numpy数组
	return numSets, labels


def main():
	# # 获取训练数据与训练标签
	trainingFileName = 'trainingDigits//'
	trainingSets, trainLabels = file2mat(trainingFileName)
	# # 获取测试数据与测试标签
	testFileName = "testDigits//"
	testSets, testLabels = file2mat(testFileName)

	# # 初始化支持向量机
	clf = svm.SVC(kernel='linear', C=1000)
	clf.fit(trainingSets, trainLabels)      # 训练数据集
	sc = clf.score(testSets, testLabels)        # 测试训练集
	print("The predict mean accuracy of the test set is :", sc)


if __name__ == '__main__':
	main()
