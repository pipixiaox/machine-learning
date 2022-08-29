# -*- coding: utf-8 -*-
"""
@Time    : 2022/8/19 17:03
@Author  : xiao.chen
"""

import pandas as pd
from sklearn.linear_model import LogisticRegression

## 从txt文件中获取数据集，并将其转换为numpy array格式输出
def file2mat(fileName):
	with open(fileName) as f:
		# # pandas 读取txt, 间隔为一个或多个空格, 无列名
		txt = pd.read_csv(f, sep='\\s+', header=None)
		# # iloc获取特定行列元素
		numSet = txt.iloc[:, :txt.shape[1]-1]      # 获取前n-1列数据作为训练集
		numLabels = txt.iloc[:, txt.shape[1]-1]        # 获取最后1列数据作为训练集标签
	# # 将数据转换为numpy数组输出
	return numSet.to_numpy(), numLabels.to_numpy()

## 主程序，调用 sklearn logistic 回归算法训练并测试数据
def main():
	fileName = 'horseColicTraining.txt'
	trainingSet, trainingLabels = file2mat(fileName)
	solverList = ['liblinear', 'sag']
	for sol in solverList:
		# # 在运用'sag'优化算法时，迭代次数需到5000次才能收敛; 因为当前的数据集量太少
		# # 在训练数据量较少时，应使用(batch)梯度下降法 liblinear
		# # 随机梯度下降法 sag 更适用于数据集较大的场景
		clf = LogisticRegression(solver=sol, max_iter=5000)
		clf.fit(trainingSet, trainingLabels)
		print(clf.get_params())
		fileName = 'horseColicTest.txt'
		testSet, testLabels = file2mat(fileName)
		predScore = clf.score(testSet, testLabels)
		print("By using the function:", sol, end=', ')
		print("the predict score is:", predScore)


if __name__ == '__main__':
	main()
