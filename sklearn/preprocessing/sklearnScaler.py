# -*- coding: utf-8 -*-
"""
@Time    : 2022/8/26 13:43
@Author  : xiao.chen
"""
import matplotlib.pyplot as plt
from sklearn import preprocessing
from sklearn import datasets
import numpy as np

def main():
	# 从数据库导入 手写数字库
	digitsX, digitsy = datasets.load_digits(return_X_y=True)
	# 将数组以矩阵的形式显示在图像中，左上角为原点
	plt.matshow(digitsX[0, :].reshape(8, 8))

	# # 数据标准化
	scaler1 = preprocessing.StandardScaler()
	standardX = scaler1.fit_transform(digitsX, digitsy)
	# 对于digits，标准化后手写数字的特征没有之前明显
	plt.matshow(standardX[0, :].reshape(8, 8))
	print(standardX.mean(axis=0))   # 标准化后数据的均值为0

	# # 数据均一化
	normalizer = preprocessing.Normalizer()
	normalizeX = normalizer.transform(digitsX)
	plt.matshow(normalizeX[0, :].reshape(8, 8))

	plt.show()


if __name__ == '__main__':
	main()
