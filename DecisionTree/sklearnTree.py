# -*- coding: utf-8 -*-
"""
@Time    : 2022/8/16 14:43
@Author  : xiao.chen
"""
from sklearn import tree
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt
import pandas as pd

with open('dataSet.txt') as f:
	lensePd = pd.read_csv(f, sep="\\s+")     # \s+ 将tab和多个空格都当成一样的分隔符
	# print(lensePd.columns)

	le = LabelEncoder()     # 序列化字符串
	for col in lensePd.columns[:4]:     # 仅对前4列进行 字符串->数字 转换
		lensePd[col] = le.fit_transform(lensePd[col])       # 将字符串转换为增量值
	# print(lensePd.columns.tolist()[:4])
	# print(lensePd.to_numpy()[:,:4].tolist())
	# # 新建决策树,限制树深度为4
	clf = tree.DecisionTreeClassifier(max_depth=4)
	# # 训练决策树，X,y 行大小一致
	lense = clf.fit(lensePd.to_numpy()[:, :4].tolist(), lensePd.to_numpy()[:, 4].tolist())
	plt.figure()
	tree.plot_tree(lense)       # 绘制决策树
	plt.show()      # 程序阻塞
