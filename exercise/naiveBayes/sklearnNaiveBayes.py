# -*- coding: utf-8 -*-
"""
@Time    : 2022/8/18 11:35
@Author  : xiao.chen
"""
import os
import re
import numpy as np
from sklearn.naive_bayes import MultinomialNB

# # 字符串划分为单词
def splitWords(str1):
	words = re.split('\\W+', str1)
	return [word.lower() for word in words if len(word) >= 2]

# # 文档转换为单词表
def file2words(hamFile, spamFile):
	hamNames = os.listdir(hamFile)
	spamNames = os.listdir(spamFile)
	wordsList = []      # 邮件词汇列表
	emailLabels = []        # 邮件标签
	allWordsSet = set()     # 所有出现的词汇总表
	for hamname in hamNames:
		wordList = []
		with open(hamFile+hamname) as f:
			for line in f.readlines():  # 逐行读取文档
				wordList.extend(splitWords(line))  # 逐词保存
		wordsList.append(wordList)
		emailLabels.append(1)
		allWordsSet.update(wordList)
	for spamname in spamNames:
		wordList = []
		with open(hamFile + spamname) as f:
			for line in f.readlines():  # 逐行读取文档
				wordList.extend(splitWords(line))  # 逐词保存
		wordsList.append(wordList)
		emailLabels.append(0)
		allWordsSet.update(wordList)
	allWordsSet = list(allWordsSet)
	return wordsList, allWordsSet, emailLabels

# # 训练测试集生成
def words2mat(wordsList, allWordsSet, emailLabels):
	emailMat = np.zeros((len(emailLabels), len(allWordsSet)))
	for i in range(len(emailLabels)):
		for word in wordsList[i]:
			emailMat[i, allWordsSet.index(word)] += 1
	return emailMat

# # 将样本按比例划分为训练集和测试集
def ratioMat(emailMat, emailNum, testRatio, allWordsNum, emailLabels):
	# 获取随机整数, replace=False, 不重复, replace=True, 可能重复
	testMatNum = np.random.choice(emailNum - 1, size=int(emailNum * testRatio), replace=False)
	testMat = np.zeros((int(emailNum * testRatio), allWordsNum))
	testLabels = []
	trainingMat = np.zeros((int(emailNum * (1 - testRatio)), allWordsNum))
	trainLabels = []
	j, k = 0, 0
	for i in range(emailNum):
		if i in testMatNum:
			testMat[j, :] = emailMat[i, :]
			testLabels.append(emailLabels[i])
			j += 1
		else:
			trainingMat[k, :] = emailMat[i, :]
			trainLabels.append(emailLabels[i])
			k += 1
	return trainingMat, trainLabels, testMat, testLabels


# # 主程序
def main():
	hamFile = 'email//ham//'        # 非垃圾邮件文件夹
	spamFile = 'email//spam//'      # 垃圾邮件文件夹
	# 从邮件文件中获取单词表
	wordsList, allWordsSet, emailLabels = file2words(hamFile, spamFile)
	emailMat = words2mat(wordsList, allWordsSet, emailLabels)       # 获取非垃圾/垃圾邮件单词数据集
	emailNum = len(wordsList)       # 获取总数据集样本量
	testRatio = 0.20
	trainingMat, trainLabels, testMat, testLabels = ratioMat(emailMat, emailNum, testRatio,
	                                                         len(allWordsSet), emailLabels)

	clf = MultinomialNB()
	clf.fit(trainingMat, trainLabels)
	testAccuracy = clf.score(testMat, testLabels)
	print(testAccuracy)


if __name__ == '__main__':
	main()
