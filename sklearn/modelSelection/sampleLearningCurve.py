import numpy as np
import matplotlib.pyplot as plt
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.datasets import load_digits
from sklearn.model_selection import learning_curve
from sklearn.model_selection import ShuffleSplit

"""
	Generate 3 plots: the test and training learning curve, the training
	samples vs fit times curve, the fit times vs score curve.

	Parameters
	----------
	estimator : estimator instance
		An estimator instance implementing `fit` and `predict` methods which
		will be cloned for each validation.

	title : str
		Title for the chart.

	X : array-like of shape (n_samples, n_features)
		Training vector, where ``n_samples`` is the number of samples and
		``n_features`` is the number of features.

	y : array-like of shape (n_samples) or (n_samples, n_features)
		Target relative to ``X`` for classification or regression;
		None for unsupervised learning.

	axes : array-like of shape (3,), default=None
		Axes to use for plotting the curves.

	ylim : tuple of shape (2,), default=None
		Defines minimum and maximum y-values plotted, e.g. (ymin, ymax).

	cv : int, cross-validation generator or an iterable, default=None
		Determines the cross-validation splitting strategy.
		Possible inputs for cv are:

		  - None, to use the default 5-fold cross-validation,
		  - integer, to specify the number of folds.
		  - :term:`CV splitter`,
		  - An iterable yielding (train, test) splits as arrays of indices.

		For integer/None inputs, if ``y`` is binary or multiclass,
		:class:`StratifiedKFold` used. If the estimator is not a classifier
		or if ``y`` is neither binary nor multiclass, :class:`KFold` is used.

		Refer :ref:`User Guide <cross_validation>` for the various
		cross-validators that can be used here.

	n_jobs : int or None, default=None
		Number of jobs to run in parallel.
		``None`` means 1 unless in a :obj:`joblib.parallel_backend` context.
		``-1`` means using all processors. See :term:`Glossary <n_jobs>`
		for more details.

	scoring : str or callable, default=None
		A str (see model evaluation documentation) or
		a scorer callable object / function with signature
		``scorer(estimator, X, y)``.
		(https://scikit-learn.org/stable/modules/model_evaluation.html#scoring-parameter)
	train_sizes : array-like of shape (n_ticks,)
		Relative or absolute numbers of training examples that will be used to
		generate the learning curve. If the ``dtype`` is float, it is regarded
		as a fraction of the maximum size of the training set (that is
		determined by the selected validation method), i.e. it has to be within
		(0, 1]. Otherwise, it is interpreted as absolute sizes of the training
		sets. Note that for classification the number of samples usually have
		to be big enough to contain at least one sample from each class.
		(default: np.linspace(0.1, 1.0, 5))
"""

def plot_learning_curve(
	estimator,      # 估计器 或 预测器
	title,      # 图表标题
	X,      # 数据集 m * n
	y,      # 数据标签 m * 1
	axes=None,      # 绘制坐标轴，默认绘制3个图
	ylim=None,      # 曲线绘制范围 (y_min, y_max)
	cv=None,        # 交叉验证生成器
	n_jobs=None,    # 并行线程，-1为使用所有线程
	scoring=None,   # 模型评估规则
	train_sizes=np.linspace(0.1, 1.0, 5),       # 用于学习曲线的样本相对量
):

	if axes is None:
		_, axes = plt.subplots(1, 3, figsize=(20, 5))

	axes[0].set_title(title)
	if ylim is not None:
		axes[0].set_ylim(*ylim)
	axes[0].set_xlabel("Training examples")
	axes[0].set_ylabel("Score")

	# # 学习曲线，确定不同训练集大小的交叉验证训练和测试分数
	# train_sizes_abs: 用于生成学习曲线的训练示例的数量，fit_times: 拟合花费时间
	train_sizes, train_scores, test_scores, fit_times, _ = learning_curve(
		estimator,      # 估计器
		X,      # 数据集
		y,      # 数据标签
		scoring=scoring,    # 模型评估规则, precision, F1, recall ...
		cv=cv,      # 交叉验证生成器
		n_jobs=n_jobs,  # 并行线程
		train_sizes=train_sizes,    # 用于学习曲线的训练数据比例
		return_times=True,  # 是否返回 拟合时间
	)
	train_scores_mean = np.mean(train_scores, axis=1)
	train_scores_std = np.std(train_scores, axis=1)
	test_scores_mean = np.mean(test_scores, axis=1)
	test_scores_std = np.std(test_scores, axis=1)
	fit_times_mean = np.mean(fit_times, axis=1)
	fit_times_std = np.std(fit_times, axis=1)

	# Plot learning curve
	axes[0].grid()
	axes[0].fill_between(       # 填充两条水平曲线之间区域
		train_sizes,        # x轴 覆盖范围
		train_scores_mean - train_scores_std,       # y轴 覆盖下限
		train_scores_mean + train_scores_std,       # y轴 覆盖上限
		alpha=0.1,      # 覆盖区域透明度， 值越大越不透明
		color="r",      # 覆盖区域颜色
	)
	axes[0].fill_between(
		train_sizes,
		test_scores_mean - test_scores_std,
		test_scores_mean + test_scores_std,
		alpha=0.1,
		color="g",
	)
	axes[0].plot(
		train_sizes, train_scores_mean, "o-", color="r", label="Training score"
	)
	axes[0].plot(
		train_sizes, test_scores_mean, "o-", color="g", label="Cross-validation score"
	)
	axes[0].legend(loc="best")

	# Plot n_samples vs fit_times
	axes[1].grid()
	axes[1].plot(train_sizes, fit_times_mean, "o-")
	axes[1].fill_between(
		train_sizes,
		fit_times_mean - fit_times_std,
		fit_times_mean + fit_times_std,
		alpha=0.1,
	)
	axes[1].set_xlabel("Training examples")
	axes[1].set_ylabel("fit_times")
	axes[1].set_title("Scalability of the model")

	# Plot fit_time vs score
	fit_time_argsort = fit_times_mean.argsort()
	fit_time_sorted = fit_times_mean[fit_time_argsort]
	test_scores_mean_sorted = test_scores_mean[fit_time_argsort]
	test_scores_std_sorted = test_scores_std[fit_time_argsort]
	axes[2].grid()
	axes[2].plot(fit_time_sorted, test_scores_mean_sorted, "o-")
	axes[2].fill_between(
		fit_time_sorted,
		test_scores_mean_sorted - test_scores_std_sorted,
		test_scores_mean_sorted + test_scores_std_sorted,
		alpha=0.1,
	)
	axes[2].set_xlabel("fit_times")
	axes[2].set_ylabel("Score")
	axes[2].set_title("Performance of the model")

	return plt


def main():
	fig, axes = plt.subplots(3, 2, figsize=(10, 15))

	# 获取手写数据集， return_X_y 控制返回元组(T)或字典(F)
	X, y = load_digits(return_X_y=True)

	title = "Learning Curves (Naive Bayes)"
	# # Cross validation with 50 iterations to get smoother mean test and train
	# score curves, each time with 20% data randomly selected as a validation set.
	# # 随机排列交叉验证器，产生将数据拆分为训练集和测试集的索引
	# n_splits 迭代次数； test_size 测试集占比； random_state 使用int将在不同调用中产生相同结果
	cv = ShuffleSplit(n_splits=50, test_size=0.2, random_state=0)
	estimator = GaussianNB()
	plot_learning_curve(
		estimator, title, X, y,	axes=axes[:, 0], ylim=(0.7, 1.01), cv=cv, n_jobs=4,
		scoring="accuracy",
	)

	title = r"Learning Curves (SVM, RBF kernel, $\gamma=0.001$)"
	# SVC is more expensive, so we do a lower number of CV iterations:
	cv = ShuffleSplit(n_splits=5, test_size=0.2, random_state=0)
	estimator = SVC(gamma=0.001)
	plot_learning_curve(
		estimator, title, X, y, axes=axes[:, 1], ylim=(0.7, 1.01), cv=cv, n_jobs=4
	)

	plt.show()
	print(estimator)


if __name__ == '__main__':
	main()
