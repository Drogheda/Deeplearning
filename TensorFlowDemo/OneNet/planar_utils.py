import matplotlib.pyplot as plt
import pylab
import numpy as np
import sklearn
import sklearn.datasets
import sklearn.linear_model

def plot_decision_boundary(model, X, y):
    # Set min and max values and give it some padding
    x_min, x_max = X[0, :].min() - 1, X[0, :].max() + 1
    y_min, y_max = X[1, :].min() - 1, X[1, :].max() + 1
    h = 0.01
    # 产生网格数据，返回的xx是竖向扩展，扩展倍数为y中值的个数，yy横向扩展，扩展倍数为x中值的个数
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
    # print(xx)
    # print(".................................")
    # print(yy)
    # print(".................................")
	
    # xx.ravel()将矩阵展成一个一维数组，即所有的数用一个一维数组表示，表示x轴的坐标
    # print(xx.ravel())
    # print(".................................")
    # print(yy.ravel())
    # print(".................................")
	
	# model是传入的函数,将两个数组按行连接后作为参数传入预测函数clf.predict()
    Z = model(np.c_[xx.ravel(), yy.ravel()])
    # print(Z)
    # print(".................................")
    Z = Z.reshape(xx.shape)
    # print(Z)
    # print(".................................")
    # 绘制等高线，输入的xx和yy为网格数据，z表示高度
    plt.contourf(xx, yy, Z, cmap = plt.cm.Spectral)
    plt.ylabel('x2')
    plt.xlabel('x1')
    # cmap = plt.cm.Spectral是用来给不同的点不同的颜色
    # colors = plt.cm.Spectral(np.arange(5))
    # print(y)
    plt.scatter(X[0, :], X[1, :], c = y, cmap = plt.cm.Spectral)
    pylab.show()
	
def sigmoid(x):
    s = 1 / (1 + np.exp(-x))
    return s

def load_planar_dataset():
    np.random.seed(1)	# 使每次产生的随机数都相同
    m = 400 # 样本数据量
    N = int(m / 2) # 每个类别的样本量是200
    D = 2 # 维数
    X = np.zeros((m, D)) # 初始化x为 mxD的数组
    Y = np.zeros((m, 1), dtype='uint8') # 初始化y的标签为0(0 for red, 1 for blue),uint8是无符号整数(0~255)
    a = 4 # 花的长度为4

    for j in range(2):	#j = 0、1
        ix = range(N * j, N * (j + 1))	#ix=0~200 200~400
		# t是N个元素的数组，np.linspace()函数用来构造等差数列；randn()返回一个具有标准正态分布的数组
        t = np.linspace(j * 3.12, (j + 1) * 3.12, N) + np.random.randn(N) * 0.2
        r = a * np.sin(4 * t) + np.random.randn(N) * 0.2 # 半径
        #  np.c_[a,b]是按行连接两个矩阵，np.r_[a, b]是按列连接两个矩阵
        X[ix] = np.c_[r * np.sin(t), r * np.cos(t)]
        Y[ix] = j
        
    X = X.T
    Y = Y.T

    return X, Y

def load_extra_datasets():  
    N = 200
    noisy_circles = sklearn.datasets.make_circles(n_samples = N, factor = .5, noise = .3)
    noisy_moons = sklearn.datasets.make_moons(n_samples = N, noise = .2)
    blobs = sklearn.datasets.make_blobs(n_samples = N, random_state = 5, n_features = 2, centers = 6)
    gaussian_quantiles = sklearn.datasets.make_gaussian_quantiles(mean = None, cov = 0.5, n_samples = N, n_features = 2, n_classes = 2, shuffle = True, random_state = None)
    no_structure = np.random.rand(N, 2), np.random.rand(N, 2)
    
    return noisy_circles, noisy_moons, blobs, gaussian_quantiles, no_structure