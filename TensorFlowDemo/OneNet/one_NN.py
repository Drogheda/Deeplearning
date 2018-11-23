"""
	Planar data classification with a hidden layer.一个隐藏层的平面数据分类
	18/11/18
"""
import numpy as np
import matplotlib.pyplot as plt
import pylab	# 需要导入这个模块
from testCases_v2 import *	# 自定义文件，封装了一些用于测试样本，用于评估算法的有效性。
import sklearn	# 包含大量数据集,提供一些简单而有效的工具用于数据挖掘和数据分析
import sklearn.datasets
# 也可以这么写：from sklearn import datasets
import sklearn.linear_model
# from sklearn.linear_model import LinearRegression
from planar_utils import plot_decision_boundary, sigmoid, load_planar_dataset, load_extra_datasets	#自定义文件，封装了一些作用用到的相关函数。

# get_ipython().magic('matplotlib inline')

np.random.seed(1) # set a seed so that the results are consistent


# load dataset
X, Y = load_planar_dataset()
# print(type(X))
# print(type(Y))
# print(X.shape)
# print(Y.shape)
# print(X.shape[1])  # training set size
# print(Y)
# print(type(Y))
# visualize the data（绘制原始数据的散点图）:
Y = np.squeeze(Y)
plt.scatter(X[0, :], X[1, :], c = Y, s = 40, cmap = plt.cm.Spectral)	#前三个参数类型都是ndarray
# pylab.show()	# 必须有这一句才能显示


# 利用Logistic回归的效果
# Train the logistic regression classifier
clf = sklearn.linear_model.LogisticRegressionCV();
clf.fit(X.T, Y.T);
# 根据X和Y得到拟合后的数据作为等高线的高度，绘制等高线
plot_decision_boundary(lambda x: clf.predict(x), X, Y)
plt.title("Logistic Regression")
# Print accuracy
LR_predictions = clf.predict(X.T)
print ('Accuracy of logistic regression: %d ' % float((np.dot(Y, LR_predictions) + np.dot(1 - Y, 1 - LR_predictions)) / float(Y.size) * 100) +
       '% ' + "(percentage of correctly labelled datapoints)")


# 利用含有一个隐藏层的神经网络的效果
def layer_sizes(X, Y):
    n_x = X.shape[0] # size of input layer = 2
    n_h = 4          # 自己规定的隐藏层的结点个数
    n_y = Y.shape[0] # size of output layer = 1
    return (n_x, n_h, n_y)

# 这个是生成两个随机数数组，用来测试一下layer_sizes()这个函数好使不好使
# X_assess, Y_assess = layer_sizes_test_case()
# (n_x, n_h, n_y) = layer_sizes(X, Y)


# 初始化参数
def initialize_parameters(n_x, n_h, n_y):
    np.random.seed(2) # we set up a seed so that your output matches ours although the initialization is random.
    
    W1 = np.random.randn(n_h, n_x) * 0.01 #获得一个较小的z有利于梯度下降
    b1 = np.zeros((n_h, 1))
    W2 = np.random.randn(n_y, n_h) * 0.01
    b2 = np.zeros((n_y, 1))

    assert (W1.shape == (n_h, n_x))
    assert (b1.shape == (n_h, 1))
    assert (W2.shape == (n_y, n_h))
    assert (b2.shape == (n_y, 1))
    
    parameters = {"W1": W1,
                  "b1": b1,
                  "W2": W2,
                  "b2": b2}
    
    return parameters

# 这是用来测试initialize_parameters()函数的
# n_x, n_h, n_y = initialize_parameters_test_case()
# parameters = initialize_parameters(n_x, n_h, n_y)


# 前向传播
def forward_propagation(X, parameters):
    W1 = parameters["W1"]
    b1 = parameters["b1"]
    W2 = parameters["W2"]
    b2 = parameters["b2"]

    Z1 = np.dot(W1, X) + b1
    A1 = np.tanh(Z1)
    Z2 = np.dot(W2, A1) + b2
    A2 = sigmoid(Z2)

    assert(A2.shape == (1, X.shape[1]))
    
    cache = {"Z1": Z1,
             "A1": A1,
             "Z2": Z2,
             "A2": A2}
    
    return A2, cache

# 这是用来测试forward_propagation()函数的
# X_assess, parameters = forward_propagation_test_case()
# A2, cache = forward_propagation(X_assess, parameters)


def compute_cost(A2, Y, parameters):
    m = Y.shape[1]
    # np.multiply()对应位置相乘
    logprobs =  np.multiply(np.log(A2), Y) + np.multiply(np.log(1 - A2), 1 - Y)
    cost = - np.sum(logprobs) / m
    
    cost = np.squeeze(cost)
    # 判断cost的类型是否为float
    assert(isinstance(cost, float))
    
    return cost

# 用来测试compute_cost()函数
# A2, Y_assess, parameters = compute_cost_test_case()
# print("cost = " + str(compute_cost(A2, Y_assess, parameters)))


# 反向传播
def backward_propagation(parameters, cache, X, Y):
    m = X.shape[1]
    W1 = parameters["W1"]
    W2 = parameters["W2"]
    A1 = cache["A1"]
    A2 = cache["A2"]
    
    dZ2 = A2 - Y
    dW2 = np.dot(dZ2, A1.T) / m
    db2 = np.sum(dZ2, axis = 1, keepdims = True) / m
    dZ1 = np.multiply(np.dot(W2.T, dZ2), 1 - np.power(A1, 2))
    dW1 = np.dot(dZ1, X.T) / m
    db1 = np.sum(dZ1, axis=1, keepdims = True) / m
    
    grads = {"dW1": dW1,
             "db1": db1,
             "dW2": dW2,
             "db2": db2}
    
    return grads

# 用来测试backward_propagation()
# parameters, cache, X_assess, Y_assess = backward_propagation_test_case()
# grads = backward_propagation(parameters, cache, X_assess, Y_assess)


# 更新参数
def update_parameters(parameters, grads, learning_rate = 1.2):
    W1 = parameters["W1"]
    b1 = parameters["b1"]
    W2 = parameters["W2"]
    b2 = parameters["b2"]
    
    # Retrieve each gradient from the dictionary "grads"
    dW1 = grads["dW1"]
    db1 = grads["db1"]
    dW2 = grads["dW2"]
    db2 = grads["db2"]
    
    # Update rule for each parameter
    W1 = W1-learning_rate * dW1
    b1 = b1-learning_rate * db1
    W2 = W2-learning_rate * dW2
    b2 = b2-learning_rate * db2
    
    parameters = {"W1": W1,
                  "b1": b1,
                  "W2": W2,
                  "b2": b2}
    
    return parameters

# 测试update_parameters()函数
# parameters, grads = update_parameters_test_case()
# parameters = update_parameters(parameters, grads)


# 神经网络
def nn_model(X, Y, n_h, num_iterations = 10000, print_cost = False):
    np.random.seed(3)
    Y = Y.reshape((1, 400))
    n_x = layer_sizes(X, Y)[0]	# 输入层节点数 = 2
    n_y = layer_sizes(X, Y)[2]	# 输出层节点数 = 1

    parameters = initialize_parameters(n_x, n_h, n_y)
    W1 = parameters["W1"]
    b1 = parameters["b1"]
    W2 = parameters["W2"]
    b2 = parameters["b2"]
    
    for i in range(0, num_iterations):
         
        A2, cache = forward_propagation(X, parameters)
        
        cost = compute_cost(A2, Y, parameters)
 
        grads = backward_propagation(parameters, cache, X, Y)
 
        parameters = update_parameters(parameters, grads, learning_rate = 1.2)
                
        # Print the cost every 1000 iterations
        if print_cost and i % 1000 == 0:
            print ("Cost after iteration %i: %f" % (i, cost))

    return parameters

# 测试nn_model()函数
# X_assess, Y_assess = nn_model_test_case()
# parameters = nn_model(X_assess, Y_assess, 4, num_iterations = 10000, print_cost=True)


def predict(parameters, X): 
    # Computes probabilities using forward propagation, and classifies to 0/1 using 0.5 as the threshold.
    A2, cache = forward_propagation(X, parameters)
    predictions = (A2 > 0.5)
    
    return predictions

# 测试predict()函数
# parameters, X_assess = predict_test_case()
# predictions = predict(parameters, X_assess)


# 调用神经网络模型
parameters = nn_model(X, Y, n_h = 5, num_iterations = 10000, print_cost = True)

plot_decision_boundary(lambda x: predict(parameters, x.T), X, Y)
pylab.show()
plt.title("Decision Boundary for hidden layer size " + str(4))

# Print accuracy
predictions = predict(parameters, X)
print ('Accuracy: %d' % float((np.dot(Y,predictions.T) + np.dot(1 - Y, 1 - predictions.T)) / float(Y.size) * 100) + '%')

# 隐藏层神经元个数对结果的影响
# plt.figure(figsize = (16, 32))
# hidden_layer_sizes = [1, 2, 3, 4, 5, 20, 50]
# for i, n_h in enumerate(hidden_layer_sizes):
    # plt.subplot(5, 2, i + 1)
    # plt.title('Hidden Layer of size %d' % n_h)
    # parameters = nn_model(X, Y, n_h, num_iterations = 5000)
    # plot_decision_boundary(lambda x: predict(parameters, x.T), X, Y)
    # predictions = predict(parameters, X)
    # accuracy = float((np.dot(Y, predictions.T) + np.dot(1 - Y, 1 - predictions.T)) / float(Y.size) * 100)
    # print ("Accuracy for {} hidden units: {} %".format(n_h, accuracy))


# Performance on other datasets
# noisy_circles, noisy_moons, blobs, gaussian_quantiles, no_structure = load_extra_datasets()

# datasets = {"noisy_circles": noisy_circles,
            # "noisy_moons": noisy_moons,
            # "blobs": blobs,
            # "gaussian_quantiles": gaussian_quantiles}

# dataset = "noisy_moons"

# X, Y = datasets[dataset]
# X, Y = X.T, Y.reshape(1, Y.shape[0])

# if dataset == "blobs":
    # Y = Y % 2

# plt.scatter(X[0, :], X[1, :], c = Y, s = 40, cmap = plt.cm.Spectral);
# pylab.show()