import numpy as np
import matplotlib.pyplot as plt	#绘图库
import pylab
import h5py							#基于NumPy来做高等数学、信号处理、优化、统计和许多其它科学任务的拓展库
import scipy
from PIL import Image
from scipy import ndimage
from lr_utils import load_dataset

# Loading the data (cat/non-cat)
train_set_x_orig, train_set_y, test_set_x_orig, test_set_y, classes = load_dataset()
# Example of a picture
#index = 25
#plt.show(train_set_x_orig[index])
#需要加如下语句才能显示图片
#pylab.show()
#train_set_y是（1,209）的矩阵，train_set_y[:, index]表示输出所有行的第index列
#np.squeeze(）表示从数组的形状中删除单维度条目，即把shape中为1的维度去掉，默认删除所有单维度条目
#print ("y = " + str(train_set_y[:, index]) + ", it's a '" + classes[np.squeeze(train_set_y[:, index])].decode("utf-8") +  "' picture.")

#m_train=209 m_test=50 num_px=64
m_train = train_set_x_orig.shape[0]
m_test = test_set_x_orig.shape[0]
num_px = train_set_x_orig.shape[1]
#print(str(m_train))
#print(test_set_x_orig.shape)
train_set_x_flatten = train_set_x_orig.reshape(train_set_x_orig.shape[0], -1).T
test_set_x_flatten = test_set_x_orig.reshape(test_set_x_orig.shape[0], -1).T
#print(train_set_x_flatten.shape)
#print(test_set_x_flatten.shape)

train_set_x = train_set_x_flatten / 255.
test_set_x = test_set_x_flatten / 255.

def sigmoid(z):
	s = 1 / (1 + np.exp(-z))
	return s
	
def initialize_with_zeros(dim):
	w = np.zeros((dim, 1))
	b = 0

	assert(w.shape == (dim, 1))
	#判断两个类型是否相同
	assert(isinstance(b, float) or isinstance(b, int))
#	assert(isinstance(b, (float, int))
	return w, b

def propagate(w, b, X, Y):
	m = X.shape[1]	
#	print(m)
	Z = np.dot(w.T, X) + b
	A = sigmoid(Z)
	cost = -1 / m * np.sum(Y * np.log(A) + (1 - Y) * np.log(1 - A))
	
	dz = A - Y
	db = 1 / m * np.sum(dz)
	dw = 1 / m * np.dot(X, dz.T)
	
	assert(Z.shape == (1, m))
	assert(A.shape == (1, m))
	assert(dz.shape == (1, m))
	assert(isinstance(db, (int, float)))

	#以字典的形式存储
	grads = {"dw" : dw, "db" : db}
	
	return grads, cost

def optimize(w, b, X, Y, num_iterations, learning_rate, print_cost = False):
	#在优化过程中的所有成本列表，用于绘制学习曲线
	costs = []
	for i in range(num_iterations):
		# print(i)
		#通过propagate函数得到dw和db的更新值，根据learning_rate得到w和d的更新值
		grads, cost = propagate(w, b, X, Y)
		dw = grads["dw"]
		db = grads["db"]
		w = w - learning_rate * dw
		b = b - learning_rate * db
		if i % 100 == 0:
			costs.append(cost)
		if print_cost and i % 100 == 0:
			print("Cost after iteration %i: %f" % (i, cost))
			
	#这里要注意缩进
	params = {"w" : w, "b" : b}
	grads = {"dw" : dw, "db" : db}
	return params, grads, costs

def predict(w, b, X):
	m = X.shape[1]
	Y_prediction = np.zeros((1, m))
	w = w.reshape(X.shape[0], 1)
	
	A = sigmoid(np.dot(w.T, X) + b)
	
	for i in range(A.shape[1]):
		if A[0, i] <= 0.5:					
			Y_prediction[0, i] = 0
		else:
			Y_prediction[0, i] = 1
	assert(Y_prediction.shape == (1, m))
	
	return Y_prediction
	
def model(X_train, Y_train, X_test, Y_test, num_iterations = 2000, learning_rate = 0.5, print_cost = False):
	#初始化
	w, b = initialize_with_zeros(X_train.shape[0])
	
	#建立模型+训练模型
	parameters, grads, costs = optimize(w, b, X_train, Y_train, num_iterations, learning_rate, print_cost)
	
	w = parameters["w"]
	b = parameters["b"]
	
	#预测
	Y_prediction_test = predict(w, b, X_test)
	Y_prediction_train = predict(w, b, X_train)
	
	print("train accuracy: {} %".format(100 - np.mean(np.abs(Y_prediction_train - Y_train)) * 100))
	print("test accuracy: {} %".format(100 - np.mean(np.abs(Y_prediction_test - Y_test)) * 100))
	
	d = {"costs" : costs,
		 "Y_prediction_test" : Y_prediction_test,
		 "Y_prediction_train" : Y_prediction_train,
		 "w" : w,
		 "b" : b,
		 "learning_rate" : learning_rate,
		 "num_iterations" : num_iterations}
	return d
	
if __name__ == '__main__':
	#迭代次数越多，效果越好
	d = model(train_set_x, train_set_y, test_set_x, test_set_y, num_iterations = 50000, learning_rate = 0.005, print_cost = True)
	
	#测试当前模型对图片的分类能力：
	# index = 2
	# plt.imshow(test_set_x[:,index].reshape((num_px, num_px, 3)))
	# pylab.show()
	# print ("y = " + str(test_set_y[0,index]) + ", you predicted that it is a \"" + classes[int(d["Y_prediction_test"][0,index])].decode("utf-8") +  "\" picture.")

	#显示成本的值和梯度：
	# costs = np.squeeze(d["costs"])
	# plt.plot(costs)
	# plt.ylabel('cost')
	# plt.xlabel('iterations(per hundreds)')
	# plt.title('Learning rate =' + str(d["learning_rate"]))
	# plt.show()
	
	#自己的图片测试
	my_image = "cat2_by_tw.jpg"   # change this to the name of your image file 
	fname = my_image
	image = np.array(ndimage.imread(fname, flatten = False))
	my_image = scipy.misc.imresize(image, size = (num_px, num_px)).reshape((1, num_px * num_px * 3)).T
	my_predicted_image = predict(d["w"], d["b"], my_image)
	plt.imshow(image)
	pylab.show()
	print("y = " + str(np.squeeze(my_predicted_image)) + ", your algorithm predicts a \"" + classes[int(np.squeeze(my_predicted_image)),].decode("utf-8") +  "\" picture.")













