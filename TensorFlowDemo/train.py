import tensorflow as tf
import numpy as np
import h5py
#broadcasting:先将实数或向量扩展再对对应元素进行运算
A = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9], [10, 11, 12]])
print(A)
print(A.shape[1])	#输出为3
print(A + 100)			

#reshape()函数,注意reshape以后不能丢失数据:1x12 12x1 2x6 6x2 3x4 4x3
#-1表示未知
print(A.reshape(-1))
print(A.reshape(-1, 1))
print(A.reshape(-1, 2))
print(A.reshape(3, -1))
print(A.reshape(2, 2, 3))	#三维矩阵
print(A.shape)
#[[[ 1  2  3]
#  [ 4  5  6]]
#[[ 7  8  9]
#  [10 11 12]]]
#从数组的形状中删除单维度条目，即把shape中为1的维度去掉，默认删除所有单维度条目
print(np.squeeze(A))
#[[ 1  2  3]
# [ 4  5  6]
# [ 7  8  9]
# [10 11 12]]

#a既不属于行向量，也不是列向量
#a = np.array([1, 2, 3])
#print(a)
#print(a.shape)
#print(a.T)
#print(np.dot(a.T, a))
a = np.random.randn(5)
print(a)
print(a.shape)
#a.T还是它本身
print(a.T)
#二者做内积应该是一个矩阵但实际结果是一个数
print(np.dot(a.T, a))

def sigmoid(x):
	s = 1 / (1 + np.exp(-x))
	return s
#当.py文件被直接运行时，if __name__ == '__main__'之下的代码块将被运行；
#当.py文件以模块形式被导入时，if __name__ == '__main__'之下的代码块不被运行。
#python xxx.py，直接运行xxx.py文件
#python -m xxx，把xxx.py当做模块运行

#compute real number
if __name__ == '__main__':
	x = 3
	s = sigmoid(x)
	print(s)

#compute array
if __name__ == '__main__':
	x = np.array([2, 3, 4])
	print(x.shape)
	s = sigmoid(x)
	print(s)

#compute matrix
if __name__ == '__main__':
	x = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
	print(x.shape)
	s = sigmoid(x)
	print(s)

#h5py用来存放数据集（dataset）或组（group）
#在当前目录下创建mfh5py.hdf5文件，或mfh5py.h5文件
f = h5py.File("mfh5py.hdf5", "w")
#创建dataset数据集,i表示元素类型,代表int
d1 = f.create_dataset("dset1", (20,), 'i')
for key in f.keys():
	print(key)
	print(f[key].name)
	print(f[key].shape)
   #初始化默认为0
	print(f[key].value)
	
#赋值的两种方式
d1[...] = np.arange(20)
print(d1)
#单引号也可以
f['dset2'] = np.arange(15)
for key in f.keys():
	print(f[key].name)
	print(f[key].value)
#直接将numpy数组传给参数data
a = np.arange(20)
d1 = f.create_dataset("dset3", data = a)

#创建一个名字为bar的组
g1 = f.create_group("bar")
#在bar这个组里面分别创建name为dset1,dset2的数据集并赋值。
g1["ddset1"] = np.arange(10)
g1["ddset2"] = np.arange(12).reshape((3, 4))
for key in g1.keys():
	print(g1[key].name)
	print(g1[key].value)

#创建组bar1,组bar2，数据集dset
g1=f.create_group("bar1")
g2=f.create_group("bar2")
d=f.create_dataset("dset",data=np.arange(10))

#在bar1组里面创建一个组car1和一个数据集dset1。
c1=g1.create_group("car1")
d1=g1.create_dataset("dset1",data=np.arange(10))

#在bar2组里面创建一个组car2和一个数据集dset2
c2=g2.create_group("car2")
d2=g2.create_dataset("dset2",data=np.arange(10))

#根目录下的组和数据集
print(".............")
for key in f.keys():
    print(f[key].name)

#bar1这个组下面的组和数据集
print(".............")
for key in g1.keys():
    print(g1[key].name)


#bar2这个组下面的组和数据集
print(".............")
for key in g2.keys():
    print(g2[key].name)

#顺便看下car1组和car2组下面都有什么，估计你都猜到了为空。
print(".............")
print(c1.keys())
print(c2.keys())

#python列表和numpy的数组
a = [1, 2, 3, 4]				#a表示数组，长度是4
arr = np.array([1, 2, 3, 4])	#arr表示向量
print(a, len(a))
print(arr, arr.shape)

#python元组的列表和numpy数组
b = [(1, 2), (3, 4)]
brr = np.array([(1, 2), (3, 4)])
crr = np.array([[1, 2], [3, 4]])
print(b[0][0], len(b))		#b是一个二维数组，也可以看成是一个含有两个元组的列表
print(brr.T, brr.shape)		#brr是一个2x2的矩阵
print(crr.T, crr.shape)		#crr和brr效果相同

#eval()函数用来执行一个字符串表达式，并返回表达式的值
print(eval("2 * 3 + 4"))

# ndarray多维数组
x = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]	#list
x = np.array(x)	# 将任意序列类型的对象转换成ndarray数组
# 或直接这样定义x：x = np.arange(10)
print(type(x))	# ndarray

# sklearn中，与逻辑回归有关的主要是这三个类：LogisticRegression， LogisticRegressionCV 和logistic_regression_path。
# 其中LogisticRegression和LogisticRegressionCV的主要区别是LogisticRegressionCV使用了交叉验证来选择正则化系数C。
# 而LogisticRegression需要自己每次指定一个正则化系数。

# 方法：
# fit(X,y[,sample_weight])：训练模型。
# predict(X)：用模型进行预测，返回预测值。
# score(X,y[,sample_weight])：返回（X，y）上的预测准确率（accuracy）。
# predict_log_proba（X）：返回一个数组，数组的元素依次是 X 预测为各个类别的概率的对数值。 
# predict_proba（X）：返回一个数组，数组元素依次是 X 预测为各个类别的概率的概率值。







