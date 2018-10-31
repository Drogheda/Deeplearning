import tensorflow as tf
import numpy as np

#broadcasting:先将实数或向量扩展再对对应元素进行运算
A = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9], [10, 11, 12]])
print(A)
print(A + 100)			

#reshape()函数,注意reshape以后不能丢失数据:1x12 12x1 2x6 6x2 3x4 4x3
#-1表示未知
print(A.reshape(-1))
print(A.reshape(-1, 1))
print(A.reshape(-1, 2))
print(A.reshape(3, -1))

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









