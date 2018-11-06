import numpy as np
import h5py
    
    
def load_dataset():
	#datasets/train_catvnoncat.h5文件下有三个数据集：/list_classes、/train_set_x、/train_set_y

	train_dataset = h5py.File('datasets/train_catvnoncat.h5', "r")
#	print(train_dataset["list_classes"].value)  #[b'non-cat' b'cat']
#	print(train_dataset["train_set_x"].value)	#输入
#	print(train_dataset["train_set_y"].value)	#输出
#	将输入输出放进numpy数组
	train_set_x_orig = np.array(train_dataset["train_set_x"].value)
	train_set_y_orig = np.array(train_dataset["train_set_y"].value)
	
	test_dataset = h5py.File('datasets/test_catvnoncat.h5', "r")
	test_set_x_orig = np.array(test_dataset["test_set_x"][:])
	test_set_y_orig = np.array(test_dataset["test_set_y"][:])

	classes = np.array(test_dataset["list_classes"][:])
#	print(classes)
#	train_set_y_orig.shape[0]返回它的行数；
#	print(train_set_y_orig)
#	print(train_set_y_orig.shape[0])
	train_set_y_orig = train_set_y_orig.reshape((1, train_set_y_orig.shape[0]))
	test_set_y_orig = test_set_y_orig.reshape((1, test_set_y_orig.shape[0]))

	return train_set_x_orig, train_set_y_orig, test_set_x_orig, test_set_y_orig, classes

#当.py文件被直接运行时，if __name__ == '__main__'之下的代码块将被运行；
#当.py文件以模块形式被导入时，if __name__ == '__main__'之下的代码块不被运行。
#python xxx.py，直接运行xxx.py文件
#python -m xxx，把xxx当做模块运行
if __name__ == '__main__':
	train_set_x_orig, train_set_y, test_set_x_orig, test_set_y, classes = load_dataset()


















