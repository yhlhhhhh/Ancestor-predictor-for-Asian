import os
import numpy as np
import pandas as pd
from keras.utils import to_categorical
from tensorflow.python.keras import regularizers
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Dense, InputLayer


os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
# 读取数据
df = pd.read_csv('dataset.csv', index_col = 0, header = 0)
# 数据预处理
feature = np.array(df)[:, 0:-1]
target = np.array(df.index)
idx = dict(zip(np.unique(target), np.arange(9)))
target = [(idx[i]) for i in target]
target = to_categorical(target, num_classes = 9)

# 开始构建神经网络, 初始化容器
model = Sequential(name = 'nn_e11_model')
'''
# 定义输入层神经元数
model.add(InputLayer(input_shape=(11,)))
# 定义隐藏层
model.add(Dense(19, activation = 'sigmoid', name = 'dense_1', kernel_regularizer = regularizers.l2(1.9261e-06)))
# 定义输出层
model.add(Dense(9, activation = 'softmax', name = 'output', kernel_regularizer = regularizers.l2(1.9261e-06)))
# 定义优化器、损失函数
model.compile(metrics = ['accuracy'], optimizer = 'adam', loss = 'categorical_crossentropy')
#model.fit(feature, target, validation_freq = 0.3, epochs=150)
#model.summary()
# 保存模型
#model.save('e11_nn.h5')
'''
print(idx)