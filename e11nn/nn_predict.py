import pandas as pd
import numpy as np
from nn import idx
from keras.models import load_model


# 读取待预测数据
df_predict = pd.read_csv('e11_test.csv', index_col=0, header=0)
arr = np.array(df_predict)
sample = list(df_predict.index)
# 读取模型
model = load_model('e11_nn.h5')
result_freq = model.predict(arr)
# 处理结果
code = np.where(result_freq == np.max(result_freq, axis = 1).reshape(-1, 1))[-1]
classification = list(idx.keys())
result = [classification[i] for i in code]
pd.Series(dict(zip(sample, result))).to_csv('e11_nn_result.csv')