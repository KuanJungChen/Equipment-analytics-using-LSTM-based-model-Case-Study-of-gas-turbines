# -*- coding: utf-8 -*-
"""
Created on Sun Dec 12 15:05:45 2021

@author: User
"""

# =============================================================================
# NOX
# 特徵選擇 Mutual Information
# 
# =============================================================================
#%%
#匯入必須的函式庫
from sklearn import preprocessing
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
from sklearn import metrics

from pandas import read_csv
from pandas import datetime
from pandas import DataFrame
from pandas import concat
from matplotlib import pyplot
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import MinMaxScaler

import warnings
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import datetime
#%%
#年份設定
year = '2011'
#目標值設定
target = 'NOX'

#%%
#檔案匯入
df_2011 = pd.read_csv("gt_2011.csv") #2011年資料
df_2012 = pd.read_csv("gt_2012.csv") #2012年資料
df_2013 = pd.read_csv("gt_2013.csv") #2013年資料
df_2014 = pd.read_csv("gt_2014.csv") #2014年資料
df_2015 = pd.read_csv("gt_2015.csv") #2015年資料

#新增年分欄位
df_2011.insert(0,'year','2011', allow_duplicates = False)
df_2012.insert(0,'year','2012', allow_duplicates = False)
df_2013.insert(0,'year','2013', allow_duplicates = False)
df_2014.insert(0,'year','2014', allow_duplicates = False)
df_2015.insert(0,'year','2015', allow_duplicates = False)

#新增序列
df_2011['index'] = [i+1 for i in range(len(df_2011))]
df_2012['index'] = [i+1 for i in range(len(df_2012))]
df_2013['index'] = [i+1 for i in range(len(df_2013))]
df_2014['index'] = [i+1 for i in range(len(df_2014))]
df_2015['index'] = [i+1 for i in range(len(df_2015))]

#將2011-2015資料創建在一起
data = df_2011.append(df_2012)
data = data.append(df_2013)
data = data.append(df_2014)
data = data.append(df_2015)

#將index重新設定
data.reset_index(inplace = True, drop = True)
#%%
#可視化-時序圖
values = data.values
groups = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]
plt.figure(figsize = (30, 20))
i = 1
for group in groups:
    plt.subplot(len(groups), 1, i)
    plt.plot(values[:, group])
    plt.title(data.columns[group], y = 0.5, loc = 'right')
    i += 1
plt.show()
#%%
#將序列轉換成監督式學習
def series_to_supervised(data, n_in = 1, n_out = 1, dropnan = True):
    n_vars = 1 if type(data) is list else data.shape[1]
    df = pd.DataFrame(data)
    cols, names = list(), list()
    #輸入序列(t-n, ..., t-1)
    for i in range(n_in, 0, -1):
        cols.append(df.shift(i))
        names += [('var%d(t-%d)' % (j+1, i)) for j in range(n_vars)]
    #預測序列(t, t+1, ..., t+n)
    for i in range(0, n_out):
        cols.append(df.shift(-i))
        if i == 0:
            names += [('var%d(t)' % (j+1)) for j in range(n_vars)]
        else:
            names += [('var%d(t+%d)' % (j+1, i)) for j in range(n_vars)]
    #整合在一起
    agg = pd.concat(cols, axis = 1)
    agg.columns = names
    #刪除那些包含空值(NaN)的行
    if dropnan:
        agg.dropna(inplace = True)
    return agg

#設置預測未來1小時排放
n_hours = 1
#將時間序列轉換成監督式學習數據集
data_reframed = series_to_supervised(data, n_hours, 1)
#%%
#資料分割
train_size = int(len(data[data['year'] <= year])) #訓練資料筆數
test_size = len(data) - train_size  #測試資料筆數
data_reframed.drop(data_reframed.columns[[0, 10, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 25]],
                   axis = 1,
                   inplace = True)
train, test = data_reframed[0:train_size], data_reframed[train_size:len(data_reframed)] #資料分割 train、test
#%%
#分離出特徵集與標籤
train_X, train_Y = train[train.columns[:-1]], train[train.columns[-1:]]
test_X, test_Y = test[test.columns[:-1]], test[test.columns[-1:]]
#%%
#特徵選取 Mutual information
#MI score > 0.3
from sklearn.feature_selection import mutual_info_regression as MIR
mi_score = MIR(train_X, train_Y)
mi_score_selected_index = np.where(mi_score > 0.2)[0]

train_X_2 = train_X[train_X.columns[mi_score_selected_index]]
test_X_2 = test_X[train_X.columns[mi_score_selected_index]]
#%%
#數據正規化
scaler_X = MinMaxScaler(feature_range = (-1, 1))
scaler_Y = MinMaxScaler(feature_range = (-1, 1))

train_X_scaler = scaler_X.fit_transform(train_X_2)
train_Y_scaler = scaler_Y.fit_transform(train_Y)

test_X_scaler = scaler_X.fit_transform(test_X_2)
test_Y_scaler = scaler_Y.fit_transform(test_Y)

#轉換成3維數據組 [樣本數, 時間步數, 特徵數]
train_X_scaler = np.reshape(train_X_scaler, (train_X_scaler.shape[0], 1, train_X_scaler.shape[1]))
test_X_scaler = np.reshape(test_X_scaler, (test_X_scaler.shape[0], 1, test_X_scaler.shape[1]))

#%%
# Import the Keras libraries and packages
import keras
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers import Dropout, BatchNormalization
from keras.layers import TimeDistributed
from keras.callbacks import EarlyStopping
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold


def LSTM_model(n_cells):  
    #創建模型
    # Initialising the LSTM
    model = Sequential()
    
    # Adding the first LSTM layer and some Dropout regularisation
    model.add(LSTM(n_cells,
                   input_shape = (train_X_scaler.shape[1], train_X_scaler.shape[2]),
                   return_sequences = True))
    
    # Adding a second LSTM layer and some Dropout regularisation
    model.add(LSTM(n_cells, return_sequences = True))
    
    # Adding the output layer
    model.add(Dense(1))
    
    # Compiling
    model.compile(optimizer = 'adam', loss = 'mean_squared_error')
    
    return model
#%%
from keras.wrappers.scikit_learn import KerasRegressor

def LSTM_cv(n_cells, n_batch, epochs):
    regressor = KerasRegressor(build_fn = LSTM_model,
                               n_cells = int(n_cells),
                               epochs = epochs,
                               verbose = True,
                               shuffle = False,
                               batch_size = int(n_batch),
                               validation_data = (test_X_scaler, test_Y_scaler))

    #交叉驗證
    val = cross_val_score(estimator = regressor, 
                          X = train_X_scaler, 
                          y = train_Y_scaler, 
                          cv = 10,
                          n_jobs = 1,
                          scoring = 'neg_mean_squared_error').mean()
    return val
#%%
n_cells = 300
n_batch = 64
epochs = 300

model = LSTM_model(int(n_cells))

# 進行訓練
history = model.fit(train_X_scaler, train_Y_scaler,
                    epochs = epochs,
                    batch_size = int(n_batch),
                    validation_data = (test_X_scaler, test_Y_scaler),
                    verbose = True,
                    shuffle = False)

plt.title('loss_function')
plt.ylabel('loss')
plt.xlabel('Epoch')
plt.plot(history.history['loss'], label = 'train')
plt.plot(history.history['val_loss'], label = 'test')
plt.legend()
plt.show()
#%%
#對測試集進行預測
predicted = model.predict(test_X_scaler)
predicted = predicted.reshape(predicted.shape[0], predicted.shape[2])
predicted = scaler_Y.inverse_transform(predicted)

test_Y.reset_index(inplace = True, drop = True)
#%%
#視覺化結果
# Visualising the results
plt.plot(test_Y, color = 'red', label = 'Real')  # 紅線表示實際值
plt.plot(predicted, color = 'blue', label = 'Predicted')  # 藍線表示預測值
plt.title('Prediction')
plt.xlabel('Time')
plt.ylabel(target)
plt.legend()
plt.show()
#%%
#模型評估
r2 = r2_score(test_Y, predicted)
mse = metrics.mean_squared_error(test_Y, predicted)
mae = metrics.mean_absolute_error(test_Y, predicted)
rmse = np.sqrt(metrics.mean_squared_error(test_Y, predicted))

print(str(year) + ' ' + str(target) + ' LSTM 模型評估 test')
print('R2 score: %.4f' % r2)
print('MSE score: %.4f' % mse)
print('MAE score: %.4f' % mae)
print('RMSE score: %.4f' % rmse)