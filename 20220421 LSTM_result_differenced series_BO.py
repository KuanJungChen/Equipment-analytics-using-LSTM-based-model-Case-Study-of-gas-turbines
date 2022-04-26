# -*- coding: utf-8 -*-
"""
Created on Thu Feb 10 14:53:14 2022

@author: User
"""

#%%
from pandas import DataFrame
from pandas import Series
from pandas import concat
from pandas import read_csv
from sklearn.model_selection import cross_val_score
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import r2_score
from sklearn import metrics
from keras.models import Sequential
from keras.wrappers.scikit_learn import KerasRegressor
from keras.layers import Dense
from keras.layers import LSTM
from math import sqrt
from matplotlib import pyplot
import numpy
from statsmodels.tsa.stattools import kpss

#%%
#frame a sequence as a supervised learning problem
def timeseries_to_supervised(data, lag = 1):
    df = DataFrame(data)
    columns = [df.shift(i) for i in range(1, lag+1)]
    columns.append(df)
    df = concat(columns, axis = 1)
    df.fillna(0, inplace = True)
    return df

#create a differenced series
def difference(dataset, interval = 1):
    diff = list()
    for i in range(interval, len(dataset)):
        value = dataset[i] - dataset[i - interval]
        diff.append(value)
    return Series(diff)

#invert differenced value
def inverse_difference(history, yhat, interval = 1):
    return yhat + history[-interval]

#KPSS檢驗
#define KPSS
def kpss_test(timeseries):
    print('Results of KPSS Test: ')
    kpsstest = kpss(timeseries, regression = 'c')
    kpss_output = Series(kpsstest[0:3], index = ['Test Statistic', 'p-value', 'Lags Used'])
    for key, value in kpsstest[3].items():
        kpss_output['Critical Value (%s)' %key] = value
    print(kpss_output)


#將序列轉換成監督式學習
def series_to_supervised(data, n_in = 1, n_out = 1, dropnan = True):
    n_vars = 1 if type(data) is list else data.shape[1]
    df = DataFrame(data)
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
    agg = concat(cols, axis = 1)
    agg.columns = names
    #刪除那些包含空值(NaN)的行
    if dropnan:
        agg.dropna(inplace = True)
    return agg


#scale train and test data to [-1, 1]
def scale(train, test):
    #fit scaler
    scaler = MinMaxScaler(feature_range = (-1, 1))
    scaler = scaler.fit(train)
    #transform train
    train = train.reshape(train.shape[0], train.shape[1])
    train_scaled = scaler.transform(train)
    #transform test
    test = test.reshape(test.shape[0], test.shape[1])
    test_scaled = scaler.transform(test)
    return scaler, train_scaled, test_scaled

#inverse scaling for a forecasted value
def invert_scale(scaler, X, yhat):
    new_row = [x for x in X] + [yhat]
    array = numpy.array(new_row)
    array = array.reshape(1, len(array))
    inverted = scaler.inverse_transform(array)
    return inverted[0, -1]

# #fit an LSTM network to training data
# def fit_lstm(train, batch_size, nb_epoch, neurons):
#     X, y = train[:, 0:-1], train[:, -1]
#     X = X.reshape(X.shape[0], 1, X.shape[1])
#     model = Sequential()
#     model.add(LSTM(neurons, batch_input_shape = (batch_size, X.shape[1], X.shape[2]), stateful = True))
#     model.add(Dense(1))
#     model.compile(loss = 'mean_squared_error', optimizer = 'adam')
#     for i in range(nb_epoch):
#         model.fit(X, y, epochs = 1, batch_size = batch_size, verbose = 0, shuffle = False)
#     return model

#fit an LSTM network to training data
def fit_lstm(train, batch_size, epochs, neurons):
    X, y = train[:, 0:-1], train[:, -1]
    X = X.reshape(X.shape[0], 1, X.shape[1])
    model = Sequential()
    model.add(LSTM(neurons,
                   input_shape = (X.shape[1], X.shape[2]),
                   return_sequences = True))
    model.add(Dense(1))
    model.compile(loss = 'mean_squared_error', optimizer = 'adam')
    model.fit(X, y, epochs = epochs, batch_size = batch_size, verbose = 0, shuffle = False)
    return model

#make a one-step forecast
def forecast_lstm(model, batch_size, X):
    X = X.reshape(1, 1, len(X))
    yhat = model.predict(X, batch_size = batch_size)
    return yhat[0, 0]

#%%
#load dataset
#檔案匯入
df_2011 = read_csv("gt_2011.csv") #2011年資料
df_2012 = read_csv("gt_2012.csv") #2012年資料
df_2013 = read_csv("gt_2013.csv") #2013年資料
df_2014 = read_csv("gt_2014.csv") #2014年資料
df_2015 = read_csv("gt_2015.csv") #2015年資料

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
series = df_2011.append(df_2012)
series = series.append(df_2013)
series = series.append(df_2014)
series = series.append(df_2015)

#平穩性檢測
kpss_test(series['NOX'])

#%%
#將index重新設定
series.reset_index(inplace = True, drop = True)

#transform data to be stationary
raw_values = series[['CO']].values
diff_values = difference(raw_values, 1)

#transform data to be supervised learning
supervised = timeseries_to_supervised(diff_values, 1)

#資料分割
train_size = int(len(series[series['year'] <= '2011'])) #訓練資料筆數
test_size = len(series) - train_size  #測試資料筆數

series.drop(series.columns[[0, 10, 11, 12]],
            axis = 1,
            inplace = True)
#將index重新設定
series.reset_index(inplace = True, drop = True)
data_reframed = concat([series, supervised.iloc[:, [1]]], axis = 1)

data_reframed_values = data_reframed.values

train, test = data_reframed[0:train_size], data_reframed[train_size:len(data_reframed)] #資料分割 train、test
train = train.values
test = test.values

#%%
#特徵選取 Mutual information
#MI score > 0.03
from sklearn.feature_selection import mutual_info_regression as MIR
mi_score = MIR(train[:, 0:-1], train[:, -1])
mi_score_selected_index = numpy.where(mi_score > 0.05)[0]

mi_score_selected_index = numpy.append(mi_score_selected_index, -1)
train = train[:, mi_score_selected_index]
test = test[:, mi_score_selected_index]

#transform the scale of the data
scaler, train_scaled, test_scaled = scale(train, test)

#%%
#fit the model
lstm_model = fit_lstm(train_scaled, 32, 150, 150)

#forecast the entire training dataset to build up state for forecasting
train_reshaped =train_scaled[:, 0:-1].reshape(train_scaled.shape[0], 1, train_scaled.shape[1]-1)
lstm_model.predict(train_reshaped, batch_size = 1)

#%%
#walk-forward validation on the test data
test_Y = list()
predictions = list()
for i in range(len(test_scaled)-1):
    #make one-step forecast
    X, y = test_scaled[i, 0:-1], test_scaled[i, -1]
    yhat = forecast_lstm(lstm_model, 1, X)
    #invert scaling
    yhat = invert_scale(scaler, X, yhat)
    #invert differencing
    yhat = inverse_difference(raw_values, yhat, len(test_scaled) + 1 - i)
    #store forecast 
    predictions.append(yhat)
    expected = raw_values[len(train) + i + 1]
    test_Y.append(expected)
    #print('Predicted = %f, Expected = %f' % (yhat, expected))
    
#%%
#貝葉斯優化(Bayesian Optimization)
from bayes_opt import BayesianOptimization
from hyperopt import hp

#建立貝葉斯優化對象
lstm_bo = BayesianOptimization(LSTM_cv,
                               {'n_cells':(100, 300),                                
                                'n_batch':(2, 64),
                                'epochs':(200, 500)})

#開始優化
lstm_bo_result = lstm_bo.maximize()

#尋找最大值
lstm_bo_bestresult = lstm_bo.max


n_cells = lstm_bo_bestresult['params']['n_cells']
epochs = lstm_bo_bestresult['params']['epochs']
n_batch = lstm_bo_bestresult['params']['n_batch']

#%%
#模型評估
r2 = r2_score(test_Y, predictions)
mse = metrics.mean_squared_error(test_Y, predictions)
mae = metrics.mean_absolute_error(test_Y, predictions)
rmse = numpy.sqrt(metrics.mean_squared_error(test_Y, predictions))

print('LSTM 模型評估 test')
print('R2 score: %.4f' % r2)
print('MSE score: %.4f' % mse)
print('MAE score: %.4f' % mae)
print('RMSE score: %.4f' % rmse)

#line plot of observed vs predicted
pyplot.plot(test_Y)
pyplot.plot(predictions)
pyplot.show()




























































































