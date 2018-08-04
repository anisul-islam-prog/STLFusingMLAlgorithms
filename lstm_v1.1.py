import numpy as np 
import pandas as pd
from keras.layers.core import Dense, Activation, Dropout
from keras.layers.recurrent import LSTM
from keras.models import Sequential
from sklearn.cross_validation import  train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt
from matplotlib import style
from matplotlib.pylab import rcParams
import math

# to not display the warnings of tensorflow
import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'


look_back = 7
epochs = 100
batch_size = 32

np.random.seed(7)


#Reading Data
df =pd.read_csv(r'C:\Users\FAHAD\PycharmProjects\Thesis\mainconcat_v100.csv',
                index_col=0,
                parse_dates=True)
df = df.drop(df[(df.Total_power_mainMeter < 2.5) | (df.Total_power_mainMeter > 14) ].index)
df.dropna(inplace=True)

df = df.astype('float32')

d = df.as_matrix()

scaler = MinMaxScaler(feature_range=(0, 1))
d = scaler.fit_transform(d)


train_size = int(len(d) * 0.85)
test_size = len(d) - train_size
train, test = d[0:train_size,:], d[train_size:len(d),:]

print('Split data into training set and test set... Number of training samples/ test samples:', len(train), len(test))



def create_dataset(dataset, look_back):  
    dataX, dataY = [], []
    for i in range(len(dataset)-look_back-1):  # last index - lookback
        a = dataset[i:(i+look_back), 0:52] 
        dataX.append(a)
        dataY.append(dataset[i + look_back, 52]) 
    return np.array(dataX), np.array(dataY)

# convert Apple's stock price data into time series dataset
trainX, trainY = create_dataset(train, look_back)
testX, testY = create_dataset(test, look_back)



trainY



model = Sequential()
model.add(LSTM(4, input_shape=(look_back, 52)))
model.add(Dense(1))
model.compile(loss='mse', optimizer='adam')
model.fit(trainX, trainY, nb_epoch=epochs, batch_size=batch_size)



trainPredict = model.predict(trainX)
testPredict = model.predict(testX)




trainPredict.shape




testPredict.shape




# invert predictions and targets to unscaled
#trainPredict = scaler.inverse_transform(trainPredict)
#trainY = scaler.inverse_transform(trainY)
#testPredict = scaler.inverse_transform(testPredict)
#testY = scaler.inverse_transform(testY)





trainScore = math.sqrt(mean_squared_error(trainY, trainPredict[:,0]))
print('Train Score: %.2f RMSE' % (trainScore))
testScore = math.sqrt(mean_squared_error(testY, testPredict[:,0]))
print('Test Score: %.2f RMSE' % (testScore))




trainPredictPlot = np.empty([len(d),1]) 
trainPredictPlot[:, :] = np.nan
trainPredictPlot[look_back: len(trainPredict)+look_back, :] = trainPredict

# shift predictions of test data for plotting
testPredictPlot = np.empty([len(d),1])
testPredictPlot[:, :] = np.nan
testPredictPlot[len(trainPredict)+(look_back*2)+1:len(d)-1, :] = testPredict




# plot baseline and predictions
rcParams['figure.figsize'] = 15, 10
style.use('ggplot')
plt.plot((d[:,52].reshape(-1, 1)), color="blue" , label="original")
plt.plot(trainPredictPlot, color="green" , label="train")
plt.plot(testPredictPlot, color="yellow", label="Predicted")
plt.legend(loc=2)
plt.show()

