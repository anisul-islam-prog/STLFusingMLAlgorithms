from sklearn.svm import SVR
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import style
import numpy as np
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error
from math import sqrt
import math
from sklearn import preprocessing
import datetime


#Initializing SVR Kernels
clf= SVR(kernel='linear', C=1.0, epsilon=0.2)



#Reading Data
df =pd.read_csv(r'C:\Users\FAHAD\PycharmProjects\Thesis\mainconcat_v100.csv',
                index_col=0,
                parse_dates=True)
df = df.drop(df[(df.Total_power_mainMeter < 2.5) | (df.Total_power_mainMeter > 14) ].index)
df.dropna(inplace=True)


#Declaring forcast variables



#Dividing into Train and Test Set
X = np.array(df.drop(['Total_power_mainMeter'], 1))
X = preprocessing.scale(X)
df.dropna(inplace=True)
y = np.array(df['Total_power_mainMeter'])
tscv = TimeSeriesSplit(n_splits=3)
for train_index, test_index in tscv.split(X):
    X_train, X_test = X[train_index], X[test_index]
    y_train, y_test = y[train_index], y[test_index]

#Trainnig a Model
ClflinR = clf.fit(X_train, y_train)


#Testing a Model
y_predict_lin = ClflinR.predict(X_test)
msr=mean_squared_error(y_test, y_predict_lin)
mar=mean_absolute_error(y_test, y_predict_lin)
rms_lin = sqrt(msr)
print('_score_')
print(ClflinR.score(X_test,y_test))
print('_Mean_Absolute_Error')
print(mar)
print('_Mean_Squared_Error')
print(msr)
print('_RMS_linear_')
print(rms_lin)

#plotting Graph
twoday_forecast_set=ClflinR.predict(X[-5:])
twoday_y=y[-5:]
twoday_X=df.index.values
twoday_X=twoday_X[-5:]
from matplotlib.dates import date2num
x = [datetime.datetime(2017, 4, 20),
     datetime.datetime(2017, 4, 21),
     datetime.datetime(2017, 4, 22),
     datetime.datetime(2017, 4, 23),
     datetime.datetime(2017, 4, 24),
    ]
x = date2num(x)
plt.figure(figsize=(12,7))
ax = plt.subplot()
w = 0.2
ax.bar(x-w, twoday_y,width=w,label='True Power Consumption')
ax.bar(x, twoday_forecast_set,width=w,label='Predicted Power Consumption')
ax.autoscale()
plt.xlabel('Date')
plt.ylabel('Power in KWH')
plt.title('Support Vector Machine Regression(Kernel=linear)')
plt.xticks(x-w/2, ('2017-04-20', '2017-04-21','2017-04-22', '2017-04-23','2017-04-24'))
plt.legend()
plt.show()

data = pd.DataFrame(index=df.index[-200:])
data['Support Vector Machine Regression(Kernel=linear)']= pd.DataFrame(data =y_predict_lin[-200:],  index=df.index[-200:])
data.to_csv(path_or_buf='Support Vector Machine Regression(Kernel=linear) output.csv' , sep=',' )